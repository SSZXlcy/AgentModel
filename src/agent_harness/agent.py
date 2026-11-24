"""The LLM side. Every call has a deterministic fallback, and every answer is checked against
the same whitelists the fallback obeys. The LLM can make a decision better; it cannot make the
system crash or step outside its bounds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Protocol

from . import policy as rules
from . import slots as slot_rules
from .contracts import DriftJudgment


class LLMProvider(Protocol):
    model: str

    def generate_json(self, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]: ...


KEY_FILE = Path.home() / ".agent_harness" / "llm_key"


def _read_key() -> str:
    """LLM_API_KEY, else the user's private key file. The key never enters the repository,
    the command line, or any record."""
    if os.getenv("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"].strip()
    path = Path(os.getenv("LLM_API_KEY_FILE", str(KEY_FILE)))
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


class OpenAICompatibleProvider:
    """Chat-completions endpoint (Qwen3.8-27B by default). Returns (json, tokens used)."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: int = 180):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL_ID", "qwen3.8-27b")
        self.api_key = api_key or _read_key()
        self.timeout = timeout

    def generate_json(self, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        return self.chat_json([{"role": "system", "content": system},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])

    def chat_json(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int]:
        """Multi-turn call; the last reply must be one JSON object."""
        if not self.api_key:
            raise RuntimeError("no LLM key: set LLM_API_KEY or create ~/.agent_harness/llm_key")
        body = {"model": self.model, "temperature": 0.1, "response_format": {"type": "json_object"},
                "messages": messages}
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"] or ""
        tokens = int((result.get("usage") or {}).get("total_tokens") or (len(content) + len(json.dumps(body))) // 4)
        return parse_json_object(content), tokens


def parse_json_object(content: str) -> dict[str, Any]:
    """The first JSON object in a reply; tolerates <think> blocks and ``` fences."""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    start = content.find("{")
    if start < 0:
        raise ValueError("LLM reply contains no JSON object")
    parsed, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


_FAILURES = (RuntimeError, ValueError, KeyError, TypeError, TimeoutError, OSError,
             urllib.error.URLError, json.JSONDecodeError)

_DRIFT_SYSTEM = (
    "You diagnose distribution drift for an industrial model that is retrained on a data stream. "
    "From the evidence, decide which kind of drift this window shows and which training data to use. "
    "drift_type: covariate (inputs moved), concept (the input->label relation changed), both, none. "
    "recipe: 'all' (initial training set + every labelled window), 'adaptive' (recent windows until "
    "enough samples), or 'recent:N' with N in 1..3 (only the last N windows). Less data is cheaper; "
    "older data helps when samples are scarce and hurts when the relation changed. "
    "Return JSON: {\"drift_type\":..., \"recipe\":..., \"confidence\":0..1, \"rationale\":\"one sentence\"}.")

_SLOT_SYSTEM = (
    "You improve the TRAINING CODE of a small single-hidden-layer NumPy MLP whose architecture is fixed. "
    "You may replace whole functions only, and only among the slots listed in `allowed_slots`; "
    "signatures are fixed (see `signatures`). Only `np` (numpy) and `math` are available; no I/O. "
    "Ground every edit in the evidence and name the observation it addresses. Prefer one well-argued "
    "edit over several weak ones. A parallel proposer owns the other slot group: do not touch it. "
    "Return JSON: {\"edits\": [{\"slot\": name, \"hypothesis\": text, \"source\": \"def name(...):\\n    ...\"}]} "
    "with at most `edit_budget` edits, or {\"edits\": []} to decline.")

_DISTILL_SYSTEM = (
    "You write one entry of an experience library from code-edit trials. The trials list every "
    "candidate with its measured paired difference and noise band; whether a candidate was adopted is "
    "deliberately hidden: adoption shows which version scored best, not which edit caused it. "
    "Prefer 'confounded' to an invented attribution. Return JSON {\"edits\": [{\"slot\", \"change\", "
    "\"observed\", \"verdict\": helped|hurt|no observed effect|confounded}], \"experience\": {\"cue\", "
    "\"lesson\", \"evidence\", \"confidence\"}} or {\"edits\": [...], \"experience\": \"NO_ENTRY\"}.")


_REVIEW_SYSTEM = (
    "You supervise a self-improving training system between windows of a data stream. You get the health "
    "checks of recent windows, the triggers that fired, the current strategy and the latest journal. Decide "
    "whether to keep the current strategy or to start a full exploration (which costs tokens and training work). "
    "Keep it unless the evidence shows the strategy no longer fits. Return JSON {\"decision\": \"continue\"|\"explore\", "
    "\"why\": \"one sentence\"}.")

_SUMMARY_SYSTEM = (
    "You write the summary of one improvement round for the journal, the experience library and the final report. "
    "You get the situation signature, what was explored (tool calls) and every validation experiment: the operator "
    "call, its paired difference, noise band and whether it passed validation. Whether it was finally adopted is "
    "deliberately not shown: passing says which arm scored better, not why. Prefer saying an effect is unclear to "
    "inventing a cause. Lessons must be transferable (no task or field names: say 'entity', not a field name), each "
    "{\"cue\": when it applies (in terms of signature features), \"when\": the same condition as a structured test on "
    "signature features, e.g. {\"has_entity\": true, \"entity_imbalance\": {\">\": 3}} or {\"label_timeliness\": {\"<\": 0.5}} "
    "(operators > < >= <= != in; {\"any\": [...]} for alternatives), \"lesson\": what to do, \"action\": {\"operator\": name, "
    "\"params\": {...using placeholders such as $entity}} when the lesson maps to one operator call, \"evidence\": the "
    "numbers it rests on, \"confidence\": high|medium|low}; return an empty list when nothing reliable was learned. Return JSON "
    "{\"lessons\": [...], \"next_time\": \"what to check first next round\", \"report_paragraph\": \"one paragraph in "
    "Chinese for the experiment report: trigger, what was found, what was tried, measured result, what changed\"}.")


_REPAIR_SYSTEM = (
    "A training-slot function you wrote for an official NumPy MLP trainer failed the official test-suite. Fix it. "
    "Rules: keep the exact signature; it must work for BOTH task types the tests exercise — classification "
    "(`target` holds integer class indices, n_classes > 0) and regression (`target` holds float values, n_classes == 0); "
    "return None or float64 arrays of the documented shapes; only `np` (numpy) and `math`; helpers named _x and "
    "UPPERCASE constants are allowed; no I/O. Return JSON {\"source\": \"<the complete fixed function, helpers first>\"}.")


class Agent:
    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider
        self.calls = 0
        self.successes = 0
        self.fallbacks = 0
        self.tokens = 0

    def _ask(self, system: str, payload: dict) -> tuple[dict | None, int]:
        if self.provider is None:
            return None, 0
        self.calls += 1
        try:
            answer, tokens = self.provider.generate_json(system, payload)
            self.tokens += tokens
            return answer, tokens
        except _FAILURES:
            self.fallbacks += 1
            return None, 0

    # ------------------------------------------------------------ inner layer
    def judge_drift(self, evidence: dict) -> DriftJudgment:
        fallback = rules.rule_judgment(evidence)
        answer, tokens = self._ask(_DRIFT_SYSTEM, evidence)
        if answer is None:
            return fallback
        drift_type, recipe = answer.get("drift_type"), answer.get("recipe")
        try:
            confidence = min(1.0, max(0.0, float(answer.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = -1.0
        if drift_type not in rules.DRIFT_TYPES or recipe not in rules.RECIPES or confidence < 0:
            self.fallbacks += 1
            return DriftJudgment(fallback.drift_type, fallback.recipe, fallback.confidence,
                                 fallback.rationale + " (LLM answer out of bounds)", "rule", tokens)
        self.successes += 1
        return DriftJudgment(drift_type, recipe, confidence, str(answer.get("rationale", ""))[:300], "llm", tokens)

    # ------------------------------------------------------------ outer layer
    def propose_edits(self, group: str, budget: int, task_type: str, evidence: dict, current_source: str,
                      library_text: str, tried: list[dict]) -> tuple[list[dict], str, int]:
        """(edits, source, tokens). Falls back to the local variant catalogue."""
        allowed = slot_rules.GROUPS[group]
        payload = {"group": group, "allowed_slots": list(allowed), "edit_budget": budget, "task_type": task_type,
                   "signatures": {k: list(v) for k, v in slot_rules.SIGNATURES.items() if k in allowed},
                   "current_slots_py": current_source, "evidence": evidence,
                   "experience_library": library_text, "already_tried": tried[-20:]}
        answer, tokens = self._ask(_SLOT_SYSTEM, payload)
        if answer is not None:
            edits = answer.get("edits")
            if isinstance(edits, list) and len(edits) <= budget and all(
                    isinstance(e, dict) and e.get("slot") in allowed and isinstance(e.get("source"), str) for e in edits):
                try:
                    slot_rules.apply_edits(current_source, edits)
                    self.successes += 1
                    return [{**e, "variant": "llm"} for e in edits], "llm", tokens
                except (SyntaxError, ValueError):
                    pass
            self.fallbacks += 1
        return local_proposal(group, budget, task_type, evidence, tried), "local", tokens

    def distill(self, trials: list[dict], chosen: dict, evidence: dict) -> tuple[dict, int]:
        """Experience for the library. {} means NO_ENTRY."""
        answer, tokens = self._ask(_DISTILL_SYSTEM, {"trials": trials, "evidence": evidence})
        if answer is not None:
            experience = answer.get("experience")
            if isinstance(experience, dict) and all(isinstance(experience.get(k), str) and experience[k].strip()
                                                    for k in ("cue", "lesson", "evidence")):
                self.successes += 1
                return {**experience, "edits": answer.get("edits", [])}, tokens
            if experience == "NO_ENTRY":
                self.successes += 1
                return {}, tokens
            self.fallbacks += 1
        return local_distill(trials, chosen, evidence), tokens

    # ------------------------------------------------------------ supervisor
    def review(self, payload: dict) -> tuple[str | None, str, int]:
        """Light review: one call, answer continue / explore. (None, ...) means use the rule."""
        answer, tokens = self._ask(_REVIEW_SYSTEM, payload)
        if answer is not None and answer.get("decision") in ("continue", "explore"):
            self.successes += 1
            return answer["decision"], str(answer.get("why", ""))[:300], tokens
        if answer is not None:
            self.fallbacks += 1
        return None, "", tokens

    def summarize_round(self, payload: dict) -> tuple[dict | None, int]:
        """Round summary for the journal, the library and the report. None means use the template."""
        answer, tokens = self._ask(_SUMMARY_SYSTEM, payload)
        if answer is not None and isinstance(answer.get("report_paragraph"), str) and isinstance(answer.get("lessons", []), list):
            self.successes += 1
            return {"lessons": [x for x in answer.get("lessons", []) if isinstance(x, dict)],
                    "next_time": str(answer.get("next_time", ""))[:500],
                    "report_paragraph": answer["report_paragraph"][:2000]}, tokens
        if answer is not None:
            self.fallbacks += 1
        return None, tokens

    def repair_slot(self, slot: str, source: str, failure: str) -> tuple[str | None, int]:
        """One repair attempt for an LLM-written training slot that failed the official tests."""
        payload = {"slot": slot, "signature": list(slot_rules.SIGNATURES[slot]), "source": source,
                   "official_test_failure": failure[-2500:]}
        answer, tokens = self._ask(_REPAIR_SYSTEM, payload)
        if answer is not None and isinstance(answer.get("source"), str) and answer["source"].strip():
            self.successes += 1
            return answer["source"], tokens
        if answer is not None:
            self.fallbacks += 1
        return None, tokens

    def status(self) -> dict:
        return {"provider": getattr(self.provider, "model", "local"), "calls": self.calls,
                "successes": self.successes, "fallbacks": self.fallbacks, "tokens": self.tokens}


# ---------------------------------------------------------------- deterministic fallbacks

def local_proposal(group: str, budget: int, task_type: str, evidence: dict, tried: list[dict]) -> list[dict]:
    """Untried catalogue variants, ordered by what the evidence points at; distinct slots only."""
    # An edit counts as tried once it was measured alone, or once a bundle containing it fell far
    # short of the band; a near-miss bundle leaves its edits open for single trials.
    done = {edit.get("variant") for row in tried for edit in row.get("edits", [])
            if len(row.get("edits", [])) == 1 or row.get("retired") or "retired" not in row}
    pool = [v for v in slot_rules.variants_for(task_type, group) if v.name not in done]
    errors = evidence.get("errors", {})

    def priority(variant: slot_rules.Variant) -> int:
        if variant.name == "sqrt_inverse_class_frequency" and errors.get("class_imbalance", 1.0) > 2.0:
            return 0
        if variant.name == "asymmetric_overestimate" and errors.get("over_estimate_ratio", 0.5) > 0.55:
            return 0
        if variant.name == "l2_1e-3" and evidence.get("diagnoses", {}).get("OVERFIT", 0) > 0:
            return 0
        return 1

    chosen, slots_used = [], set()
    for variant in sorted(pool, key=priority):
        if len(chosen) >= budget:
            break
        if variant.slot not in slots_used:
            chosen.append(variant.edit())
            slots_used.add(variant.slot)
    return chosen


def local_distill(trials: list[dict], chosen: dict, evidence: dict) -> dict:
    """An entry only when attribution is clean: a single-edit candidate that beat the noise band."""
    if len(chosen.get("edits", [])) != 1:
        return {}
    edit = chosen["edits"][0]
    gain, band = chosen["mean_diff"], chosen["band"]
    errors = evidence.get("errors", {})
    cue = f"{evidence.get('task_type')} task; " + ", ".join(f"{k}={v:.3g}" for k, v in errors.items()
                                                              if isinstance(v, (int, float)))
    return {"cue": cue, "lesson": f"{edit['slot']} -> {edit.get('variant')}: {edit.get('hypothesis', '')}",
            "evidence": f"paired next-window utility {gain:+.4f} vs noise band {band:.4f} over "
                        f"{chosen['n']} (seed, window) pairs",
            "confidence": "high" if gain > 2 * band else "medium",
            "edits": [{"slot": edit["slot"], "change": edit.get("variant"), "observed": f"{gain:+.4f}",
                       "verdict": "helped"}]}
