"""LLM explorer (W2): the agent explores with tools, then returns an analysis plan written in the
operator manual's vocabulary.

One JSON object per turn: a tool call, or the final plan. The harness executes tools, enforces the
step and token budgets, and falls back to the deterministic explorer on any failure. Whatever the
plan proposes is executed by the manual (operators.py): checked, then applied or validated.
"""
from __future__ import annotations

import json

from . import operators
from .agent import _FAILURES
from .tools import CATALOGUE

# Kept for callers that validate raw policy fields; the manual is the authority.
BOUNDS = {"z": [1.0, 4.0], "label_fraction": [0.05, 1.0], "gate_fraction": [0.2, 0.8], "max_epochs": [1, 8],
          "max_candidates": [1, 5], "min_samples": [10, 100000], "confirm_windows": [1, 5],
          "cooldown_windows": [0, 5], "cap_per_group": [5, 1000000], "recent_windows_only": [1, 20],
          "health_window": [2, 10], "review_every": [2, 50], "probation_windows": [1, 10],
          "min_gate_units": [2, 50], "lr_scale": [0.05, 2.0]}
ENUMS = {"split_mode": ["random", "by_group", "by_order"], "serve_mode": ["trial", "shadow", "freeze"],
         "branch_from": ["serving", "static"], "gate_mode": ["in_window", "next_window"],
         "train_trigger": ["every_window", "on_completion"], "label_sampling": ["random", "spread"]}
LIST_FIELDS = ["split_group_by", "cap_group_by"]


def system_prompt(max_steps: int, sig: dict) -> str:
    tools = "\n".join(f"  {name}: {doc}" for name, doc in CATALOGUE.items())
    return f"""You are the analysis agent of a controlled self-improving system. It keeps a small industrial model
(the official single-hidden-layer NumPy MLP; architecture fixed, continued from the official checkpoint) good on a
drifting data stream, at the lowest cost in training work, labels and LLM tokens.
You are NOT told what the task is: the situation signature below was computed from the data. Explore with tools if
you need more evidence, then return an analysis plan.

Reply with exactly ONE JSON object per turn, either
  {{"tool": "<name>", "args": {{...}}, "why": "<what you want to learn>"}}
or
  {{"plan": {{"findings": [{{"claim": "...", "evidence": "<which tool result or signature feature>"}}],
             "hypotheses": [{{"id": "H1", "statement": "...", "expected_effect": "..."}}],
             "corrections": [{{"id": "C1", "for": "H1", "operator": "<operator name>", "params": {{...}},
                              "validation": {{"group_by": [...]}}, "why": "...",
                              "experience": "<id, when it follows one>"}}],
             "risks": ["..."]}}}}

Tools:
{tools}

OPERATOR MANUAL — the only way anything changes. Parameters may use the placeholders $entity, $order,
$median_group_size, $median_class_size; the harness fills them from the signature. "validation" is optional
(it may set group_by or seeds, or raise folds / windows; the harness never lets them fall below its defaults).
{operators.prompt_manual(sig)}

How a plan is executed (you cannot bypass it):
- Tightening operators (stricter split, shadow/freeze serving, higher z / min_units / confirm, fewer labels) apply at once.
- Every other operator is validated NOW, in this round, by the experiment the manual prescribes; it applies only if the
  gain exceeds the noise band at acceptable cost. At most one improvement is adopted per round (the best one).
- There may be no later round. Every hypothesis that implies a change MUST appear as a correction in THIS plan —
  never write "test it next round". Proposing several candidate improvements is cheap and safe: the failing ones are
  simply rejected. You may also call the `experiment` tool yourself before deciding.
- Code operators: the slot function must work for BOTH task types the official tests run — classification
  (`target` = integer class indices, n_classes > 0) and regression (`target` = floats, n_classes == 0). Only `np` and
  `math`; helpers named _x and UPPERCASE constants are allowed; no I/O. Prefer a built-in variant when one fits.
- Rhythm operators (mode.*) set how the stream is followed (when to train, where a candidate is judged, how it
  serves, how labels are drawn); they apply at once and the probation that follows judges them. The context gives
  `experiences.recommended_rhythm` (the most credible experience whose condition holds on the signature) and the
  other matching experiences, with their credibility from earlier verdicts. Follow them unless your evidence
  contradicts them, and then say why. For triggers T1 and T10 the plan must contain exactly one mode.* correction.
  A correction that follows an experience carries "experience": "<its id>".
- Only data that already happened is visible. Never assume field names: use the signature and tools.
Budget: at most {max_steps} tool calls.

Situation signature:
{json.dumps(sig, ensure_ascii=False)}"""


def explore(agent, box, context: dict, max_steps: int, max_tokens: int) -> tuple[dict | None, list[dict], int, str]:
    """(plan or None, trace, tokens, note). None means: use the deterministic explorer."""
    provider = agent.provider
    if provider is None or not hasattr(provider, "chat_json"):
        return None, [], 0, "no LLM provider"
    sig = context.get("signature") or {}
    messages = [{"role": "system", "content": system_prompt(max_steps, sig)},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)}]
    trace, tokens, invalid = [], 0, 0
    for step in range(max_steps + 2):
        force = step >= max_steps or tokens >= max_tokens
        if force:
            messages.append({"role": "user", "content": "Budget exhausted: return {\"plan\": {...}} now."})
        agent.calls += 1
        try:
            reply, used = provider.chat_json(messages)
        except _FAILURES as error:
            agent.fallbacks += 1
            return None, trace, tokens, f"provider failed: {type(error).__name__}: {str(error)[:200]}"
        tokens += used
        agent.tokens += used
        messages.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
        if isinstance(reply.get("plan"), dict):
            agent.successes += 1
            return reply["plan"], trace, tokens, "ok"
        if not force and isinstance(reply.get("tool"), str) and isinstance(reply.get("args", {}), dict):
            result = box.call(reply["tool"], reply.get("args", {}))
            trace.append({"tool": reply["tool"], "args": reply.get("args", {}), "why": reply.get("why", ""),
                          "result": result})
            messages.append({"role": "user", "content": json.dumps({"tool_result": result}, ensure_ascii=False,
                                                                   default=str)})
            continue
        invalid += 1
        if invalid > 2 or force:
            agent.fallbacks += 1
            return None, trace, tokens, "invalid replies"
        messages.append({"role": "user", "content": "Reply with one JSON object: a tool call or {\"plan\": ...}."})
    agent.fallbacks += 1
    return None, trace, tokens, "no plan returned"
