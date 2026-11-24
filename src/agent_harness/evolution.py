"""Outer layer (策略迭代层): evolve the code θ and the policy π, offline, under four gates.

Code round t:
  1. edit budget b_t anneals from b_max to b_min (rrsi schedule); a stall widens it again
  2. OPT and PHM groups are proposed independently (LLM, or the local catalogue)
  3. each candidate: slot boundary check → official tests in a subprocess → replay_code
  4. paired judgment over (seed, window): gain beyond the band, no regressed seed, cost paid by gain
  5. if both groups helped, their union is tried as a composed candidate
  6. the best admissible candidate becomes the incumbent; experience is distilled blind to the verdict
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path

import yaml

from . import codebase, sandbox
from . import slots as slot_rules
from .agent import Agent
from .codebase import CodeVersion
from .config import PolicyConfig, TaskConfig
from .contracts import digest, json_ready
from .controller import Session
from .library import LeakError, Library
from .recorder import read_jsonl, write_json
from .replay import paired, replay_code, replay_policy


def edit_budget(t: int, rounds: int, b_min: int, b_max: int) -> int:
    """b_t = ceil(b_min + (b_max − b_min)·½(1 + cos(π t / T))), from rrsi/schedule.py."""
    if rounds <= 0:
        return b_max
    t = max(0, min(t, rounds))
    value = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / rounds))
    return int(math.ceil(round(value, 9)))


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


@dataclass
class Trial:
    label: str
    group: str
    edits: list[dict]
    proposer: str
    version: CodeVersion | None = None
    stage: str = "proposed"
    reason: str = ""
    tests: dict | None = None
    comparison: dict | None = None
    replay: dict | None = None
    admissible: bool = False
    tokens: int = 0
    retired: bool = False          # a multi-edit trial far from the band: its edits count as tried
    step_ratio: float | None = None

    def blind(self) -> dict:
        """What the distiller may see: everything except whether it was adopted."""
        return {"label": self.label, "group": self.group, "stage": self.stage,
                "edits": [{k: e.get(k) for k in ("slot", "variant", "hypothesis")} for e in self.edits],
                "comparison": self.comparison}


class _Artifacts:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.store = self.root / "code"
        self.promoted_path = self.root / "promoted.json"
        self.edits = self.root / "edits.jsonl"
        self.policy_edits = self.root / "policy_edits.jsonl"
        self.replays = self.root / "replays"

    def promoted(self) -> dict:
        if self.promoted_path.is_file():
            return json.loads(self.promoted_path.read_text(encoding="utf-8"))
        return {"code_version": None, "policy": None, "rounds_done": 0, "history": []}

    def save_promoted(self, value: dict) -> None:
        write_json(self.promoted_path, value)


def promoted_code(task: TaskConfig, artifacts: Path) -> CodeVersion:
    """The code version runs should use: the last promoted one, or v0."""
    paths = _Artifacts(artifacts)
    base = codebase.base_version(task.package, paths.store)
    vid = paths.promoted().get("code_version")
    return codebase.open_version(paths.store, vid) if vid else base


def promoted_policy(artifacts: Path, default: PolicyConfig) -> PolicyConfig:
    path = _Artifacts(artifacts).promoted().get("policy")
    if not path:
        return default
    return PolicyConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class CodeEvolver:
    def __init__(self, task: TaskConfig, policy: PolicyConfig, artifacts: Path, *, agent: Agent | None = None,
                 seeds: list[int] = (0, 1, 2), rounds: int = 3, b_min: int = 1, b_max: int = 3,
                 z: float = 2.0, beta0: float = 0.25, beta1: float = 5.0, log=print):
        self.task, self.policy = task, policy
        self.paths = _Artifacts(artifacts)
        self.agent = agent or Agent()
        self.seeds, self.rounds = list(seeds), rounds
        self.b_min, self.b_max = b_min, b_max
        self.z, self.beta0, self.beta1 = z, beta0, beta1
        self.library = Library(self.paths.root / "library", task.guarded_markers)
        self.log = log

    def run(self) -> dict:
        base = codebase.base_version(self.task.package, self.paths.store)
        task_type = base.profile(self.task.profile)["task"]
        promoted = self.paths.promoted()
        incumbent = codebase.open_version(self.paths.store, promoted["code_version"]) if promoted["code_version"] else base
        known = sandbox.reference_failures(base, self.paths.root / "sandbox_reference.json")
        self.log(f"[{self.task.task}] incumbent {incumbent.version_id}; replaying {len(self.seeds)} seeds")
        current = replay_code(self.task, self.policy, incumbent, self.seeds, self.paths.replays)
        tried = read_jsonl(self.paths.edits)
        stall, start = 0, promoted["rounds_done"]
        rounds_log = []
        for t in range(self.rounds):
            global_round = start + t
            self.library.retain(global_round)
            budget = self.b_max if stall >= 2 else edit_budget(t, self.rounds, self.b_min, self.b_max)
            evidence = {"task_type": task_type, "incumbent_utility": current["S"],
                        "per_window_utility": [round(float(sum(c) / len(c)), 4) for c in zip(*current["scored"])],
                        "errors": current["errors"], "train_seconds": current["C"]}
            trials = []
            for group in ("OPT", "PHM"):
                edits, proposer, tokens = self.agent.propose_edits(
                    group, budget, task_type, evidence, incumbent.slots_source(), self.library.prompt_text(), tried)
                trial = Trial(f"r{global_round}-{group.lower()}", group, edits, proposer, tokens=tokens)
                trials.append(self._evaluate(trial, incumbent, current, budget, slot_rules.GROUPS[group], known))
            # A bundle that came close is split into single-edit trials, so a real gain inside it is
            # neither lost nor misattributed; a bundle far from the band retires all its edits.
            for trial in list(trials):
                if len(trial.edits) > 1 and trial.stage == "replayed":
                    if trial.comparison["mean_diff"] > 0.5 * trial.comparison["band"]:
                        for edit in trial.edits:
                            single = Trial(f"{trial.label}-{edit['slot']}", trial.group, [edit], trial.proposer)
                            trials.append(self._evaluate(single, incumbent, current, budget,
                                                         slot_rules.GROUPS[trial.group], known))
                    else:
                        trial.retired = True
            helped = [tr for tr in trials if tr.stage == "replayed" and tr.comparison["mean_diff"] > 0
                      and len(tr.edits) == 1]
            if len(helped) == 2 and helped[0].group != helped[1].group and sum(len(tr.edits) for tr in helped) <= budget:
                composed = Trial(f"r{global_round}-compose", "COMPOSE", helped[0].edits + helped[1].edits, "compose")
                trials.append(self._evaluate(composed, incumbent, current, budget, tuple(slot_rules.SIGNATURES), known))
            for trial in trials:
                if trial.stage == "replayed":
                    trial.admissible, trial.reason = self._judge(trial.comparison, current)
            admissible = [tr for tr in trials if tr.admissible]
            winner = max(admissible, key=lambda tr: tr.comparison["mean_diff"]) if admissible else None
            before = incumbent.version_id
            for trial in trials:
                verdict = ("promoted" if trial is winner else "not_selected" if trial.admissible
                           else trial.reason or trial.stage)
                row = self._log_row(global_round, trial, before, verdict, winner)
                _append(self.paths.edits, row)
                tried.append(row)
            if winner:
                incumbent, current, stall = winner.version, winner.replay, 0
                experience, tokens = self.agent.distill([tr.blind() for tr in trials],
                                                        {**winner.blind(), **winner.comparison}, evidence)
                why = (f"{winner.label}: paired next-window utility {winner.comparison['mean_diff']:+.4f} vs band "
                       f"{winner.comparison['band']:.4f} over {winner.comparison['n']} (seed, window) pairs; "
                       f"training time change {winner.comparison['cost_change']:+.1%}.")
                try:
                    entry = self.library.add(f"r{global_round:03d}-{incumbent.version_id}", global_round,
                                             incumbent.version_id, incumbent.slots_source(), why, experience)
                except LeakError as error:
                    entry = None
                    self.log(f"  library entry refused: {error}")
                promoted["code_version"] = incumbent.version_id
                promoted["history"].append({"round": global_round, "code_version": incumbent.version_id,
                                            "from": before, "trial": winner.label, **winner.comparison,
                                            "library_entry": entry.name if entry else None})
            else:
                stall += 1
            promoted["rounds_done"] = global_round + 1
            self.paths.save_promoted(promoted)
            rounds_log.append({"round": global_round, "budget": budget,
                               "trials": [(tr.label, tr.stage, tr.reason, tr.comparison and round(tr.comparison["mean_diff"], 5))
                                          for tr in trials],
                               "winner": winner.label if winner else None, "incumbent": incumbent.version_id})
            self.log(f"  round {global_round}: budget {budget}; " + "; ".join(
                f"{tr.label}={tr.reason or tr.stage}" for tr in trials) + f" -> {incumbent.version_id}")
        return {"task": self.task.task, "incumbent": incumbent.version_id, "S": current["S"], "C": current["C"],
                "rounds": rounds_log, "agent": self.agent.status()}

    def _evaluate(self, trial: Trial, incumbent: CodeVersion, current: dict, budget: int,
                  allowed: tuple[str, ...], known: set[str]) -> Trial:
        if not trial.edits:
            trial.stage, trial.reason = "declined", "declined"
            return trial
        try:
            source = slot_rules.apply_edits(incumbent.slots_source(), trial.edits)
        except (SyntaxError, ValueError) as error:
            trial.stage, trial.reason = "rejected", f"rejected:ast ({error})"
            return trial
        errors = slot_rules.check_module(source)
        changed = slot_rules.changed_slots(incumbent.slots_source(), source)
        if errors or not changed or any(slot not in allowed for slot in changed) or len(changed) > budget:
            trial.stage = "rejected"
            trial.reason = "rejected:ast (" + ("; ".join(errors) or f"changed {changed} vs allowed {list(allowed)}, budget {budget}") + ")"
            return trial
        version = codebase.derive_version(incumbent, source, trial.edits, self.paths.store)
        (version.root / "patch.diff").write_text("".join(difflib.unified_diff(
            incumbent.slots_source().splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile=f"{incumbent.version_id}/baselines/slots.py", tofile=f"{version.version_id}/baselines/slots.py")),
            encoding="utf-8")
        trial.version = version
        ok, trial.tests = sandbox.check(version, known)
        if not ok:
            trial.stage, trial.reason = "rejected", "rejected:tests"
            return trial
        try:
            trial.replay = replay_code(self.task, self.policy, version, self.seeds, self.paths.replays)
        except Exception as error:   # a slot that passes the tests can still fail on real data
            trial.stage, trial.reason = "rejected", f"rejected:replay ({type(error).__name__}: {error})"
            return trial
        trial.comparison = paired(trial.replay, current, self.z)
        trial.step_ratio = self._step_ratio(version, incumbent)
        trial.comparison["step_time_ratio"] = trial.step_ratio
        trial.stage = "replayed"
        return trial

    def _step_ratio(self, version: CodeVersion, incumbent: CodeVersion) -> float:
        """Per-step training time of the candidate relative to the incumbent (median of 5)."""
        base = Session(self.task, incumbent).base
        x, y = base.x[:512], base.y[:512]
        a = sandbox.step_benchmark(incumbent, self.task.checkpoint, x, y)
        b = sandbox.step_benchmark(version, self.task.checkpoint, x, y)
        return b / max(a, 1e-9)

    def _judge(self, comparison: dict, current: dict) -> tuple[bool, str]:
        if comparison["mean_diff"] <= comparison["band"]:
            return False, "rejected:below_band"
        if comparison["worst_seed_diff"] < -comparison["band"]:
            return False, "rejected:worst_seed"
        relative_gain = comparison["mean_diff"] / max(abs(current["S"]), 1e-9)
        if comparison["cost_change"] > self.beta0 + self.beta1 * relative_gain:
            return False, "rejected:cost"
        if (comparison.get("step_time_ratio") or 1.0) > 2.0:
            return False, "rejected:slow"   # each training step more than twice as slow (median of 5 timings)
        return True, "admissible"

    def _log_row(self, round_index: int, trial: Trial, before: str, verdict: str, winner: Trial | None) -> dict:
        patch = trial.version.root / "patch.diff" if trial.version else None
        return {"task": self.task.task, "round": round_index, "trial": trial.label, "group": trial.group,
                "proposer": trial.proposer, "llm_tokens": trial.tokens,
                "edits": [{k: e.get(k) for k in ("slot", "variant", "hypothesis")} for e in trial.edits],
                "variant": ",".join(str(e.get("variant")) for e in trial.edits),
                "retired": trial.retired,
                "patch_sha256": "sha256:" + hashlib.sha256(patch.read_bytes()).hexdigest() if patch and patch.is_file() else None,
                "code_version_before": before, "candidate_version": trial.version.version_id if trial.version else None,
                "official_tests": trial.tests, "comparison": trial.comparison, "verdict": verdict,
                "code_version_after": winner.version.version_id if winner else before}


POLICY_GRID: dict[str, list] = {
    "label_fraction": [0.3, 0.7], "z": [1.645, 2.5], "max_epochs": [2, 4],
    "max_candidates": [2, 4], "confirm_windows": [1, 3],
}


class PolicyEvolver:
    """π search: one parameter at a time, each variant replayed with the full controller."""

    def __init__(self, task: TaskConfig, policy: PolicyConfig, artifacts: Path, *, seeds: list[int] = (0, 1),
                 grid: dict[str, list] | None = None, z: float = 2.0, beta0: float = 0.25, beta1: float = 5.0, log=print):
        self.task, self.policy = task, policy
        self.paths = _Artifacts(artifacts)
        self.seeds = list(seeds)
        self.grid = grid or POLICY_GRID
        self.z, self.beta0, self.beta1 = z, beta0, beta1
        self.log = log

    def run(self) -> dict:
        code = promoted_code(self.task, self.paths.root)
        session = Session(self.task, code)
        root = self.paths.root / "policy_replays"
        current = replay_policy(self.task, self.policy, code, self.seeds, root, session)
        trials = []
        for name, values in self.grid.items():
            for value in values:
                variant = self.policy.replace(**{name: value}, policy_id=f"{self.policy.policy_id}+{name}={value}")
                result = replay_policy(self.task, variant, code, self.seeds, root, session)
                comparison = paired(result, current, self.z)
                ok, reason = self._judge(comparison, current)
                trials.append((variant, comparison, ok, reason))
                self.log(f"  {variant.policy_id}: {comparison['mean_diff']:+.4f} (band {comparison['band']:.4f}) {reason}")
        admissible = [t for t in trials if t[2]]
        winner = max(admissible, key=lambda t: t[1]["mean_diff"]) if admissible else None
        for variant, comparison, ok, reason in trials:
            _append(self.paths.policy_edits, {"task": self.task.task, "code_version": code.version_id,
                                              "from": self.policy.policy_id, "variant": variant.policy_id,
                                              "comparison": comparison,
                                              "verdict": "promoted" if winner and variant is winner[0] else reason})
        if winner:
            path = self.paths.root / "policies" / f"{digest(winner[0].to_dict(), 8)}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(winner[0].to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
            promoted = self.paths.promoted()
            promoted["policy"] = path.as_posix()
            self.paths.save_promoted(promoted)
        return {"task": self.task.task, "code_version": code.version_id, "S": current["S"],
                "winner": winner[0].policy_id if winner else None,
                "trials": [(v.policy_id, round(c["mean_diff"], 5), r) for v, c, _, r in trials]}

    def _judge(self, comparison: dict, current: dict) -> tuple[bool, str]:
        if comparison["mean_diff"] <= comparison["band"]:
            return False, "rejected:below_band"
        if comparison["worst_seed_diff"] < -comparison["band"]:
            return False, "rejected:worst_seed"
        relative_gain = comparison["mean_diff"] / max(abs(current["S"]), 1e-9)
        limit = self.beta0 + self.beta1 * relative_gain
        if comparison["cost_change"] > limit or comparison["label_change"] > limit:
            return False, "rejected:cost"
        return True, "admissible"
