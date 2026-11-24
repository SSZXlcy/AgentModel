"""Structured experiences (经验库): condition on the situation signature → one operator call.

    configs/experiences/seed.yaml          seed experiences written from the design
    <shared>/experiences/outcomes.jsonl    probation verdicts of rounds that applied an experience
                                           (append-only; shared across runs and tasks)
    library entries with `when` + `action` experiences learned by round summaries (library.py)

An experience is prior knowledge, not a command: its action still goes through the operator manual
(improvements are validated, rhythm changes are judged by a probation). Its credibility starts at its
prior and moves with every verdict. Outcomes of the running run are visible only for earlier rounds,
so a resumed run and a fresh run choose alike.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import yaml

from . import operators
from . import signature as sig_rules

PRIORS = {"high": 0.8, "medium": 0.6, "low": 0.4}
PSEUDO_COUNT = 4.0
SEED_PATH = Path(__file__).resolve().parents[2] / "configs" / "experiences" / "seed.yaml"


@dataclass
class Experience:
    id: str
    when: dict
    action: dict
    why: str = ""
    prior: str = "medium"
    source: str = "seed"
    support: int = 0
    refute: int = 0

    @property
    def operator(self) -> str:
        return str(self.action.get("operator"))

    @property
    def rhythm(self) -> bool:
        return self.operator.startswith("mode.")

    @property
    def credibility(self) -> float:
        return (PRIORS.get(self.prior, 0.6) * PSEUDO_COUNT + self.support) / (PSEUDO_COUNT + self.support + self.refute)

    def view(self) -> dict:
        return {**asdict(self), "credibility": round(self.credibility, 3)}


def _valid(item: dict) -> bool:
    action = item.get("action")
    return (isinstance(item.get("id"), str) and isinstance(item.get("when"), dict) and isinstance(action, dict)
            and action.get("operator") in operators.REGISTRY)


class Experiences:
    def __init__(self, shared_root: Path, run_id: str, seed_path: Path | None = None):
        self.seed_path = Path(seed_path or SEED_PATH)
        self.outcomes = Path(shared_root) / "experiences" / "outcomes.jsonl"
        self.run_id = run_id

    # ------------------------------------------------------------ loading
    def _seeds(self) -> list[Experience]:
        if not self.seed_path.is_file():
            return []
        items = yaml.safe_load(self.seed_path.read_text(encoding="utf-8")) or []
        return [Experience(i["id"], i["when"], i["action"], str(i.get("why", "")).strip(), i.get("prior", "medium"))
                for i in items if _valid(i)]

    @staticmethod
    def _learned(library, before_round: int | None, run_id: str) -> list[Experience]:
        if library is None:
            return []
        out = []
        for row in library.entries():
            if row.get("run_id") == run_id and before_round is not None and row.get("round", 0) >= before_round:
                continue
            item = {"id": row["entry_id"], "when": row.get("when"), "action": row.get("action")}
            if _valid(item):
                out.append(Experience(item["id"], item["when"], item["action"], row.get("lesson", ""),
                                      row.get("confidence") if row.get("confidence") in PRIORS else "low", "learned"))
        return out

    def _verdicts(self, before_round: int | None) -> list[dict]:
        if not self.outcomes.is_file():
            return []
        rows = [json.loads(line) for line in self.outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if not (r.get("run_id") == self.run_id and before_round is not None
                                        and r.get("round", 0) >= before_round)]

    def all(self, library=None, before_round: int | None = None) -> list[Experience]:
        items = self._seeds() + self._learned(library, before_round, self.run_id)
        by_id = {e.id: e for e in items}
        for row in self._verdicts(before_round):
            e = by_id.get(row.get("experience"))
            if e is None:
                continue
            if row.get("verdict") == "confirmed":
                e.support += 1
            elif row.get("verdict") == "reverted":
                e.refute += 1
        return items

    # ------------------------------------------------------------ choosing
    def matching(self, sig: dict, library=None, before_round: int | None = None,
                 blocked: set[str] | frozenset = frozenset()) -> list[Experience]:
        """Experiences whose condition holds and whose operator applies here, most credible first."""
        out = []
        for e in self.all(library, before_round):
            op = operators.REGISTRY[e.operator]
            if e.id in blocked or not sig_rules.holds(e.when, sig):
                continue
            if op.requires and not sig_rules.holds(op.requires, sig):
                continue
            out.append(e)
        return sorted(out, key=lambda e: -e.credibility)

    def choose(self, sig: dict, library=None, before_round: int | None = None,
               blocked: set[str] | frozenset = frozenset()) -> tuple[Experience | None, list[Experience]]:
        """(the rhythm to use, the other experiences that apply)."""
        found = self.matching(sig, library, before_round, blocked)
        rhythm = next((e for e in found if e.rhythm), None)
        return rhythm, [e for e in found if not e.rhythm]

    # ------------------------------------------------------------ learning
    def record(self, experience_ids: list[str], verdict: str, round_index: int, task: str) -> None:
        if not experience_ids or verdict not in ("confirmed", "reverted"):
            return
        self.outcomes.parent.mkdir(parents=True, exist_ok=True)
        with self.outcomes.open("a", encoding="utf-8") as stream:
            for e in experience_ids:
                stream.write(json.dumps({"experience": e, "verdict": verdict, "run_id": self.run_id,
                                         "round": round_index, "task": task}, ensure_ascii=False) + "\n")


def mode_of(experience: Experience | None) -> str | None:
    return experience.operator.split(".", 1)[1] if experience is not None and experience.rhythm else None
