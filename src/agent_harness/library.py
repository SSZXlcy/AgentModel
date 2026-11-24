"""Experience library (after harness2's core/library.py).

    library/INDEX.md                       browsable list, rebuilt from disk every time
    library/entries/<id>/slots.py          the full adopted slots module, never a diff
    library/entries/<id>/EVIDENCE.md       why it was adopted, per-edit verdicts, transferable lesson
    library/entries/<id>/entry.json        round, code version, experience fields

Admission needs a supported lesson (cue, lesson, evidence); declining is a normal outcome.
Before each round the library is truncated to the causal prefix, so a resumed run and a fresh
run see the same evidence at every step. Text is checked for leak markers before it is stored,
because anything stored is served as evidence forever after.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil


class LeakError(ValueError):
    """Text destined for the library mentions data the system must never see."""


def has_experience(experience: dict | None) -> bool:
    return isinstance(experience, dict) and all(
        isinstance(experience.get(k), str) and experience[k].strip() for k in ("cue", "lesson", "evidence"))


class Library:
    def __init__(self, root: Path, leak_markers: list[str]):
        self.root = Path(root)
        self.leak_markers = [m for m in leak_markers if m]

    @property
    def index(self) -> Path:
        return self.root / "INDEX.md"

    def entries(self) -> list[dict]:
        rows = []
        for path in sorted((self.root / "entries").glob("*/entry.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(rows, key=lambda row: (row["round"], row["entry_id"]))

    def check_leaks(self, text: str) -> None:
        found = [m for m in self.leak_markers if m in text]
        if found:
            raise LeakError(f"library text mentions {found}")

    def add(self, entry_id: str, round_index: int, code_version: str, slots_source: str, why: str,
            experience: dict, extra: dict | None = None) -> Path | None:
        if not has_experience(experience):
            return None
        rows = experience.get("edits") or []
        lines = [f"# {entry_id}", "", f"- round: {round_index}", f"- code version: `{code_version}`", "",
                 "## Why this was chosen", why.strip(), "", "## What each edit did"]
        lines += [f"- slot: {r.get('slot')} / change: {r.get('change')} / observed: {r.get('observed')} / "
                  f"verdict: {r.get('verdict')}" for r in rows] or ["- (none recorded)"]
        lines += ["", "## Transferable lesson"]
        lines += [f"- **{k}**: {str(experience[k]).strip()}" for k in ("cue", "lesson", "evidence", "confidence")
                  if experience.get(k)]
        evidence_md = "\n".join(lines) + "\n"
        self.check_leaks(evidence_md + slots_source)
        directory = self.root / "entries" / entry_id
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        (directory / "slots.py").write_text(slots_source, encoding="utf-8")
        (directory / "EVIDENCE.md").write_text(evidence_md, encoding="utf-8")
        (directory / "entry.json").write_text(json.dumps(
            {"entry_id": entry_id, "round": round_index, "code_version": code_version,
             "cue": experience["cue"], "lesson": experience["lesson"], **(extra or {})},
            ensure_ascii=False, indent=2), encoding="utf-8")
        self.rebuild_index()
        return directory

    def retain(self, before_round: int) -> None:
        """Causal prefix: drop entries written in round `before_round` or later."""
        for row in self.entries():
            if row["round"] >= before_round:
                shutil.rmtree(self.root / "entries" / row["entry_id"], ignore_errors=True)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        rows = ["# Experience library", "",
                "Each entry is an adopted code change with its full slots module and the evidence for it. "
                "There it was a measured fact; here it is a hypothesis to re-check.", ""]
        entries = self.entries()
        if not entries:
            rows.append("_(empty)_")
        for row in entries:
            rows += [f"## `entries/{row['entry_id']}/` (round {row['round']}, `{row['code_version']}`)",
                     f"- cue: {row['cue']}", f"- lesson: {row['lesson']}", ""]
        self.root.mkdir(parents=True, exist_ok=True)
        self.index.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def prompt_text(self, limit: int = 6000, exclude=None) -> str:
        """INDEX plus each EVIDENCE.md, for chat endpoints that cannot browse files.
        `exclude(entry)` hides entries the caller must not see yet (causal prefix)."""
        if not self.index.is_file():
            return ""
        visible = [row for row in self.entries() if not (exclude and exclude(row))]
        parts = ["# Experience library", ""] + [f"- ({row.get('task', '?')}) cue: {row['cue']} | lesson: {row['lesson']}"
                                                 for row in visible]
        for row in reversed(visible):
            parts.append((self.root / "entries" / row["entry_id"] / "EVIDENCE.md").read_text(encoding="utf-8"))
        text = "\n".join(parts)
        return text if len(text) <= limit else text[:limit] + "\n[truncated by the harness]"
