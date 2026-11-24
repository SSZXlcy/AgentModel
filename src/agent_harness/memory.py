"""Memory (W4): what earlier rounds explored, decided and learned, in a form the agent can read
next round and the report can quote.

    <run>/journal.jsonl          one record per round (append-only); outcomes are back-filled by
                                 appending a "backfill" record, never by rewriting
    <run>/reports/round_NN.md    the report paragraph of each round
    <shared>/library/            lessons with evidence, shared across runs and tasks (library.py)

A round only sees journal rounds before it and library entries that existed before it, so a
resumed run and a fresh run read the same history at every step.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import json_ready
from .library import Library, has_experience
from .recorder import read_jsonl


class Memory:
    def __init__(self, run_dir: Path, shared_root: Path, leak_markers: list[str], run_id: str, task: str):
        self.run_dir = Path(run_dir)
        self.journal_path = self.run_dir / "journal.jsonl"
        self.reports = self.run_dir / "reports"
        self.library = Library(Path(shared_root) / "library", leak_markers)
        self.run_id = run_id
        self.task = task

    # ------------------------------------------------------------ journal
    def rounds(self, before: int | None = None) -> list[dict]:
        rows = read_jsonl(self.journal_path)
        merged: dict[int, dict] = {}
        for row in rows:
            if row.get("type") == "round":
                merged[row["round"]] = dict(row)
        for row in rows:
            if row.get("type") == "backfill" and row["round"] in merged:
                merged[row["round"]].setdefault("outcome", {}).update(row["outcome"])
        out = [merged[r] for r in sorted(merged)]
        return [r for r in out if before is None or r["round"] < before]

    def next_round(self) -> int:
        return len(self.rounds())

    def append(self, record: dict) -> dict:
        record = json_ready({"type": "round", **record})
        text = json.dumps(record, ensure_ascii=False)
        self.library.check_leaks(text)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        paragraph = (record.get("summary") or {}).get("report_paragraph")
        if paragraph:
            self.reports.mkdir(parents=True, exist_ok=True)
            (self.reports / f"round_{record['round']:02d}.md").write_text(
                f"### 第 {record['round']} 轮（触发：{record.get('trigger')}，第 {record.get('window')} 批）\n\n"
                f"{paragraph}\n", encoding="utf-8")
        return record

    def backfill(self, round_index: int, outcome: dict) -> None:
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_ready({"type": "backfill", "round": round_index, "outcome": outcome}),
                                    ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ lessons
    def add_lessons(self, round_index: int, lessons: list[dict], code_version: str, slots_source: str = "") -> list[str]:
        stored = []
        for i, lesson in enumerate(lessons or []):
            if not has_experience(lesson):
                continue
            structured = {k: lesson[k] for k in ("when", "action", "confidence")
                          if isinstance(lesson.get(k), (dict, str))}
            entry = self.library.add(f"{self.run_id}-r{round_index:02d}-{i}", round_index, code_version, slots_source,
                                     lesson.get("why", ""), lesson,
                                     extra={"run_id": self.run_id, "task": self.task, **structured})
            if entry:
                stored.append(entry.name)
        return stored

    # ------------------------------------------------------------ what the agent may read
    def prompt_view(self, last: int = 5, before_round: int | None = None) -> dict:
        before = self.next_round() if before_round is None else before_round
        journal = [{"round": r["round"], "trigger": r.get("trigger"), "window": r.get("window"),
                    "findings": [f.get("claim") for f in (r.get("plan") or {}).get("findings", [])][:5],
                    "applied": r.get("applied"), "rejected": r.get("rejected"),
                    "lessons": (r.get("summary") or {}).get("lessons"), "next_time": (r.get("summary") or {}).get("next_time"),
                    "outcome": r.get("outcome")} for r in self.rounds(before)[-last:]]
        library = self.library.prompt_text(
            limit=4000, exclude=lambda e: e.get("run_id") == self.run_id and e.get("round", 0) >= before)
        return {"journal": journal, "library": library}

    def tried_variants(self) -> list[dict]:
        rows = []
        for r in self.rounds():
            for item in (r.get("applied") or []) + (r.get("rejected") or []):
                for edit in item.get("edits") or []:
                    rows.append({"edits": [edit]})
                variant = (item.get("params") or {}).get("variant")
                if variant:
                    rows.append({"edits": [{"variant": variant}]})
        return rows
