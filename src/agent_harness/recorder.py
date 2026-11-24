"""Run records: manifest, append-only events, decision/settlement transitions, summary."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from .contracts import json_ready


def write_json(path: Path, value: Any, attempts: int = 6) -> None:
    """Atomic: write a temp file, then replace. On Windows the target may be held open for a moment by a
    scanner or indexer (access denied); retry with a short back-off instead of failing the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Recorder:
    def __init__(self, run_dir: Path, tracker=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        existing = read_jsonl(self.run_dir / "events.jsonl") + read_jsonl(self.run_dir / "transitions.jsonl")
        self.seq = max((row.get("seq", 0) for row in existing), default=0)

    def _append(self, name: str, row: dict) -> dict:
        self.seq += 1
        row = {"ts": datetime.now(timezone.utc).isoformat(), "seq": self.seq, **row}
        with (self.run_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
        return row

    def manifest(self, value: dict) -> None:
        write_json(self.run_dir / "manifest.json", value)

    def event(self, kind: str, **fields: Any) -> dict:
        return self._append("events.jsonl", {"kind": kind, **fields})

    def decision(self, **fields: Any) -> dict:
        return self._append("transitions.jsonl", {"type": "decision", **fields})

    def settlement(self, **fields: Any) -> dict:
        return self._append("transitions.jsonl", {"type": "settlement", **fields})

    def strategy_update(self, **fields: Any) -> dict:
        return self._append("transitions.jsonl", {"type": "strategy_update", **fields})

    def metrics(self, values: dict) -> None:
        if self.tracker is not None:
            try:
                self.tracker.log(values)
            except Exception as error:  # the local files are the record of truth
                self.event("tracking_error", error=f"{type(error).__name__}: {error}")

    def summary(self, value: dict) -> None:
        write_json(self.run_dir / "summary.json", value)
