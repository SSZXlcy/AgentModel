"""Loading official JSONL with the official parser, and materialising training files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codebase import CodeVersion

_CACHE: dict[tuple[str, float, str], "Table"] = {}


@dataclass
class Table:
    """Rows of one official JSONL file; `raw` keeps each original line for byte-exact rewrites."""

    ids: np.ndarray       # str
    raw: list[str]
    x: np.ndarray         # float64 [n, features], official flatten order
    y: np.ndarray         # class index (int64) or target (float64)
    groups: np.ndarray    # str entity key from the task config; the sample id when there is none
    meta: list[dict] = None  # the row's `meta` object (trace fields; never a model input)

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, index) -> "Table":
        index = np.asarray(index, dtype=np.int64)
        return Table(self.ids[index], [self.raw[i] for i in index], self.x[index], self.y[index], self.groups[index],
                     [self.meta[i] for i in index])

    def key(self, fields: list[str], index=None) -> np.ndarray:
        """Composite key over meta fields (or the row id when `fields` is empty)."""
        index = np.arange(len(self)) if index is None else np.asarray(index, dtype=np.int64)
        if not fields:
            return self.ids[index]
        return np.asarray(["|".join(str(self.meta[i].get(f)) for f in fields) for i in index])


def load_table(path: Path, code: CodeVersion, profile: dict, group_key: list[str]) -> Table:
    """Parse with the official `json_value` / `flatten` so features match what train_run sees."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime, ",".join(group_key))
    if key in _CACHE:
        return _CACHE[key]
    data = code.modules().data
    classes = profile.get("classes") or []
    ids, raw, xs, ys, groups, metas = [], [], [], [], [], []
    expected_paths = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = data.json_value(line)
        paths, values = data.flatten(row["inputs"])
        if expected_paths is None:
            expected_paths = paths
        elif paths != expected_paths:
            raise ValueError(f"{path}: feature schema changed at {row['id']}")
        answer = row["answer"]
        ys.append(classes.index(answer) if profile["task"] == "classification" else float(answer))
        meta = row.get("meta") or {}
        groups.append("|".join(str(meta.get(k)) for k in group_key) if group_key else row["id"])
        ids.append(row["id"])
        raw.append(line)
        xs.append(values)
        metas.append(meta)
    table = Table(np.asarray(ids), raw, np.asarray(xs, dtype=np.float64),
                  np.asarray(ys, dtype=np.int64 if profile["task"] == "classification" else np.float64),
                  np.asarray(groups), metas)
    _CACHE[key] = table
    return table


def write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
