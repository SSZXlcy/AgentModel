"""The stream interface. Everything about how the platform delivers data lives behind this class,
so swapping in the evaluation SDK touches only this file.

LocalStream simulates the platform with a public split: the controller sees features, pays for
labels, and never reads an answer it did not request (or that does not exist yet). Scoring is
platform-side.

Label protocols (a property of the platform, configured per task):
  immediate          a requested label is returned in the same window
  on_group_complete  a requested label is released only once its entity has left the stream;
                     the platform knows when that happens, the controller does not
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .config import TaskConfig
from .data import Table


class DataAccessError(PermissionError):
    """A label or row that does not exist yet (or was never paid for) was requested."""


class LocalStream:
    def __init__(self, table: Table, task: TaskConfig, task_type: str, protocol_keys: np.ndarray | None = None):
        self._table = table
        self.task_type = task_type
        self.windows = _cut(table, task)
        self.protocol = task.label_protocol
        self._released: dict[int, set[int]] = {}
        self._pending: dict[int, set[int]] = {}
        self._complete_at = None
        if self.protocol == "on_group_complete":
            if protocol_keys is None:
                raise ValueError("on_group_complete needs entity keys for every stream row")
            last: dict[str, int] = {}
            for w, rows in enumerate(self.windows):
                for key in protocol_keys[rows].tolist():
                    last[key] = w
            self._row_key = protocol_keys
            self._complete_at = last

    def __len__(self) -> int:
        return len(self.windows)

    def features(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.windows[window]
        return self._table.ids[rows], self._table.x[rows], self._table.groups[rows]

    def meta(self, window: int, positions: np.ndarray | None = None) -> list[dict]:
        rows = self.windows[window] if positions is None else self.windows[window][np.asarray(positions, dtype=np.int64)]
        return [self._table.meta[i] for i in rows]

    def _available(self, window: int, position: int, now: int) -> bool:
        if self._complete_at is None:
            return True
        return self._complete_at[self._row_key[self.windows[window][position]]] <= now

    def request_labels(self, window: int, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ask for labels of rows in the current window: (released positions, their labels).
        Under on_group_complete, the rest are delivered later by `deliver`."""
        positions = np.asarray(positions, dtype=np.int64)
        ready = np.asarray([p for p in positions.tolist() if self._available(window, p, window)], dtype=np.int64)
        later = set(positions.tolist()) - set(ready.tolist())
        if later:
            self._pending.setdefault(window, set()).update(later)
        self._released.setdefault(window, set()).update(ready.tolist())
        return ready, self._table.y[self.windows[window][ready]]

    def deliver(self, now: int) -> dict[int, np.ndarray]:
        """Labels of earlier windows that became available at window `now`."""
        out: dict[int, np.ndarray] = {}
        for window, waiting in list(self._pending.items()):
            ready = sorted(p for p in waiting if self._available(window, p, now))
            if ready:
                waiting.difference_update(ready)
                self._released.setdefault(window, set()).update(ready)
                out[window] = np.asarray(ready, dtype=np.int64)
            if not waiting:
                del self._pending[window]
        return out

    def released(self, window: int) -> np.ndarray:
        return np.asarray(sorted(self._released.get(window, set())), dtype=np.int64)

    def labels(self, window: int, positions: np.ndarray) -> np.ndarray:
        """Labels already released to the controller."""
        positions = np.asarray(positions, dtype=np.int64)
        if not set(positions.tolist()) <= self._released.get(window, set()):
            raise DataAccessError(f"window {window}: label requested for a row that was not released")
        return self._table.y[self.windows[window][positions]]

    def labelled_lines(self, window: int, positions: np.ndarray) -> list[str]:
        """Original JSONL lines for training files; only rows whose labels were released."""
        positions = np.asarray(positions, dtype=np.int64)
        if not set(positions.tolist()) <= self._released.get(window, set()):
            raise DataAccessError(f"window {window}: training data requested for unlabelled rows")
        rows = self.windows[window][positions]
        return [self._table.raw[i] for i in rows]

    def restore(self, released: dict[int, list[int]], pending: dict[int, list[int]]) -> None:
        """Resume: the simulated platform re-learns what it had already handed out."""
        self._released = {int(w): set(p) for w, p in released.items()}
        self._pending = {int(w): set(p) for w, p in pending.items() if p}

    def snapshot(self) -> tuple[dict, dict]:
        return ({w: sorted(p) for w, p in self._released.items()}, {w: sorted(p) for w, p in self._pending.items()})

    def score(self, window: int, outputs: np.ndarray) -> dict:
        """Platform-side evaluation of the submitted predictions on every row of the window."""
        return metrics.report(self.task_type, self._table.y[self.windows[window]], outputs)


def _cut(table: Table, task: TaskConfig) -> list[np.ndarray]:
    """Consecutive windows in file order; a short tail is merged into the previous window."""
    n = len(table)
    if task.window_groups:
        order: list[str] = []
        for group in table.groups.tolist():
            if not order or order[-1] != group:
                order.append(group)
        chunks = [order[i:i + task.window_groups] for i in range(0, len(order), task.window_groups)]
        if len(chunks) > 1 and len(chunks[-1]) < task.window_groups / 2:
            chunks[-2].extend(chunks.pop())
        return [np.flatnonzero(np.isin(table.groups, chunk)) for chunk in chunks]
    bounds = list(range(0, n, task.window_size)) + [n]
    windows = [np.arange(a, b) for a, b in zip(bounds, bounds[1:])]
    if len(windows) > 1 and len(windows[-1]) < task.window_size / 2:
        tail = windows.pop()
        windows[-1] = np.concatenate([windows[-1], tail])
    return windows
