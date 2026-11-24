"""Task-agnostic structure discovery over the `meta` fields of an official JSONL table.

Nothing here knows any field name. A field (or pair of fields) is a candidate *entity key* when
its values repeat; an integer field is a candidate *order* when, inside every entity, it counts up
by one. A regression target is a *countdown* when, inside every entity, it falls by a constant step
and ends at 0 — such a label only exists once the entity has finished (e.g. remaining life).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import islice

import numpy as np

from .data import Table


def _fields(table: Table) -> list[str]:
    names: dict[str, None] = {}
    for meta in islice(table.meta, 2000):
        names.update(dict.fromkeys(meta))
    return list(names)


def field_profile(table: Table, field: str, top: int = 8) -> dict:
    values = [m.get(field) for m in table.meta]
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    out = {"field": field, "rows": len(values), "distinct": len(counts),
           "rows_per_value": {"min": min(counts.values()), "median": float(np.median(list(counts.values()))),
                              "max": max(counts.values())},
           "top_values": ranked[:top]}
    if len(numeric) == len(values) and numeric:
        out["numeric"] = {"min": min(numeric), "max": max(numeric),
                          "integer": all(float(v).is_integer() for v in numeric)}
    return out


def candidate_keys(table: Table, low_cardinality: int = 20) -> list[list[str]]:
    n = len(table)
    fields = _fields(table)
    distinct = {f: len({str(m.get(f)) for m in table.meta}) for f in fields}
    singles = [f for f in fields if 1 < distinct[f] <= n / 3]
    low = [f for f in fields if 1 < distinct[f] <= low_cardinality]
    keys = [[f] for f in singles]
    keys += [[a, b] for a in low for b in singles if a != b and [b, a] not in keys]
    return keys


def order_candidates(table: Table) -> list[str]:
    out = []
    for f in _fields(table):
        values = [m.get(f) for m in table.meta]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values) and len(set(values)) > 1:
            out.append(f)
    return out


def sequence_check(table: Table, group_by: list[str], order_by: str | None, task_type: str) -> dict:
    """Inside each group (rows in file order): does `order_by` count up by one, and is the answer
    monotone / a countdown ending at 0?"""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(table.key(group_by)):
        groups[key].append(i)
    consecutive = countdown = decreasing = 0
    for rows in groups.values():
        if order_by is not None:
            order = [table.meta[i].get(order_by) for i in rows]
            if all(isinstance(v, int) for v in order) and order == list(range(order[0], order[0] + len(order))):
                consecutive += 1
        if task_type == "regression" and len(rows) > 1:
            y = table.y[rows]
            steps = np.diff(y)
            if np.all(steps < 0):
                decreasing += 1
                if np.allclose(steps, steps[0]) and abs(float(y[-1])) < 1e-9:
                    countdown += 1
    n = max(len(groups), 1)
    sizes = np.array([len(r) for r in groups.values()])
    return {"group_by": group_by, "order_by": order_by, "groups": len(groups),
            "rows_per_group": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
            "order_consecutive_fraction": consecutive / n if order_by else None,
            "answer_decreasing_fraction": decreasing / n if task_type == "regression" else None,
            "answer_countdown_fraction": countdown / n if task_type == "regression" else None}


def group_shares(table: Table, group_by: list[str], answer_classes: list[str] | None = None) -> dict:
    keys = table.key(group_by) if group_by != ["answer"] else np.asarray(
        [answer_classes[int(v)] if answer_classes else str(v) for v in table.y])
    values, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    share = counts / counts.sum()
    return {"group_by": group_by, "groups": int(len(values)),
            "largest_share": float(share.max()), "top2_share": float(np.sort(share)[-2:].sum()) if len(share) > 1 else 1.0,
            "max_over_median": float(counts.max() / max(np.median(counts), 1)),
            "max_over_min": float(counts.max() / max(counts.min(), 1)),
            "median_count": float(np.median(counts)),
            "top": [(str(values[i]), int(counts[i])) for i in order[:8]]}


SEGMENT_FIELD = "_countdown_run"        # synthetic meta field holding a recovered entity (never a model input)


def countdown_runs(y: np.ndarray, min_share: float = 0.9) -> dict | None:
    """Entities recovered from the order of the answers alone, for rows that carry no `meta`.

    A remaining-life target falls by one constant step from row to row while the same unit is observed
    (N-1, N-2, ..., 0); where that chain breaks, the next unit starts. Accepted only when the step is
    positive and holds for at least `min_share` of consecutive pairs; otherwise rows are independent.
    Returns {"run": run index per row, "step": step, "share": share} or None."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return None
    diffs = np.round(y[:-1] - y[1:], 6)
    values, counts = np.unique(diffs, return_counts=True)
    step = float(values[np.argmax(counts)])
    share = float(counts.max() / len(diffs))
    if step <= 0 or share < min_share:
        return None
    run = np.concatenate([[0], np.cumsum(diffs != step)])
    return {"run": run, "step": step, "share": round(share, 4)}


def detect(table: Table, task_type: str) -> dict:
    """Best entity key and order field, or {"entity": None} when rows look independent."""
    orders = order_candidates(table)
    checks = []
    for key in candidate_keys(table):
        for order in [o for o in orders if o not in key] or [None]:
            checks.append(sequence_check(table, key, order, task_type))
    timed = [c for c in checks if c["order_by"] and (c["order_consecutive_fraction"] or 0) >= 0.95]
    if not timed:
        return {"entity": None, "order": None, "countdown": False, "checked": len(checks)}
    best = max(timed, key=lambda c: (c["answer_countdown_fraction"] or 0, c["order_consecutive_fraction"],
                                     -len(c["group_by"]), c["groups"]))
    return {"entity": best["group_by"], "order": best["order_by"],
            "countdown": (best["answer_countdown_fraction"] or 0) >= 0.95, "groups": best["groups"],
            "rows_per_group": best["rows_per_group"], "checked": len(checks)}
