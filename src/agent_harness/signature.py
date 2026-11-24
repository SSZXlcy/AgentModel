"""Situation signature: a small set of task-agnostic features that describe "where we are".

Operators read it to fill default parameters ($entity, $median_group_size, ...); experiences use
it as their applicability condition. Nothing here names a task or a field: entity and order fields
are whatever structure.detect() discovered.
"""
from __future__ import annotations

import numpy as np

from . import structure

MIN_TIMELINESS_WINDOWS = 3
MIN_TIMELINESS_LABELS = 50
# Input novelty: share of a window's rows farther from the training rows than 95% of training rows are from
# each other (nearest neighbour, standardised inputs). Training-like inputs give about 0.05; a window is
# "covered" by the training data up to three times that.
NOVELTY_QUANTILE = 0.95
COVERED_NOVELTY = 0.15
NOVELTY_REFERENCE_ROWS = 2000

PLACEHOLDERS = {
    "$entity": "entity",                       # discovered entity key fields, e.g. ["bearing"]
    "$order": "order",                         # discovered order field
    "$median_group_size": "median_group_size", # training rows of the median entity
    "$median_class_size": "median_class_size", # training rows of the median class
}


def compute(ctrl) -> dict:
    s = ctrl.session
    found = s.structure
    entity = found.get("entity") or []
    sig: dict = {"task_type": s.task_type, "base_rows": int(len(s.base)), "has_entity": bool(entity),
                 "entity": entity, "order": found.get("order"), "countdown": bool(found.get("countdown")),
                 "n_entities": int(found.get("groups") or 0)}
    if entity:
        shares = structure.group_shares(s.base, entity)
        sig.update(entity_imbalance=round(shares["max_over_median"], 3), top2_entity_share=round(shares["top2_share"], 3),
                   median_group_size=int(max(shares["median_count"], 1)))
    else:
        sig.update(entity_imbalance=1.0, top2_entity_share=None, median_group_size=None)
    if s.task_type == "classification":
        shares = structure.group_shares(s.base, ["answer"], s.classes)
        sig.update(class_imbalance=round(shares["max_over_min"], 3), median_class_size=int(max(shares["median_count"], 1)))
    else:
        sig.update(class_imbalance=None, median_class_size=None)

    st = getattr(ctrl, "state", None) or {}
    health = st.get("health", [])
    sig["stream_format"] = stream_format(ctrl, entity)
    novelty = input_novelty(ctrl)
    sig["input_novelty"] = novelty
    sig["inputs_covered"] = None if novelty is None else bool(novelty <= COVERED_NOVELTY)
    sig.update(windows_total=len(getattr(ctrl, "window_ids", []) or []), windows_done=len(st.get("rows", [])),
               labels_released=int(sum(len(v) for v in st.get("pool", {}).values())),
               promotions=st.get("promotions", 0), rollbacks=st.get("rollbacks", 0),
               entities_completed=len(st.get("completed", [])))
    requested = st.get("labels_requested_total", 0)
    asked = sum(1 for h in health if h.get("requested"))
    # Cumulative share of requested answers returned in their own window. One window can mislead (the
    # first window may hold entities that happen to finish in it), so the share counts only once
    # MIN_TIMELINESS_WINDOWS windows and MIN_TIMELINESS_LABELS requests are behind it.
    sig["label_timeliness"] = (round(st.get("labels_released_at_once", 0) / requested, 3)
                               if asked >= MIN_TIMELINESS_WINDOWS and requested >= MIN_TIMELINESS_LABELS else None)
    policy = getattr(ctrl, "policy", None)
    sig["drift_exploitable"] = drift_exploitable(health, policy.health_window if policy is not None else 3)
    if policy is not None:
        sig.update(mode=policy.mode, serve_mode=policy.serve_mode, split_mode=policy.split_mode,
                   gate_mode=policy.gate_mode, train_trigger=policy.train_trigger,
                   capped=bool(policy.cap_per_group), recent_only=policy.recent_windows_only)
    drifts = [h["drift"] for h in health if h.get("drift")]
    sig["drift_ratio"] = round(drifts[-1]["mmd"] / max(drifts[-1]["threshold"], 1e-12), 3) if drifts else None
    settled = [h["settled"] for h in health if h.get("settled") and h["settled"].get("predicted") is not None]
    if settled:
        realized = float(np.mean([x["gain"] for x in settled]))
        predicted = float(np.mean([x["predicted"] for x in settled]))
        sig["gate_overestimate"] = round(predicted / realized, 3) if realized > 0 else None
        sig["gate_realized_negative"] = realized <= 0
    else:
        sig.update(gate_overestimate=None, gate_realized_negative=None)
    vs = [h["vs_static"]["diff"] for h in health[-3:] if h.get("vs_static")]
    sig["recent_vs_static"] = round(float(np.mean(vs)), 6) if vs else None
    return sig


def stream_format(ctrl, entity: list[str]) -> str | None:
    """How the stream presents entities, read from the inputs of its first window (never its answers):
    independent  no entity structure
    sequence     an entity appears with several rows (its life unfolds over the stream)
    snapshot     each entity appears once (one observation per unit, e.g. a cut-off last cycle)"""
    if not entity:
        return "independent"
    st = getattr(ctrl, "state", None)
    if st is not None and st.get("stream_format"):
        return st["stream_format"]
    stream, ids = getattr(ctrl, "stream", None), getattr(ctrl, "window_ids", None)
    if stream is None or not ids:
        return None
    counts: dict[str, int] = {}
    for meta in stream.meta(ids[0]):
        key = "|".join(str(meta.get(f)) for f in entity)
        counts[key] = counts.get(key, 0) + 1
    found = "snapshot" if counts and float(np.median(list(counts.values()))) <= 1 else "sequence"
    if st is not None:
        st["stream_format"] = found
    return found


def _nearest(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    distances = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    return np.sqrt(np.maximum(distances.min(1), 0.0))


def input_novelty(ctrl) -> float | None:
    """Novelty of the latest window whose inputs are visible (the first window at the opening), read from
    inputs only. Low novelty means the training data already covers these inputs: retraining on them has
    little to teach (the stream may still move between regimes the model knows)."""
    normalize, stream, ids = getattr(ctrl, "normalize", None), getattr(ctrl, "stream", None), \
        getattr(ctrl, "window_ids", None)
    st = getattr(ctrl, "state", None)
    if normalize is None or stream is None or not ids or st is None:
        return None
    seen = [w for w in ids if w < st.get("next_window", ids[0])]
    window = seen[-1] if seen else ids[0]
    cache = st.setdefault("novelty", {})
    if str(window) in cache:
        return cache[str(window)]
    reference = getattr(ctrl, "_novelty_reference", None)
    if reference is None:
        s = ctrl.session
        base = normalize(s.base.x[s.base_rows])
        if len(base) > NOVELTY_REFERENCE_ROWS:
            base = base[np.random.default_rng(0).choice(len(base), NOVELTY_REFERENCE_ROWS, replace=False)]
        reference = (base, float(np.quantile(_nearest(base, base, exclude_self=True), NOVELTY_QUANTILE)))
        ctrl._novelty_reference = reference
    base, radius = reference
    x = normalize(stream.features(window)[1])
    cache[str(window)] = round(float(np.mean(_nearest(x, base) > radius)), 4)
    return cache[str(window)]


def drift_exploitable(health: list[dict], k: int) -> bool | None:
    """Can updating earn anything from this stream? Judged by what changes of the serving model actually
    earned on the next window, never by input movement alone (inputs may move between regimes the model
    already knows).

    True   at least two settled changes, and their summed next-window gain clears the combined band
    False  the settled changes earned nothing in sum; or training ran in k of the recent windows, nothing
           was promoted, and the serving model did not degrade
    None   not enough evidence yet
    """
    recent = health[-2 * k:]
    settled = [h for h in recent if h.get("settled") and h["settled"].get("changed")]
    if len(settled) >= 2:
        total = float(sum(h["settled"]["gain"] for h in settled))
        band = float(np.sqrt(sum((h.get("vs_static") or {}).get("band", 0.0) ** 2 for h in settled)))
        if total > band:
            return True
        # giving up needs more evidence than following: one window's answers may cover other classes or
        # entities than the next window's, so a few settled changes can look worthless by chance
        if total <= 0 and len(settled) >= 2 * k:
            return False
    trained = sum(1 for h in recent if h.get("trained"))
    promoted = sum(1 for h in recent if h.get("promoted"))
    degraded = any((h.get("drift") or {}).get("degraded") for h in recent)
    if trained >= k and promoted == 0 and not degraded:
        return False
    return None


def resolve(value, sig: dict):
    """Replace a placeholder such as "$entity" by the value it has for this task."""
    if isinstance(value, str) and value in PLACEHOLDERS:
        resolved = sig.get(PLACEHOLDERS[value])
        if resolved in (None, [], ""):
            raise ValueError(f"{value} is undefined for this task")
        return resolved
    if isinstance(value, list):
        return [resolve(v, sig) for v in value]
    return value


def holds(condition: dict, sig: dict) -> bool:
    """A structured applicability condition: {"feature": value} for equality (null = not known yet), or
    {"feature": {">": x}} / {"<": x} / {">=": x} / {"<=": x} / {"in": [...]} / {"!=": x}; all must hold.
    {"any": [condition, ...]} holds when one of the conditions holds, {"all": [...]} when every one does."""
    for feature, expected in (condition or {}).items():
        if feature == "any":
            if not any(holds(c, sig) for c in expected):
                return False
            continue
        if feature == "all":
            if not all(holds(c, sig) for c in expected):
                return False
            continue
        actual = sig.get(feature)
        if isinstance(expected, dict):
            for op, ref in expected.items():
                if op == "!=":
                    if actual == ref:
                        return False
                    continue
                if actual is None:
                    return False
                if op == ">":
                    ok = actual > ref
                elif op == "<":
                    ok = actual < ref
                elif op == ">=":
                    ok = actual >= ref
                elif op == "<=":
                    ok = actual <= ref
                elif op == "in":
                    ok = actual in ref
                else:
                    raise ValueError(f"unknown comparison {op!r}")
                if not ok:
                    return False
        elif actual != expected:
            return False
    return True
