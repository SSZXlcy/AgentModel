"""Is the model saturated under the current (code, data, hyper-parameters), and why?

Input is the per-epoch curve of the best candidate chain in a window: training loss from the
official history.jsonl, and gate utility measured by the harness after each epoch.
"""
from __future__ import annotations

NOT_SATURATED = "NOT_SATURATED"   # still improving at the last epoch -> allow more epochs
OVERFIT = "OVERFIT"               # best epoch is earlier and loss kept falling -> lower lr / branch earlier
STALLED = "STALLED"               # nothing beat the serving model by more than the noise band
CAPACITY = "CAPACITY"             # stalled and training loss barely moves -> outer layer
OK = "OK"
NONE = "NONE"                     # no training happened this window


def diagnose_chain(curve: list[dict], serving_utility: float, band: float) -> str:
    if not curve:
        return NONE
    utilities = [point["utility"] for point in curve]
    losses = [point["loss"] for point in curve]
    best = max(range(len(curve)), key=lambda i: utilities[i])
    if utilities[best] <= serving_utility + band:
        flat = len(losses) > 1 and abs(losses[-1] - losses[0]) <= 0.01 * max(abs(losses[0]), 1e-12)
        return CAPACITY if flat else STALLED
    if best == len(curve) - 1 and len(curve) > 1 and utilities[-1] > utilities[-2]:
        return NOT_SATURATED
    if best < len(curve) - 1 and losses[-1] < losses[best]:
        return OVERFIT
    return OK
