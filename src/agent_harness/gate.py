"""Promotion gate (in-window), confirmation (next window), and the static-baseline floor.

All comparisons are paired: both models predict the same rows, and the noise band is the
bootstrap sd of the *difference*, resampled by entity (see metrics.resamples).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import metrics
from .config import PolicyConfig


@dataclass
class GateResult:
    accepted: bool
    reason: str
    gain: float
    band: float
    cost: float
    roi: float
    anchor_drop: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def promotion_gate(task: str, expected: np.ndarray, candidate: np.ndarray, serving: np.ndarray,
                   groups: np.ndarray, cost: float, policy: PolicyConfig, *,
                   anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, seed: int = 0) -> GateResult:
    gain, sd = metrics.paired_difference(task, expected, candidate, serving, groups, policy.bootstrap_reps, seed)
    band = policy.z * sd * metrics.small_sample_factor(groups)
    roi = gain / max(cost, 1e-9)
    drop = None
    if anchor is not None:
        y, cand_out, serve_out = anchor
        drop = metrics.utility(task, y, serve_out) - metrics.utility(task, y, cand_out)
    units = len(np.unique(groups))
    if 1 < units < policy.min_gate_units:
        return GateResult(False, f"only {units} independent entities in the gate rows (< {policy.min_gate_units}): "
                                 "too few to decide", gain, band, cost, roi, drop)
    if gain <= band:
        return GateResult(False, f"gain {gain:+.4f} within noise band {band:.4f}", gain, band, cost, roi, drop)
    if roi < policy.min_roi:
        return GateResult(False, f"roi {roi:.4g} < min_roi {policy.min_roi}", gain, band, cost, roi, drop)
    if drop is not None and drop > policy.anchor_veto * max(abs(metrics.utility(task, anchor[0], anchor[2])), 1e-9):
        return GateResult(False, f"anchor utility dropped by {drop:.4f} (catastrophic forgetting)", gain, band, cost, roi, drop)
    return GateResult(True, f"gain {gain:+.4f} > band {band:.4f}", gain, band, cost, roi, drop)


def confirmation(task: str, expected: np.ndarray, provisional: np.ndarray, confirmed: np.ndarray,
                 groups: np.ndarray, policy: PolicyConfig, seed: int = 0) -> tuple[str, float, float]:
    """On rows neither model trained on: roll back only when provisional is significantly worse."""
    diff, sd = metrics.paired_difference(task, expected, provisional, confirmed, groups, policy.bootstrap_reps, seed)
    band = policy.z * sd
    return ("rollback" if diff < -band else "hold"), diff, band


def cumulative_worse(track: list) -> bool:
    """[(diff, band), ...] of consecutive windows: is the summed difference below minus the combined band
    (bands add in quadrature)? Catches losses too small for any single window to show."""
    if len(track) < 2:
        return False
    total = float(sum(d for d, _ in track))
    band = float(np.sqrt(sum(b * b for _, b in track)))
    return total < -band


def below_static(task: str, expected: np.ndarray, serving: np.ndarray, static: np.ndarray,
                 groups: np.ndarray, policy: PolicyConfig, seed: int = 0) -> tuple[bool, float, float]:
    diff, sd = metrics.paired_difference(task, expected, serving, static, groups, policy.bootstrap_reps, seed)
    band = policy.z * sd
    return diff < -band, diff, band
