"""Task metrics (same formulas as the official train.metrics) and grouped bootstrap noise bands."""
from __future__ import annotations

import numpy as np


def macro_f1(expected: np.ndarray, labels: np.ndarray) -> float:
    observed = sorted(set(expected.tolist()) | set(labels.tolist()))
    scores = [2 * np.sum((labels == c) & (expected == c)) / (np.sum(labels == c) + np.sum(expected == c))
              for c in observed]
    return float(np.mean(scores))


def nasa_score(expected: np.ndarray, predicted: np.ndarray) -> float:
    """C-MAPSS asymmetric score: over-estimating remaining life costs more. Lower is better."""
    d = predicted - expected
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def _point(task: str, expected: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Classification outputs are reduced to labels once; regression outputs pass through."""
    return out.argmax(axis=1) if task == "classification" and out.ndim == 2 else out


def utility(task: str, expected: np.ndarray, out: np.ndarray) -> float:
    """One number, higher is better: Macro-F1, or −RMSE for regression."""
    point = _point(task, expected, out)
    if task == "classification":
        return macro_f1(expected, point)
    return -float(np.sqrt(np.mean((point - expected) ** 2)))


def report(task: str, expected: np.ndarray, out: np.ndarray) -> dict:
    point = _point(task, expected, out)
    if task == "classification":
        return {"accuracy": float(np.mean(point == expected)), "macro_f1": macro_f1(expected, point),
                "utility": macro_f1(expected, point), "n": int(len(expected))}
    error = point - expected
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))), "nasa": nasa_score(expected, point),
            "over_estimate_ratio": float(np.mean(error > 0)), "utility": -rmse, "n": int(len(expected))}


_T975 = [(1, 12.71), (2, 4.30), (3, 3.18), (4, 2.78), (5, 2.57), (6, 2.45), (7, 2.36), (8, 2.31), (9, 2.26),
         (10, 2.23), (15, 2.13), (20, 2.09), (30, 2.04), (60, 2.00)]


def t_factor(df: int) -> float:
    """t(0.975, df) / 1.96: how much a 95% band must widen when the sd rests on df+1 values."""
    if df >= 60:
        return 1.0
    df = max(df, 1)
    for (d0, t0), (d1, t1) in zip(_T975, _T975[1:]):
        if d0 <= df <= d1:
            return (t0 + (t1 - t0) * (df - d0) / (d1 - d0)) / 1.96
    return _T975[0][1] / 1.96


def small_sample_factor(groups: np.ndarray) -> float:
    """How much wider a band must be when it rests on few independent units.

    A bootstrap over g clusters understates the spread when g is small; scale by t(0.975, g−1)/1.96.
    Rows without entities (g = n) are unaffected; a block bootstrap counts its ~10 blocks.
    """
    g = len(np.unique(groups))
    units = g if g >= 5 else 10
    df = max(units - 1, 1)
    if df >= 60:
        return 1.0
    for (d0, t0), (d1, t1) in zip(_T975, _T975[1:]):
        if d0 <= df <= d1:
            t = t0 + (t1 - t0) * (df - d0) / (d1 - d0)
            return t / 1.96
    return _T975[0][1] / 1.96


def resamples(groups: np.ndarray, reps: int, seed: int):
    """Index resamples that respect correlation: whole entities when there are ≥5 of them,
    otherwise contiguous blocks (rows of one bearing are a time series)."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    unique = np.unique(groups)
    if len(unique) >= 5:
        members = [np.flatnonzero(groups == g) for g in unique]
        for _ in range(reps):
            yield np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        return
    block = max(1, n // 10)
    count = -(-n // block)
    for _ in range(reps):
        starts = rng.integers(0, max(1, n - block + 1), count)
        yield np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def paired_difference(task: str, expected: np.ndarray, out_a: np.ndarray, out_b: np.ndarray,
                      groups: np.ndarray, reps: int, seed: int = 0) -> tuple[float, float]:
    """(U(a) − U(b), bootstrap sd of that difference) on the same rows."""
    a, b = _point(task, expected, out_a), _point(task, expected, out_b)
    observed = utility(task, expected, a) - utility(task, expected, b)
    draws = [utility(task, expected[i], a[i]) - utility(task, expected[i], b[i])
             for i in resamples(groups, reps, seed)]
    return observed, float(np.std(draws))


def utility_sd(task: str, expected: np.ndarray, out: np.ndarray, groups: np.ndarray, reps: int, seed: int = 0) -> float:
    point = _point(task, expected, out)
    return float(np.std([utility(task, expected[i], point[i]) for i in resamples(groups, reps, seed)]))
