"""Unlabelled drift evidence: RBF-MMD with a calibrated null threshold, and PSI per feature."""
from __future__ import annotations

import numpy as np


def mmd_rbf(a: np.ndarray, b: np.ndarray, max_samples: int = 256, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if len(a) > max_samples:
        a = a[rng.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[rng.choice(len(b), max_samples, replace=False)]
    joined = np.vstack([a, b])
    squared = np.sum(joined ** 2, axis=1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * joined @ joined.T, 0.0)
    bandwidth = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    kernel = np.exp(-distances / max(bandwidth, 1e-12))
    n = len(a)
    value = kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
    return float(max(value, 0.0))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> tuple[float, float]:
    """Population stability index per feature on reference-quantile bins: (max, mean)."""
    values = []
    for column in range(reference.shape[1]):
        edges = np.unique(np.quantile(reference[:, column], np.linspace(0, 1, bins + 1)[1:-1]))
        expected = np.bincount(np.searchsorted(edges, reference[:, column]), minlength=len(edges) + 1)
        actual = np.bincount(np.searchsorted(edges, current[:, column]), minlength=len(edges) + 1)
        p = np.maximum(expected / expected.sum(), 1e-4)
        q = np.maximum(actual / actual.sum(), 1e-4)
        values.append(float(np.sum((q - p) * np.log(q / p))))
    return float(np.max(values)), float(np.mean(values))


class DriftMeter:
    """Compares a window with the data the serving model last adapted to.

    The threshold is not a constant: it is the `quantile` of MMD between two disjoint samples of
    the reference set itself, at the size actually being compared.
    """

    def __init__(self, reference: np.ndarray, quantile: float, seed: int = 0, draws: int = 20):
        self.reference = reference
        self.quantile = quantile
        self.seed = seed
        self.draws = draws
        self._thresholds: dict[int, float] = {}

    def threshold(self, size: int) -> float:
        size = max(8, min(size, len(self.reference) // 2, 256))
        if size not in self._thresholds:
            rng = np.random.default_rng(self.seed)
            null = []
            for draw in range(self.draws):
                order = rng.permutation(len(self.reference))
                null.append(mmd_rbf(self.reference[order[:size]], self.reference[order[size:2 * size]], seed=draw))
            self._thresholds[size] = float(np.quantile(null, self.quantile))
        return self._thresholds[size]

    def measure(self, current: np.ndarray, previous: np.ndarray | None) -> dict:
        against = previous if previous is not None and len(previous) >= 8 else self.reference
        threshold = self.threshold(len(current))
        recent = mmd_rbf(against, current, seed=self.seed)
        psi_max, psi_mean = psi(self.reference, current)
        return {"mmd_vs_recent": recent, "mmd_vs_base": mmd_rbf(self.reference, current, seed=self.seed),
                "threshold": threshold, "shifted": bool(recent > threshold),
                "psi_max": psi_max, "psi_mean": psi_mean}
