"""Seeing the shape of a big scatter: binned means and a local linear (LOWESS-style) smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd


def binned(x, y, bins: int = 20) -> pd.DataFrame:
    """Equal-count bins of x; mean x, mean y and the standard error of mean y in each."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b + 1, "n": int(m.sum()), "x_mean": x[m].mean(), "y_mean": y[m].mean(),
                     "y_se": y[m].std(ddof=1) / np.sqrt(m.sum())})
    return pd.DataFrame(rows)


def local_linear(x, y, grid=None, frac: float = 0.3, robust_iters: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Local linear regression with tricube weights (the core of LOWESS).

    At each grid point x₀ take the k = frac·n nearest points, weight them w = (1 − (d/d_max)³)³,
    fit a weighted straight line and read it at x₀. robust_iters > 0 adds LOWESS's bisquare
    down-weighting of large residuals (fitted at the data points, so O(n²)).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    grid = np.linspace(x.min(), x.max(), 60) if grid is None else np.asarray(grid, float)
    k = max(3, int(np.ceil(frac * x.size)))
    rw = np.ones_like(x)

    def at(points):
        out = np.empty(points.size)
        for i, x0 in enumerate(points):
            d = np.abs(x - x0)
            near = np.argpartition(d, k - 1)[:k]
            dmax = d[near].max() or 1.0
            w = (1 - (d[near] / dmax) ** 3) ** 3 * rw[near]
            X = np.column_stack([np.ones(k), x[near] - x0])
            sw = np.sqrt(w)
            out[i] = np.linalg.lstsq(X * sw[:, None], y[near] * sw, rcond=None)[0][0]
        return out

    for _ in range(robust_iters):
        e = y - at(x)
        s = np.median(np.abs(e)) * 6 or 1.0
        rw = np.clip(1 - (e / s) ** 2, 0, None) ** 2
    return grid, at(grid)
