"""Robust line fits: Theil-Sen and Huber M-estimation (IRLS)."""
from __future__ import annotations

import numpy as np


def theil_sen(x, y) -> tuple[float, float]:
    """Slope = median of all pairwise slopes (y_j − y_i)/(x_j − x_i); intercept = median(y − slope·x).

    Breakdown point ≈ 29%: up to that share of points can be arbitrarily bad before the slope is.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    keep = dx != 0
    slope = float(np.median((y[j] - y[i])[keep] / dx[keep]))
    return float(np.median(y - slope * x)), slope


def huber(x, y, c: float = 1.345, iters: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Huber regression by iteratively reweighted least squares.

    Loss is quadratic for |u| ≤ c and linear beyond, u = e / scale, scale = MAD/0.6745.
    Weight w = min(1, c/|u|). c = 1.345 keeps 95% efficiency when errors are normal.
    It down-weights large residuals but not high-leverage x positions.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        e = y - X @ beta
        scale = np.median(np.abs(e - np.median(e))) / 0.6745
        u = np.abs(e) / max(scale, 1e-12)
        w = np.minimum(1.0, c / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])
