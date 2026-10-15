"""Correlation coefficients written out from their formulas."""
from __future__ import annotations

import numpy as np
from scipy import stats


def pearson(x, y) -> float:
    """r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² · Σ(y−ȳ)²)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    return float(dx @ dy / np.sqrt((dx @ dx) * (dy @ dy)))


def pearson_p(r: float, n: int) -> float:
    """Two-sided p-value for H0: ρ = 0 using t = r√(n−2)/√(1−r²) on n−2 df."""
    if abs(r) >= 1:
        return 0.0
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r * r)
    return float(2 * stats.t.sf(abs(t), n - 2))


def rank(a) -> np.ndarray:
    """Ranks 1..n; tied values share the average of the ranks they span."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    r = np.empty(a.size)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y) -> float:
    """ρ = Pearson correlation of the ranks (handles ties; equals 1 − 6Σd²/(n(n²−1)) without ties)."""
    return pearson(rank(x), rank(y))


def kendall(x, y) -> float:
    """τ-b = (C − D) / √((n0 − n1)(n0 − n2)), counting concordant/discordant pairs; O(n²)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(x.size, 1)
    sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    s = float((sx * sy).sum())  # C − D
    n0 = i.size
    tx, ty = float((sx == 0).sum()), float((sy == 0).sum())
    return s / np.sqrt((n0 - tx) * (n0 - ty))


def summary(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = pearson(x, y)
    return {"n": x.size, "mean_x": x.mean(), "mean_y": y.mean(), "var_x": x.var(ddof=1), "var_y": y.var(ddof=1),
            "pearson_r": r, "pearson_p": pearson_p(r, x.size), "spearman_rho": spearman(x, y),
            "kendall_tau": kendall(x, y)}
