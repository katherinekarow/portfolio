"""Pareto / ABC classification, Lorenz curve and Gini coefficient."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_table(labels, values, a_cut: float = 80.0, b_cut: float = 95.0) -> pd.DataFrame:
    """Sort descending; share % and cumulative %. An item is A if the cumulative share *before* it is < a_cut
    (so the item that crosses 80% is still A), B if before it is < b_cut, else C."""
    df = pd.DataFrame({"item": list(labels), "value": np.asarray(list(values), float)})
    df = df.sort_values(["value", "item"], ascending=[False, True]).reset_index(drop=True)
    df["share_pct"] = 100 * df.value / df.value.sum()
    df["cum_pct"] = df.share_pct.cumsum()
    before = df.cum_pct - df.share_pct
    df["abc"] = np.where(before < a_cut - 1e-9, "A", np.where(before < b_cut - 1e-9, "B", "C"))
    return df


def lorenz(values):
    """Points (cumulative population share, cumulative value share), smallest first, starting at (0, 0)."""
    v = np.sort(np.asarray(values, float))
    return np.r_[0, np.arange(1, v.size + 1) / v.size], np.r_[0, v.cumsum() / v.sum()]


def gini(values) -> float:
    """G = Σᵢ (2i − n − 1)·x₍ᵢ₎ / (n·Σx), x sorted ascending, i = 1…n. 0 = equal; (n−1)/n = one item holds all."""
    v = np.sort(np.asarray(values, float))
    n = v.size
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * v).sum() / (n * v.sum()))
