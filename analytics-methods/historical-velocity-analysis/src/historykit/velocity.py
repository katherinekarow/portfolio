"""Velocity / lead-time analysis: how the orders for one service date arrive.

Arrays are indexed [service date, days_out] with days_out = 0..L-1.
C[d, k] = orders for date d that were received when k or more days were left.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative(by_lead: np.ndarray) -> np.ndarray:
    """C[:, k] = sum of orders with lead >= k (a reverse cumulative sum)."""
    return np.cumsum(by_lead[:, ::-1], axis=1)[:, ::-1]


def velocity(C: np.ndarray) -> np.ndarray:
    """Share of the final count already received at each days-out: V[:, k] = C[:, k] / C[:, 0]."""
    final = C[:, :1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(final > 0, C / final, np.nan)


def pickup(gross: np.ndarray, cancels: np.ndarray | None = None) -> np.ndarray:
    """Incremental orders taken on the day with k days left (net of cancellations if given)."""
    return gross if cancels is None else gross - cancels


def percentiles(V: np.ndarray, q=(25, 50, 75)) -> np.ndarray:
    """Pointwise percentiles across service dates, shape (len(q), L)."""
    return np.nanpercentile(V, q, axis=0)


def pace_vs_history(c_now: float, C_ref: np.ndarray, k: int) -> float:
    """Current cumulative / historical median cumulative at the same days-out (1.0 = on pace)."""
    med = np.median(C_ref[:, k])
    return float(c_now / med) if med > 0 else float("nan")


def pace_projection(c_now: float, V_ref: np.ndarray, k: int) -> dict:
    """Projected final = cumulative today / typical (P50) share received by now.

    The 50% interval divides by the P75 share (low) and the P25 share (high); the 80% interval
    uses P90 and P10 the same way. A share of 0 gives no projection (NaN).
    """
    p10, p25, p50, p75, p90 = np.nanpercentile(V_ref[:, k], (10, 25, 50, 75, 90))

    def div(share):
        return c_now / share if share > 0 else float("nan")

    return {"low": div(p75), "point": div(p50), "high": div(p25), "low80": div(p90), "high80": div(p10),
            "share_p25": p25, "share_p50": p50, "share_p75": p75}


def reference_rows(dates: pd.DatetimeIndex, target: pd.Timestamp, last_complete: int,
                   weeks: int = 52, same_weekday: bool = True) -> np.ndarray:
    """Indices of complete service dates used as history for `target`."""
    idx = np.arange(last_complete)
    keep = dates[idx] > dates[last_complete - 1] - pd.Timedelta(weeks=weeks)
    if same_weekday:
        keep &= dates[idx].weekday == target.weekday()
    return idx[keep]


def backtest_projection(gross: np.ndarray, dates: pd.DatetimeIndex, test_rows, days_out=(7, 14, 28),
                        weeks: int = 52) -> pd.DataFrame:
    """Project held-out complete dates as if seen k days before, using only earlier dates as history."""
    C = cumulative(gross)
    V = velocity(C)
    rows = []
    for d in test_rows:
        for k in days_out:
            # history = dates already complete when date d was k days out
            last_complete = d - k
            ref = reference_rows(dates, dates[d], last_complete + 1, weeks)
            pr = pace_projection(C[d, k], V[ref], k)
            final = C[d, 0]
            rows.append({"date": dates[d], "days_out": k, "observed": C[d, k], "final": final,
                         "projection": pr["point"], "low": pr["low"], "high": pr["high"],
                         "rel_error": pr["point"] / final - 1,
                         "inside": pr["low"] <= final <= pr["high"],
                         "inside80": pr["low80"] <= final <= pr["high80"]})
    return pd.DataFrame(rows)
