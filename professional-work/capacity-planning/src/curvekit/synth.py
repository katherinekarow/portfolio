"""Synthetic appointment data for a clinic network's screening sessions.

Everything here is invented: sessions, capacities, demand shapes and cancellations
are generated from simple, documented rules so the methods can be shown publicly.

Vocabulary
  session     one scheduled screening session (a clinic day or multi-day drive) with a fixed
              number of appointment slots
  days_out    days to session (365 ... 0)
  segment     group of sessions with similar demand behaviour (reach x season x length)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = {
    # name: (median appointment day, steepness, final fill rate mean)
    "regional / flu season / multi-day": (150, 0.022, 0.97),
    "regional / off-season / multi-day": (110, 0.025, 0.90),
    "local / flu season / one-day": (70, 0.035, 0.95),
    "local / off-season / one-day": (45, 0.045, 0.86),
}
MAX_DAYS_OUT = 365


def _cum_shape(days_out: np.ndarray, mid: float, k: float) -> np.ndarray:
    """Share of final appointments already on the books at `days_out` days out (logistic in time)."""
    raw = 1.0 / (1.0 + np.exp(k * (days_out - mid)))
    start = 1.0 / (1.0 + np.exp(k * (MAX_DAYS_OUT - mid)))
    end = 1.0 / (1.0 + np.exp(k * (0 - mid)))
    return (raw - start) / (end - start)


def make_sessions(n_hist: int = 240, n_future: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    rows = []
    for i in range(n_hist + n_future):
        seg = names[i % len(names)]
        mid, k, fill = SEGMENTS[seg]
        rows.append({
            "session_id": f"S{i:04d}",
            "segment": seg,
            "capacity": int(rng.choice([1200, 1800, 2400, 3000])),
            "is_future": i >= n_hist,
            "mid": mid * rng.uniform(0.85, 1.15),
            "k": k * rng.uniform(0.85, 1.15),
            "final_fill": float(np.clip(rng.normal(fill, 0.04), 0.6, 1.0)),
            # future sessions are observed partway through their appointment window
            "days_out_today": int(rng.integers(30, 240)) if i >= n_hist else 0,
        })
    return pd.DataFrame(rows)


def make_appointments(sessions: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Daily cumulative gross and net appointments per session.

    Future sessions only have rows from MAX_DAYS_OUT down to their `days_out_today`.
    """
    rng = np.random.default_rng(seed)
    days_out = np.arange(MAX_DAYS_OUT, -1, -1)
    out = []
    for d in sessions.itertuples():
        net_final = d.final_fill * d.capacity
        shape = _cum_shape(days_out, d.mid, d.k)
        noise = np.cumsum(rng.normal(0, 0.002, len(days_out)))
        net_cum = np.clip(net_final * (shape + noise - noise[-1] * shape), 0, d.capacity)
        net_cum = np.maximum.accumulate(net_cum)
        # cancellations: higher far out, falling to ~0 close in
        cx_share = 0.18 * (days_out / MAX_DAYS_OUT) + 0.02
        gross_cum = net_cum / (1 - cx_share * 0.6)
        frame = pd.DataFrame({
            "session_id": d.session_id,
            "days_out": days_out,
            "gross_appointments": np.round(gross_cum).astype(int),
            "net_appointments": np.round(net_cum).astype(int),
        })
        if d.is_future:
            frame = frame[frame.days_out >= d.days_out_today]
        out.append(frame)
    return pd.concat(out, ignore_index=True)
