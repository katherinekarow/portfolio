"""Fill-curve analytics: historical profiles, target curves, projections and glide paths.

All quantities are per session and indexed by days_out (days to session).
  fill rate (fill)      = appointments / capacity
  share of final        = appointments at days_out / appointments at session
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_profiles(appointments: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Per historical session and days_out: fill rate and share-of-final (net)."""
    hist_ids = sessions.loc[~sessions.is_future, ["session_id", "segment", "capacity"]]
    h = appointments.merge(hist_ids, on="session_id")
    final = h[h.days_out == 0].set_index("session_id").net_appointments
    h["fill"] = h.net_appointments / h.capacity
    h["share_of_final"] = h.net_appointments / h.session_id.map(final).clip(lower=1)
    # survival ratio is only meaningful once a session has a reasonable number of appointments
    h["net_to_gross"] = (h.net_appointments / h.gross_appointments.clip(lower=1)).clip(upper=1).where(h.gross_appointments >= 20)
    return h


def segment_curves(profiles: pd.DataFrame, target_pct: float = 60) -> pd.DataFrame:
    """Benchmarks per segment and days_out.

    target_fill    - percentile `target_pct` of historical fill rate (an ambitious-but-seen pace)
    median_share - typical share of final appointments already on the books
    share_std    - volatility of that share (used for risk bands)
    n            - number of historical sessions supporting the benchmark
    """
    g = profiles.groupby(["segment", "days_out"])
    out = pd.DataFrame({
        "target_fill": g.fill.quantile(target_pct / 100),
        "median_fill": g.fill.median(),
        "median_share": g.share_of_final.median(),
        "share_std": g.share_of_final.std(),
        "net_to_gross": g.net_to_gross.median(),
        "n": g.session_id.nunique(),
    }).reset_index()
    # targets must never go down as session approaches
    out = out.sort_values(["segment", "days_out"], ascending=[True, False])
    out["target_fill"] = out.groupby("segment").target_fill.cummax().clip(upper=1.0)
    out["median_share"] = out.groupby("segment").median_share.cummax().clip(upper=1.0)
    # far from session there are too few appointments to estimate survival: use the nearest estimate
    out["net_to_gross"] = out.groupby("segment").net_to_gross.transform(lambda s: s.bfill().ffill())
    final_fill = out[out.days_out == 0].set_index("segment").median_fill
    out["median_final_fill"] = out.segment.map(final_fill)
    return out


def current_position(appointments: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    fut = sessions[sessions.is_future]
    b = appointments.merge(fut[["session_id", "days_out_today"]], on="session_id")
    now = b[b.days_out == b.days_out_today][["session_id", "net_appointments", "gross_appointments"]]
    return fut.merge(now, on="session_id")


def project_and_decompose(position: pd.DataFrame, curves: pd.DataFrame, z: float = 1.0) -> pd.DataFrame:
    """Project appointments at session and split the gap into understandable pieces.

    projected_final = current / typical share-of-final at today's days_out   (pickup method)
                      blended with the segment's typical final fill rate while the
                      appointment window is young (credibility weighting: little data -> lean on history)
    gap_to_full     = capacity - projected_final
    gap_to_target   = target final appointments - projected_final
    pace_gap_today  = target appointments for today - current appointments
    A volatility band (+/- z * std of share) gives low/high projections.
    """
    p = position.merge(curves, left_on=["segment", "days_out_today"], right_on=["segment", "days_out"])
    share = p.median_share.clip(lower=0.02)
    w = (p.median_share / 0.30).clip(0, 1)                      # credibility of the pickup estimate
    prior = p.median_final_fill * p.capacity

    def blend(pickup):
        return np.minimum(w * pickup + (1 - w) * prior, p.capacity).clip(lower=p.net_appointments).round()

    p["projected_final"] = blend(p.net_appointments / share)
    p["projected_low"] = blend(p.net_appointments / (share + z * p.share_std).clip(upper=1))
    p["projected_high"] = blend(p.net_appointments / (share - z * p.share_std).clip(lower=0.02))
    final_target = curves[curves.days_out == 0].set_index("segment").target_fill
    p["target_final"] = (p.segment.map(final_target) * p.capacity).round()
    p["target_today"] = (p.target_fill * p.capacity).round()
    p["gap_to_full"] = p.capacity - p.projected_final
    p["gap_to_target"] = (p.target_final - p.projected_final).clip(lower=0)
    p["pace_gap_today"] = p.target_today - p.net_appointments
    # status combines today's pace and the projected outcome
    behind_pace = p.pace_gap_today / p.capacity
    short_final = p.gap_to_target / p.capacity
    p["status"] = np.select(
        [(behind_pace <= 0) & (short_final <= 0.01), (behind_pace <= 0.03) & (short_final <= 0.05)],
        ["on pace", "watch"], "behind")
    cols = ["session_id", "segment", "capacity", "days_out_today", "net_appointments", "gross_appointments",
            "projected_low", "projected_final", "projected_high", "target_today", "target_final",
            "pace_gap_today", "gap_to_target", "gap_to_full", "status", "n"]
    return p[cols]


def glide_path(row: pd.Series, curves: pd.DataFrame, goal: float) -> pd.DataFrame:
    """Daily path from today's appointments to `goal` at session, following the segment's typical shape.

    Remaining demand is allocated in proportion to how appointments usually arrive from today onward
    (residual-demand allocation), so the path is realistic rather than a straight line.
    """
    c = curves[(curves.segment == row.segment) & (curves.days_out <= row.days_out_today)].sort_values("days_out", ascending=False)
    s = c.median_share.to_numpy()
    s0 = s[0]
    progress = (s - s0) / max(1 - s0, 1e-9)
    path = row.net_appointments + (goal - row.net_appointments) * progress
    return pd.DataFrame({"session_id": row.session_id, "days_out": c.days_out.to_numpy(), "glide_path": path.round()})
