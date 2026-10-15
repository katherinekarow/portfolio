"""Outreach sizing: how many extra appointments must come from outreach (reminders, waitlist calls, partner referrals), and when.

Two allocation strategies are compared:
  baseline   - spread the gap in proportion to how appointments normally arrive (stable, low risk)
  milestones - front-load the gap using cumulative checkpoints (faster closure, more execution risk)

Eligibility rules protect against over-allocation:
  * the gap must be at least `min_gap_units` (and `min_gap_pct` of capacity)
  * the segment benchmark must be supported by at least `min_history` past sessions
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _remaining_shape(curves: pd.DataFrame, segment: str, days_out_today: int) -> pd.DataFrame:
    c = curves[(curves.segment == segment) & (curves.days_out <= days_out_today)].sort_values("days_out", ascending=False)
    inc = np.diff(np.concatenate([[c.median_share.iloc[0]], c.median_share.to_numpy()]))
    inc = np.clip(inc, 0, None)
    w = inc / inc.sum() if inc.sum() > 0 else np.full(len(inc), 1 / len(inc))
    return pd.DataFrame({"days_out": c.days_out.to_numpy(), "weight": w})


def allocate(plan_row: pd.Series, curves: pd.DataFrame, strategy: str = "baseline",
             milestones=((0.25, 0.50), (0.50, 0.80), (1.00, 1.00))) -> pd.DataFrame:
    gap = float(plan_row.gap_to_target)
    shape = _remaining_shape(curves, plan_row.segment, int(plan_row.days_out_today))
    if strategy == "baseline":
        units = gap * shape.weight.to_numpy()
    elif strategy == "milestones":
        n = len(shape)
        t = np.arange(1, n + 1) / n                      # share of remaining time elapsed
        xs = np.concatenate([[0.0], [m[0] for m in milestones]])
        ys = np.concatenate([[0.0], [m[1] for m in milestones]])
        cum = np.interp(t, xs, ys)
        units = gap * np.diff(np.concatenate([[0.0], cum]))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return pd.DataFrame({"session_id": plan_row.session_id, "strategy": strategy,
                         "days_out": shape.days_out, "outreach_units": units})


def eligible(plan: pd.DataFrame, min_gap_units: int = 25, min_gap_pct: float = 0.01,
             min_history: int = 20) -> pd.Series:
    return (plan.gap_to_target >= min_gap_units) & \
           (plan.gap_to_target >= min_gap_pct * plan.capacity) & \
           (plan.n >= min_history)


def build_outreach_plan(plan: pd.DataFrame, curves: pd.DataFrame, strategy: str, **rules) -> pd.DataFrame:
    ok = eligible(plan, **rules)
    parts = [allocate(r, curves, strategy) for _, r in plan[ok].iterrows()]
    if not parts:
        return pd.DataFrame(columns=["session_id", "strategy", "days_out", "outreach_units"])
    return pd.concat(parts, ignore_index=True)


def reconcile(plan: pd.DataFrame, allocations: pd.DataFrame, tol: float = 0.5) -> pd.DataFrame:
    """Control total: allocated units per session must equal its gap."""
    got = allocations.groupby("session_id").outreach_units.sum()
    exp = plan.set_index("session_id").gap_to_target.reindex(got.index)
    out = pd.DataFrame({"expected": exp, "allocated": got.round(2)})
    out["ok"] = (out.expected - out.allocated).abs() <= tol
    return out.reset_index()
