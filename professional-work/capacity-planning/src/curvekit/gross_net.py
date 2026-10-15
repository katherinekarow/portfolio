"""Gross vs. net planning.

Gross appointments include appointments that will later cancel; net appointments are what actually shows up.
Planning on gross alone overstates the outcome, so targets are translated with a
days_out-dependent survival factor learned from history:

    survival(days_out)   = median( net appointments / gross appointments ) at that days_out
    net_target(days_out) = gross_target(days_out) * survival(days_out)
    cancel_rate(days_out)= 1 - survival(days_out)
"""
from __future__ import annotations

import pandas as pd


def cancellation_profile(curves: pd.DataFrame) -> pd.DataFrame:
    out = curves[["segment", "days_out", "net_to_gross"]].rename(columns={"net_to_gross": "survival"})
    out["cancel_rate"] = 1 - out.survival
    return out


def gross_to_net(position: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Expected net appointments from the gross appointments currently on the books."""
    p = position.merge(profile, left_on=["segment", "days_out_today"], right_on=["segment", "days_out"])
    p["expected_net_from_gross"] = (p.gross_appointments * p.survival).round()
    p["gross_net_gap"] = p.gross_appointments - p.expected_net_from_gross
    return p[["session_id", "segment", "days_out_today", "gross_appointments", "net_appointments",
              "cancel_rate", "expected_net_from_gross", "gross_net_gap"]]
