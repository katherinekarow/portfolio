"""Weekly program roll-up: volume, acceptance, revenue and reconciliation health by week."""
from __future__ import annotations

import pandas as pd

from .reconcile import EXCEPTIONS


def weekly_summary(vendor_clean: pd.DataFrame, detail: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """One row per week (Mon-Sun). Weeks that are not finished as of `as_of` are flagged and
    excluded from week-over-week change so a partial Monday never looks like a collapse."""
    v = vendor_clean.assign(week=pd.to_datetime(vendor_clean.updated_at).dt.to_period("W-SUN").dt.start_time.dt.date)
    acc = v.status == "ACCEPTED"
    wk = v.groupby("week").agg(
        offers=("offer_id", "size"),
        accepted=("status", lambda s: (s == "ACCEPTED").sum()),
        declined=("status", lambda s: (s == "DECLINED").sum()),
        cancelled=("status", lambda s: (s == "CANCELLED").sum()),
    )
    wk["accepted_revenue_usd"] = v[acc].groupby("week").amount_usd.sum().round(2)
    wk["avg_accepted_bid_usd"] = v[acc].groupby("week").amount_usd.mean().round(2)
    decided = wk.accepted + wk.declined
    wk["acceptance_rate"] = (wk.accepted / decided.where(decided > 0)).round(3)

    d = detail.merge(v[["offer_id", "week"]], on="offer_id", how="left")
    d["week"] = d.week.fillna(pd.to_datetime(d.i_report_date).dt.to_period("W-SUN").dt.start_time.dt.date)
    exc = d[d.recon_class.isin(EXCEPTIONS)]
    wk["exceptions"] = exc.groupby("week").size()
    wk["revenue_at_risk_usd"] = exc.groupby("week").revenue_at_risk_usd.sum().round(2)
    wk = wk.fillna({"exceptions": 0, "revenue_at_risk_usd": 0.0, "accepted_revenue_usd": 0.0})
    wk["exceptions"] = wk.exceptions.astype(int)
    wk["exception_rate"] = (wk.exceptions / wk.accepted.where(wk.accepted > 0)).round(3)
    wk = wk.reset_index()
    as_of = pd.Timestamp(as_of or pd.to_datetime(vendor_clean.updated_at).max()).date()
    wk["complete_week"] = [(w + pd.Timedelta(days=6)) < as_of for w in pd.to_datetime(wk.week).dt.date]
    full = wk.accepted_revenue_usd.where(wk.complete_week)
    wk["wow_revenue_change"] = full.pct_change(fill_method=None).round(3)
    wk.loc[~wk.complete_week, "exception_rate"] = float("nan")     # charges for an open week are still landing
    return wk


def tier_pair_mix(vendor_clean: pd.DataFrame) -> pd.DataFrame:
    """Acceptance and average bid by grade path - which paths carry the program."""
    v = vendor_clean.assign(path=vendor_clean.from_tier + " → " + vendor_clean.to_tier)
    g = v.groupby("path")
    out = pd.DataFrame({
        "offers": g.size(),
        "accepted": g.status.apply(lambda s: (s == "ACCEPTED").sum()),
        "avg_bid_usd": g.amount_usd.mean().round(2),
        "accepted_revenue_usd": v[v.status == "ACCEPTED"].groupby("path").amount_usd.sum().round(2),
    }).fillna(0)
    out["acceptance_share"] = (out.accepted / out.offers).round(3)
    return out.sort_values("accepted_revenue_usd", ascending=False).reset_index()
