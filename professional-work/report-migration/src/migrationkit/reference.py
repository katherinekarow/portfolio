"""Independent pandas implementations of what the legacy reports return under ORACLE semantics.

They are written from the report definitions, not from the translated SQL, so comparing the
two is a real check: if the translator got DECODE's NULL handling or an outer join wrong,
the parity harness will show it.
"""
from __future__ import annotations

import pandas as pd


def _decode_status(s: pd.Series) -> pd.Series:
    # DECODE(status, 'A','Active', 'C','Cancelled', NULL,'Unknown', 'Other'): DECODE matches NULL to NULL
    return s.map({"A": "Active", "C": "Cancelled"}).where(s.notna(), "Unknown").fillna("Other")


def appointment_activity(t: dict[str, pd.DataFrame], as_of: str = "2026-03-01", charge_nvl: bool = True) -> pd.DataFrame:
    """charge_nvl=False reproduces an older version of the report that summed charge * units without NVL:
    Oracle's SUM skips NULLs, so the only visible difference is NULL revenue for groups where every charge is NULL."""
    b = t["appointments"].merge(t["sessions"], on="session_id")                          # inner join
    b = b.merge(t["referrers"][["referrer_id", "channel"]], on="referrer_id", how="left")         # (+) outer join
    cutoff = (pd.Timestamp(as_of) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    b = b[b.session_date >= cutoff]
    out = b.assign(
        appointment_month=pd.to_datetime(b.appointment_date).dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d"),
        status_label=_decode_status(b.status),
        channel=b.channel.fillna("DIRECT"),
        rev=(b.charge.fillna(0) if charge_nvl else b.charge) * b.units,
    ).groupby(["session_id", "appointment_month", "status_label", "channel"], as_index=False).agg(
        units=("units", "sum"), revenue=("rev", lambda x: x.sum(min_count=1)))
    return out


def capacity_position(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d, b = t["sessions"], t["appointments"]
    b = b.assign(net=b.units.where(b.status != "C", 0))       # DECODE(status,'C',0,units): NULL status keeps units
    g = b.groupby("session_id").agg(booked=("net", "sum"), referrers=("referrer_id", "count")).reset_index()
    out = d.merge(g, on="session_id", how="left")
    out["remaining"] = out.capacity - out.booked
    out["referral_mix"] = out.referrers.gt(0).map({True: "REFERRED", False: "DIRECT"})
    return out[["session_id", "capacity", "booked", "remaining", "referral_mix"]]


def top_sessions(t: dict[str, pd.DataFrame], n: int = 5) -> pd.DataFrame:
    b = t["appointments"]
    b = b[b.status.fillna("A") != "C"]
    r = (b.charge.fillna(0) * b.units).groupby(b.session_id).sum().rename("revenue").reset_index()
    return r.sort_values(["revenue", "session_id"], ascending=[False, True]).head(n).reset_index(drop=True)
