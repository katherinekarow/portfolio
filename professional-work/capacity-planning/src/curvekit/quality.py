"""Data-quality checks, a validation gate, and snapshot-first history.

Checks return a tidy table (check, passed, detail) so results can be logged, shown in a
dashboard, or used to block a publish.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd


def run_checks(appointments: pd.DataFrame, sessions: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    cap = sessions.set_index("session_id").capacity
    b = appointments.assign(capacity=appointments.session_id.map(cap))
    results = [
        ("no duplicate session/days_out rows", not appointments.duplicated(["session_id", "days_out"]).any(), ""),
        ("no missing values", not appointments.isna().any().any(), ""),
        ("appointments within 0..capacity", bool(((b.net_appointments >= 0) & (b.net_appointments <= b.capacity)).all()), ""),
        ("net never exceeds gross", bool((b.net_appointments <= b.gross_appointments).all()), ""),
    ]
    mono = (b.sort_values(["session_id", "days_out"], ascending=[True, False])
              .groupby("session_id").net_appointments.apply(lambda s: s.is_monotonic_increasing))
    results.append(("cumulative appointments never decrease", bool(mono.all()), f"{(~mono).sum()} sessions failed"))
    outliers = plan[(plan.net_appointments > plan.projected_high) | (plan.projected_final < 0.5 * plan.capacity)]
    results.append(("projection outliers flagged for review", True, f"{len(outliers)} sessions flagged"))
    return pd.DataFrame(results, columns=["check", "passed", "detail"])


def validation_gate(new: pd.DataFrame, previous: pd.DataFrame | None, key: str, col: str,
                    max_abs_change_pct: float = 0.05) -> tuple[bool, pd.DataFrame]:
    """Block a publish if any value moved more than the tolerance since the last published run."""
    if previous is None or previous.empty:
        return True, pd.DataFrame()
    m = new[[key, col]].merge(previous[[key, col]], on=key, suffixes=("_new", "_prev"))
    m["change_pct"] = (m[f"{col}_new"] - m[f"{col}_prev"]).abs() / m[f"{col}_prev"].clip(lower=1)
    breaches = m[m.change_pct > max_abs_change_pct]
    return breaches.empty, breaches


def _fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def snapshot_first(conn: sqlite3.Connection, table: str, current: pd.DataFrame,
                   now: datetime | None = None) -> str:
    """Copy the current output into HIST_<table> before it is overwritten.

    * every snapshot gets a history_added_at timestamp
    * re-running on the same day with identical data keeps a single (latest) snapshot
    * different data on the same day is kept as an additional snapshot
    """
    now = now or datetime.now(timezone.utc)
    hist = f"HIST_{table}"
    fp = _fingerprint(current)
    day = now.date().isoformat()
    snap = current.assign(history_added_at=now.isoformat(), snapshot_day=day, fingerprint=fp)
    exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (hist,)).fetchone()
    if exists:
        same = conn.execute(f"SELECT COUNT(*) FROM {hist} WHERE snapshot_day=? AND fingerprint=?", (day, fp)).fetchone()[0]
        if same:
            conn.execute(f"DELETE FROM {hist} WHERE snapshot_day=? AND fingerprint=?", (day, fp))
            snap.to_sql(hist, conn, if_exists="append", index=False)
            return "replaced identical same-day snapshot"
    snap.to_sql(hist, conn, if_exists="append", index=False)
    return "added new snapshot"
