"""Synthetic source tables for the legacy reports, with the NULLs that make migrations hard."""
from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd


def make_tables(seed: int = 15, n_sessions: int = 40, n_appointments: int = 3000) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 9, 1)
    deps = pd.DataFrame({
        "session_id": [f"D{i:03d}" for i in range(n_sessions)],
        "session_date": [(start + dt.timedelta(days=int(d))).isoformat() for d in rng.integers(0, 720, n_sessions)],
        "region": rng.choice(["north", "south", "east", "west"], n_sessions),
        "capacity": rng.choice([200, 300, 400], n_sessions),
    })
    referrers = pd.DataFrame({
        "referrer_id": [f"A{i:02d}" for i in range(8)],
        "referrer_name": [f"Referrer {c}" for c in "ABCDEFGH"],
        "channel": ["ONLINE", "PHONE", "PHONE", None, "ONLINE", "PARTNER", "PHONE", "ONLINE"],   # one referrer has no channel
        "parent_referrer_id": [None, "A00", "A00", None, "A03", "A03", "A01", "A04"],
    })
    dep_idx = rng.integers(0, n_sessions, n_appointments)
    dep_dates = pd.to_datetime(deps.session_date.to_numpy()[dep_idx])
    lead = rng.integers(1, 300, n_appointments)
    appointments = pd.DataFrame({
        "appointment_id": [f"B{i:05d}" for i in range(n_appointments)],
        "session_id": deps.session_id.to_numpy()[dep_idx],
        "appointment_date": (dep_dates - pd.to_timedelta(lead, unit="D")).strftime("%Y-%m-%d"),
        "status": rng.choice(np.array(["A", "C", None, "X"], dtype=object), n_appointments, p=[.8, .12, .05, .03]),
        "referrer_id": np.where(rng.random(n_appointments) < 0.4, None, rng.choice(referrers.referrer_id, n_appointments)),
        "units": rng.integers(1, 5, n_appointments),
        "charge": np.where(rng.random(n_appointments) < 0.03, np.nan, rng.normal(900, 250, n_appointments).round(2)),
    })
    appointments["referrer_id"] = appointments.referrer_id.astype(object).where(appointments.referrer_id.notna(), None)
    return {"sessions": deps, "referrers": referrers, "appointments": appointments}


def to_sqlite(tables: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, con, index=False)
    return con
