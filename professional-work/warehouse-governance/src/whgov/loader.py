"""Load synthetic raw data into a layered SQLite warehouse.

Each layer is its own attached database file (raw, stg, core, mart, rpt), which gives real
`layer.table` names in SQL - the same shape as schemas in a cloud warehouse.

The raw feed contains the problems real feeds have, recorded in `INJECTED` so tests can
check that staging rejects exactly those rows:
  * re-sent events (same event_id twice)            -> kept once
  * session codes with stray spaces / lower case  -> standardized, not rejected
  * blank session codes                           -> MISSING_SESSION
  * zero or negative units                          -> INVALID_UNITS
  * unknown event types                             -> UNKNOWN_EVENT_TYPE
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np

LAYERS = ["raw", "stg", "core", "mart", "rpt"]
AS_OF = dt.date(2026, 3, 2)
REGIONS = {"north": ("flu-season", "multi-day"), "south": ("off-season", "one-day"), "east": ("flu-season", "one-day"), "west": ("off-season", "multi-day")}


def connect(folder: Path | None = None) -> sqlite3.Connection:
    """In-memory by default; pass a folder to keep one .sqlite file per layer."""
    con = sqlite3.connect(":memory:" if folder is None else str(Path(folder) / "main.sqlite"))
    for layer in LAYERS:
        target = ":memory:" if folder is None else str(Path(folder) / f"{layer}.sqlite")
        con.execute(f"ATTACH DATABASE '{target}' AS {layer}")
    return con


def load_raw(con: sqlite3.Connection, n_sessions: int = 80, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    deps, events = [], []
    eid = 0
    for i in range(n_sessions):
        region = list(REGIONS)[i % 4]
        season, length = REGIONS[region]
        open_ = i >= int(n_sessions * 0.75)
        ddate = AS_OF + dt.timedelta(days=int(rng.integers(20, 250))) if open_ else AS_OF - dt.timedelta(days=int(rng.integers(5, 400)))
        cap = int(rng.choice([300, 400, 500, 600]))
        deps.append((f"S{i:04d}", ddate.isoformat(), region, season, length, cap))
        mid = {"flu-season": 150, "off-season": 90}[season] * rng.uniform(0.8, 1.2)
        target_units = cap * rng.uniform(0.78, 0.98)
        booked = 0
        while booked < target_units:
            days_out = int(np.clip(rng.logistic(mid, 30), 0, 365))
            day = ddate - dt.timedelta(days=days_out)
            units = int(rng.integers(1, 5))
            booked += units
            if day > AS_OF:
                continue                                         # future demand not booked yet
            eid += 1
            code = f"S{i:04d}"
            events.append((f"E{eid:07d}", f"B{eid:07d}", code, "NEW", units, f"{day.isoformat()} {rng.integers(0, 24):02d}:00:00"))
            if rng.random() < 0.08:                              # cancellation, later than the appointment
                cday = min(day + dt.timedelta(days=int(rng.integers(1, max(2, days_out)))), ddate, AS_OF)
                eid += 1
                events.append((f"E{eid:07d}", f"B{eid - 1:07d}", code, "CANCEL", units, f"{cday.isoformat()} 12:00:00"))

    injected = {"duplicates": set(), "MISSING_SESSION": set(), "INVALID_UNITS": set(), "UNKNOWN_EVENT_TYPE": set(), "messy_codes": set()}
    idx = rng.permutation(len(events))
    for k in idx[:40]:                                           # messy but fixable codes
        e = list(events[k]); e[2] = f"  {e[2].lower()} "; events[k] = tuple(e); injected["messy_codes"].add(e[0])
    for k in idx[40:52]:
        e = list(events[k]); e[2] = "   "; events[k] = tuple(e); injected["MISSING_SESSION"].add(e[0])
    for k in idx[52:61]:
        e = list(events[k]); e[4] = int(rng.choice([0, -1, -2])); events[k] = tuple(e); injected["INVALID_UNITS"].add(e[0])
    for k in idx[61:66]:
        e = list(events[k]); e[3] = "XFER"; events[k] = tuple(e); injected["UNKNOWN_EVENT_TYPE"].add(e[0])
    resend = [events[k] for k in idx[66:96]]
    injected["duplicates"] = {e[0] for e in resend}
    events += resend

    con.executescript("""
        DROP TABLE IF EXISTS raw.appointments_feed; DROP TABLE IF EXISTS raw.session_master; DROP TABLE IF EXISTS raw.run_context;
        CREATE TABLE raw.appointments_feed (event_id TEXT, appointment_id TEXT, session_code TEXT, event_type TEXT, units INTEGER, event_ts TEXT);
        CREATE TABLE raw.session_master (session_code TEXT, session_date TEXT, region TEXT, season TEXT, length_class TEXT, capacity INTEGER);
        CREATE TABLE raw.run_context (as_of_date TEXT);
    """)
    con.executemany("INSERT INTO raw.appointments_feed VALUES (?,?,?,?,?,?)", events)
    con.executemany("INSERT INTO raw.session_master VALUES (?,?,?,?,?,?)", deps)
    con.execute("INSERT INTO raw.run_context VALUES (?)", (AS_OF.isoformat(),))
    con.commit()
    return injected
