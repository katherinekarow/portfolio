"""Build a benchmark database: ~1M appointments, ~2.5M status events, 2,000 sessions, 40 referrers.

Generated with NumPy and loaded in chunks, so it takes a few seconds. No indexes are created
here except primary keys - each tuning case adds the indexes it argues for.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

DAY0 = np.datetime64("2024-01-01", "D")
_DAYS = {}


def day(offset: int) -> str:
    """ISO date `offset` days after 2024-01-01 (cached - called millions of times)."""
    if offset not in _DAYS:
        _DAYS[offset] = str(DAY0 + np.timedelta64(int(offset), "D"))
    return _DAYS[offset]


def build(path: str | Path = ":memory:", n_appointments: int = 1_000_000, seed: int = 29) -> sqlite3.Connection:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(str(path))
    con.executescript("""
        PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA temp_store = MEMORY; PRAGMA cache_size = -200000;
        DROP TABLE IF EXISTS sessions; DROP TABLE IF EXISTS referrers; DROP TABLE IF EXISTS appointments; DROP TABLE IF EXISTS appointment_events;
        CREATE TABLE sessions (session_id INTEGER PRIMARY KEY, session_date TEXT, region TEXT, capacity INTEGER);
        CREATE TABLE referrers (referrer_id INTEGER PRIMARY KEY, referrer_name TEXT, channel TEXT);
        CREATE TABLE appointments (appointment_id INTEGER PRIMARY KEY, session_id INTEGER, referrer_id INTEGER,
                               booked_ts TEXT, units INTEGER, charge REAL);
        CREATE TABLE appointment_events (event_id INTEGER PRIMARY KEY, appointment_id INTEGER, event_ts TEXT, status TEXT);
    """)
    n_dep = 2000
    dep_days = rng.integers(0, 900, n_dep)
    con.executemany("INSERT INTO sessions VALUES (?,?,?,?)",
                    [(i, day(d), r, int(c)) for i, (d, r, c) in
                     enumerate(zip(dep_days, rng.choice(["north", "south", "east", "west"], n_dep), rng.choice([300, 600, 900], n_dep)))])
    con.executemany("INSERT INTO referrers VALUES (?,?,?)",
                    [(i, f"Referrer {i:02d}", ["ONLINE", "PHONE", "PARTNER"][i % 3]) for i in range(40)])

    chunk = 200_000
    eid = 0
    for start in range(0, n_appointments, chunk):
        n = min(chunk, n_appointments - start)
        ids = np.arange(start, start + n)
        dep = rng.integers(0, n_dep, n)
        lead = rng.integers(1, 365, n)
        bday = dep_days[dep] - lead
        secs = rng.integers(0, 86400, n)
        ts = [f"{day(d)} {s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}" for d, s in zip(bday, secs)]
        agent = np.where(rng.random(n) < 0.3, -1, rng.integers(0, 40, n))
        con.executemany("INSERT INTO appointments VALUES (?,?,?,?,?,?)",
                        [(int(i), int(d), None if a < 0 else int(a), t, int(u), float(f)) for i, d, a, t, u, f in
                         zip(ids, dep, agent, ts, rng.integers(1, 5, n), rng.normal(900, 250, n).round(2))])
        # 1-4 status events per appointment: CONFIRMED, then maybe RESCHEDULED / CANCELLED, later timestamps
        k = rng.integers(1, 5, n)
        rows = []
        for i, d, s, kk in zip(ids, bday, secs, k):
            day_ = int(d)
            for j in range(int(kk)):
                status = "CONFIRMED" if j == 0 else ("CANCELLED" if j == kk - 1 and rng.random() < 0.35 else "RESCHEDULED")
                rows.append((eid, int(i), f"{day(day_ + j * 3)} {s // 3600:02d}:00:00", status))
                eid += 1
        con.executemany("INSERT INTO appointment_events VALUES (?,?,?,?)", rows)
    con.commit()
    con.execute("ANALYZE")
    return con


if __name__ == "__main__":
    t = time.time()
    c = build("bench.sqlite")
    print({tb: c.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0] for tb in ("sessions", "appointments", "appointment_events")},
          f"{time.time() - t:.1f}s")
