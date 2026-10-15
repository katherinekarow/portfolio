"""Run every case: plan + timing + result check, slow vs fast."""
from __future__ import annotations

import re
import sqlite3
import statistics
import time

from .cases import CASES, Case


def _index_names(sqls: list[str]) -> list[str]:
    return [re.search(r"index\s+(\w+)", s, re.I).group(1) for s in sqls]


def _drop(con: sqlite3.Connection, sqls: list[str]):
    for n in _index_names(sqls):
        con.execute(f"DROP INDEX IF EXISTS {n}")


def _create(con: sqlite3.Connection, sqls: list[str]):
    for s in sqls:
        con.execute(s.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS"))
    con.execute("ANALYZE")


def plan(con: sqlite3.Connection, sql: str) -> list[str]:
    return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql)]


def _time(con: sqlite3.Connection, sql: str, repeat: int) -> tuple[float, list]:
    times, rows = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        rows = con.execute(sql).fetchall()
        times.append(time.perf_counter() - t)
    return statistics.median(times), rows


def _normalize(rows: list) -> list:
    rows = [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows]
    return sorted(rows, key=lambda r: tuple((1, 0) if v is None else (0, v) for v in r))


def run_case(con: sqlite3.Connection, case: Case, repeat: int = 3) -> dict:
    all_idx = case.shared_indexes + case.fast_indexes
    _drop(con, all_idx + [s for c in CASES for s in c.shared_indexes + c.fast_indexes])   # clean slate
    _create(con, case.shared_indexes)
    slow_plan = plan(con, case.slow_sql)
    slow_t, slow_rows = _time(con, case.slow_sql, repeat)
    _create(con, case.fast_indexes)
    fast_plan = plan(con, case.fast_sql)
    fast_t, fast_rows = _time(con, case.fast_sql, repeat)
    alt = None
    if case.alt_sql:
        alt_t, alt_rows = _time(con, case.alt_sql, repeat)
        alt = {"label": case.alt_label, "seconds": alt_t, "same_result": _normalize(alt_rows) == _normalize(fast_rows),
               "plan": plan(con, case.alt_sql)}
    _drop(con, all_idx)
    same = _normalize(slow_rows) == _normalize(fast_rows)
    out = {"case": case.name, "title": case.title, "slow_s": slow_t, "fast_s": fast_t,
           "speedup": slow_t / fast_t if fast_t > 0 else float("inf"),
           "same_result": same, "expected_same": case.same_result,
           "slow_plan": slow_plan, "fast_plan": fast_plan, "rows": len(fast_rows), "alt": alt}
    if not case.same_result:
        # how wrong was the slow version? compare the last numeric column in total
        s = sum(r[-1] for r in slow_rows if isinstance(r[-1], (int, float)))
        f = sum(r[-1] for r in fast_rows if isinstance(r[-1], (int, float)))
        out["overstatement_pct"] = (s / f - 1) * 100 if f else float("nan")
    return out


def run_all(con: sqlite3.Connection, repeat: int = 3, only: list[str] | None = None) -> list[dict]:
    return [run_case(con, c, repeat) for c in CASES if not only or c.name in only]
