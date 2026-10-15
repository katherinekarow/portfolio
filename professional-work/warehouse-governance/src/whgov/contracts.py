"""Data contracts declared in model headers, executed against the built warehouse.

    unique(col[, col...])                 no duplicate keys
    not_null(col[, col...])               no NULLs
    accepted_values(col: a, b, c)         only the listed values
    expression(<sql predicate>)           every row satisfies the predicate
    relationship(col -> layer.table.col)  every value exists in the parent (no orphans)
    assert_empty(<select>)                custom check: the query must return no rows

Plus documentation coverage: every column in the built table must be described in the header,
and every documented column must exist.
"""
from __future__ import annotations

import re
import sqlite3

from .project import Model


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def compile_test(model: Model, test: str) -> tuple[str, str]:
    """-> (label, SQL returning the number of failing rows)."""
    t = model.name
    m = re.match(r"^(\w+)\((.*)\)$", test.strip(), re.S)
    if not m:
        raise ValueError(f"{t}: cannot parse test {test!r}")
    kind, arg = m.group(1), m.group(2).strip()
    if kind == "unique":
        cols = ", ".join(_split(arg))
        return f"unique({cols})", f"SELECT COUNT(*) FROM (SELECT {cols} FROM {t} GROUP BY {cols} HAVING COUNT(*) > 1)"
    if kind == "not_null":
        cond = " OR ".join(f"{c} IS NULL" for c in _split(arg))
        return f"not_null({arg})", f"SELECT COUNT(*) FROM {t} WHERE {cond}"
    if kind == "accepted_values":
        col, vals = arg.split(":", 1)
        lits = ", ".join("'" + v.replace("'", "''") + "'" for v in _split(vals))
        return f"accepted_values({col.strip()})", f"SELECT COUNT(*) FROM {t} WHERE CAST({col.strip()} AS TEXT) NOT IN ({lits})"
    if kind == "expression":
        return f"expression({arg})", f"SELECT COUNT(*) FROM {t} WHERE NOT ({arg})"
    if kind == "relationship":
        col, parent = [x.strip() for x in arg.split("->")]
        ptable, pcol = parent.rsplit(".", 1)
        return (f"relationship({col} -> {parent})",
                f"SELECT COUNT(*) FROM {t} AS c WHERE c.{col} IS NOT NULL AND NOT EXISTS "
                f"(SELECT 1 FROM {ptable} AS p WHERE p.{pcol} = c.{col})")
    if kind == "assert_empty":
        return "assert_empty(custom)", f"SELECT COUNT(*) FROM ({arg})"
    raise ValueError(f"{t}: unknown test type {kind}")


def run_tests(con: sqlite3.Connection, models: dict[str, Model]) -> list[dict]:
    results = []
    for m in models.values():
        for test in m.tests:
            label, sql = compile_test(m, test)
            try:
                failing = con.execute(sql).fetchone()[0]
                results.append({"model": m.name, "test": label, "failing_rows": failing, "passed": failing == 0})
            except sqlite3.Error as e:
                results.append({"model": m.name, "test": label, "failing_rows": None, "passed": False, "error": str(e)})
        built = [r[1] for r in con.execute(f"PRAGMA {m.layer}.table_info({m.name.split('.')[1]})")]
        undocumented = [c for c in built if c not in m.columns]
        stale = [c for c in m.columns if c not in built]
        results.append({"model": m.name, "test": "documentation", "failing_rows": len(undocumented) + len(stale),
                        "passed": not undocumented and not stale,
                        **({"error": f"undocumented {undocumented}, documented but missing {stale}"} if undocumented or stale else {})})
    return results
