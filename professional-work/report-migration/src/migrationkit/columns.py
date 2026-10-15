"""Output-column audit: the root cause of "merged" and "broken" columns after a migration.

A report tool binds to the NAMES of a query's output columns. Legacy queries often left
expressions unaliased, so the name was whatever the database generated:

    Oracle     SUM(DECODE(B.STATUS,'C',0,B.UNITS))
    Snowflake  SUM(CASE WHEN B.STATUS = 'C' THEN 0 ELSE B.UNITS END)    <- different after translation

or selected two columns with the same name (d.session_id, b.session_id), which many
tools silently collapse into one. This module lists every output column with the name
each engine would give it and flags the ones a report can't bind to reliably.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .sqltext import find_top_keyword, mask, split_top


@dataclass
class OutputColumn:
    position: int
    expression: str
    name: str | None            # explicit or implicit alias / bare column name
    issue: str = ""


def _select_list(sql: str) -> str:
    m = mask(sql)
    s = find_top_keyword(m, "select")
    f = find_top_keyword(m, "from", s)
    return sql[s + 6: f if f >= 0 else len(sql)]


def output_columns(sql: str) -> list[OutputColumn]:
    cols = []
    for i, raw in enumerate(split_top(_select_list(sql)), start=1):
        expr = " ".join(re.sub(r"--[^\n]*", "", raw).split())
        expr = re.sub(r"^/\*\+.*?\*/\s*", "", expr)
        m_as = re.match(r"^(.*?)\s+as\s+(\"?[A-Za-z_][\w$#]*\"?)$", expr, re.I | re.S)
        m_implicit = re.match(r"^(.*[\)\w'])\s+([A-Za-z_][\w$#]*)$", expr, re.S)
        if expr == "*" or expr.endswith(".*"):
            cols.append(OutputColumn(i, expr, None, "SELECT * - column set depends on the table definition"))
        elif m_as:
            cols.append(OutputColumn(i, m_as.group(1), m_as.group(2).strip('"').lower()))
        elif re.fullmatch(r"(\w+\.)?\w+", expr):
            cols.append(OutputColumn(i, expr, expr.split(".")[-1].lower()))
        elif m_implicit and not re.search(r"\b(end|and|or|then|else|null)$", m_implicit.group(0), re.I) \
                and m_implicit.group(2).upper() not in {"END", "NULL", "DESC", "ASC"}:
            cols.append(OutputColumn(i, m_implicit.group(1), m_implicit.group(2).lower(),
                                     "implicit alias without AS - easy to break by editing; add AS"))
        else:
            cols.append(OutputColumn(i, expr, None, "expression without alias - engine-generated name changes between databases"))
    seen = {}
    for c in cols:
        if c.name:
            if c.name in seen:
                c.issue = (c.issue + "; " if c.issue else "") + f"duplicate output name '{c.name}' (also column {seen[c.name]}) - BI tools merge these"
            else:
                seen[c.name] = c.position
    return cols


def suggest_alias(expr: str) -> str:
    """A readable snake_case alias from an expression, e.g. 'd.capacity - SUM(...)' -> 'capacity_minus_sum'."""
    e = re.sub(r"'[^']*'", "", expr.lower())
    e = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", e)                     # drop table qualifiers (b.units -> units)
    e = e.replace(" - ", " minus ").replace(" + ", " plus ")
    words = [w for w in re.findall(r"[a-z_]+", e) if w not in {"case", "when", "then", "else", "end", "is", "not", "null", "decode", "nvl", "coalesce"}]
    words = [w.split(".")[-1] for w in words]
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    name = "_".join(out[:4]) or "expr"
    return re.sub(r"_+", "_", name)[:40]


def fix_aliases(sql: str) -> tuple[str, list[str]]:
    """Add AS aliases to unaliased expressions and de-duplicate names. Returns (sql, notes)."""
    cols = output_columns(sql)
    sel = _select_list(sql)
    items = split_top(sel)
    used, notes, new_items = set(), [], []
    for item, c in zip(items, cols):
        lead = item[:len(item) - len(item.lstrip())]
        trail = item[len(item.rstrip()):]
        body = item.strip()
        name = c.name
        if c.name is None and not c.expression.endswith("*"):
            name = suggest_alias(c.expression)
            body = f"{body} AS {name}"
        elif c.issue.startswith("implicit alias"):
            body = f"{c.expression} AS {c.name}"
        if name and name in used:
            qual = c.expression.split(".")[0].lower() if "." in c.expression else f"col{c.position}"
            new = f"{qual}_{name}"
            body = re.sub(r"\s+as\s+\w+$", "", body, flags=re.I) + f" AS {new}"
            notes.append(f"column {c.position}: renamed duplicate '{name}' to '{new}'")
            name = new
        elif name != c.name:
            notes.append(f"column {c.position}: added alias '{name}'")
        used.add(name)
        new_items.append(lead + body + trail)
    start = sql.find(sel)
    return sql[:start] + ",".join(new_items) + sql[start + len(sel):], notes
