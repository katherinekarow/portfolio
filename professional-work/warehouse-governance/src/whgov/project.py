"""Models as code: parse SQL model files, build the dependency graph, materialize in order.

A model file is one SELECT statement with a metadata header:

    -- model: mart.session_plan
    -- description: ...
    -- grain: one row per open session
    -- owner: operations analytics
    -- column session_id: session identifier
    -- test: unique(session_id)

Dependencies are not declared by hand - they are read from the FROM / JOIN clauses
(CTE names are excluded), so the graph can't drift from the SQL.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

LAYERS = ["raw", "stg", "core", "mart", "rpt"]
# FROM x / JOIN x, plus comma-joined tables (FROM a, core.b) - only schema-qualified names are relations
REF_RE = re.compile(r"(?:\b(?:from|join)\s+|,\s*)((?:raw|stg|core|mart|rpt)\.[a-z0-9_]+)\b(?!\s*\()", re.I)
CTE_RE = re.compile(r"(?:with(?:\s+recursive)?|,)\s*([a-z_][a-z0-9_]*)\s*(?:\([^)]*\))?\s+as\s*\(", re.I)


@dataclass
class Model:
    name: str
    path: Path
    sql: str                      # the SELECT body (header removed)
    meta: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)
    tests: list = field(default_factory=list)
    header_lines: int = 0

    @property
    def layer(self) -> str:
        return self.name.split(".")[0]

    @property
    def refs(self) -> set[str]:
        ctes = {c.lower() for c in CTE_RE.findall(strip_comments(self.sql))}
        return {r.lower() for r in REF_RE.findall(strip_comments(self.sql)) if r.split(".")[0].lower() not in ctes}


def strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", "", sql)


def parse_model(path: Path) -> Model:
    text = Path(path).read_text(encoding="utf-8")
    meta, cols, tests, n = {}, {}, [], 0
    for line in text.splitlines():
        m = re.match(r"--\s*(\w+)(?:\s+([a-z0-9_]+))?\s*:\s*(.*)$", line)
        if not line.startswith("--") or not m:
            break
        n += 1
        key, arg, val = m.group(1), m.group(2), m.group(3).strip()
        if key == "column" and arg:
            cols[arg] = val
        elif key == "test":
            tests.append(val)
        else:
            meta[key] = val
    body = "\n".join(text.splitlines()[n:])
    name = meta.get("model") or f"{Path(path).parent.name}.{Path(path).stem}"
    return Model(name=name, path=Path(path), sql=body.strip().rstrip(";"), meta=meta, columns=cols, tests=tests,
                 header_lines=n)


def load_models(folder: Path) -> dict[str, Model]:
    models = {}
    for p in sorted(Path(folder).glob("*/*.sql")):
        m = parse_model(p)
        if m.name in models:
            raise ValueError(f"duplicate model name {m.name} ({p} and {models[m.name].path})")
        models[m.name] = m
    return models


def graph(models: dict[str, Model], sources: set[str]) -> nx.DiGraph:
    g = nx.DiGraph()
    for m in models.values():
        g.add_node(m.name, kind="model", layer=m.layer)
        for r in m.refs:
            if r not in models and r not in sources:
                raise ValueError(f"{m.name} references unknown relation {r}")
            g.add_node(r, kind="model" if r in models else "source", layer=r.split(".")[0])
            g.add_edge(r, m.name)
    cycles = list(nx.simple_cycles(g))
    if cycles:
        raise ValueError("dependency cycle: " + "; ".join(" -> ".join(c + [c[0]]) for c in cycles))
    return g


def sources_in(con: sqlite3.Connection) -> set[str]:
    out = set()
    for layer in LAYERS:
        try:
            rows = con.execute(f"SELECT name FROM {layer}.sqlite_master WHERE type = 'table'").fetchall()
        except sqlite3.OperationalError:
            continue
        out |= {f"{layer}.{r[0]}" for r in rows}
    return out


def build(con: sqlite3.Connection, models: dict[str, Model], select: str | None = None) -> list[dict]:
    """Materialize models in dependency order. `select='+mart.session_plan'` builds a model and its upstream."""
    raw_sources = {s for s in sources_in(con) if s.startswith("raw.")}
    g = graph(models, raw_sources)
    order = [n for n in nx.lexicographical_topological_sort(g) if n in models]
    if select:
        target = select.lstrip("+")
        keep = ({target} | nx.ancestors(g, target)) if select.startswith("+") else {target}
        order = [n for n in order if n in keep]
    log = []
    for name in order:
        m = models[name]
        t = time.perf_counter()
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS {m.sql}")
        rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        log.append({"model": name, "rows": rows, "seconds": round(time.perf_counter() - t, 3)})
    con.commit()
    return log
