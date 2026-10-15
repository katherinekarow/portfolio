"""SQL convention linter for warehouse models.

| code  | severity | rule |
|-------|----------|------|
| WG001 | error    | no `SELECT *` / `t.*` in models - columns must be explicit so upstream changes can't leak through |
| WG002 | error    | every computed column needs an explicit `AS alias` (unaliased expressions get engine-generated names that collide or change between databases - the root cause of "merged" report columns) |
| WG003 | error    | no implicit comma joins - write `INNER JOIN ... ON` |
| WG004 | error    | layer rules: stg<-raw, core<-raw/stg/core, mart<-core/mart, rpt<-mart only |
| WG005 | error    | header must have description, grain, owner and at least one uniqueness test (the grain key) |
| WG006 | error    | no hard-coded dates - read the run date from raw.run_context |
| WG007 | error    | file must live at models/<layer>/<table>.sql and match its declared name; aliases snake_case |
| WG008 | warning  | no top-level ORDER BY in a model (tables have no order; sort in the report) |
| WG009 | warning  | SQL keywords in upper case |
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .project import Model

ALLOWED = {"stg": {"raw"}, "core": {"raw", "stg", "core"}, "mart": {"core", "mart"}, "rpt": {"mart"}}
KEYWORDS = ["select", "from", "where", "join", "inner join", "left join", "group by", "order by", "having",
            "with", "as", "on", "and", "or", "case", "when", "then", "else", "end", "union all", "over", "partition by"]


@dataclass(frozen=True)
class Finding:
    model: str
    line: int
    code: str
    severity: str
    message: str

    def __str__(self):
        return f"{self.model}:{self.line}  {self.code} [{self.severity}] {self.message}"


def mask(sql: str) -> str:
    """Blank out comments and string literals (keeping length and newlines) so rules see only code."""
    out = list(sql)
    for m in re.finditer(r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'", sql, re.S):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " " if m.group(0).startswith("-") or m.group(0).startswith("/") else ("'" if i in (m.start(), m.end() - 1) else "x")
    return "".join(out)


def _depths(s: str) -> list[int]:
    d, out = 0, []
    for ch in s:
        if ch == "(":
            d += 1
        out.append(d)
        if ch == ")":
            d -= 1
    return out


def select_lists(code: str):
    """Yield (start_offset, items) for every SELECT list, items as (offset, text)."""
    depth = _depths(code)
    low = code.lower()
    for m in re.finditer(r"\bselect\b", low):
        start, d = m.end(), depth[m.start()]
        i, items, cur_start = start, [], start
        while i < len(code):
            # inner parentheses (and their closing ')') sit deeper than d, so a ')' at depth d closes our subquery
            if depth[i] < d or (code[i] == ")" and depth[i] == d and d > 0):
                break
            if depth[i] == d and re.match(r"(from|union|where|window)\b", low[i:i + 7]) and not (low[i - 1].isalnum() or low[i - 1] == "_"):
                break
            if code[i] == "," and depth[i] == d:
                items.append((cur_start, code[cur_start:i]))
                cur_start = i + 1
            i += 1
        items.append((cur_start, code[cur_start:i]))
        yield m.start(), [(o, t) for o, t in items if t.strip()]


def lint_model(m: Model, models_root: Path | None = None) -> list[Finding]:
    text = m.path.read_text(encoding="utf-8")
    body_offset = len("\n".join(text.splitlines()[:m.header_lines])) + (1 if m.header_lines else 0)
    code = mask(text[body_offset:])
    line_of = lambda off: text[:body_offset + off].count("\n") + 1
    f: list[Finding] = []
    add = lambda off, code_, sev, msg: f.append(Finding(m.name, line_of(off), code_, sev, msg))

    for _, items in select_lists(code):
        for off, item in items:
            it = " ".join(item.split())
            lead = off + len(item) - len(item.lstrip())
            if re.fullmatch(r"(distinct\s+)?(\w+\.)?\*", it, re.I):
                add(lead, "WG001", "error", "SELECT * - list the columns explicitly")
                continue
            if re.fullmatch(r"(distinct\s+)?\w+(\.\w+)?", it, re.I):
                continue
            alias = re.search(r"\bas\s+(\w+)\s*$", it, re.I)
            if not alias:
                add(lead, "WG002", "error", f"expression without an alias: {it[:60]}")
            elif not re.fullmatch(r"[a-z][a-z0-9_]*", alias.group(1)):
                add(lead, "WG007", "error", f"alias {alias.group(1)} is not snake_case")

    for mm in re.finditer(r"\bfrom\s+[\w.]+(\s+(as\s+)?\w+)?\s*,", code, re.I):
        add(mm.start(), "WG003", "error", "implicit comma join - use INNER JOIN ... ON")

    for mm in re.finditer(r"(?:\b(?:from|join)\s+|,\s*)(raw|stg|core|mart|rpt)\.([a-z0-9_]+)", code, re.I):
        src_layer = mm.group(1).lower()
        if m.layer in ALLOWED and src_layer in {"raw", "stg", "core", "mart", "rpt"} and src_layer not in ALLOWED[m.layer]:
            add(mm.start(), "WG004", "error", f"{m.layer} model reads {src_layer}.{mm.group(2)} (allowed: {', '.join(sorted(ALLOWED[m.layer]))})")

    for key in ("description", "grain", "owner"):
        if not m.meta.get(key):
            f.append(Finding(m.name, 1, "WG005", "error", f"header is missing '{key}'"))
    if not any(t.startswith("unique(") for t in m.tests):
        f.append(Finding(m.name, 1, "WG005", "error", "no unique(...) test declaring the grain key"))

    for mm in re.finditer(r"'\d{4}-\d{2}-\d{2}'", mask_comments_only(text[body_offset:])):
        add(mm.start(), "WG006", "error", f"hard-coded date {mm.group(0)} - use raw.run_context")

    expected = None
    if models_root is not None:
        rel = m.path.resolve().relative_to(Path(models_root).resolve())
        expected = f"{rel.parts[0]}.{Path(rel.parts[-1]).stem}"
    if expected and expected != m.name:
        f.append(Finding(m.name, 1, "WG007", "error", f"declared as {m.name} but the file path says {expected}"))
    if not re.fullmatch(r"(stg|core|mart|rpt)\.[a-z][a-z0-9_]*", m.name):
        f.append(Finding(m.name, 1, "WG007", "error", "model name must be <layer>.<snake_case>"))

    depth = _depths(code)
    for mm in re.finditer(r"\border\s+by\b", code, re.I):
        if depth[mm.start()] == 0 and not re.search(r"\blimit\b", code[mm.end():], re.I):
            # ORDER BY inside OVER (...) is at depth >= 1; WINDOW clauses are named and also nested
            add(mm.start(), "WG008", "warning", "top-level ORDER BY in a model")

    for kw in KEYWORDS:
        for mm in re.finditer(rf"(?<![\w.]){kw.replace(' ', r'\s+')}(?![\w])", code):
            if mm.group(0) != mm.group(0).upper():
                add(mm.start(), "WG009", "warning", f"keyword '{mm.group(0)}' should be upper case")
    return sorted(set(f), key=lambda x: (x.line, x.code))


def mask_comments_only(sql: str) -> str:
    return re.sub(r"--[^\n]*|/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), sql, flags=re.S)


def lint(models: dict[str, Model], models_root: Path | None = None) -> list[Finding]:
    return [x for m in models.values() for x in lint_model(m, models_root)]
