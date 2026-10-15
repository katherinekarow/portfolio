"""Rule-based Oracle -> Snowflake (or SQLite) SQL translator.

It does the mechanical 80% of a report migration and, just as importantly, writes down
what it changed and what a person still has to look at. Every rewrite is logged with
its line number; constructs it cannot translate safely are flagged, never guessed.

Targets
  snowflake   what the migrated views run on
  sqlite      lets the test-suite EXECUTE translated queries and prove they return what
              Oracle semantics say they should (DECODE's NULL matching, (+) joins, ROWNUM)

Rules
  NVL(a, b)                  -> COALESCE(a, b)
  NVL2(a, b, c)              -> CASE WHEN a IS NOT NULL THEN b ELSE c END
  DECODE(x, s1, r1, ..., d)  -> CASE WHEN x = s1 THEN r1 ... ELSE d END
                                (a NULL search value becomes `x IS NULL`; a non-literal search value
                                 becomes `(x = s OR (x IS NULL AND s IS NULL))` - DECODE treats NULLs as equal)
  TRUNC(date[, 'MM'|'YYYY']) -> DATE_TRUNC(...) / DATE(...)     (numeric TRUNC is flagged)
  ADD_MONTHS(d, n)           -> DATEADD(MONTH, n, d) / DATE(d, '+n months')
  TO_CHAR(d, 'YYYY-MM')      -> TO_CHAR / STRFTIME
  TO_DATE('2026-03-01', 'YYYY-MM-DD') -> TO_DATE / DATE
  SYSDATE                    -> CURRENT_TIMESTAMP()
  a.x = b.y(+)               -> FROM a LEFT JOIN b ON a.x = b.y   (comma joins rewritten as explicit joins)
  WHERE ... ROWNUM <= n      -> LIMIT n   (flagged when combined with ORDER BY: Oracle applies ROWNUM first)
  FROM DUAL                  -> removed
  MINUS                      -> EXCEPT
  /*+ hints */               -> removed
  CONNECT BY, PRIOR, MERGE, ROWID, .NEXTVAL -> flagged for manual rewrite
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sqltext import find_top_keyword, line_of, mask, matching_paren, split_and, split_top

TRUNC_UNITS = {"MM": "MONTH", "MONTH": "MONTH", "MON": "MONTH", "YYYY": "YEAR", "YEAR": "YEAR", "Y": "YEAR",
               "DD": "DAY", "DDD": "DAY", "J": "DAY", "Q": "QUARTER", "IW": "WEEK"}
SQLITE_TRUNC = {"MONTH": "'start of month'", "YEAR": "'start of year'"}
TO_CHAR_FMT = [("YYYY", "%Y"), ("MM", "%m"), ("DD", "%d"), ("HH24", "%H"), ("MI", "%M"), ("SS", "%S")]
DATEISH = re.compile(r"date|_dt\b|_ts\b|time|sysdate|current_|day", re.I)
MANUAL = [(r"\bconnect\s+by\b", "CONNECT BY hierarchy - rewrite as a recursive CTE"),
          (r"\bprior\b", "PRIOR (hierarchical query) - rewrite as a recursive CTE"),
          (r"\bmerge\s+into\b", "MERGE - check matched/not-matched semantics by hand"),
          (r"\browid\b", "ROWID has no equivalent - use a real key"),
          (r"\.nextval\b", "sequence .NEXTVAL - use an identity column or a Snowflake sequence"),
          (r"\(\+\)", None)]                                      # (+) handled below; listed so leftovers are caught


@dataclass
class Change:
    line: int
    rule: str
    before: str
    after: str


@dataclass
class Result:
    sql: str
    changes: list[Change] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.warnings)


class Translator:
    def __init__(self, target: str = "snowflake"):
        if target not in ("snowflake", "sqlite"):
            raise ValueError("target must be snowflake or sqlite")
        self.target = target

    # ------------------------------------------------------------------ public
    def translate(self, sql: str) -> Result:
        res = Result(sql=sql)
        src = sql
        for pat, msg in MANUAL:
            for m in re.finditer(pat, mask(src), re.I):
                if msg:
                    res.warnings.append(f"line {line_of(src, m.start())}: {msg}")
        s = self._hints(src, res)
        s = self._functions(s, res, src)
        s = self._sysdate(s, res, src)
        s = self._dual(s, res, src)
        s = re.sub(r"\bminus\b", lambda m: self._log(res, src, m.group(0), "MINUS -> EXCEPT", "EXCEPT"), s, flags=re.I) \
            if re.search(r"\bminus\b", mask(s), re.I) else s
        s = self._outer_joins(s, res, src)
        s = self._rownum(s, res, src)
        if "(+)" in mask(s):
            res.warnings.append("(+) outer join could not be rewritten automatically (nested query or OR condition)")
        res.sql = re.sub(r"\n[ \t]+(?=\n)", "", s)        # drop lines left empty by removed clauses
        res.__dict__.pop("_masked", None); res.__dict__.pop("_cursor", None)
        return res

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _locate(res: Result, src: str, text: str) -> int:
        """Line of the next occurrence of `text` in code (not in comments), so repeated
        expressions (SELECT list and GROUP BY) are reported on their own lines."""
        masked = res.__dict__.setdefault("_masked", mask(src))
        cursor = res.__dict__.setdefault("_cursor", {})
        start = cursor.get(text, -1) + 1
        while True:
            pos = src.find(text, start)
            if pos < 0:
                pos = src.upper().find(text.upper(), start)
            if pos < 0:
                return 0
            if masked[pos] != " ":                    # not inside a comment
                cursor[text] = pos
                return line_of(src, pos)
            start = pos + 1

    def _log(self, res: Result, src: str, before: str, rule: str, after: str) -> str:
        res.changes.append(Change(self._locate(res, src, before), rule, " ".join(before.split())[:80], " ".join(after.split())[:80]))
        return after

    def _hints(self, s: str, res: Result) -> str:
        def repl(m):
            res.changes.append(Change(line_of(s, m.start()), "optimizer hint removed", m.group(0), ""))
            return ""
        return re.sub(r"/\*\+.*?\*/\s*", repl, s, flags=re.S)

    def _sysdate(self, s: str, res: Result, src: str) -> str:
        new = "CURRENT_TIMESTAMP()" if self.target == "snowflake" else "CURRENT_TIMESTAMP"
        out, last = [], 0
        for m in re.finditer(r"\bsysdate\b", mask(s), re.I):
            out += [s[last:m.start()], new]
            last = m.end()
            res.changes.append(Change(self._locate(res, src, "SYSDATE"), "SYSDATE -> CURRENT_TIMESTAMP", "SYSDATE", new))
        return "".join(out) + s[last:]

    def _dual(self, s: str, res: Result, src: str) -> str:
        m = re.search(r"\s+from\s+dual\b", mask(s), re.I)
        if not m:
            return s
        res.changes.append(Change(self._locate(res, src, "dual"), "FROM DUAL removed", s[m.start():m.end()].strip(), ""))
        return s[:m.start()] + s[m.end():]

    # ------------------------------------------------------------------ functions
    FUNCS = re.compile(r"\b(nvl2|nvl|decode|trunc|add_months|to_char|to_date)\s*\(", re.I)

    def _functions(self, s: str, res: Result, src: str) -> str:
        """Rewrite function calls left to right; arguments are translated first (inner calls)."""
        out, i = [], 0
        while True:
            m = self.FUNCS.search(mask(s), i)
            if not m:
                out.append(s[i:])
                return "".join(out)
            open_idx = m.end() - 1
            close = matching_paren(mask(s), open_idx)
            name = m.group(1).upper()
            args = [self._functions(a.strip(), res, src) for a in split_top(s[open_idx + 1:close])]
            original = s[m.start():close + 1]
            new = self._rewrite(name, args, original, res, src)
            out.append(s[i:m.start()])
            out.append(new)
            i = close + 1

    def _rewrite(self, name: str, a: list[str], original: str, res: Result, src: str) -> str:
        sf = self.target == "snowflake"
        rec = lambda rule, new: self._log(res, src, original, rule, new)
        if name == "NVL" and len(a) == 2:
            return rec("NVL -> COALESCE", f"COALESCE({a[0]}, {a[1]})")
        if name == "NVL2" and len(a) == 3:
            return rec("NVL2 -> CASE", f"CASE WHEN {a[0]} IS NOT NULL THEN {a[1]} ELSE {a[2]} END")
        if name == "DECODE" and len(a) >= 3:
            x, rest = a[0], a[1:]
            whens = []
            for k in range(0, len(rest) - 1, 2):
                srch, out = rest[k], rest[k + 1]
                if srch.upper() == "NULL":
                    cond = f"{x} IS NULL"
                elif re.fullmatch(r"'(?:''|[^'])*'|-?\d+(\.\d+)?", srch):
                    cond = f"{x} = {srch}"
                else:
                    cond = f"({x} = {srch} OR ({x} IS NULL AND {srch} IS NULL))"
                whens.append(f"WHEN {cond} THEN {out}")
            default = f" ELSE {rest[-1]}" if len(rest) % 2 == 1 else ""
            return rec("DECODE -> CASE (NULL-safe)", "CASE " + " ".join(whens) + default + " END")
        if name == "TRUNC":
            unit = TRUNC_UNITS.get(a[1].strip("'").upper()) if len(a) == 2 else "DAY"
            looks_date = bool(DATEISH.search(a[0])) or len(a) == 2 and a[1].startswith("'")
            if not looks_date or unit is None:
                res.warnings.append(f"line {self._locate(res, src, original)}: TRUNC({a[0]}) - numeric or unknown unit, left unchanged")
                return f"TRUNC({', '.join(a)})"
            if sf:
                return rec("TRUNC(date) -> DATE_TRUNC", f"DATE_TRUNC('{unit}', {a[0]})")
            if unit == "DAY":
                return rec("TRUNC(date) -> DATE", f"DATE({a[0]})")
            if unit in SQLITE_TRUNC:
                return rec("TRUNC(date) -> DATE", f"DATE({a[0]}, {SQLITE_TRUNC[unit]})")
            res.warnings.append(f"TRUNC to {unit} not supported for sqlite target")
            return original
        if name == "ADD_MONTHS" and len(a) == 2:
            if sf:
                return rec("ADD_MONTHS -> DATEADD", f"DATEADD(MONTH, {a[1]}, {a[0]})")
            if re.fullmatch(r"-?\d+", a[1]):
                return rec("ADD_MONTHS -> DATE", f"DATE({a[0]}, '{int(a[1]):+d} months')")
            return rec("ADD_MONTHS -> DATE", f"DATE({a[0]}, ({a[1]}) || ' months')")
        if name == "TO_DATE" and len(a) == 2 and a[1].strip("'").upper() in ("YYYY-MM-DD", "YYYYMMDD"):
            if sf:
                return f"TO_DATE({a[0]}, {a[1]})"
            if a[1].strip("'").upper() == "YYYYMMDD" and a[0].startswith("'"):
                v = a[0].strip("'")
                return rec("TO_DATE -> DATE", f"DATE('{v[:4]}-{v[4:6]}-{v[6:]}')")
            return rec("TO_DATE -> DATE", f"DATE({a[0]})")
        if name == "TO_CHAR" and len(a) == 2 and a[1].startswith("'"):
            if sf:
                return f"TO_CHAR({a[0]}, {a[1]})"
            fmt = a[1].strip("'")
            for o, p in TO_CHAR_FMT:
                fmt = fmt.replace(o, p)
            return rec("TO_CHAR -> STRFTIME", f"STRFTIME('{fmt}', {a[0]})")
        return f"{name}({', '.join(a)})"

    # ------------------------------------------------------------------ query shape
    def _clauses(self, s: str):
        m = mask(s)
        f = find_top_keyword(m, "from")
        w = find_top_keyword(m, "where", f) if f >= 0 else -1
        ends = [p for p in (find_top_keyword(m, k, max(f, 0)) for k in ("group", "order", "having", "union", "except", "limit")) if p >= 0]
        end = min(ends) if ends else len(s)
        return f, w, end

    def _outer_joins(self, s: str, res: Result, src: str) -> str:
        if "(+)" not in mask(s):
            return s
        f, w, end = self._clauses(s)
        if f < 0 or w < 0:
            return s
        from_items = [x.strip() for x in split_top(s[f + 4:w])]
        if any(re.search(r"\bjoin\b|\(", mask(x), re.I) for x in from_items):
            return s                                           # already ANSI / subqueries - leave to warning
        alias = {}
        for item in from_items:
            parts = item.split()
            alias[(parts[-1] if len(parts) > 1 else parts[0].split(".")[-1]).lower()] = item
        conds = split_and(s[w + 5:end])
        if any(re.search(r"\bor\b", mask(c), re.I) and "(+)" in c for c in conds):
            return s
        joins, inner_on, keep = {}, {}, []
        for c in conds:
            refs = {r.lower() for r in re.findall(r"\b([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*", mask(c), re.I)} & set(alias)
            if "(+)" in c:
                outer = {r.lower() for r in re.findall(r"\b([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\s*\(\+\)", c, re.I)}
                if len(outer) != 1:
                    return s
                joins.setdefault(outer.pop(), []).append(c.replace("(+)", "").strip())
            elif len(refs) == 2:
                inner_on.setdefault(frozenset(refs), []).append(c)
            else:
                keep.append(c)
        order = list(alias)
        base = order[0]
        placed, sql_from = {base}, alias[base]
        pending = [a for a in order[1:]]
        while pending:
            progressed = False
            for a in list(pending):
                if a in joins:
                    deps = {r.lower() for cond in joins[a] for r in re.findall(r"\b([a-z_][a-z0-9_]*)\.", cond)} - {a}
                    if deps <= placed:
                        sql_from += f"\nLEFT JOIN {alias[a]} ON " + " AND ".join(joins[a])
                        placed.add(a); pending.remove(a); progressed = True
                else:
                    ons = [c for k, v in inner_on.items() if a in k and (k - {a}) <= placed for c in v]
                    if ons:
                        for k in [k for k in inner_on if a in k and (k - {a}) <= placed]:
                            del inner_on[k]
                        sql_from += f"\nINNER JOIN {alias[a]} ON " + " AND ".join(ons)
                        placed.add(a); pending.remove(a); progressed = True
            if not progressed:
                return s
        keep += [c for v in inner_on.values() for c in v]
        new = s[:f] + "FROM " + sql_from + ("\nWHERE " + "\n  AND ".join(keep) if keep else "") + "\n" + s[end:].lstrip()
        res.changes.append(Change(self._locate(res, src, "(+)"), "(+) outer join -> LEFT JOIN",
                                  f"FROM {', '.join(from_items)} ... (+)", "explicit INNER/LEFT JOIN ... ON"))
        return new.rstrip() + ("\n" if s.endswith("\n") else "")

    def _rownum(self, s: str, res: Result, src: str) -> str:
        m = re.search(r"\b(?:and\s+)?rownum\s*(<=|<)\s*(\d+)(\s+and\b)?", mask(s), re.I)
        if not m:
            if re.search(r"\brownum\b", mask(s), re.I):
                res.warnings.append("ROWNUM used in a way that can't be mapped to LIMIT - review by hand")
            return s
        n = int(m.group(2)) - (1 if m.group(1) == "<" else 0)
        cut = s[:m.start()] + (" " if m.group(3) and not m.group(0).lower().startswith("and") else "") + s[m.end():]
        cut = re.sub(r"\bwhere\s+(group|order|$)", r"\1", cut, flags=re.I)
        cut = re.sub(r"\bwhere\s*\n\s*(group|order)", r"\1", cut, flags=re.I)
        cut = re.sub(r"\bWHERE\s*$", "", cut.rstrip(), flags=re.I)
        if find_top_keyword(mask(s), "order") >= 0:
            res.warnings.append(f"line {self._locate(res, src, 'ROWNUM')}: ROWNUM with ORDER BY - Oracle picks the rows "
                                "BEFORE sorting; LIMIT sorts first. Confirm which result the report intended.")
        res.__dict__.get("_cursor", {}).pop("ROWNUM", None)
        res.changes.append(Change(self._locate(res, src, "ROWNUM"), "ROWNUM -> LIMIT", m.group(0).strip(), f"LIMIT {n}"))
        return cut.rstrip().rstrip(";") + f"\nLIMIT {n}"
