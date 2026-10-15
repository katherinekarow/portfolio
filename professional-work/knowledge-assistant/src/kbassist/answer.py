"""Route a question to the right kind of answer, always with sources.

Order matters and mirrors the browser version:
  1. guardrails      - refuse credentials / personal data / write actions; redirect decisions
  2. lineage         - "what feeds X", "what breaks if X changes"
  3. column lookup   - "which view has gap_to_full"
  4. definition      - "what is a glide path", "difference between target curve and glide path"
  5. query help      - "sql for ...", "how do I get ..."
  6. search          - BM25 over views, terms and project write-ups
  7. don't know      - below the confidence threshold, say so and suggest where to look
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .search import BM25, expand, tokens


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class Answer:
    type: str
    text: str
    sources: list[str] = field(default_factory=list)
    score: float = 0.0
    sql: str = ""


class Assistant:
    def __init__(self, kb: dict):
        self.kb = kb
        self.views = {v["name"]: v for v in kb["views"]}
        self.terms = kb["terms"]
        self.index = BM25(kb["docs"])
        self.guard = kb["guardrails"]
        self.columns: dict[str, list[str]] = {}
        for v in kb["views"]:
            for c in v["columns"]:
                self.columns.setdefault(c, []).append(v["name"])

    # ------------------------------------------------------------------ helpers
    def _mentioned_views(self, q: str) -> list[str]:
        ql = q.lower()
        hits = [n for n in self.views if n in ql]
        if not hits:   # allow the short name without schema, but only identifier-like ones ("session_plan", not "sessions")
            hits = [n for n in self.views if "_" in n.split(".")[1] and re.search(rf"\b{re.escape(n.split('.')[1])}\b", ql)]
        return sorted(hits, key=lambda n: -len(n))

    def _term_hits(self, q: str) -> list[dict]:
        """Glossary entries whose name or alias appears in the question, in order of appearance.
        Longer phrases win, so 'target fill rate' is not also read as 'fill rate'."""
        ql = " " + norm(q) + " "
        found, spans = [], []
        cands = sorted(((p, t) for t in self.terms for p in [t["term"]] + t.get("aliases", [])), key=lambda x: -len(x[0]))
        for phrase, t in cands:
            i = ql.find(" " + norm(phrase) + " ")
            if i >= 0 and not any(a <= i < b for a, b in spans) and all(t is not f[1] for f in found):
                found.append((i, t))
                spans.append((i, i + len(norm(phrase)) + 2))
        return [t for _, t in sorted(found, key=lambda x: x[0])]

    @staticmethod
    def _term_text(t: dict) -> str:
        s = f"**{t['term']}** - {t['definition']}"
        if t.get("formula"):
            s += f"\n`{t['formula']}`"
        if t.get("example"):
            s += f"\nExample: {t['example']}"
        return s

    # ------------------------------------------------------------------ main
    def ask(self, q: str) -> Answer:
        q = q.strip()
        if not q:
            return Answer("empty", "Ask about a view, a column, a term, or how part of the pipeline works.")
        for rule in self.guard["block"]:
            if any(re.search(p, q, re.I) for p in rule["patterns"]):
                return Answer("blocked", rule["reply"], [f"guardrail:{rule['id']}"])
        for rule in self.guard["redirect"]:
            if any(re.search(p, q, re.I) for p in rule["patterns"]):
                return Answer("redirect", rule["reply"], [f"guardrail:{rule['id']}"])

        ql = q.lower()
        views = self._mentioned_views(q)
        if views and re.search(r"\b(upstream|feeds?|fed|sources?|depends?|comes? from|built from|lineage|downstream|impact|breaks?|affected|uses?|used by|reads?)\b", ql):
            v = views[0]
            lin = self.kb["lineage"][v]
            down = re.search(r"\b(downstream|impact|breaks?|affected|used by|uses)\b", ql) and not re.search(r"\b(upstream|feeds?|built from|comes? from)\b", ql)
            if down:
                txt = (f"**{v}** is read by {', '.join(lin['direct_down']) or 'nothing'}."
                       + (f" Changing it affects {len(lin['downstream'])} views in total: {', '.join(lin['downstream'])}." if lin["downstream"] else ""))
                return Answer("lineage", txt, [f"view:{v}"] + [f"view:{x}" for x in lin["direct_down"]])
            txt = (f"**{v}** is built from {', '.join(lin['direct_up']) or 'no other views (it is a raw source)'}."
                   + (f" Full upstream chain: {', '.join(lin['upstream'])}." if lin["upstream"] else ""))
            return Answer("lineage", txt, [f"view:{v}"] + [f"view:{x}" for x in lin["direct_up"]])

        cols = [c for c in self.columns if re.search(rf"\b{re.escape(c)}\b", ql) and ("_" in c or re.search(r"\bcolumn|field\b", ql))]
        if cols and re.search(r"\b(which|what|where)\b.*\b(view|table|column|field)s?\b|\bcolumn\b|\bfield\b", ql):
            c = sorted(cols, key=len)[-1]
            where = self.columns[c]
            desc = self.views[where[0]]["columns"][c]
            return Answer("column", f"`{c}` ({desc}) is in: " + ", ".join(f"**{w}**" for w in where) + ".",
                          [f"view:{w}" for w in where])

        terms = self._term_hits(q)
        if terms and re.search(r"\b(differen\w*|vs\.?|versus|compare)\b", ql) and len(terms) >= 2:
            a, b = terms[0], terms[1]
            return Answer("definition", self._term_text(a) + "\n\n" + self._term_text(b),
                          [f"term:{a['term']}", f"term:{b['term']}"])
        if terms and re.search(r"^(what|define|meaning|explain)\b|\bmean\b|\bdefinition\b|\bstand for\b|\bformula\b", ql):
            t = terms[0]
            return Answer("definition", self._term_text(t), [f"term:{t['term']}"])

        qtok = expand(q, self.kb["synonyms"])
        if re.search(r"\b(sql|query|select)\b|^how (do|can) i (get|find|see|list|pull|check)", ql):
            # prefer views matching more of the question, then BM25 score
            hits = self.index.search(qtok, k=3, kinds={"view"}, coverage_first=True)
            if views:
                v = self.views[views[0]]
            elif hits:
                v = self.views[hits[0][1]["title"]]
            else:
                v = None
            if v:
                sql = v.get("sample_query", "").strip() or starter_sql(v)
                return Answer("query", f"Start from **{v['name']}** ({v['grain']}):", [f"view:{v['name']}"],
                              hits[0][0] if hits else 0.0, sql)

        hits = self.index.search(qtok, k=3)
        # don't guess: the best passage must score well AND contain at least two different words of a multi-word question
        weak = not hits or hits[0][0] < self.guard["min_score"] or \
            (len(set(qtok)) >= 2 and self.index.coverage(qtok, hits[0][1]["id"]) < 2)
        if weak:
            sug = ", ".join(t["term"] for t in self.terms[:4])
            return Answer("unknown", "I don't know that one yet - I'd rather not guess. Try a view name, a column, or a term "
                                     f"such as {sug}. If it's missing, add it to the catalog or glossary.",
                          [], hits[0][0] if hits else 0.0)
        top = hits[0][1]
        if top["kind"] == "view":
            v = self.views[top["title"]]
            txt = f"**{v['name']}** - {v['description']} Grain: {v['grain']}."
        elif top["kind"] == "term":
            txt = self._term_text(next(t for t in self.terms if t["term"] == top["title"]))
        else:
            txt = f"From *{top['title']}*: {snippet(top['text'], qtok)}"
        return Answer("search", txt, [h[1]["id"] for h in hits], hits[0][0])


def starter_sql(v: dict) -> str:
    """A read-only starting query built from the catalog when the view has no curated example."""
    cols = list(v["columns"])
    return "SELECT " + ",\n       ".join(cols[:6]) + f"\nFROM {v['name']}\nLIMIT 100;"


def snippet(text: str, qtok: list[str], width: int = 45) -> str:
    """The ~45-word window with the most distinct query terms (ties: earliest)."""
    words = text.split()
    if len(words) <= width:
        return text
    want = set(qtok)
    best, best_s = 0, -1
    for i in range(0, len(words) - width + 1, 5):
        s = len(want & set(tokens(" ".join(words[i:i + width]))))
        if s > best_s:
            best, best_s = i, s
    out = " ".join(words[best:best + width])
    return ("... " if best else "") + out + (" ..." if best + width < len(words) else "")
