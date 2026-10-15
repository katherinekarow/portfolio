"""Tokenizing and BM25 ranking.

Kept deliberately small and dependency-free so the browser version (docs/assistant/app.js)
can implement exactly the same steps - the self-test checks both give the same answers.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""a an and are as at be by can do does for from has have how i in into is it its me my of on or our
show tell that the their them then there these this to was we what when where which who why will with you your
about get find see list give between mean means""".split())


def stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokens(text: str) -> list[str]:
    out = []
    for raw in re.findall(r"[a-z0-9_]+", text.lower()):
        parts = [raw] + ([p for p in raw.split("_") if p] if "_" in raw else [])
        for p in parts:
            if p not in STOP and len(p) > 1:
                out.append(stem(p))
    return out


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.2, b: float = 0.75, title_weight: int = 2):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = []
        for d in docs:
            toks = tokens(d["title"]) * title_weight + tokens(d["text"])
            self.tf.append(Counter(toks))
        self.pos = {d["id"]: i for i, d in enumerate(docs)}
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query_tokens: list[str], k: int = 5, kinds: set[str] | None = None,
               coverage_first: bool = False) -> list[tuple[float, dict]]:
        q = list(dict.fromkeys(query_tokens))           # unique, order kept
        scored = []
        for i, d in enumerate(self.docs):
            if kinds and d["kind"] not in kinds:
                continue
            tf, s, cov = self.tf[i], 0.0, 0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    cov += 1
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((round(s, 6), cov, d))
        scored.sort(key=lambda x: ((-x[1] if coverage_first else 0), -x[0], x[2]["id"]))
        return [(s, d) for s, _, d in scored[:k]]


    def coverage(self, query_tokens: list[str], doc_id: str) -> int:
        """How many distinct query terms the document contains."""
        return len(set(query_tokens) & set(self.tf[self.pos[doc_id]]))


def expand(query: str, synonyms: dict[str, list[str]]) -> list[str]:
    """Add synonym tokens when the query contains a known phrase (e.g. 'days_out' -> days to session)."""
    q = " " + re.sub(r"[^a-z0-9_]+", " ", query.lower()) + " "
    toks = tokens(query)
    for phrase, alts in synonyms.items():
        if f" {phrase} " in q:
            for a in alts:
                toks.extend(tokens(a))
    return toks
