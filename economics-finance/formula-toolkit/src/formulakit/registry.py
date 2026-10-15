"""A small registry so every formula carries its own documentation, inputs and worked example.

The same registry drives the command line (python -m formulakit), the formula reference
(FORMULAS.md, generated) and the in-browser calculator (docs/formulas/), whose JavaScript
versions are tested against these Python ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

REGISTRY: dict[str, "Formula"] = {}
GROUPS = {
    "finance": "Time value of money",
    "macro": "Money & interest rates",
    "demand": "Demand & price elasticity",
    "curves": "Demand curves & pacing",
    "targets": "Gross vs net targets",
    "stats": "Benchmarking, credibility & data checks",
}


@dataclass
class Param:
    name: str
    label: str
    default: object
    kind: str = "num"          # num | list (comma-separated numbers) | int


@dataclass
class Formula:
    key: str
    group: str
    title: str
    formula: str               # plain-text formula shown in docs and on the page
    explain: str               # one or two sentences: what it is for
    params: list[Param]
    fn: Callable
    out: str = "num"           # num | pct | money | list | table
    examples: list[dict] = field(default_factory=list)

    def __call__(self, **kw):
        return self.fn(**kw)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def run(self, raw: dict | None = None):
        """Evaluate from raw (string or number) inputs, filling in defaults."""
        raw = {**self.defaults(), **(raw or {})}
        return self.fn(**{p.name: parse(raw[p.name], p.kind) for p in self.params})


def formula(key, group, title, formula, explain, params, out="num", examples=None):
    def wrap(fn):
        f = Formula(key, group, title, formula, explain, [Param(*p) for p in params], fn, out, examples or [])
        if not f.examples:
            f.examples = [f.defaults()]
        REGISTRY[key] = f
        return fn
    return wrap


def parse(value: str, kind: str):
    if kind == "list":
        return [float(x) for x in str(value).replace(";", ",").split(",") if x.strip()]
    if kind == "groups":
        return [[float(x) for x in g.split(",") if x.strip()] for g in str(value).split("|") if g.strip()]
    if kind == "int":
        return int(float(value))
    return float(value)
