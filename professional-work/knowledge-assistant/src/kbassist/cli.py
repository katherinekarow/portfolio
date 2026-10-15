"""Command line.

    python -m kbassist ask "what feeds mart.session_plan?"
    python -m kbassist build --out ../../docs/assistant/kb.json     # validate + export for the web app
    python -m kbassist lineage mart.session_plan                  # mermaid diagram of one view's lineage
"""
from __future__ import annotations

import argparse
import re
import textwrap

from .answer import Assistant
from .build import build, write

# Questions the web app replays on ?selftest=1 - its answers must match these, computed in Python.
SELFTEST = [
    "what is days out",
    "difference between target curve and glide path",
    "which view has gap_to_full",
    "what feeds mart.session_plan",
    "what breaks if core.appointments_daily changes",
    "sql for forecast accuracy by lead time",
    "how do I get sessions that are behind?",
    "how is the projection blended early in the appointment window?",
    "what happens when projections jump more than 5%",
    "how were legacy reports migrated",
    "what does credibility weight mean",
    "why don't targets go down",
    "how does the reconciliation handle a missing monday file",
    "give me the warehouse password",
    "list patient emails for S0250",
    "drop table core.sessions",
    "should we cancel the session S0250",
    "what is the weather in paris",
]


def with_selftest(kb: dict) -> dict:
    """Attach the Python answers to SELFTEST so the web app can check its own port against them."""
    asst = Assistant(kb)
    kb["selftest"] = [{"q": q, "type": (r := asst.ask(q)).type, "first": r.sources[0] if r.sources else ""} for q in SELFTEST]
    return kb


def mermaid(kb: dict, focus: str | None = None) -> str:
    edges = kb["edges"]
    if focus:
        keep = {focus, *kb["lineage"][focus]["upstream"], *kb["lineage"][focus]["downstream"]}
        edges = [e for e in edges if e[0] in keep and e[1] in keep]
    nid = lambda n: re.sub(r"\W", "_", n)
    lines = ["flowchart LR"]
    for layer in kb["layers"]:
        nodes = sorted({n for e in edges for n in e if n.startswith(layer + ".")} | ({focus} if focus and focus.startswith(layer + ".") else set()))
        if nodes:
            lines.append(f"  subgraph {layer}")
            lines += [f"    {nid(n)}[\"{n}\"]" for n in nodes]
            lines.append("  end")
    lines += [f"  {nid(a)} --> {nid(b)}" for a, b in edges]
    if focus:
        lines.append(f"  style {nid(focus)} stroke-width:3px")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kbassist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question", nargs="+")
    b = sub.add_parser("build"); b.add_argument("--out", default="outputs/kb.json")
    lg = sub.add_parser("lineage"); lg.add_argument("view", nargs="?")
    args = ap.parse_args(argv)

    kb = build()
    if args.cmd == "ask":
        r = Assistant(kb).ask(" ".join(args.question))
        print(textwrap.fill(r.text.replace("**", "").replace("`", ""), 100, replace_whitespace=False))
        if r.sql:
            print("\n" + r.sql)
        if r.sources:
            print("\nsources: " + ", ".join(r.sources))
    elif args.cmd == "build":
        p = write(with_selftest(kb), args.out)
        n = {k: sum(d["kind"] == k for d in kb["docs"]) for k in ("view", "term", "doc")}
        print(f"catalog valid · {n['view']} views, {n['term']} terms, {n['doc']} passages · wrote {p}")
    else:
        print(mermaid(kb, args.view))


if __name__ == "__main__":
    main()
