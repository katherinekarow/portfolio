"""python -m dashkit build [--out ../../docs/dashboards] [--only slug ...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .builders import BUILDERS, HERE


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def build(out: Path, only: list[str] | None = None) -> dict[str, int]:
    sizes = {}
    for slug, fn in BUILDERS.items():
        if only and slug not in only:
            continue
        d = fn()
        p = out / slug / "data.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(d, default=_default, separators=(",", ":"))
        p.write_text(txt)
        sizes[slug] = len(txt)
    return sizes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dashkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data.json for every dashboard")
    b.add_argument("--out", type=Path, default=HERE.parents[4] / "docs" / "dashboards")
    b.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    for slug, n in build(a.out, a.only).items():
        print(f"{slug:24s} {n/1024:7.1f} KB")


if __name__ == "__main__":
    main()
