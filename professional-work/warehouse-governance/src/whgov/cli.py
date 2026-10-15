"""python -m whgov <command>

    lint                  check every model against the conventions (exit 1 on errors)
    run                   load synthetic raw data, build all models in dependency order, run the contracts
    docs                  run, then write CATALOG.md and the lineage chart
    impact <model>        what is downstream of a model (what to re-test if it changes)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

from . import contracts, docs, lint, loader, project

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def run_all(models_dir: Path = MODELS):
    models = project.load_models(models_dir)
    con = loader.connect()
    injected = loader.load_raw(con)
    log = project.build(con, models)
    results = contracts.run_tests(con, models)
    g = project.graph(models, {s for s in project.sources_in(con) if s.startswith("raw.")})
    return con, models, g, log, results, injected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="whgov")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lint")
    sub.add_parser("run")
    d = sub.add_parser("docs"); d.add_argument("--out", default=str(ROOT))
    i = sub.add_parser("impact"); i.add_argument("model")
    a = ap.parse_args(argv)

    if a.cmd == "lint":
        models = project.load_models(MODELS)
        found = lint.lint(models, MODELS)
        for f in found:
            print(f)
        errors = sum(f.severity == "error" for f in found)
        print(f"{len(models)} models · {errors} errors · {len(found) - errors} warnings")
        return 1 if errors else 0

    con, models, g, log, results, _ = run_all()
    if a.cmd == "impact":
        down = sorted(nx.descendants(g, a.model))
        print(f"{a.model} feeds {len(down)} models: " + (", ".join(down) or "none"))
        return 0
    print("build order")
    for r in log:
        print(f"  {r['model']:<24}{r['rows']:>8,} rows  {r['seconds']:.3f}s")
    failed = [r for r in results if not r["passed"]]
    print(f"\ncontracts: {len(results) - len(failed)}/{len(results)} passed")
    for r in failed:
        print(f"  FAIL {r['model']} {r['test']}: {r.get('error') or str(r['failing_rows']) + ' failing rows'}")
    rej = con.execute("SELECT reason_code, COUNT(*) FROM stg.rejected_rows GROUP BY reason_code ORDER BY reason_code").fetchall()
    print("rejected raw rows: " + ", ".join(f"{k} {v}" for k, v in rej))
    if a.cmd == "docs":
        out = Path(a.out)
        (out / "CATALOG.md").write_text(docs.catalog_markdown(models, g, con, results), encoding="utf-8")
        (out / "figures").mkdir(exist_ok=True)
        docs.lineage_png(g, out / "figures" / "lineage.png")
        docs.lineage_png(g, out / "figures" / "lineage_session_plan.png", highlight="mart.session_plan")
        print(f"wrote {out / 'CATALOG.md'} and figures/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
