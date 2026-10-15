"""python -m migrationkit <command>

    translate [--target snowflake|sqlite]   translate every legacy/*.sql into migrated/, with a change log
    columns                                 audit output columns of every legacy report
    parity                                  run the parity demo (legacy extract vs migrated view)
    plan                                    score the inventory and propose migration waves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import columns, inventory, parity, reference, synth
from .translate import Translator

ROOT = Path(__file__).resolve().parents[2]


def translate_all(target: str = "snowflake", out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or ROOT / "migrated"
    out_dir.mkdir(exist_ok=True)
    log = []
    for p in sorted((ROOT / "legacy").glob("*.sql")):
        sql = p.read_text()
        r = Translator(target).translate(sql)
        fixed, notes = columns.fix_aliases(r.sql)
        header = [f"-- migrated from legacy/{p.name} by migrationkit ({target})"]
        header += [f"-- line {c.line}: {c.rule}" for c in r.changes]
        header += [f"-- {n}" for n in notes]
        header += [f"-- REVIEW: {w}" for w in r.warnings]
        body = "\n".join(l for l in fixed.splitlines() if not l.lstrip().startswith("--")).strip()
        (out_dir / p.name).write_text("\n".join(header) + "\n" + body + "\n")
        log.append({"report": p.stem, "rewrites": len(r.changes), "aliases_fixed": len(notes),
                    "needs_review": len(r.warnings), "status": "REVIEW" if r.warnings else "auto"})
    return log


def parity_demo(seed: int = 15) -> parity.ParityResult:
    """Legacy extract (Oracle semantics + the quirks the old extract really had) vs the migrated view."""
    t = synth.make_tables(seed)
    con = synth.to_sqlite(t)
    sql = Translator("sqlite").translate((ROOT / "legacy" / "appointment_activity.sql").read_text()).sql
    new = pd.read_sql_query(sql, con)
    # the legacy extract: an older query without NVL on charge, written to a file that rounded money to whole units
    legacy = reference.appointment_activity(t, charge_nvl=False)
    legacy["revenue"] = legacy.revenue.round(0)
    return parity.compare(legacy, new, keys=["session_id", "appointment_month", "status_label", "channel"],
                          tolerance={"revenue": 0.005})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="migrationkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate"); t.add_argument("--target", default="snowflake")
    sub.add_parser("columns"); sub.add_parser("parity"); sub.add_parser("plan")
    a = ap.parse_args(argv)
    if a.cmd == "translate":
        print(pd.DataFrame(translate_all(a.target)).to_string(index=False))
        print(f"\nwrote migrated/*.sql")
    elif a.cmd == "columns":
        for p in sorted((ROOT / "legacy").glob("*.sql")):
            bad = [c for c in columns.output_columns(p.read_text()) if c.issue]
            print(f"{p.stem}: {'ok' if not bad else ''}")
            for c in bad:
                print(f"   col {c.position}: {c.expression[:50]:<50} {c.issue}")
    elif a.cmd == "parity":
        r = parity_demo()
        print(r.summary())
        print(r.totals.to_string(index=False))
    else:
        df = inventory.plan(ROOT / "inventory.yaml", ROOT / "legacy")
        print(df[["wave", "name", "criticality", "frequency", "users", "value", "effort", "manual_items", "column_issues"]].to_string(index=False))


if __name__ == "__main__":
    main()
