"""python -m orderrec run --date 2026-03-02 --out outputs/"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from . import bid_optimizer, ingest, normalize, reconcile, report, synth, weekly


def run(run_date: dt.date, out: Path, seed: int = 17, figures_dir: Path | None = None) -> dict:
    out = Path(out)
    world = synth.make_world(run_date=run_date, seed=seed)

    # 1. inputs -> real files, the way they arrive
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    world.vendor.to_csv(inputs / f"vendor_offers_{run_date:%Y%m%d}.csv", index=False)
    for day, df in world.internal_daily.items():
        df.to_csv(inputs / f"internal_substitutions_{day:%Y%m%d}.csv", index=False)
    books = ingest.write_books(world.upload_books, inputs)

    # 2. clean, combine, reconcile
    clean, vendor_rejects = normalize.normalize_vendor(world.vendor)
    grid, grid_rejects = ingest.combine_uploads(books)
    recon = reconcile.run(clean, world.internal_daily, run_date)
    wk = weekly.weekly_summary(clean, recon.detail, as_of=run_date)
    mix = weekly.tier_pair_mix(clean)

    # 3. decide pending bids + floor recommendations
    plan = bid_optimizer.plan(clean, world.inventory, grid)
    floors = bid_optimizer.recommended_floors(world.inventory, sorted(set(zip(grid.from_tier, grid.to_tier))))

    # 4. outputs
    xlsx = report.write_excel(out / f"substitution_recon_{run_date:%Y%m%d}.xlsx", recon=recon, vendor_rejects=vendor_rejects,
                              grid=grid, grid_rejects=grid_rejects, weekly=wk, mix=mix, plan=plan, floors=floors,
                              run_date=run_date)
    figs = report.figures(figures_dir or out / "figures", recon=recon, weekly=wk, plan=plan, floors=floors, grid=grid)
    s = plan.summary.set_index("policy")
    summary = {
        "run_date": str(run_date),
        "internal_cutoff": str(recon.internal_file_date),
        "warnings": recon.warnings,
        "vendor_duplicates_removed": int(clean.attrs.get("duplicates_removed", 0)),
        "vendor_rejects": vendor_rejects.reason_code.value_counts().to_dict(),
        "upload_rejects": grid_rejects.reason_code.value_counts().to_dict(),
        "classes": recon.control["counts"],
        "revenue_at_risk_usd": recon.control["revenue_at_risk_usd"],
        "rows_reconciled": bool(recon.control["rows_reconciled"]),
        "money_reconciled": bool(recon.control["money_reconciled"]),
        "pending_bids": len(plan.decisions),
        "accepted_baseline": int(s.loc["baseline", "bids_accepted"]),
        "accepted_optimized": int(s.loc["optimized", "bids_accepted"]),
        "lift_baseline_usd": float(s.loc["baseline", "lift_vs_accept_none_usd"]),
        "lift_optimized_usd": float(s.loc["optimized", "lift_vs_accept_none_usd"]),
        "excel": str(xlsx), "figures": [str(f) for f in figs],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orderrec", description="Substitution-program reconciliation and bid decisions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate inputs, reconcile, optimize and write the Excel report")
    r.add_argument("--date", type=dt.date.fromisoformat, default=dt.date(2026, 3, 2))
    r.add_argument("--out", type=Path, default=Path("outputs"))
    r.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    s = run(a.date, a.out, a.seed)
    print(f"run {s['run_date']}  (internal cut-off {s['internal_cutoff']})")
    for w in s["warnings"]:
        print(f"  ! {w}")
    print(f"  vendor duplicates removed: {s['vendor_duplicates_removed']}   rejects: {s['vendor_rejects']}")
    print(f"  upload rejects: {s['upload_rejects']}")
    for k, v in sorted(s["classes"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24}{v:>6}")
    print(f"  revenue at risk: ${s['revenue_at_risk_usd']:,.2f}   rows reconciled: {s['rows_reconciled']}"
          f"   money reconciled: {s['money_reconciled']}")
    print(f"  pending bids: {s['pending_bids']}   accept (rule of thumb): {s['accepted_baseline']}"
          f" -> {s['lift_baseline_usd']:+,.0f}   accept (optimized): {s['accepted_optimized']}"
          f" -> {s['lift_optimized_usd']:+,.0f} expected USD")
    print(f"  report: {s['excel']}")


if __name__ == "__main__":
    main()
