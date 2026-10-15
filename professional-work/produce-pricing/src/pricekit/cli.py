"""python -m pricekit run [--out outputs/]  - estimate, back-test, recommend, validate, chart."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import elasticity, optimize, synth, validation

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def run(out: Path, figures: Path | None = None) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    hist, truth = synth.make_history()
    deps = synth.make_lots()
    est = elasticity.estimate(hist)
    bt = elasticity.backtest(hist)
    rec = optimize.recommend(deps, est)
    checks = validation.run(est, rec)
    est.merge(truth, on=["segment", "tier"]).to_csv(out / "elasticity.csv", index=False)
    rec.to_csv(out / "recommendations.csv", index=False)
    checks.to_csv(out / "validation.csv", index=False)
    if figures:
        charts(Path(figures), est.merge(truth, on=["segment", "tier"]), rec)
    return {"cells": len(est), "shrunk": int((est.method != "cell estimate").sum()),
            "backtest_wape_model": round(float((bt.wape_model * bt.n).sum() / bt.n.sum()), 3),
            "backtest_wape_naive": round(float((bt.wape_naive * bt.n).sum() / bt.n.sum()), 3),
            "rows": len(rec), "revenue_impact": round(float(rec.revenue_impact.sum()), 0),
            "contribution_impact": round(float(rec.contribution_impact.sum()), 0),
            "checks_passed": int(checks.passed.sum()), "checks": len(checks),
            "reasons": rec.reason.str.replace(r" \(.*\)| \+ ladder fix", "", regex=True).value_counts().to_dict(),
            "ladder_fixes": int(rec.reason.str.contains("ladder").sum())}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)


def charts(folder: Path, est: pd.DataFrame, rec: pd.DataFrame):
    folder.mkdir(parents=True, exist_ok=True)
    # 1. recovered elasticities vs truth
    e = est.sort_values(["tier", "segment"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(e))
    ok = e.method == "cell estimate"
    ax.errorbar(e.raw_elasticity[ok], y[ok], xerr=1.96 * e.raw_se[ok], fmt="o", color=S[0], ms=5, lw=1, label="cell estimate ± 95%")
    ax.errorbar(e.raw_elasticity[~ok], y[~ok], xerr=1.96 * e.raw_se[~ok], fmt="o", color=GRID, mec=MUTED, ms=5, lw=1, label="thin cell (raw)")
    ax.scatter(e.elasticity[~ok], y[~ok], marker="D", color=S[1], s=36, zorder=4, label="after shrinkage")
    ax.scatter(e.true_elasticity, y, marker="|", color=INK, s=160, zorder=5, label="true value")
    ax.set_yticks(y); ax.set_yticklabels(e.tier + " · " + e.segment, fontsize=8)
    ax.set_xlabel("Price elasticity (higher = more price-sensitive)", color=MUTED)
    ax.set_xlim(-0.5, 4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Estimated vs true elasticity per market segment and grade")
    fig.tight_layout(); fig.savefig(folder / "elasticity_recovery.png", dpi=150); plt.close(fig)

    # 2. contribution curve for two contrasting cells
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    clean = ~rec.reason.str.contains("ladder")
    picks = [rec[clean & (rec.tier == "standard") & (rec.reason.str.startswith("capacity"))].head(1),
             rec[clean & (rec.tier == "organic") & rec.reason.str.startswith("inelastic")].head(1)]
    for ax, r in zip(axes, picks):
        if r.empty:
            continue
        r = r.iloc[0]
        p = np.linspace(r.current_price * 0.7, r.current_price * 1.3, 300)
        ax.plot(p, optimize.contribution(p, r) / 1000, color=S[0], lw=2)
        ax.axvspan(r.current_price * 0.85, r.current_price * 1.15, color=GRID, alpha=0.5, lw=0)
        ax.axvline(r.current_price, color=MUTED, ls="--", lw=1)
        ax.axvline(r.recommended_price, color=S[1], lw=2)
        ax.text(r.recommended_price, ax.get_ylim()[1], f" recommended ${r.recommended_price:,.0f}", color=S[1], fontsize=8, va="top")
        ax.text(r.current_price, ax.get_ylim()[0], f" today ${r.current_price:,.0f}", color=MUTED, fontsize=8, va="bottom")
        ax.set_xlabel("Price", color=MUTED); ax.set_ylabel("Contribution ($k)", color=MUTED)
        _style(ax, f"{r.tier.title()}, elasticity {r.elasticity:.2f}\n{r.reason.split(' (')[0].split(':')[0].capitalize()}")
    fig.tight_layout(); fig.savefig(folder / "price_response.png", dpi=150); plt.close(fig)

    # 3. recommended moves by grade
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, t in enumerate(synth.TIERS):
        v = rec[rec.tier == t].price_change_pct * 100
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), color=S[i], s=22, alpha=0.8)
    ax.axvline(0, color=MUTED, lw=1)
    for x in (-15, 15):
        ax.axvline(x, color=GRID, lw=2)
    ax.set_yticks(range(4)); ax.set_yticklabels([t.title() for t in synth.TIERS])
    ax.set_xlabel("Recommended price change (%) · grey lines = ±15% guardrails", color=MUTED)
    _style(ax, "Recommended price moves by grade - low-elasticity grades hit the +15% guardrail")
    fig.tight_layout(); fig.savefig(folder / "recommended_moves.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pricekit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", default="outputs"); r.add_argument("--figures", default=None)
    a = ap.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None)
    print(f"elasticity cells: {s['cells']} ({s['shrunk']} thin cells shrunk to their grade pool)")
    print(f"back-test WAPE: model {s['backtest_wape_model']:.1%} vs naive {s['backtest_wape_naive']:.1%}")
    print(f"recommendations: {s['rows']} lot-grades · ladder fixes {s['ladder_fixes']}")
    for k, v in s["reasons"].items():
        print(f"   {v:>3}  {k}")
    print(f"expected revenue impact {s['revenue_impact']:+,.0f} · contribution impact {s['contribution_impact']:+,.0f}")
    print(f"validation queries: {s['checks_passed']}/{s['checks']} passed")


if __name__ == "__main__":
    main()
