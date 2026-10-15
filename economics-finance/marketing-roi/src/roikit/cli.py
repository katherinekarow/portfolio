"""python -m roikit run [--out sample_output] [--figures figures] [--config config.yaml]"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import allocate, config, pipeline

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GOOD, CRIT = "#0ca30c", "#d03b3b"


def write_outputs(r: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    r["param_table"].to_csv(out / "fitted_params.csv", index=False)
    r["allocation"].to_csv(out / "allocation.csv", index=False)
    r["mc_summary"].to_csv(out / "roi_simulation_summary.csv", index=False)
    r["ab_plan"].to_csv(out / "ab_plan.csv", index=False)
    pd.DataFrame([r["ab_result"]]).round(6).to_csv(out / "ab_result.csv", index=False)
    r["peeking"].to_csv(out / "peeking_fpr.csv", index=False)
    r["cash_flows"].round(2).to_csv(out / "cash_flows.csv", index=False)
    fin = pd.DataFrame(r["financials"]).T.rename_axis("scenario").reset_index()
    fin.round(4).to_csv(out / "financial_summary.csv", index=False)
    r["kpi"].to_csv(out / "kpi_decisions.csv", index=False)


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def charts(r: dict, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    chans, labels, P, T = r["chans"], r["labels"], r["params"], r["truth"]
    cfg = r["cfg"]
    W, m = cfg["budget"]["weeks"], cfg["finance"]["gross_margin"]
    opt, naive, media = r["opt"], r["naive"], r["media"]
    mu = r["kkt"]["mu"]

    # 1. response curves + marginal ROI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    B = np.linspace(0, 0.62 * media, 300)
    for i, c in enumerate(chans):
        axes[0].plot(B / 1e3, allocate.gross_profit(B, P[c], W, m) / 1e3, color=S[i], lw=2, label=labels[c])
        axes[0].plot(B / 1e3, allocate.gross_profit(B, T[c], W, m) / 1e3, color=S[i], lw=1, ls=":")
        axes[0].scatter(opt[i] / 1e3, allocate.gross_profit(opt[i], P[c], W, m) / 1e3, color=S[i], s=45, zorder=5, ec="white")
        axes[0].scatter(naive[i] / 1e3, allocate.gross_profit(naive[i], P[c], W, m) / 1e3, facecolor="white", ec=S[i], s=40, zorder=5, lw=1.5)
        mr = allocate.marginal_roi(B[1:], P[c], W, m)
        axes[1].plot(B[1:] / 1e3, mr, color=S[i], lw=2, label=labels[c])
        axes[1].scatter(opt[i] / 1e3, r["kkt"]["marginal"][i], color=S[i], s=45, zorder=5, ec="white")
    axes[1].axhline(mu, color=INK, lw=1, ls="--")
    axes[1].text(0.62 * media / 1e3, mu, f"common marginal ROI {mu:.2f}", color=INK, fontsize=8, ha="right", va="bottom")
    axes[1].set_ylim(0, 4)
    axes[0].set_xlabel("18-month channel spend ($k)", color=MUTED)
    axes[0].set_ylabel("Incremental gross profit ($k)", color=MUTED)
    axes[1].set_xlabel("18-month channel spend ($k)", color=MUTED)
    axes[1].set_ylabel(r"Marginal ROI (\$ profit per extra \$)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].text(0.99, 0.02, "solid = fitted, dotted = truth\nfilled = optimal, hollow = naive split",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    _style(axes[0], "Fitted response curves (adstock + Hill)")
    _style(axes[1], "At the optimum every channel earns the same\nmarginal ROI")
    fig.tight_layout()
    fig.savefig(folder / "response_curves.png", dpi=150)
    plt.close(fig)

    # 2. allocation naive vs optimal
    a = r["allocation"]
    fin = r["financials"]
    up = fin["optimal (truth)"]["gross_profit"] / fin["naive (truth)"]["gross_profit"] - 1
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(chans))[::-1]
    h = 0.36
    ax.barh(y + h / 2, a.naive_spend / 1e3, h, color=S[0], label="Naive split (25/60/10/5)")
    ax.barh(y - h / 2, a.optimal_spend / 1e3, h, color=S[1], label="Optimal (equal marginal ROI)")
    for yi, (n_, o_, lo_, hi_) in zip(y, a[["naive_spend", "optimal_spend", "min", "max"]].to_numpy()):
        ax.text(n_ / 1e3 + 3, yi + h / 2, f"\\${n_ / 1e3:,.0f}k", va="center", fontsize=8, color=MUTED)
        ax.text(o_ / 1e3 + 3, yi - h / 2, f"\\${o_ / 1e3:,.0f}k", va="center", fontsize=8, color=INK)
        ax.plot([lo_ / 1e3, hi_ / 1e3], [yi - 0.44] * 2, color=MUTED, lw=1)
        ax.plot([lo_ / 1e3] * 2, [yi - 0.47, yi - 0.41], color=MUTED, lw=1)
        ax.plot([hi_ / 1e3] * 2, [yi - 0.47, yi - 0.41], color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in chans])
    ax.set_xlabel(f"18-month media spend (\\$k) of \\${media / 1e3:,.0f}k · grey bracket = allowed min-max", color=MUTED)
    ax.set_xlim(0, a["max"].max() / 1e3 * 1.12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, f"Reallocating the same budget: {up:+.1%} incremental gross profit")
    fig.tight_layout()
    fig.savefig(folder / "allocation.png", dpi=150)
    plt.close(fig)

    # 3. ROI distribution
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bins = np.linspace(0.4, 2.8, 61)
    for i, (name, mc) in enumerate([("Naive split", r["mc_naive"]), ("Optimal allocation", r["mc"])]):
        ax.hist(mc.roi_multiple, bins=bins, histtype="stepfilled", alpha=0.25, color=S[i])
        ax.hist(mc.roi_multiple, bins=bins, histtype="step", lw=1.8, color=S[i],
                label=f"{name}: P(ROI > 1) = {(mc.roi_multiple > 1).mean():.0%}")
    q = r["mc"].roi_multiple.quantile([0.1, 0.5, 0.9])
    top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, top)
    for k, v in zip(("P10", "P50", "P90"), q):
        ax.axvline(v, color=S[1], lw=1, ls="--")
        ax.text(v, top * 0.98, f" {k} {v:.2f}", color=S[1], fontsize=8, va="top")
    ax.axvline(1, color=CRIT, lw=1.5)
    ax.text(1, top * 0.80, "break-even ", color=CRIT, fontsize=8, ha="right")
    ax.set_xlabel("ROI multiple = incremental gross profit / total spend (24 months)", color=MUTED)
    ax.set_ylabel("Simulations", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _style(ax, f"ROI over {len(r['mc']):,} draws of fit and market-transfer uncertainty")
    fig.tight_layout()
    fig.savefig(folder / "roi_distribution.png", dpi=150)
    plt.close(fig)

    # 4. peeking
    pk = r["peeking"]
    alpha = cfg["ab"]["alpha"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(pk.looks, pk.fpr_naive * 100, "-o", color=S[0], lw=2, label="Stop at first p < 0.05 (naive peeking)")
    ax.plot(pk.looks, pk.fpr_bonferroni * 100, "-o", color=S[1], lw=2, label="Threshold alpha / looks (Bonferroni)")
    ax.axhline(alpha * 100, color=MUTED, ls="--", lw=1)
    ax.text(pk.looks.max(), alpha * 100 + 0.6, f"nominal {alpha:.0%}", color=MUTED, fontsize=8, ha="right")
    for x, v in zip(pk.looks, pk.fpr_naive):
        ax.text(x, v * 100 + 1.0, f"{v:.0%}", color=S[0], fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(pk.looks)
    ax.set_xticklabels(pk.looks)
    ax.set_ylim(0, max(pk.fpr_naive) * 100 * 1.18)
    ax.set_xlabel("Number of interim looks at an A/A test (log scale)", color=MUTED)
    ax.set_ylabel("False-positive rate (%)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Peeking at a test with no real effect inflates false positives")
    fig.tight_layout()
    fig.savefig(folder / "peeking_fpr.png", dpi=150)
    plt.close(fig)


def report(r: dict) -> str:
    fin, a, k = r["financials"], r["allocation"], r["kkt"]
    mc = r["mc_summary"].set_index(["allocation", "metric"])
    ab = r["ab_result"]
    lines = [f"fit: residual sd {r['fit']['sigma']:,.0f}/wk · carry-over (fitted/true): " +
             ", ".join(f"{c} {r['params'][c]['lam']:.2f}/{r['truth'][c]['lam']:.2f}" for c in r["chans"])]
    lines.append("allocation of ${:,.0f} media:".format(r["media"]))
    for _, row in a.iterrows():
        lines.append(f"   {row.label:<22} naive ${row.naive_spend:>9,.0f}  optimal ${row.optimal_spend:>9,.0f}"
                     f"  marginal ROI {row.marginal_roi_optimal:.3f} ({row.bound_status})")
    lines.append(f"KKT: common marginal ROI {k['mu']:.3f}, spread across interior channels {k['spread']:.4f}, "
                 f"conditions hold: {k['conditions_hold']}")
    for s in ("optimal (truth)", "naive (truth)"):
        f = fin[s]
        lines.append(f"{s:<16} ROI {f['roi_multiple']:.2f}x · NPV ${f['npv']:,.0f} · IRR {f['irr_monthly']:.1%}/month"
                     f" · payback month {f['payback_month']}")
    up = fin["optimal (truth)"]["gross_profit"] / fin["naive (truth)"]["gross_profit"] - 1
    lines.append(f"uplift from reallocation: {up:+.1%} gross profit")
    o = mc.loc[("optimal", "roi_multiple")]
    lines.append(f"Monte Carlo ROI (optimal): P10 {o.p10:.2f} · P50 {o.p50:.2f} · P90 {o.p90:.2f} · P(ROI>1) {o.prob_above_breakeven:.1%}")
    lines.append("A/B plan: " + "; ".join(f"{t.n_per_arm:,}/arm, {t.weeks_needed} wk" for t in r["ab_plan"].itertuples()))
    lines.append(f"A/B result ({ab['test']}): lift {ab['rel_lift']:+.1%}, 95% CI [{ab['ci_low']:+.4f}, {ab['ci_high']:+.4f}] abs, p = {ab['p_value']:.4f}")
    pk = r["peeking"]
    lines.append("peeking FPR: " + ", ".join(f"{int(l)} looks {v:.1%}" for l, v in zip(pk.looks, pk.fpr_naive)))
    lines.append("KPI review: " + ", ".join(f"{v} {s}" for s, v in r["kpi"].status.value_counts().items()))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roikit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="fit, allocate, simulate, plan tests, write CSVs and figures")
    rp.add_argument("--out", default="sample_output")
    rp.add_argument("--figures", default=None)
    rp.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    r = pipeline.run(config.load(args.config))
    write_outputs(r, Path(args.out))
    if args.figures:
        charts(r, Path(args.figures))
    print(report(r))


if __name__ == "__main__":
    main()
