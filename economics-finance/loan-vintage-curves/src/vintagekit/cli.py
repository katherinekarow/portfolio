"""python -m vintagekit run [--out sample_output] [--figures figures] [--seed 7]"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest, curves, plan, project, reference, synth
from .synth import TERM

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GOOD, CRIT = "#0ca30c", "#d03b3b"


def analyse(seed: int = 7) -> dict:
    c = curves.build(synth.simulate(seed=seed))
    ref = reference.reference(c)
    models = project.fit(c, ref)
    lgd = reference.lgd(c)
    proj = plan.compare(project.project_latest(c, models), models, lgd)
    return {"curves": c, "ref": ref, "models": models, "lgd": lgd, "proj": proj, "el": plan.portfolio_el(proj)}


def run(out: Path, figures: Path | None = None, seed: int = 7) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    a = analyse(seed)
    detail, bt = backtest.run(a["curves"])
    proj, ref, el = a["proj"], a["ref"], a["el"]
    cols = ["segment", "vintage", "vintage_month", "mob", "orig_balance", "balance", "balance_pct", "cum_prepay_pct",
            "cum_default_pct", "k", "z", "pace", "theta", "proj_chain_ladder", "proj_reference", "proj_credibility",
            "proj_low", "proj_high", "plan_to_date_pct", "plan_lifetime_pct", "pace_gap_pct", "future_gap_pct",
            "lifetime_gap_pct", "plan_ratio", "status", "lgd", "ead", "pd_remaining", "expected_loss",
            "lifetime_gap_net_loss"]
    proj[cols].round(5).to_csv(out / "vintage_projection.csv", index=False)
    plan_col = ref.p50.rename("plan_pct")
    ref.assign(plan_pct=plan_col).round(5).to_csv(out / "segment_reference.csv", index=False)
    bt.round(5).to_csv(out / "backtest.csv", index=False)
    detail.round(5).to_csv(out / "backtest_detail.csv", index=False)
    el.rename_axis("segment").round(5).to_csv(out / "expected_loss.csv")
    if figures:
        charts(Path(figures), a, bt, seed)
    immature = proj[proj.mob < TERM]
    shocked = immature.vintage >= synth.N_VINTAGES - synth.SHOCK_VINTAGES
    return {"vintages": len(proj), "immature": len(immature), "a": a, "bt": bt,
            "status_counts": immature.status.value_counts().to_dict(),
            "shocked_behind": int((immature[shocked].status == "behind").sum()), "shocked_n": int(shocked.sum()),
            "unshocked_behind": int((proj[proj.vintage < synth.N_VINTAGES - synth.SHOCK_VINTAGES].status == "behind").sum()),
            "unshocked_n": int((proj.vintage < synth.N_VINTAGES - synth.SHOCK_VINTAGES).sum())}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)


def charts(folder: Path, a: dict, bt: pd.DataFrame, seed: int = 7):
    folder.mkdir(parents=True, exist_ok=True)
    c, ref, proj, models = a["curves"], a["ref"], a["proj"], a["models"]
    segs = list(synth.SEGMENTS)
    shock_from = synth.N_VINTAGES - synth.SHOCK_VINTAGES
    is_seasoned = reference.seasoned(c)

    # 1. vintage curves vs reference band
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, seg in zip(axes.flat, segs):
        r = ref[ref.segment == seg]
        g = c[c.segment == seg]
        for i, (v, x) in enumerate(g[is_seasoned.loc[g.index]].groupby("vintage")):
            ax.plot(x.mob, x.cum_default_pct, color="#b9b8b2", lw=0.7, label="seasoned vintages" if i == 0 else None)
        ax.fill_between(r.mob, r.p25, r.p75, color=S[0], alpha=0.18, lw=0, label="reference P25-P75")
        ax.plot(r.mob, r.p50, color=S[0], lw=2, label="reference P50 (= plan)")
        imm = g[~is_seasoned.loc[g.index]]
        for i, (v, x) in enumerate(imm[imm.vintage < shock_from].groupby("vintage")):
            ax.plot(x.mob, x.cum_default_pct, color=S[2], lw=0.9, alpha=0.8,
                    label="immature vintages" if i == 0 else None)
        for i, (v, x) in enumerate(imm[imm.vintage >= shock_from].groupby("vintage")):
            ax.plot(x.mob, x.cum_default_pct, color=S[1], lw=1.6, marker="o", ms=2.5,
                    label=f"last {synth.SHOCK_VINTAGES} vintages (downturn)" if i == 0 else None)
        ax.set_xlim(0, TERM)
        _style(ax, seg)
        ax.set_ylabel("Cumulative default, % of original balance", color=MUTED, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Months on book", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Cumulative default by vintage against the seasoned reference band", x=0.01, ha="left",
                 fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(folder / "vintage_curves.png", dpi=150); plt.close(fig)

    # 2. projection fan for four immature vintages, with the hidden future drawn in
    full = curves.build(synth.simulate(seed=seed, full=True))
    seg = "Personal B"
    m = models[seg]
    picks = [shock_from + 4, shock_from, 36, 30]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, v in zip(axes.flat, picks):
        p = proj[(proj.segment == seg) & (proj.vintage == v)].iloc[0]
        n = int(p.mob)
        truth = full[(full.segment == seg) & (full.vintage == v)]
        ax.plot(m.ref.mob, m.ref.p50, color=MUTED, lw=1.2, ls="--", label="plan (reference P50)")
        mm, lo = m.path(p.cum_default_pct, n, (p.proj_low - p.cum_default_pct) / max(m.r(TERM) - m.r(n), 1e-9))
        _, hi = m.path(p.cum_default_pct, n, (p.proj_high - p.cum_default_pct) / max(m.r(TERM) - m.r(n), 1e-9))
        _, mid = m.path(p.cum_default_pct, n, p.theta)
        ax.fill_between(mm, lo, hi, color=S[0], alpha=0.18, lw=0, label="80% band")
        ax.plot(mm, mid, color=S[0], lw=2, label="credibility projection")
        ax.plot(truth.mob[truth.mob >= n], truth.cum_default_pct[truth.mob >= n], color=INK, lw=1, ls=":",
                label="what happened next (hidden)")
        ax.plot(truth.mob[truth.mob <= n], truth.cum_default_pct[truth.mob <= n], color=S[1], lw=2.2,
                label="observed")
        ax.scatter([TERM], [p.proj_chain_ladder], marker="D", color=S[3], s=30, zorder=5, label="chain ladder")
        ax.axvline(n, color=GRID, lw=1.5)
        _style(ax, f"{seg}, {p.vintage_month} - MOB {n}, Z = {p.z:.2f}, status: {p.status}")
        ax.set_xlim(0, TERM + 1)
        ax.set_ylabel("Cumulative default, %", color=MUTED, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Months on book", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Projecting immature vintages to lifetime: little data leans on the reference, more data on the vintage",
                 x=0.01, ha="left", fontweight="bold", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(folder / "projection_fan.png", dpi=150); plt.close(fig)

    # 3. back-test error by MOB of projection
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for i, name in enumerate(backtest.METHODS.values()):
        x = bt[bt.method == name]
        ax.plot(x.cut_mob, 100 * x.mape, color=S[i], lw=2, marker="o", ms=4, label=name)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10, 30, 100]); ax.set_yticklabels(["0.3%", "1%", "3%", "10%", "30%", "100%"])
    ax.set_xlabel("Months on book when the projection is made", color=MUTED)
    ax.set_ylabel("MAPE of lifetime default rate (log scale)", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, f"Back-test error, {int(bt.n.iloc[0])} seasoned vintages, leave-one-out")
    ax = axes[1]
    cov = bt[bt.method == "credibility blend"]
    xs = np.arange(len(cov))
    ax.bar(xs, 100 * cov.band_coverage, width=0.65, color=S[0])
    ax.set_xticks(xs); ax.set_xticklabels(cov.cut_mob.astype(int))
    ax.axhline(80, color=MUTED, ls="--", lw=1)
    ax.text(-0.3, 82, "target 80%", color=MUTED, fontsize=8)
    ax.set_ylim(0, 105); ax.set_xlabel("Months on book at projection", color=MUTED)
    ax.set_ylabel("Actual inside the 80% band, %", color=MUTED, fontsize=8)
    _style(ax, "Credibility band: share of actuals inside")
    fig.tight_layout(); fig.savefig(folder / "backtest_error.png", dpi=150); plt.close(fig)

    # 4. loss vs plan by vintage
    colours = {"on-pace": GOOD, "watch": S[3], "behind": CRIT}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, seg in zip(axes.flat, segs):
        p = proj[proj.segment == seg]
        y = 100 * (p.plan_ratio - 1)
        lo = 100 * (p.proj_low / p.plan_lifetime_pct - 1)
        hi = 100 * (p.proj_high / p.plan_lifetime_pct - 1)
        ax.axvspan(shock_from - 0.5, synth.N_VINTAGES - 0.5, color=GRID, alpha=0.6, lw=0)
        for st, col in colours.items():
            k = p.status == st
            ax.bar(p.vintage[k], y[k], color=col, width=0.8, label=st)
        ax.errorbar(p.vintage, y, yerr=[y - lo, hi - y], fmt="none", ecolor=MUTED, lw=0.6, capsize=0)
        for t in (plan.WATCH, plan.BEHIND):
            ax.axhline(100 * (t - 1), color=MUTED, ls=":", lw=0.8)
        ax.axhline(0, color=INK, lw=0.8)
        _style(ax, seg)
        ax.set_ylabel("Projected lifetime default vs plan, %", color=MUTED, fontsize=8)
    for ax in axes[1]:
        ticks = list(range(0, synth.N_VINTAGES, 6))
        ax.set_xticks(ticks); ax.set_xticklabels([str(synth.START + t) for t in ticks], fontsize=7.5)
        ax.set_xlabel("Vintage (grey band = downturn vintages)", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Loss vs plan by vintage: dotted lines are the watch (+10%) and behind (+25%) thresholds",
                 x=0.01, ha="left", fontweight="bold", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(folder / "loss_vs_plan.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vintagekit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="sample_output"); r.add_argument("--figures", default=None)
    r.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None, a.seed)
    res = s["a"]
    print(f"vintages: {s['vintages']} ({len(synth.SEGMENTS)} segments x {synth.N_VINTAGES} months), "
          f"{s['immature']} still immature")
    print("credibility k ($ of expected default): " +
          ", ".join(f"{k} {m.k / 1e3:,.0f}k" for k, m in res["models"].items()))
    print("status of immature vintages: " + ", ".join(f"{k} {v}" for k, v in sorted(s["status_counts"].items())))
    print(f"downturn vintages flagged behind: {s['shocked_behind']}/{s['shocked_n']}; "
          f"other vintages flagged behind: {s['unshocked_behind']}/{s['unshocked_n']}")
    bt = s["bt"].pivot(index="cut_mob", columns="method", values="mape")
    print("back-test MAPE by MOB of projection:")
    print((100 * bt).round(1).to_string())
    el = res["el"].loc["Portfolio"]
    print(f"expected loss on outstanding book: EAD {el.ead / 1e6:,.0f}m x PD {el.pd:.2%} x LGD {el.lgd:.1%} "
          f"= {el.expected_loss / 1e6:,.1f}m ({el.el_pct_of_ead:.2f}% of EAD)")


if __name__ == "__main__":
    main()
