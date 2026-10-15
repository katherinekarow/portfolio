"""python -m ratekit run [--out outputs/] [--figures figures/]"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import index, matching, quality, synth

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GOOD, WARN, CRIT, INK, MUTED, GRID = "#0ca30c", "#fab219", "#d03b3b", "#0b0b0b", "#52514e", "#e6e5e0"


def run(out: Path | None = None, figures: Path | None = None, seed: int = 43) -> dict:
    w = synth.make(seed)
    matches, unmapped = matching.match(w["ours"], w["theirs"])
    clean, rep = quality.clean(w["observations"], w["dates"])
    idx = index.daily_index(w["our_prices"], matches, clean, w["dates"])
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        matches.to_csv(out / "product_matches.csv", index=False)
        unmapped.to_csv(out / "unmapped_labels.csv", index=False)
        idx.to_csv(out / "price_index_daily.csv", index=False)
    if figures:
        charts(Path(figures), w, clean, rep, idx)
    last = idx[idx.date == idx.date.max()]
    return {"world": w, "matches": matches, "unmapped": unmapped, "clean": clean, "report": rep, "index": idx,
            "summary": {
                "our_products": len(w["ours"]), "matched_products": len(matches),
                "matched_cells": int(matches.groupby(["lot_id", "tier"]).ngroups),
                "fuzzy_labels": int(matches.tier_rule.str.startswith("similar").sum()), "unmapped_labels": len(unmapped),
                "duplicates_removed": rep["duplicates_removed"], "outliers": rep["outliers"],
                "stale": {k: len(v) for k, v in rep["stale_days"].items()},
                "missing": {k: len(v) for k, v in rep["missing_days"].items()},
                "positions": last.position.value_counts().to_dict()}}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def charts(folder: Path, w, clean, rep, idx):
    folder.mkdir(parents=True, exist_ok=True)
    days = w["dates"]
    comps = list(synth.COMPETITORS)
    # 1. feed health
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for i, c in enumerate(comps):
        stale = set(rep["stale_days"].get(c, [])); miss = set(rep["missing_days"].get(c, []))
        outl = clean[(clean.competitor == c) & clean.is_outlier].groupby("quote_date").size()
        for d in days:
            col = CRIT if d in miss else WARN if d in stale else "#cfe6cf"
            ax.add_patch(plt.Rectangle((mdates.date2num(d) - 0.45, i - 0.35), 0.9, 0.7, color=col, lw=0))
            if d in outl.index:
                ax.text(mdates.date2num(d), i, "×", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xlim(mdates.date2num(days[0]) - 1, mdates.date2num(days[-1]) + 1); ax.set_ylim(-0.6, len(comps) - 0.4)
    ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#cfe6cf", WARN, CRIT)]
    ax.legend(handles + [plt.Line2D([], [], ls="", marker="$×$", color=INK)], ["delivered", "stale (frozen prices)", "missing", "outlier price"],
              frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax, "Competitor feed health, day by day")
    fig.tight_layout(); fig.savefig(folder / "feed_health.png", dpi=150); plt.close(fig)

    # 2. one lot over time: our price vs the market band
    pick = idx.groupby(["lot_id", "tier"]).competitors.min().reset_index()
    pick = pick[(pick.competitors >= 2) & (pick.tier == "premium")].iloc[0]
    d = idx[(idx.lot_id == pick.lot_id) & (idx.tier == pick.tier)].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(d.date, d.market_min, d.market_max, color=S[1], alpha=0.15, lw=0, label="competitor range")
    ax.plot(d.date, d.market_median, color=S[1], lw=1.6, ls="--", label="market median")
    ax.plot(d.date, d.price, color=S[0], lw=2.2, label="our price")
    st = rep["stale_days"].get("Competitor B", [])
    if st:
        ax.axvspan(min(st), max(st), color=WARN, alpha=0.15, lw=0)
        ax.text(min(st), ax.get_ylim()[1], " Competitor B stale - excluded", fontsize=8, va="top", color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.set_ylabel("USD", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    _style(ax, f"{pick.lot_id} {pick.tier}: our price against matched competitors")
    fig.tight_layout(); fig.savefig(folder / "price_vs_market.png", dpi=150); plt.close(fig)

    # 3. latest index by tier
    last = idx[(idx.date == idx.date.max()) & idx.price_index.notna()]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axvspan(0.95, 1.05, color=GRID, alpha=0.7, lw=0)
    for i, t in enumerate(synth.TIERS):
        v = last[last.tier == t].price_index
        col = np.where(v < 0.95, S[2], np.where(v > 1.05, S[1], MUTED))
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), c=col, s=28)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_yticks(range(4)); ax.set_yticklabels([t.title() for t in synth.TIERS])
    ax.set_xlabel("Price index = our price / market median   (grey band = at market, ±5%)", color=MUTED)
    _style(ax, "Where each lot sits against the market today")
    fig.tight_layout(); fig.savefig(folder / "price_index_by_tier.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ratekit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", default="outputs"); r.add_argument("--figures", default=None)
    a = ap.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None)["summary"]
    print(f"products matched: {s['matched_products']} competitor products -> {s['matched_cells']} of our {s['our_products']} lot-grades")
    print(f"grade labels: {s['fuzzy_labels']} fixed by fuzzy match, {s['unmapped_labels']} left unmapped for review")
    print(f"feed: {s['duplicates_removed']} duplicates removed, {s['outliers']} outlier prices excluded")
    print(f"stale days: {s['stale']}   missing days: {s['missing']}")
    print("today's positioning: " + ", ".join(f"{k} {v}" for k, v in s["positions"].items()))


if __name__ == "__main__":
    main()
