"""python -m clusterkit run --out sample_output --figures figures"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import synth
from .pipeline import run_analysis

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
OTHER = "#a3a29c"


def run(out: Path, figures: Path | None = None, bootstrap_B: int = 50, gap_B: int = 20) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = run_analysis(bootstrap_B=bootstrap_B, gap_B=gap_B)
    s = r["sessions"]
    s.round(4).to_csv(out / "sessions_clustered.csv", index=False)
    kt = r["k_table"].assign(data="after screen").merge(
        r["k_table_all"][["k", "inertia", "silhouette", "gap", "s_k"]], on="k", suffixes=("", "_all_rows"))
    kt.drop(columns="data").round(4).to_csv(out / "choose_k.csv", index=False)
    r["profiles"].round(4).to_csv(out / "segment_profiles.csv", index=False)
    r["avg_curves"].round(4).to_csv(out / "segment_curves.csv", index=False)
    r["confusion"].to_csv(out / "confusion_matrix.csv")
    r["scaling_demo"].round(4).to_csv(out / "scaling_demo.csv", index=False)
    r["distance_demo"].round(4).to_csv(out / "distance_demo.csv", index=False)
    pd.DataFrame(r["linkage"], columns=["a", "b", "height", "size"]).round(6).to_csv(out / "ward_linkage.csv", index=False)
    planted = s.is_outlier.to_numpy()
    flagged = r["flagged"]
    summary = {
        "sessions": len(s), "k": r["k"], "picks_after_screen": r["picks"], "picks_all_rows": r["picks_all"],
        "screened_before_k": int(s.screened_before_k.sum()),
        "ari": {k: round(v, 3) for k, v in r["ari"].items()},
        "bootstrap_jaccard": [round(float(x), 3) for x in r["stability"]],
        "pca_explained_2": [round(float(x), 3) for x in r["pca"].explained_variance_ratio[:2]],
        "flagged_outliers": int(flagged.sum()), "planted_outliers_caught": f"{int((flagged & planted).sum())}/{int(planted.sum())}",
        "kmeans_iterations": r["kmeans"].n_iter,
    }
    pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                  for k, v in summary.items()]).to_csv(out / "summary.csv", index=False)
    if figures:
        charts(Path(figures), r)
    return summary


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def cluster_names(r):
    return [f"{c}: {n}" for c, n in zip(r["profiles"].cluster, r["profiles"].matches_true_segment)]


def charts(folder: Path, r: dict):
    folder.mkdir(parents=True, exist_ok=True)
    k = r["k"]
    names = cluster_names(r)

    # 1. elbow + silhouette
    kt, ka = r["k_table"], r["k_table_all"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    ax.plot(kt.k, kt.inertia, "-o", color=S[0], lw=2, ms=6)
    ax.axvline(r["picks"]["elbow"], color=MUTED, lw=1, ls=":")
    ax.annotate(f"knee rule picks k = {r['picks']['elbow']}", (r["picks"]["elbow"], kt.inertia.max() * 0.9),
                xytext=(8, 0), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set_xlabel("k (number of clusters)", color=MUTED)
    ax.set_ylabel("Inertia (within-cluster SS)", color=MUTED)
    _style(ax, "Elbow: inertia falls with every extra cluster")
    ax = axes[1]
    ax.plot(kt.k[1:], kt.silhouette[1:], "-o", color=S[0], lw=2, ms=6, label="after outlier screen")
    ax.plot(ka.k[1:], ka.silhouette[1:], "-o", color=OTHER, lw=1.5, ms=5, label="all 300 rows")
    best = kt.loc[kt.silhouette.idxmax()]
    ax.scatter([best.k], [best.silhouette], s=110, facecolor="none", edgecolor=INK, lw=1.5, zorder=5)
    ax.annotate(f"best k = {int(best.k)}", (best.k, best.silhouette), xytext=(8, 6),
                textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_xlabel("k (number of clusters)", color=MUTED)
    ax.set_ylabel("Mean silhouette", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "Silhouette: how well each point fits its cluster")
    fig.tight_layout()
    fig.savefig(folder / "choose_k.png", dpi=150)
    plt.close(fig)

    # 2. PCA scatter
    s = r["sessions"]
    p = r["pca"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fl = s.flagged_outlier.to_numpy()
    for c in range(k):
        m = (s.cluster.to_numpy() == c) & ~fl
        ax.scatter(s.pc1[m], s.pc2[m], s=22, color=S[c], alpha=0.85, edgecolor="white", lw=0.5, label=names[c])
    pl = s.is_outlier.to_numpy()
    ax.scatter(s.pc1[fl & pl], s.pc2[fl & pl], s=60, marker="X", color=INK, label="flagged outlier (planted)")
    ax.scatter(s.pc1[fl & ~pl], s.pc2[fl & ~pl], s=50, marker="o", facecolor="none", edgecolor=INK,
               lw=1.2, label="flagged outlier (ordinary row in the tail)")
    ev = p.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)", color=MUTED)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="best")
    _style(ax, "k-means segments seen through the first two principal components")
    fig.tight_layout()
    fig.savefig(folder / "pca_clusters.png", dpi=150)
    plt.close(fig)

    # 3. average demand curve per cluster, with P25-P75 band
    ac = r["avg_curves"]
    days = synth.GRID_DAYS
    cols = [f"d{x}" for x in days]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for c in range(k):
        mean = ac[(ac.cluster == c) & (ac.stat == "mean")][cols].to_numpy()[0]
        lo = ac[(ac.cluster == c) & (ac.stat == "p25")][cols].to_numpy()[0]
        hi = ac[(ac.cluster == c) & (ac.stat == "p75")][cols].to_numpy()[0]
        ax.fill_between(days, lo, hi, color=S[c], alpha=0.15, lw=0)
        ax.plot(days, mean, color=S[c], lw=2, label=names[c])
    ax.invert_xaxis()
    ax.set_xlabel("Days before the session", color=MUTED)
    ax.set_ylabel("Share of final bookings on the books", color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax, "Average booking curve per segment (band = P25 to P75)")
    fig.tight_layout()
    fig.savefig(folder / "segment_curves.png", dpi=150)
    plt.close(fig)

    # 4. dendrogram (linkage computed by hierarchy.ward_linkage; scipy only draws it)
    from scipy.cluster import hierarchy as sch
    Z = r["linkage"]
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2            # height between the k-th and (k-1)-th last merge
    fig, ax = plt.subplots(figsize=(10, 4.6))
    # colour each top-level branch by the k-means cluster most of its leaves belong to
    from .hierarchy import cut_tree
    ward = cut_tree(Z, k)
    kl = r["labels"][r["keep"]]
    ward_to_km = {w: np.bincount(kl[ward == w], minlength=k).argmax() for w in range(k)}
    n = len(Z) + 1
    leaf_group = {i: ward[i] for i in range(n)}
    members = {i: [i] for i in range(n)}
    for t, row in enumerate(Z):
        members[n + t] = members[int(row[0])] + members[int(row[1])]

    def colour(node):
        groups = {leaf_group[i] for i in members[node]}
        return S[ward_to_km[groups.pop()]] if len(groups) == 1 else MUTED

    sch.dendrogram(Z, ax=ax, truncate_mode="lastp", p=40, link_color_func=colour,
                   no_labels=True, above_threshold_color=MUTED)
    ax.axhline(cut, color=INK, lw=1, ls="--")
    ax.annotate(f"cut here -> {k} clusters", (0.2, cut), xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="left", color=INK, fontsize=8.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=S[c], lw=2, label=names[c]) for c in range(k)]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    ax.set_ylabel("Ward merge height", color=MUTED)
    ax.set_xlabel("Sessions (last 40 merges shown; leaves are groups of sessions)", color=MUTED)
    _style(ax, "Ward hierarchy: four branches merge far above everything else")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(folder / "dendrogram.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="clusterkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="generate data, cluster, validate, write CSVs and figures")
    pr.add_argument("--out", default="sample_output")
    pr.add_argument("--figures", default=None)
    pr.add_argument("--bootstrap", type=int, default=50, help="bootstrap resamples for stability")
    pr.add_argument("--gap-refs", type=int, default=20, help="reference sets for the gap statistic")
    a = ap.parse_args(argv)
    summary = run(Path(a.out), Path(a.figures) if a.figures else None, a.bootstrap, a.gap_refs)
    print(json.dumps(summary, indent=2))
