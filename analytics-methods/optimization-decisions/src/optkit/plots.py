"""Figures. Fixed palette, left-aligned titles, light grid, no dual axes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import lp
from . import scenarios as S

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, OTHER = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c"


def _style(ax, title, grid=True):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if grid:
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def _save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(Path(folder) / name, dpi=150)
    plt.close(fig)


def feasible_region(folder, lp_result):
    f = S.FARM
    A, b, c = np.array(f["A"], float), np.array(f["b"]), np.array(f["profit"])
    verts = lp.vertices_2d(A, b)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill(verts[:, 0], verts[:, 1], color=SERIES[0], alpha=0.10, lw=0)
    xs = np.linspace(0, 120, 200)
    for k, (row, rhs, name) in enumerate(zip(A, b, f["constraints"])):
        ax.plot(xs, (rhs - row[0] * xs) / row[1], color=SERIES[k], lw=1.6, label=f"{name} ≤ {rhs:g}")
    for z in (20000, 32000, 44000):
        ax.plot(xs, (z - c[0] * xs) / c[1], color=MUTED, lw=1, ls="--")
        lx = 88 if z == 44000 else 2
        ax.text(lx, (z - c[0] * lx) / c[1] + 1.5, f"profit ${z:,.0f}", color=MUTED, fontsize=8)
    path = np.array(lp_result.path)
    for k in range(len(path) - 1):
        ax.annotate("", xy=path[k + 1], xytext=path[k],
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, shrinkA=4, shrinkB=4))
    for k, p in enumerate(path[:-1]):
        ax.scatter(*p, s=26, color=INK, zorder=5)
        off = {0: (4, 6), 1: (-6, 10), 2: (8, 4)}.get(k, (6, 6))
        ax.annotate(f"pivot {k}" if k else "start (0, 0)", p, textcoords="offset points", xytext=off,
                    fontsize=8, color=INK)
    x = lp_result.x
    ax.scatter(*x, s=110, facecolor="white", edgecolor=INK, lw=2, zorder=6)
    ax.annotate(f"optimum (pivot {len(path) - 1}): {x[0]:.0f} grain, {x[1]:.0f} veg acres\nprofit ${lp_result.objective:,.0f}",
                x, textcoords="offset points", xytext=(14, 16), fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Grain (acres)", color=MUTED)
    ax.set_ylabel("Vegetables (acres)", color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax, "Planting plan: feasible region, iso-profit lines and the simplex path")
    _save(fig, folder, "lp_feasible_region.png")


def shadow_prices(folder, con):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(con))[::-1]
    ax.barh(y, con.shadow_price, color=SERIES[0], height=0.55)
    units = ["acres", "hours", "acre-ft"]
    for yi, (_, r), u in zip(y, con.iterrows(), units):
        hi = "∞" if np.isinf(r.rhs_range_high) else f"{r.rhs_range_high:.0f}"
        txt = (f"${r.shadow_price:,.0f} per extra unit, holds for {r.rhs_range_low:.0f}–{hi} {u}"
               if r.shadow_price > 0 else f"$0: {r.slack:.0f} {u} unused (slack)")
        ax.text(r.shadow_price + 4, yi, txt, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(con.constraint)
    ax.set_xlim(0, 420)
    ax.set_xlabel("Shadow price: profit from one more unit of the resource ($)", color=MUTED)
    _style(ax, "Shadow prices: what one more acre, labour hour or acre-foot is worth")
    ax.grid(axis="y", visible=False)
    _save(fig, folder, "shadow_prices.png")


def knapsack_gap(folder, gaps):
    g = gaps.gap_pct.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, np.ceil(g.max()) + 0.5, 0.5)
    ax.hist(g[g > 0], bins=bins, color=SERIES[0], edgecolor="white")
    share_opt = (g == 0).mean()
    ax.text(0.98, 0.95, f"greedy exactly optimal: {share_opt:.0%} of instances\n"
                        f"median gap {np.median(g):.2f}% · 90th pct {np.quantile(g, .9):.2f}% · max {g.max():.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlabel("Margin lost by greedy vs the DP optimum (%, instances with a gap)", color=MUTED)
    ax.set_ylabel("Instances", color=MUTED)
    _style(ax, f"Greedy ratio heuristic vs exact DP on {len(g):,} random knapsacks (20 items)")
    _save(fig, folder, "knapsack_gap.png")


def convergence(folder, cv):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    k = cv.iteration
    ax.semilogy(k, cv.gradient_error, "o-", ms=3, color=SERIES[0], label="gradient ascent (fixed step): linear")
    ne = cv.newton_error.where(cv.newton_error > 1e-15)
    ax.semilogy(k, ne, "s-", ms=4, color=SERIES[1], label="Newton: quadratic")
    ax.semilogy(k, cv.golden_width, "^-", ms=3, color=SERIES[2], label="golden-section bracket width: ×0.618 per step")
    ax.set_xlabel("Iteration", color=MUTED)
    ax.set_ylabel("Distance from optimal price ($, log scale)", color=MUTED)
    ax.set_xlim(-0.5, 30)
    ax.set_ylim(1e-12, 50)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax, "Finding the profit-maximising price p* = $6: three methods")
    _save(fig, folder, "convergence.png")


def newsvendor_curve(folder, nv, q_star, cr):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nv.stock, nv.expected_profit_exact, color=SERIES[0], lw=2, label="exact (sum over Poisson pmf)")
    ax.scatter(nv.stock, nv.expected_profit_simulated, color=SERIES[1], s=14, zorder=4,
               label="simulation (200,000 days)")
    ax.axvline(q_star, color=MUTED, lw=1, ls="--")
    ax.text(q_star - 0.4, nv.expected_profit_exact.min() + 5,
            f"critical ratio {cr:.3f} → stock {q_star} boxes", fontsize=8.5, color=INK, ha="right")
    ax.set_xlabel("Boxes stocked", color=MUTED)
    ax.set_ylabel("Expected daily profit ($)", color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax, "Newsvendor: the critical-ratio rule lands on the simulated optimum")
    _save(fig, folder, "newsvendor.png")


def two_way(folder, tw):
    opts = list(S.OFFERS.index)
    ph = np.sort(tw.p_high.unique())
    fx = np.sort(tw.fixed_offer.unique())
    grid = tw.pivot(index="fixed_offer", columns="p_high", values="best_option").loc[fx, ph]
    codes = grid.apply(lambda col: col.map({o: i for i, o in enumerate(opts)})).to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    dp, df = ph[1] - ph[0], fx[1] - fx[0]
    ax.imshow(codes, origin="lower", aspect="auto", cmap=ListedColormap(SERIES[:3]), vmin=-0.5, vmax=2.5,
              extent=[ph[0] - dp / 2, ph[-1] + dp / 2, fx[0] - df / 2, fx[-1] + df / 2], alpha=0.85)
    ax.scatter([S.OFFER_PROBS[-1]], [S.OFFERS.loc["fixed contract", "low"]], marker="*", s=180,
               color="white", edgecolor=INK, zorder=5)
    ax.annotate("today's inputs", (S.OFFER_PROBS[-1], 50), textcoords="offset points", xytext=(10, -14),
                fontsize=8.5, color=INK)
    ax.legend(handles=[Patch(color=SERIES[i], label=o) for i, o in enumerate(opts)], frameon=False,
              fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("Probability of a high market", color=MUTED)
    ax.set_ylabel("Fixed-contract offer ($k)", color=MUTED)
    _style(ax, "Two-way sensitivity: which offer has the highest expected value", grid=False)
    _save(fig, folder, "sensitivity_two_way.png")


# ---------- decision tree ----------
def decision_tree(folder, tree):
    """Draws the rolled-back tree. Branches not chosen are grey and collapsed to their EV."""
    col_x = [0, 2.3, 4.6, 6.9, 9.2]
    items = {"nodes": [], "edges": []}
    counter = [0]

    def place(node, depth, active):
        """Returns (x, y) of the node; leaves take the next row, parents sit at their children's mean."""
        kind = node["kind"]
        if kind == "end":
            y = counter[0]
            counter[0] += 1
            items["nodes"].append((col_x[-1], y, node, active))
            return col_x[-1], y
        kids = []
        if kind == "decision":
            for label, child in node["options"]:
                chosen = label == node["best"]
                if chosen or child["kind"] == "end":
                    cx, cy = place(child, depth + 1, active and chosen)
                else:   # collapsed alternative: one row showing its EV
                    cx, cy = col_x[depth + 1], counter[0]
                    counter[0] += 1
                    items["nodes"].append((cx, cy, {"kind": "collapsed", "ev": child["ev"]}, False))
                kids.append((cx, cy, label, None, active and chosen))
        else:
            for label, p, child in node["branches"]:
                cx, cy = place(child, depth + 1, active)
                kids.append((cx, cy, label, p, active))
        x, y = col_x[depth], float(np.mean([k[1] for k in kids]))
        items["nodes"].append((x, y, node, active))
        for cx, cy, lab, p, act in kids:
            items["edges"].append((x, y, cx, cy, lab, p, act))
        return x, y

    place(tree, 0, True)
    n_rows = counter[0]
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * n_rows + 1.4))
    for x0, py, x1, cy, lab, p, act in items["edges"]:
        color = INK if act else OTHER
        ax.plot([x0, x0 + 0.25, x1], [py, cy, cy], color=color, lw=1.4 if act else 1.0)
        text = lab if p is None else f"{lab} ({p:.2f})"
        ax.text(x0 + 0.32, cy - 0.08, text, fontsize=7.8, color=color, va="bottom")
    for x, y, node, act in items["nodes"]:
        color = INK if act else OTHER
        k = node["kind"]
        if k == "decision":
            ax.scatter(x, y, marker="s", s=140, color=SERIES[0] if act else OTHER, zorder=5)
            ax.text(x - 0.08, y + 0.22, f"EV {node['ev']:.2f}", fontsize=8, color=color, ha="right", va="top",
                    fontweight="bold")
        elif k == "chance":
            ax.scatter(x, y, marker="o", s=110, color=SERIES[1] if act else OTHER, zorder=5)
            ax.text(x - 0.08, y + 0.22, f"EV {node['ev']:.2f}", fontsize=8, color=color, ha="right", va="top")
        elif k == "collapsed":
            ax.scatter(x, y, marker="o", s=60, color=OTHER, zorder=5)
            ax.text(x + 0.18, y, f"EV {node['ev']:.2f} (not chosen)", fontsize=7.8, color=OTHER, va="center")
        else:
            ax.scatter(x, y, marker="<", s=60, color=SERIES[2] if act else OTHER, zorder=5)
            ax.text(x + 0.18, y, f"{node['value']:g}", fontsize=8, color=color, va="center")
    ax.set_xlim(-1.3, col_x[-1] + 1.6)
    ax.set_ylim(n_rows - 0.4, -0.9)
    ax.axis("off")
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=SERIES[0], label="decision"),
                       plt.Line2D([], [], marker="o", ls="", color=SERIES[1], label="chance"),
                       plt.Line2D([], [], marker="<", ls="", color=SERIES[2], label="payoff ($k)")],
              frameon=False, fontsize=8.5, loc="lower left", ncol=3)
    ax.set_title("Crop-offer decision tree, rolled back (forecast costs $2k; grey = not chosen)",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK)
    _save(fig, folder, "decision_tree.png")
