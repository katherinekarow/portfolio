"""Figures (matplotlib, fixed palette, no dual axes)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, OTHER = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c"
GOOD, CRIT = "#0ca30c", "#d03b3b"


def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=10)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def velocity_figure(V_ref, bands, target, path: Path):
    """bands: (3, L) P25/P50/P75 shares; target: dict for the current service date."""
    L = V_ref.shape[1]
    x = np.arange(L)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(0)
    for row in V_ref[rng.choice(len(V_ref), min(30, len(V_ref)), replace=False)]:
        a.plot(x, row, color=OTHER, lw=0.5, alpha=0.5)
    a.fill_between(x, bands[0], bands[2], color=SERIES[0], alpha=0.22, lw=0, label="P25–P75 of history")
    a.plot(x, bands[1], color=SERIES[0], lw=2, label="P50 (median)")
    k = target["k"]
    a.axvline(k, color=MUTED, lw=0.8, ls="--")
    a.annotate(f"today: {k} days out\nP50 share = {bands[1][k]:.0%}", (k, bands[1][k]), (k + 14, 0.52),
               fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.set_xlim(L - 1, 0)
    a.set_ylim(0, 1.02)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.set_xlabel("days before the service date", color=MUTED, fontsize=9)
    a.set_ylabel("share of final orders received", color=MUTED, fontsize=9)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    _style(a, f"Velocity: how orders for a {target['weekday']} arrive (last 52 weeks)")

    obs = target["observed_curve"]                   # cumulative for days_out >= k
    b.plot(x, target["hist_median"], color=OTHER, lw=1.6, ls="--", label="history median (same weekday)")
    b.plot(x[k:], obs[k:], color=SERIES[0], lw=2.2, label="this date so far")
    proj_path = obs[k] * bands[1][: k + 1] / bands[1][k]
    b.plot(x[: k + 1], proj_path, color=SERIES[1], lw=1.6, ls=":", label="pace projection (P50 shape)")
    lo, pt, hi = target["low"], target["point"], target["high"]
    lo80, hi80 = target["low80"], target["high80"]
    b.errorbar([0], [pt], yerr=[[pt - lo80], [hi80 - pt]], fmt="none", color=SERIES[1], capsize=3, lw=0.9)
    b.errorbar([0], [pt], yerr=[[pt - lo], [hi - pt]], fmt="o", color=SERIES[1], ms=5, capsize=0, lw=3.5)
    b.plot([0], [target["actual"]], marker="D", color=INK, ms=5, ls="none", label="actual final (held back)")
    b.annotate(f"projected {pt:.0f}\n50%: {lo:.0f}–{hi:.0f}\n80%: {lo80:.0f}–{hi80:.0f}", (0, pt), (16, pt * 0.92),
               fontsize=8, color=INK, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    b.set_xlim(L - 1, -2)
    b.set_ylim(0, max(hi80, target["actual"]) * 1.1)
    b.set_xlabel("days before the service date", color=MUTED, fontsize=9)
    b.set_ylabel("cumulative orders", color=MUTED, fontsize=9)
    b.legend(frameon=False, fontsize=8, loc="upper left")
    _style(b, f"Current period: {target['date']:%a %d %b %Y}, pace {target['pace']:.2f}× history")
    fig.tight_layout()
    _save(fig, path)


def decomposition_figure(index, dec, path: Path, shift_date):
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    parts = [("observed", "Weekly orders (observed)"), ("trend", "Trend: 2×52 centred moving average"),
             ("seasonal", "Seasonal index (multiplicative, averages 1)"), ("resid", "Residual (observed ÷ trend ÷ seasonal)")]
    for ax, (key, title) in zip(axes, parts):
        ax.plot(index, dec[key], color=SERIES[0] if key != "resid" else MUTED, lw=1.4)
        if key == "resid":
            ax.axhline(1, color=MUTED, lw=0.8)
        if key == "trend":
            ax.axvline(shift_date, color=SERIES[1], lw=1, ls="--")
            ax.text(shift_date, np.nanmin(dec["trend"]), "  planted +12% level shift", color=SERIES[1],
                    fontsize=8, va="bottom")
        _style(ax, title)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    _save(fig, path)


def weekday_figure(est: dict, truth: dict, path: Path):
    names = list(est)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - 0.2, [est[n] for n in names], 0.4, color=SERIES[0], label="estimated (decomposition)")
    ax.bar(x + 0.2, [truth[n] for n in names], 0.4, color=OTHER, label="planted")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(x, names)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Weekday indices (1.0 = average day)")
    fig.tight_layout()
    _save(fig, path)


def backtest_figure(acc: pd.DataFrame, path: Path):
    acc = acc.sort_values("WAPE_%", ascending=False)
    y = np.arange(len(acc))
    best = acc["WAPE_%"].idxmin()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    a.barh(y, acc["WAPE_%"], color=[SERIES[0] if i == best else OTHER for i in acc.index])
    for yi, v, m in zip(y, acc["WAPE_%"], acc["MASE"]):
        a.text(v + 0.8, yi, f"{v:.1f}%  (MASE {m:.2f})", va="center", fontsize=8, color=INK)
    a.set_yticks(y, acc["method"])
    a.set_xlim(0, acc["WAPE_%"].max() * 1.45)
    a.set_xlabel("WAPE, % of volume", color=MUTED, fontsize=9)
    _style(a, "Accuracy: 13 rolling origins × 14 days")
    b.barh(y, acc["bias_%"], color=[SERIES[1] if abs(v) > 2 else OTHER for v in acc["bias_%"]])
    b.axvline(0, color=MUTED, lw=0.8)
    for yi, v in zip(y, acc["bias_%"]):
        b.text(v + (1 if v >= 0 else -1), yi, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
               fontsize=8, color=INK)
    lim = acc["bias_%"].abs().max() * 1.35
    b.set_xlim(-lim, lim * 0.6)
    b.set_xlabel("bias, % of volume (+ = over-forecast)", color=MUTED, fontsize=9)
    _style(b, "Bias")
    fig.tight_layout()
    _save(fig, path)


def yoy_figure(tab: pd.DataFrame, det: dict, shift_date, path: Path):
    fig = plt.figure(figsize=(10, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1, 1], hspace=0.45)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    c = fig.add_subplot(gs[2], sharex=b)
    zoom = tab.loc["2025-02-03":"2025-03-30"]
    a.plot(zoom.index, zoom.yoy_calendar, color=SERIES[1], lw=1.4, marker="o", ms=2.5, label="same calendar date")
    a.plot(zoom.index, zoom.yoy_weekday, color=SERIES[0], lw=1.8, marker="o", ms=2.5,
           label="same weekday (364 days back)")
    a.axhline(0, color=MUTED, lw=0.8)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    a.set_ylim(min(zoom.yoy_calendar.min(), -0.6) * 1.1, zoom.yoy_calendar.max() * 1.35)
    _style(a, "Daily YoY %, Feb–Mar 2025: calendar alignment compares different weekdays")
    win = tab.loc[tab.yoy_weekday.first_valid_index():]
    b.plot(win.index, win.yoy_roll28, color=SERIES[0], lw=1.6)
    b.axvline(shift_date, color=SERIES[1], lw=1, ls="--")
    b.text(shift_date, 0.02, "  planted +12% level shift", color=SERIES[1], fontsize=8, va="bottom")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    _style(b, "YoY % of 28-day rolling sums (weekday aligned)")
    sp = det["s_pos"]
    c.plot(sp.index, sp, color=SERIES[0], lw=1.6, label="CUSUM S+ (upward drift)")
    c.plot(det["s_neg"].index, det["s_neg"], color=OTHER, lw=1.0, label="CUSUM S− (downward drift)")
    c.axhline(det["h"], color=CRIT, lw=1, ls="--", label=f"alarm threshold h = {det['h']:g}")
    c.axvline(shift_date, color=SERIES[1], lw=1, ls="--", label="planted shift")
    if det["alarm_up"] is not None:
        c.plot([det["alarm_up"]], [sp.loc[det["alarm_up"]]], "o", color=CRIT, ms=6)
        lag = (det["alarm_up"] - shift_date).days
        c.annotate(f"alarm {det['alarm_up']:%d %b %Y} ({lag} days after)", (det["alarm_up"], sp.loc[det['alarm_up']]),
                   (det["alarm_up"] + pd.Timedelta(days=25), det["h"] * 3), fontsize=8, color=INK,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    c.set_ylim(0, min(sp.max() * 1.05, det["h"] * 8))
    c.legend(frameon=False, fontsize=8, loc="upper left")
    _style(c, "CUSUM on standardised YoY residuals")
    c.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(b.get_xticklabels(), visible=False)
    _save(fig, path)
