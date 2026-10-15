"""Excel exception report + charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

USD = FuncFormatter(lambda v, _: f"${v:,.0f}")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CLASS_COLOR = {"CANCELLED_BUT_CHARGED": CRIT, "MISSING_INTERNAL": CRIT, "MISSING_VENDOR": WARN,
               "PRICE_MISMATCH": WARN}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def write_excel(path: Path, *, recon, vendor_rejects, grid, grid_rejects, weekly, mix, plan, floors, run_date) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    det = recon.detail
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        h1 = wb.add_format({"bold": True, "font_size": 14})
        bold = wb.add_format({"bold": True})
        money = wb.add_format({"num_format": "$#,##0.00"})
        warn = wb.add_format({"font_color": "#9c5700", "bg_color": "#fff2cc"})
        ok = wb.add_format({"font_color": "#0b6b0b"})
        bad = wb.add_format({"font_color": "#9c0006"})

        # ---- Summary
        ws = wb.add_worksheet("Summary")
        xw.sheets["Summary"] = ws
        ws.write(0, 0, f"Substitution program reconciliation - run date {run_date}", h1)
        ws.write(1, 0, f"Internal file cut-off used: {recon.internal_file_date}")
        r = 3
        for w in recon.warnings:
            ws.write(r, 0, f"⚠ {w}", warn); r += 1
        r += 1
        ctl = recon.control
        ws.write(r, 0, "Control totals", bold); r += 1
        for k in ["vendor_offers", "internal_charges", "union_offers", "classified_rows", "internal_total_usd",
                  "detail_internal_total_usd", "revenue_at_risk_usd"]:
            ws.write(r, 0, k); ws.write(r, 1, ctl[k], money if k.endswith("usd") else None); r += 1
        for k in ["rows_reconciled", "money_reconciled"]:
            ws.write(r, 0, k); ws.write(r, 1, "PASS" if ctl[k] else "FAIL", ok if ctl[k] else bad); r += 1
        r += 1
        ws.write(r, 0, "Class", bold); ws.write(r, 1, "Offers", bold); ws.write(r, 2, "Revenue at risk", bold); r += 1
        by = det.groupby("recon_class").agg(n=("offer_id", "size"), risk=("revenue_at_risk_usd", "sum"))
        for cls, row in by.sort_values("risk", ascending=False).iterrows():
            ws.write(r, 0, cls); ws.write(r, 1, int(row.n)); ws.write(r, 2, row.risk, money); r += 1
        r += 1
        ws.write(r, 0, "Rejected inputs", bold); r += 1
        ws.write(r, 0, "vendor rows"); ws.write(r, 1, len(vendor_rejects)); r += 1
        ws.write(r, 0, "price-grid rows"); ws.write(r, 1, len(grid_rejects)); r += 1
        ws.set_column(0, 0, 34); ws.set_column(1, 2, 18)

        # ---- data tabs
        tabs = {
            "Exceptions": recon.exceptions,
            "In transit": det[det.recon_class == "IN_TRANSIT"],
            "Matched": det[det.recon_class == "MATCHED"],
            "Vendor rejects": vendor_rejects,
            "Weekly": weekly,
            "Grade paths": mix,
            "Bid decisions": plan.decisions,
            "Policy comparison": plan.summary,
            "Recommended floors": floors,
            "Combined upload": grid,
            "Upload rejects": grid_rejects,
        }
        for name, df in tabs.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
            for j, col in enumerate(df.columns):
                width = min(max(len(str(col)), int(df[col].map(lambda x: len(str(x))).quantile(0.9)) if len(df) else 0) + 2, 48)
                ws.set_column(j, j, width, money if str(col).endswith("_usd") else None)
        ex = xw.sheets["Exceptions"]
        n = len(recon.exceptions)
        if n:
            ex.conditional_format(1, 1, n, 1, {"type": "text", "criteria": "begins with", "value": "CANCELLED", "format": bad})
            ex.conditional_format(1, 1, n, 1, {"type": "text", "criteria": "begins with", "value": "MISSING_INTERNAL", "format": bad})
    return path


def figures(folder: Path, *, recon, weekly, plan, floors, grid) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    # 1. exceptions by class
    exc = recon.exceptions.groupby("recon_class").agg(n=("offer_id", "size"), risk=("revenue_at_risk_usd", "sum"))
    exc = exc.sort_values("risk")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(exc.index.str.replace("_", " ").str.title(), exc.risk, color=[CLASS_COLOR[c] for c in exc.index])
    for b, (cls, row) in zip(bars, exc.iterrows()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  ${row.risk:,.0f}  ({int(row.n)} offers)",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, exc.risk.max() * 1.45)
    ax.set_xlabel("Revenue at risk", color=MUTED)
    ax.xaxis.set_major_formatter(USD)
    _style(ax, "Exceptions found in one run, by type")
    fig.tight_layout(); p = folder / "exceptions_by_class.png"; fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    # 2. weekly trend
    wk = weekly.copy()
    labels = [f"{pd.Timestamp(w):%b %d}" + ("" if c else "\n(partial)") for w, c in zip(wk.week, wk.complete_week)]
    # two panels sharing the week axis - never two y-scales on one chart
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax.bar(labels, wk.accepted_revenue_usd / 1000, color=[SERIES[0] if c else GRID for c in wk.complete_week])
    ax.set_ylabel("Revenue ($k)", color=MUTED)
    _style(ax, "Accepted substitution revenue per week (grey = week still open)")
    rate = (wk.exception_rate * 100).where(wk.complete_week)   # open weeks are still settling
    ax2.plot(labels, rate, color=SERIES[0], marker="o", lw=2)
    ax2.set_ylabel("Per 100 accepted", color=MUTED)
    ax2.set_ylim(0, max(wk.exception_rate.max() * 100 * 1.4, 1))
    _style(ax2, "Exceptions per 100 accepted orders")
    fig.tight_layout(); p = folder / "weekly_trend.png"; fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    # 3. optimizer vs baseline
    d = plan.decisions.assign(path=plan.decisions.from_tier.str.title() + "\n→ " + plan.decisions.to_tier.str.title())
    g = d.groupby("path")[["accept_baseline", "accept_optimized"]].sum()
    s = plan.summary.set_index("policy")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    lifts = s.loc[["baseline", "optimized"], "lift_vs_accept_none_usd"] / 1000
    cols = [CRIT if v < 0 else GOOD for v in lifts]
    a1.bar(["Accept if ≥ grid min", "Optimized"], lifts, color=cols)
    for i, v in enumerate(lifts):
        a1.text(i, v, f"{v:+,.1f}k", ha="center", va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel("Expected revenue vs. accepting none ($k)", color=MUTED)
    _style(a1, "Expected revenue lift")
    x = range(len(g))
    a2.bar([i - 0.2 for i in x], g.accept_baseline, 0.4, color=GRID, label="rule of thumb")
    a2.bar([i + 0.2 for i in x], g.accept_optimized, 0.4, color=SERIES[0], label="optimized")
    a2.set_xticks(list(x)); a2.set_xticklabels(g.index, fontsize=8)
    a2.set_ylabel("Bids accepted", color=MUTED)
    a2.legend(frameon=False, fontsize=9)
    _style(a2, "Accepted bids by grade path")
    fig.tight_layout(); p = folder / "optimizer_vs_baseline.png"; fig.savefig(p, dpi=150); plt.close(fig); out.append(p)

    # 4. recommended floors vs current grid minimum
    cur = grid.groupby(["from_tier", "to_tier"]).min_bid_usd.median().rename("grid_min").reset_index()
    f = floors.merge(cur, on=["from_tier", "to_tier"])
    f["path"] = f.from_tier.str.title() + " → " + f.to_tier.str.title()
    f = f.sort_values("recommended_floor_usd")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = range(len(f))
    ax.hlines(y, f.grid_min, f.recommended_floor_usd, color=GRID, lw=3)
    ax.scatter(f.grid_min, y, color=MUTED, s=50, label="current grid minimum (median)", zorder=3)
    ax.scatter(f.median_break_even_usd, y, color=SERIES[0], s=50, label="break-even (median lot)", zorder=3)
    ax.scatter(f.recommended_floor_usd, y, color=SERIES[1], s=50, label="recommended floor (75th pct)", zorder=3)
    ax.set_yticks(list(y)); ax.set_yticklabels(f.path)
    ax.set_xscale("log"); ax.set_xlabel("Per substitution (log scale)", color=MUTED)
    ax.xaxis.set_major_formatter(USD); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if str(int(v))[0] in "25" else ""))
    ax.tick_params(which="minor", colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Bid floors: current grid vs. displacement-based recommendation")
    fig.tight_layout(); p = folder / "floors_vs_grid.png"; fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    return out
