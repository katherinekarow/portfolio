"""python -m statkit run [--out sample_output] [--figures figures]  - every analysis, CSVs and charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import capability, descriptive as ds, inference as inf, montecarlo as mc, pareto, queueing as q, spc, synth

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRIT = "#d03b3b"
CLINIC_MU = 4.0           # consults per doctor-hour (15-minute mean)


# ------------------------------------------------------------------ analyses
def packing_analysis():
    w = synth.packing_line()
    ch = spc.xbar_r(w, baseline=20)
    sig_mean = ch["sigma_within"] / np.sqrt(ch["n"])
    cu = spc.cusum(ch["xbar"], ch["center"], sig_mean)
    ew = spc.ewma(ch["xbar"], ch["center"], sig_mean)
    we = spc.western_electric(ch["xbar"], ch["center"], sig_mean)
    base = w[:20].ravel()
    cap = capability.indices(ds.mean(base), ch["sigma_within"], ds.std(base), synth.LSL, synth.USL)
    cap_after = capability.indices(ds.mean(w[synth.SHIFT_AT:].ravel()), ch["sigma_within"],
                                   ds.std(w[synth.SHIFT_AT:].ravel()), synth.LSL, synth.USL)
    table = pd.DataFrame({"subgroup": np.arange(len(w)), "xbar": ch["xbar"], "range": ch["range"],
                          "lcl_x": ch["lcl_x"], "ucl_x": ch["ucl_x"], "ucl_r": ch["ucl_r"],
                          "flag_x": ch["flag_x"], "cusum_plus": cu["c_plus"], "cusum_minus": cu["c_minus"],
                          "ewma": ew["z"], "ewma_ucl": ew["ucl"], "ewma_lcl": ew["lcl"]})
    table["we_rules"] = [",".join(str(r) for i, r in we if i == s) for s in table.subgroup]
    return w, ch, cu, ew, we, cap, cap_after, table


def pipeline_analysis():
    df = synth.pipeline_rows()
    b = df.day < synth.BASELINE_DAYS
    expected = df.weekday.map(df[b].groupby("weekday").rows.mean())
    df["expected"] = expected.round()
    df["resid_pct"] = 100 * (df.rows / expected - 1)
    ch = spc.imr(df.resid_pct, baseline=synth.BASELINE_DAYS)
    cu = spc.cusum(df.resid_pct, 0.0, ch["sigma"])
    ew = spc.ewma(df.resid_pct, 0.0, ch["sigma"])
    df["i_flag"] = ch["flag"]
    df["cusum_plus"], df["cusum_minus"] = cu["c_plus"], cu["c_minus"]
    df["ewma"], df["ewma_flag"] = ew["z"], ew["flag"]
    after = df.day >= synth.DROP_AT
    first_i = df[after & df.i_flag].day.min()
    first_cu = next((i for i in range(synth.DROP_AT, len(df)) if max(cu["c_plus"][i], cu["c_minus"][i]) > cu["h"]), None)
    first_ew = df[after & df.ewma_flag].day.min()
    return df, ch, cu, {"i_chart": None if pd.isna(first_i) else int(first_i), "cusum": first_cu,
                        "ewma": None if pd.isna(first_ew) else int(first_ew)}


def inspection_analysis():
    d = synth.inspections()
    pc = spc.p_chart(d.defects, d.inspected, baseline=30)
    we = spc.western_electric(pc["z"], 0.0, 1.0)
    d["p"], d["lcl"], d["ucl"], d["flag"] = pc["p"], pc["lcl"], pc["ucl"], pc["flag"]
    d["we_rules"] = [",".join(str(r) for i, r in we if i == s) for s in d.day]
    return d, pc, we


def queue_analysis(seed: int = 21):
    rng = np.random.default_rng(seed)
    rows = []
    for c in (1, 2, 3):
        for rho in (0.5, 0.7, 0.8, 0.9):
            lam = rho * c * CLINIC_MU
            th, sim = q.mmc(lam, CLINIC_MU, c), q.simulate(lam, CLINIC_MU, c, n=200000, rng=rng)
            rows.append({"doctors": c, "utilisation": rho, "arrivals_per_hour": lam,
                         "Wq_min_formula": 60 * th["Wq"], "Wq_min_sim": 60 * sim["Wq"],
                         "P_wait_formula": th["P_wait"], "P_wait_sim": sim["P_wait"],
                         "L_time_avg_sim": sim["L_time_avg"], "lambda_W_sim": sim["lambda_W"]})
    scen = []
    for c in (3, 4):
        th = q.mmc(10.0, CLINIC_MU, c)
        sim = q.simulate(10.0, CLINIC_MU, c, n=200000, rng=rng)
        scen.append({"doctors": c, "arrivals_per_hour": 10.0, "utilisation": th["rho"], "P_wait": th["P_wait"],
                     "Wq_min_formula": 60 * th["Wq"], "Wq_min_sim": 60 * sim["Wq"], "Wq_p90_min_sim": 60 * sim["Wq_p90"],
                     "L_formula": th["L"], "L_sim": sim["L_time_avg"]})
    return pd.DataFrame(rows), pd.DataFrame(scen)


def inference_analysis(seed: int = 31):
    rng = np.random.default_rng(seed)
    a, b = synth.clinic_waits()
    wt = inf.welch_t(b, a)
    tci = inf.t_ci_mean(a)
    boot = inf.bootstrap_ci(a, np.median, B=5000, rng=rng)
    insp = synth.inspections()
    wk = insp.iloc[:7]
    wil = inf.wilson_ci(int(wk.defects.sum()), int(wk.inspected.sum()))
    ct = synth.defects_by_shift()
    chi = inf.chi2_independence(ct.values)
    sd = ds.std(np.r_[a - a.mean(), b - b.mean()])
    n80 = inf.n_for_power(3.0, sd)
    err = inf.simulated_error_rates(rng, 3.0, sd, n80)
    rows = [
        ("Site A mean wait (min)", ds.mean(a), *tci, "t interval, 95%"),
        ("Site A median wait (min)", ds.median(a), *boot, "percentile bootstrap, 95%, B=5000"),
        ("Site B − Site A mean wait (min)", wt["diff"], np.nan, np.nan, f"Welch t={wt['t']:.2f}, df={wt['df']:.1f}, p={wt['p']:.2g}"),
        ("Defect rate, first week", wk.defects.sum() / wk.inspected.sum(), *wil, "Wilson, 95%"),
        ("Defect reason vs shift", chi["chi2"], np.nan, np.nan, f"chi-square df={chi['dof']}, p={chi['p']:.2g}"),
        ("n per site to detect 3 min (80% power)", n80, np.nan, np.nan,
         f"sim: type I {err['type_I']:.3f}, power {err['power']:.3f}, sd {sd:.1f}"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "estimate", "ci_low", "ci_high", "method"]), {
        "welch": wt, "chi2": chi, "n80": n80, "errors": err, "a": a, "b": b}


def run(out: Path, figures: Path | None = None) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    clt = pd.DataFrame(ds.clt_demo(rng))
    clt.to_csv(out / "clt_demo.csv", index=False)
    w, ch, cu, ew, we, cap, cap_after, xt = packing_analysis()
    xt.to_csv(out / "packing_xbar_r.csv", index=False)
    pd.DataFrame([{"period": "baseline (subgroups 0-19)", **cap}, {"period": f"after shift (from {synth.SHIFT_AT})", **cap_after}]
                 ).to_csv(out / "capability.csv", index=False)
    pipe, pch, pcu, detect = pipeline_analysis()
    pipe.to_csv(out / "pipeline_monitor.csv", index=False)
    insp, pc, pwe = inspection_analysis()
    insp.to_csv(out / "p_chart.csv", index=False)
    inft, extra = inference_analysis()
    inft.to_csv(out / "inference.csv", index=False)
    a = extra["a"]
    desc = pd.DataFrame([{"series": "clinic A wait (min)", **ds.summary(a)}, {"series": "clinic B wait (min)", **ds.summary(extra["b"])},
                         {"series": "box weight baseline (kg)", **ds.summary(w[:20].ravel())}])
    desc.to_csv(out / "descriptive.csv", index=False)
    par = pareto.pareto_table(synth.DEFECT_REASONS.keys(), synth.DEFECT_REASONS.values())
    par.to_csv(out / "pareto_defects.csv", index=False)
    sk = synth.sku_revenue()
    abc = pareto.pareto_table(sk.sku, sk.revenue)
    abc.to_csv(out / "abc_skus.csv", index=False)
    qt, qs = queue_analysis()
    qt.to_csv(out / "queue_validation.csv", index=False)
    qs.to_csv(out / "clinic_scenarios.csv", index=False)
    sim = mc.simulate()
    mcs = mc.summary(sim)
    tor = pd.DataFrame(mc.tornado())
    tor.to_csv(out / "tornado.csv", index=False)
    pd.DataFrame([mcs]).to_csv(out / "monte_carlo.csv", index=False)
    if figures:
        charts(Path(figures), ch, cu, pipe, pch, pcu, detect, par, qt, tor, mcs, insp)

    first_x = int(np.flatnonzero(ch["flag_x"][synth.SHIFT_AT:])[0] + synth.SHIFT_AT) if ch["flag_x"][synth.SHIFT_AT:].any() else None
    abc_counts = abc.abc.value_counts().to_dict()
    return {
        "clt_skew_n1_vs_n30": [round(clt.skew_of_means[0], 2), round(float(clt.loc[clt.n == 30, "skew_of_means"].iloc[0]), 2)],
        "xbar_limits": [round(ch["lcl_x"], 4), round(ch["center"], 4), round(ch["ucl_x"], 4)], "rbar": round(ch["rbar"], 4),
        "packing_shift_at": synth.SHIFT_AT, "packing_first_xbar_flag": first_x,
        "packing_first_cusum": cu["first_signal"], "packing_first_ewma": ew["first_signal"],
        "packing_we_first": next(((i, r) for i, r in we if i >= synth.SHIFT_AT), None),
        "false_flags_before_shift": int(ch["flag_x"][:synth.SHIFT_AT].sum()),
        "capability_baseline": {k: round(v, 3) for k, v in cap.items()},
        "capability_after": {k: round(v, 3) for k, v in cap_after.items()},
        "pipeline_drop_at": synth.DROP_AT, "pipeline_detect_day": detect,
        "pipeline_i_flags": pipe[pipe.i_flag].day.tolist(), "pipeline_sigma_pct": round(pch["sigma"], 3),
        "p_chart_pbar": round(pc["pbar"], 4), "p_chart_flags": insp[insp.flag].day.tolist(),
        "p_chart_we_first_after": next(((i, r) for i, r in pwe if i >= synth.P_SHIFT_AT), None),
        "inference": inft.round(4).to_dict("records"),
        "pareto_top3_cum": round(float(par.cum_pct.iloc[2]), 1), "pareto_A": par[par.abc == "A"].item.tolist(),
        "abc_counts": abc_counts, "abc_A_share": round(float(abc[abc.abc == "A"].share_pct.sum()), 1),
        "gini_skus": round(pareto.gini(sk.revenue), 3),
        "clinic": qs.round(3).to_dict("records"),
        "queue_max_abs_err_min": round(float((qt.Wq_min_formula - qt.Wq_min_sim).abs().max()), 2),
        "monte_carlo": {k: round(v, 3) for k, v in mcs.items()},
        "tornado_order": tor.label.tolist(),
    }


# ------------------------------------------------------------------ charts
def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _limit(ax, y, label, x_end, ls="--", x0=-0.5):
    ax.hlines(y, x0, x_end, color=MUTED, lw=1, ls=ls)
    ax.text(x_end, y, f" {label}", va="center", ha="left", fontsize=8, color=MUTED)


def charts(folder, ch, cu, pipe, pch, pcu, detect, par, qt, tor, mcs, insp):
    folder.mkdir(parents=True, exist_ok=True)
    k = np.arange(len(ch["xbar"])); xe = k[-1] + 0.6
    # 1. X̄–R chart
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a1.axvspan(synth.SHIFT_AT - 0.5, k[-1] + 0.5, color=GRID, alpha=0.55, lw=0)
    a1.text(synth.SHIFT_AT - 0.3, ch["ucl_x"] + 0.012, "planted +0.025 kg filler drift", fontsize=8, color=MUTED, va="bottom")
    a1.plot(k, ch["xbar"], color=S[0], lw=2, marker="o", ms=4.5, label="subgroup mean (n = 5)")
    f = ch["flag_x"]
    if f.any():
        a1.scatter(k[f], ch["xbar"][f], s=70, color=CRIT, zorder=5, label="beyond 3σ limit")
    sig_mean = ch["sigma_within"] / np.sqrt(ch["n"])
    we = spc.western_electric(ch["xbar"], ch["center"], sig_mean)
    wi = sorted({i for i, _ in we})
    if wi:
        a1.scatter(k[wi], ch["xbar"][wi], s=90, facecolor="none", edgecolor=CRIT, lw=1.6, zorder=6,
                   label="Western Electric zone rule (2-of-3 > 2σ, 4-of-5 > 1σ, 8 in a row)")
    if cu["first_signal"] is not None:
        a1.axvline(cu["first_signal"], color=CRIT, lw=1, ls=":", ymax=0.74)
        a1.text(cu["first_signal"] + 0.2, ch["lcl_x"] + 0.004, f"CUSUM signal\n(subgroup {cu['first_signal']})", fontsize=8, color=INK, va="bottom")
    for y, lab in ((ch["ucl_x"], f"UCL {ch['ucl_x']:.3f}"), (ch["lcl_x"], f"LCL {ch['lcl_x']:.3f}")):
        _limit(a1, y, lab, xe)
    _limit(a1, ch["center"], f"X̿ {ch['center']:.3f}", xe, ls="-")
    a1.set_ylim(ch["lcl_x"] - 0.03, ch["ucl_x"] + 0.05)
    _style(a1, "X̄ chart: no point crosses 3σ, but zone rules and CUSUM catch the drift", ylabel="mean box weight (kg)")
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    a2.plot(k, ch["range"], color=S[0], lw=2, marker="o", ms=4)
    fr = ch["flag_r"]; a2.scatter(k[fr], ch["range"][fr], s=70, color=CRIT, zorder=5)
    _limit(a2, ch["ucl_r"], f"UCL {ch['ucl_r']:.3f}", xe); _limit(a2, ch["rbar"], f"R̄ {ch['rbar']:.3f}", xe, ls="-")
    _style(a2, "R chart: spread within each subgroup", "subgroup (hourly sample of 5 boxes)", "range (kg)")
    a2.set_xlim(-0.8, k[-1] + 5)
    fig.tight_layout(); fig.savefig(folder / "xbar_r_chart.png", dpi=150); plt.close(fig)

    # 2. pipeline residuals + CUSUM
    d = pipe.day.values; de = d[-1] + 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for ax in (a1, a2):
        ax.axvspan(synth.DROP_AT - 0.5, d[-1] + 0.5, color=GRID, alpha=0.55, lw=0)
    a1.plot(d, pipe.resid_pct, color=S[0], lw=1.5, marker="o", ms=3, label="rows vs weekday baseline")
    fl = pipe.i_flag.values
    a1.scatter(d[fl], pipe.resid_pct[fl], s=60, color=CRIT, zorder=5, label="beyond 3σ (I chart)")
    _limit(a1, pch["ucl"], "UCL", de); _limit(a1, pch["lcl"], "LCL", de); _limit(a1, pch["center"], "centre", de, ls="-")
    a1.text(synth.DROP_AT + 0.5, pch["ucl"] + 0.9, "planted −1% drop", fontsize=8, color=MUTED)
    a1.text(synth.SPIKE_AT + 1.2, pipe.resid_pct[synth.SPIKE_AT], "planted duplicate load (+6%)", fontsize=8, color=MUTED, va="center")
    a1.set_ylim(min(pipe.resid_pct.min(), pch["lcl"]) - 0.8, max(pipe.resid_pct.max(), pch["ucl"]) + 1.8)
    _style(a1, "Daily row count of a pipeline table: individuals chart", ylabel="deviation from baseline (%)")
    a1.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    a2.plot(d, pcu["c_minus"], color=S[1], lw=2, label="C⁻ (downward)")
    a2.plot(d, pcu["c_plus"], color=S[0], lw=2, label="C⁺ (upward)")
    _limit(a2, pcu["h"], "h = 5σ", de)
    if detect["cusum"] is not None:
        a2.scatter([detect["cusum"]], [pcu["c_minus"][detect["cusum"]]], s=70, color=CRIT, zorder=5)
        a2.annotate(f"signal day {detect['cusum']} ({detect['cusum'] - synth.DROP_AT} days after the drop)",
                    (detect["cusum"], pcu["c_minus"][detect["cusum"]]), xytext=(-210, 22), textcoords="offset points",
                    fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a2.set_ylim(0, max(pcu["c_minus"].max(), pcu["h"]) * 1.15)
    _style(a2, "CUSUM (k = 0.5σ) catches the small sustained drop", "day", "cumulative sum (σ units)")
    a2.legend(frameon=False, fontsize=8, loc="upper left"); a2.set_xlim(-1, d[-1] + 9)
    fig.tight_layout(); fig.savefig(folder / "pipeline_cusum.png", dpi=150); plt.close(fig)

    # 3. Pareto — both on one % axis
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(par))
    cols = [S[0] if c == "A" else ("#a3a29c" if c == "C" else "#7fa9e0") for c in par.abc]
    ax.bar(x, par.share_pct, color=cols, width=0.72, edgecolor="white", lw=2)
    (cum_line,) = ax.plot(x, par.cum_pct, color=S[1], lw=2, marker="o", ms=5, label="cumulative share")
    ax.axhline(80, color=MUTED, lw=1, ls="--"); ax.text(len(par) - 0.5, 81.5, "80%", ha="right", fontsize=8, color=MUTED)
    for xi, s in zip(x, par.share_pct):
        if s > 10:
            ax.text(xi, s - 2.5, f"{s:.0f}%", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(xi, s + 1.2, f"{s:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, par["item"], rotation=30, ha="right")
    ax.set_ylim(0, 105)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S[0], label="share of defects, class A"), Patch(color="#7fa9e0", label="class B"),
                       Patch(color="#a3a29c", label="class C"), cum_line], frameon=False, fontsize=8, loc="center right")
    _style(ax, f"Pareto of packing-line defect reasons (n = {int(par.value.sum())})", ylabel="% of all defects")
    fig.tight_layout(); fig.savefig(folder / "pareto_defects.png", dpi=150); plt.close(fig)

    # 4. Erlang C wait vs utilisation
    fig, ax = plt.subplots(figsize=(8.5, 5))
    rr = np.linspace(0.3, 0.93, 200)
    for i, c in enumerate((1, 2, 3)):
        wq = [60 * q.mmc(r * c * CLINIC_MU, CLINIC_MU, c)["Wq"] for r in rr]
        ax.plot(rr * 100, wq, color=S[i], lw=2, label=f"{c} doctor{'s' if c > 1 else ''} (formula)")
        sub = qt[qt.doctors == c]
        ax.scatter(sub.utilisation * 100, sub.Wq_min_sim, color=S[i], s=60, edgecolor="white", lw=2, zorder=5)
    ax.scatter([], [], color=MUTED, s=50, label="simulation, 200k patients each")
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    _style(ax, "Mean wait before a 15-minute consult vs doctor utilisation", "utilisation ρ = λ / (c·μ) (%)", "mean wait Wq (minutes, log scale)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(folder / "erlang_c_wait.png", dpi=150); plt.close(fig)

    # 5. tornado: separate low/high bars on one axis
    fig, ax = plt.subplots(figsize=(9, 4.6))
    base = tor.base_profit.iloc[0] / 1e3
    y = np.arange(len(tor))[::-1]
    h = 0.36
    ax.barh(y + h / 2, tor.profit_at_p10 / 1e3 - base, left=base, height=h, color=S[0], label="input at its P10")
    ax.barh(y - h / 2, tor.profit_at_p90 / 1e3 - base, left=base, height=h, color=S[1], label="input at its P90")
    ax.axvline(base, color=INK, lw=1)
    ax.text(base + 2, y[0] + 0.62, f"base case (all inputs at median): ${base:,.0f}k", ha="left", fontsize=8, color=INK)
    for yi, (lo, hi) in zip(y, zip(tor.profit_at_p10 / 1e3, tor.profit_at_p90 / 1e3)):
        for v, off in ((lo, h / 2), (hi, -h / 2)):
            ax.text(v + (3 if v > base else -3), yi + off, f"{v:,.0f}", va="center", ha="left" if v > base else "right", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, tor.label)
    ax.set_ylim(-0.7, len(tor) - 0.1)
    lo_all, hi_all = (tor[["profit_at_p10", "profit_at_p90"]].min().min() / 1e3, tor[["profit_at_p10", "profit_at_p90"]].max().max() / 1e3)
    ax.set_xlim(lo_all - 45, hi_all + 45)
    _style(ax, "Tornado: season profit when one input moves to its P10 / P90", "season profit ($ thousand)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(folder / "tornado.png", dpi=150); plt.close(fig)

    # 6. p-chart
    fig, ax = plt.subplots(figsize=(9, 4.4))
    dd = insp.day.values
    ax.axvspan(synth.P_SHIFT_AT - 0.5, dd[-1] + 0.5, color=GRID, alpha=0.55, lw=0)
    ax.step(dd, insp.ucl * 100, where="mid", color=MUTED, lw=1, ls="--", label="3σ limits (vary with daily sample size)")
    ax.step(dd, insp.lcl * 100, where="mid", color=MUTED, lw=1, ls="--")
    pbar = 100 * insp.defects[:30].sum() / insp.inspected[:30].sum()
    _limit(ax, pbar, f"p̄ {pbar:.1f}%\n(days 0-29)", dd[-1] + 0.8, ls="-", x0=-0.8)
    ax.plot(dd, insp.p * 100, color=S[0], lw=1.5, marker="o", ms=3.5, label="daily defect %")
    fl = insp.flag.values
    ax.scatter(dd[fl], insp.p[fl] * 100, s=60, color=CRIT, zorder=5, label="beyond 3σ")
    rule = (insp.we_rules != "") & (insp.we_rules != "1") & ~fl
    ax.scatter(dd[rule], insp.p[rule] * 100, s=70, facecolor="none", edgecolor=CRIT, lw=1.5, zorder=5, label="Western Electric rule 2-4")
    ax.text(synth.P_SHIFT_AT + 0.3, 0.6, "planted 3.5% → 6%", fontsize=8, color=MUTED)
    _style(ax, "p-chart: share of inspected boxes with a defect", "day", "defective (%)")
    ax.set_ylim(0, max(insp.p.max(), insp.ucl.max()) * 100 + 1.5)
    ax.set_xlim(-1, dd[-1] + 7)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(folder / "p_chart.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="statkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every analysis, write CSVs and figures")
    r.add_argument("--out", default="sample_output")
    r.add_argument("--figures", default=None)
    args = ap.parse_args(argv)
    res = run(Path(args.out), Path(args.figures) if args.figures else None)
    print(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
