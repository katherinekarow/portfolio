"""python -m overbook run --out sample_output --figures figures
python -m overbook export --out ../../docs/overbooking/data.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import estimate as est_mod
from .profit import marginal_rule_limit, normal_limit, optimal_limit, profit_curve, service_limit
from .simulate import simulate_years

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
DEFAULTS = {"C": 20, "p": 0.85, "r": 150.0, "b": 300.0, "rho": 0.0, "lam": 0.0}
RHO_DEMO, LAM_DEMO, SESSIONS_PER_YEAR, YEARS = 0.05, 22.0, 500, 200


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- analyses
def analyses(d: dict = DEFAULTS, replications: int = 30) -> dict:
    C, p, r, b = d["C"], d["p"], d["r"], d["b"]
    lmax = 40
    base = profit_curve(C, p, r, b, 0.0, None, lmax)
    disp = profit_curve(C, p, r, b, RHO_DEMO, None, lmax)
    dem = profit_curve(C, p, r, b, 0.0, LAM_DEMO, lmax)
    curve = pd.DataFrame({"limit": base.limit, "profit_binomial": base.profit, "profit_overdispersed": disp.profit,
                          "profit_demand_capped": dem.profit, "expected_bumps": base.bumps,
                          "expected_empty_slots": base["empty"], "p_any_bump": base.pbump,
                          "bumps_overdispersed": disp.bumps, "empty_overdispersed": disp["empty"]})
    opt = {"binomial": optimal_limit(C, p, r, b), "overdispersed": optimal_limit(C, p, r, b, RHO_DEMO),
           "demand_capped": optimal_limit(C, p, r, b, 0.0, LAM_DEMO)}

    # three methods
    rows = []
    for Cg in (10, 20, 50, 100, 200, 400):
        for pg in (0.75, 0.85, 0.95):
            for ratio in (0.5, 2.0, 5.0):
                bg = r * ratio
                row = {"capacity": Cg, "p": pg, "bump_to_revenue": ratio,
                       "enumeration": optimal_limit(Cg, pg, r, bg)["limit"],
                       "marginal_rule": marginal_rule_limit(Cg, pg, r, bg),
                       "normal_approx": normal_limit(Cg, pg, r, bg)}
                if Cg <= 50:
                    row["enumeration_rho03"] = optimal_limit(Cg, pg, r, bg, 0.03)["limit"]
                    row["marginal_rule_rho03"] = marginal_rule_limit(Cg, pg, r, bg, 0.03)
                    row["normal_approx_rho03"] = normal_limit(Cg, pg, r, bg, 0.03)
                rows.append(row)
    methods = pd.DataFrame(rows)

    # sensitivity
    rows = []
    for pg in (0.70, 0.80, 0.90, 0.95):
        for ratio in np.round(np.geomspace(0.1, 20, 31), 4):
            o = optimal_limit(C, pg, r, r * ratio)
            rows.append({"p": pg, "bump_to_revenue": ratio, "optimal_limit": o["limit"],
                         "overbook": o["limit"] - C, "p_any_bump": o["pbump"], "expected_bumps": o["bumps"],
                         "expected_empty": o["empty"], "profit": o["profit"],
                         "service_limit_5pct": service_limit(C, pg, 0.0, 0.05)})
    sens = pd.DataFrame(rows)

    # demand
    rows = []
    unc = opt["binomial"]
    for lam in (12, 16, 18, 20, 22, 24, 26, 30, 40):
        o = optimal_limit(C, p, r, b, 0.0, lam)
        rows.append({"lambda": lam, "optimal_limit": o["limit"], "unconstrained_limit": unc["limit"],
                     "profit": o["profit"], "profit_at_capacity_limit": profit_curve(C, p, r, b, 0, lam, lmax).profit[C],
                     "unconstrained_profit": unc["profit"]})
    demand = pd.DataFrame(rows)
    demand["gain_from_overbooking"] = demand.profit - demand.profit_at_capacity_limit

    # Monte Carlo: 200 independent simulated years (500 sessions each) per limit and model
    rows = []
    for j, (name, rho, lam) in enumerate((("binomial", 0.0, None), ("overdispersed", RHO_DEMO, None),
                                          ("demand_capped", 0.0, LAM_DEMO))):
        cur = profit_curve(C, p, r, b, rho, lam, lmax)
        for k, L in enumerate((C, opt[name]["limit"], opt[name]["limit"] + 3)):
            sy = simulate_years(L, C, p, r, b, rho, lam, YEARS, SESSIONS_PER_YEAR, seed=10 * j + k)
            y0, exact = sy["years"][0], cur.profit[L]
            cover = np.abs(sy["means"] - exact) <= sy["half_width"]
            rows.append({"model": name, "limit": L, "analytic": exact, "year1_mean": y0["mean"], "year1_lo": y0["lo"],
                         "year1_hi": y0["hi"], "years": YEARS, "share_of_years_ci_covers": cover.mean(),
                         "pooled_mean": sy["pooled_mean"], "pooled_lo": sy["pooled_lo"], "pooled_hi": sy["pooled_hi"],
                         "pooled_inside": sy["pooled_lo"] <= exact <= sy["pooled_hi"],
                         "analytic_bumps": cur.bumps[L], "mc_bumps": float(np.mean([y["bumps"] for y in sy["years"]]))})
    sim = pd.DataFrame(rows)

    # estimation
    hist, truth = est_mod.make_history()
    est = est_mod.estimate(hist).merge(truth, on=["clinic", "type"])
    reg = est_mod.regret_by_clinic(est.drop(columns="true_p"), truth, C, r, b)
    reps = []
    for sd in range(replications):
        h, t = est_mod.make_history(1000 + sd)
        e = est_mod.estimate(h).merge(t, on=["clinic", "type"])
        g = est_mod.regret_by_clinic(e.drop(columns="true_p"), t, C, r, b)
        per = g.filter(like="regret_").div(g.sessions, axis=0)
        small = g.sessions <= 30
        reps.append({"seed": 1000 + sd, "mse_raw": ((e.raw_p - e.true_p) ** 2).mean(),
                     "mse_shrunk": ((e.shrunk_p - e.true_p) ** 2).mean(),
                     "regret_per_session_raw": per.regret_raw.mean(), "regret_per_session_shrunk": per.regret_shrunk.mean(),
                     "regret_per_session_no_dispersion": per.regret_shrunk_no_dispersion.mean(),
                     "small_clinic_regret_raw": per.regret_raw[small].mean(),
                     "small_clinic_regret_shrunk": per.regret_shrunk[small].mean(), "rho_hat": e.rho_hat.mean()})
    reps = pd.DataFrame(reps)
    p_hats = np.round(np.arange(0.70, 0.9801, 0.005), 3)
    rc = pd.concat([est_mod.regret_curve(C, p, r, bb, p_hats).assign(bump_cost=bb) for bb in (b, 3 * b)])
    return {"curve": curve, "opt": opt, "methods": methods, "sensitivity": sens, "demand": demand,
            "simulation": sim, "estimates": est, "regret_by_clinic": reg, "replications": reps, "regret_curve": rc}


# ---------------------------------------------------------------- figures
def charts(a: dict, folder: Path, d: dict = DEFAULTS):
    folder.mkdir(parents=True, exist_ok=True)
    C = d["C"]
    cur, opt, sim = a["curve"], a["opt"], a["simulation"]
    sl = cur[(cur.limit >= C - 4) & (cur.limit <= 32)]

    # 1. profit by limit
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (col, key, lab) in enumerate((("profit_binomial", "binomial", "Binomial show-ups"),
                                         ("profit_overdispersed", "overdispersed", f"Over-dispersed (ρ = {RHO_DEMO})"),
                                         ("profit_demand_capped", "demand_capped", f"Requests ~ Poisson({LAM_DEMO:g})"))):
        ax.plot(sl.limit, sl[col], color=SERIES[i], lw=2, label=lab)
        o = opt[key]
        ax.scatter([o["limit"]], [o["profit"]], color=SERIES[i], s=60, zorder=5, edgecolor="white", lw=1.5)
    s = sim[sim.model == "binomial"]
    ax.errorbar(s.limit, s.year1_mean, yerr=[s.year1_mean - s.year1_lo, s.year1_hi - s.year1_mean], fmt="none",
                ecolor=INK, capsize=4, lw=1.2, zorder=6, label="One simulated year, 95% CI (binomial)")
    ob = opt["binomial"]
    ax.annotate(f"best limit {ob['limit']}: ${ob['profit']:,.0f}/session", (ob["limit"], ob["profit"]),
                xytext=(ob["limit"] + 1.5, ob["profit"] + 60), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axvline(C, color=MUTED, ls="--", lw=1)
    ax.text(C + 0.15, 1600, " capacity", color=MUTED, fontsize=8)
    ax.set_xlabel("Booking limit (patients booked per session)", color=MUTED)
    ax.set_ylabel("Expected profit per session ($)", color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, f"Expected profit by booking limit: {C} slots, show rate {d['p']:.0%}, \\${d['r']:.0f}/visit, \\${d['b']:.0f}/bump")
    fig.tight_layout(); fig.savefig(folder / "profit_by_limit.png", dpi=150); plt.close(fig)

    # 2. bumps & empty slots
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(sl.limit, sl.expected_empty_slots, color=SERIES[0], lw=2, label="Empty slots (binomial)")
    ax.plot(sl.limit, sl.expected_bumps, color=SERIES[1], lw=2, label="Bumped patients (binomial)")
    ax.plot(sl.limit, sl.empty_overdispersed, color=SERIES[0], lw=1.5, ls="--", label=f"Empty slots (ρ = {RHO_DEMO})")
    ax.plot(sl.limit, sl.bumps_overdispersed, color=SERIES[1], lw=1.5, ls="--", label=f"Bumped patients (ρ = {RHO_DEMO})")
    ax.axvline(ob["limit"], color=MUTED, ls=":", lw=1.2)
    ax.text(ob["limit"] + 0.15, 2.9, f" best limit {ob['limit']}\n "
            f"{ob['empty']:.2f} empty, {ob['bumps']:.2f} bumped", color=INK, fontsize=8, va="top")
    ax.set_xlabel("Booking limit (patients booked per session)", color=MUTED)
    ax.set_ylabel("Patients per session", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    _style(ax, "Overbooking trades empty slots for bumped patients")
    fig.tight_layout(); fig.savefig(folder / "bumps_and_empty_slots.png", dpi=150); plt.close(fig)

    # 3. sensitivity
    sens = a["sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, (pg, g) in enumerate(sens.groupby("p")):
        axes[0].step(g.bump_to_revenue, g.overbook, where="post", color=SERIES[i], lw=2, label=f"show rate {pg:.0%}")
        axes[1].plot(g.bump_to_revenue, g.p_any_bump * 100, color=SERIES[i], lw=2, label=f"show rate {pg:.0%}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 20]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "20"])
        ax.set_xlabel("Bump cost ÷ revenue per visit (b / r, log scale)", color=MUTED)
    axes[0].set_ylabel("Extra bookings above capacity", color=MUTED)
    axes[1].set_ylabel("Sessions with at least one bump (%)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0], "Optimal overbooking falls as bumps get dearer")
    _style(axes[1], "...and so does the bump risk it accepts")
    fig.tight_layout(); fig.savefig(folder / "sensitivity.png", dpi=150); plt.close(fig)

    # 4. estimation
    e = a["estimates"][a["estimates"]["type"] == "new patient"].sort_values("sessions")
    rc = a["regret_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    y = np.arange(len(e))
    ax.scatter(e.raw_p * 100, y, color=SERIES[0], s=30, label="raw rate", zorder=3)
    ax.scatter(e.shrunk_p * 100, y, color=SERIES[1], marker="D", s=30, label="shrunk (empirical Bayes)", zorder=4)
    ax.scatter(e.true_p * 100, y, color=INK, marker="|", s=170, label="true rate", zorder=5)
    for yy, (a1, a2) in zip(y, zip(e.raw_p, e.shrunk_p)):
        ax.plot([a1 * 100, a2 * 100], [yy, yy], color=GRID, lw=2, zorder=2)
    ax.axvline(e.prior_mean.iloc[0] * 100, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} ({n})" for c, n in zip(e.clinic, e.sessions)], fontsize=8)
    ax.set_xlabel("New-patient show rate (%); sessions of history in brackets", color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    _style(ax, "Thin clinics are pulled toward the pool")
    ax = axes[1]
    for i, (bb, g) in enumerate(rc.groupby("bump_cost")):
        ax.plot(g.p_hat * 100, g.regret, color=SERIES[i], lw=2, label=f"bump cost ${bb:,.0f}")
    ax.axvline(d["p"] * 100, color=MUTED, ls="--", lw=1)
    ax.text(d["p"] * 100 + 0.3, ax.get_ylim()[1] * 0.92, f"true rate {d['p']:.0%}", color=MUTED, fontsize=8)
    ax.set_xlabel("Show rate used for planning (%)", color=MUTED)
    ax.set_ylabel("Profit lost per session ($)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _style(ax, "What mis-estimating the show rate costs")
    fig.tight_layout(); fig.savefig(folder / "estimation_and_regret.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- commands
def run(out: Path, figures: Path | None = None, replications: int = 30) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    a = analyses(replications=replications)
    a["curve"].round(6).to_csv(out / "profit_curve.csv", index=False)
    a["methods"].to_csv(out / "method_agreement.csv", index=False)
    a["sensitivity"].round(6).to_csv(out / "sensitivity.csv", index=False)
    a["demand"].round(4).to_csv(out / "demand_constrained.csv", index=False)
    a["simulation"].round(4).to_csv(out / "simulation_check.csv", index=False)
    a["estimates"].round(5).to_csv(out / "show_rate_estimates.csv", index=False)
    a["regret_by_clinic"].round(2).to_csv(out / "regret_by_clinic.csv", index=False)
    a["replications"].round(6).to_csv(out / "shrinkage_replications.csv", index=False)
    a["regret_curve"].round(4).to_csv(out / "regret_curve.csv", index=False)
    if figures:
        charts(a, Path(figures))
    return a


def export(path: Path) -> dict:
    """Defaults, input bounds, presets (with the Python answer) and estimated rates for the web page."""
    hist, truth = est_mod.make_history()
    est = est_mod.estimate(hist)
    seg = est.groupby("type", sort=False).agg(prior_mean=("prior_mean", "first"), rho_hat=("rho_hat", "first")).reset_index()
    presets = [{"name": "Default", **DEFAULTS}]
    for _, s in seg.iterrows():
        presets.append({"name": f"Estimated {s['type']}", **DEFAULTS, "p": round(float(s.prior_mean), 3),
                        "rho": round(float(s.rho_hat), 3)})
    presets += [{"name": "Costly bumps", **DEFAULTS, "b": 1500.0},
                {"name": "Thin demand", **DEFAULTS, "lam": 21.0},
                {"name": "Large session", **DEFAULTS, "C": 120, "p": 0.9}]
    for pr in presets:
        o = optimal_limit(pr["C"], pr["p"], pr["r"], pr["b"], pr["rho"], pr["lam"] or None)
        pr["python"] = {"limit": o["limit"], "profit": round(o["profit"], 6)}
    data = {"defaults": DEFAULTS, "presets": presets,
            "bounds": {"C": [1, 300], "p": [0.3, 0.999], "r": [1, 10000], "b": [0, 100000], "rho": [0, 0.5], "lam": [0, 1000]},
            "estimates": est[["clinic", "type", "sessions", "raw_p", "shrunk_p"]].round(4).to_dict("records")}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="overbook")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", default="outputs"); r.add_argument("--figures", default=None)
    r.add_argument("--replications", type=int, default=30)
    e = sub.add_parser("export"); e.add_argument("--out", default="../../docs/overbooking/data.json")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        d = export(Path(a.out))
        print(f"wrote {a.out}: {len(d['presets'])} presets, {len(d['estimates'])} clinic estimates")
        return
    res = run(Path(a.out), Path(a.figures) if a.figures else None, a.replications)
    o, d = res["opt"], DEFAULTS
    cur = res["curve"]
    print(f"session: {d['C']} slots, show rate {d['p']:.0%}, ${d['r']:.0f}/visit, ${d['b']:.0f}/bump "
          f"(critical ratio r/(r+b) = {d['r'] / (d['r'] + d['b']):.3f})")
    print(f"  book to capacity ({d['C']}): ${cur.profit_binomial[d['C']]:,.2f}/session")
    for k, v in o.items():
        print(f"  best limit {k:<14} {v['limit']:>3}  ${v['profit']:,.2f}/session  bumps {v['bumps']:.3f}  "
              f"empty {v['empty']:.3f}  P(bump>=1) {v['pbump']:.1%}")
    m = res["methods"]
    print(f"methods: enumeration = marginal rule in {(m.enumeration == m.marginal_rule).sum()}/{len(m)} cases; "
          f"normal approx exact in {(m.enumeration == m.normal_approx).sum()}/{len(m)}, "
          f"max gap {(m.enumeration - m.normal_approx).abs().max()} (C>=100: {(m[m.capacity >= 100].enumeration - m[m.capacity >= 100].normal_approx).abs().max()})")
    s = res["simulation"]
    print(f"Monte Carlo ({YEARS} years x {SESSIONS_PER_YEAR} sessions, 9 limit/model cases): one-year 95% CIs cover the "
          f"exact value in {s.share_of_years_ci_covers.min():.1%}-{s.share_of_years_ci_covers.max():.1%} of years; "
          f"pooled estimate inside its CI in {s.pooled_inside.sum()}/{len(s)}; "
          f"max |pooled - exact| ${(s.pooled_mean - s.analytic).abs().max():.2f}")
    rp = res["replications"].mean()
    print(f"estimation ({len(res['replications'])} synthetic histories): MSE raw {rp.mse_raw:.2e} vs shrunk {rp.mse_shrunk:.2e}; "
          f"regret/session raw ${rp.regret_per_session_raw:.2f} vs shrunk ${rp.regret_per_session_shrunk:.2f} "
          f"(small clinics ${rp.small_clinic_regret_raw:.2f} vs ${rp.small_clinic_regret_shrunk:.2f}); "
          f"ignoring over-dispersion ${rp.regret_per_session_no_dispersion:.2f}")


if __name__ == "__main__":
    main()
