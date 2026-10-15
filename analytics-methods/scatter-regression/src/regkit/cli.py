"""python -m regkit run --out sample_output --figures figures"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analysis as A
from . import ols, robust, synth

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def run(out: Path, figures: Path | None = None, sims: int = 4000) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = synth.make_stores()
    promo = synth.make_promo()                                   # hetero + outliers
    clean = synth.make_promo(hetero=False, outliers=False)
    ans = A.anscombe_table()
    coef = A.coefficients(df)
    forms = A.functional_forms(df)
    simp = A.simpson(df)
    diag = A.promo_diagnostics(promo)
    clean_diag = A.promo_diagnostics(clean)
    vif = A.vif_table(df)
    rob = A.robust_compare(promo)
    cov = A.coverage(sims)
    temp = A.temperature_shape(df)

    ans.round(4).to_csv(out / "anscombe.csv", index=False, float_format="%.6g")
    coef.to_csv(out / "elasticity_model.csv", index=False, float_format="%.6g")
    forms.to_csv(out / "functional_forms.csv", index=False, float_format="%.6g")
    pd.DataFrame([{"fit": "pooled (no controls)", "ln_price_slope": simp["pooled"]},
                  {"fit": "controlling for ln size", "ln_price_slope": simp["controlled"]},
                  {"fit": "format fixed effects", "ln_price_slope": simp["format_fe"]},
                  {"fit": "partial regression (FWL)", "ln_price_slope": simp["partial_fwl"]}]
                 + [{"fit": f"within {k} stores", "ln_price_slope": v} for k, v in simp["within"].items()]
                 ).to_csv(out / "simpson.csv", index=False, float_format="%.6g")
    m = diag["model"]
    pd.DataFrame({"store": promo.store, "promo_spend": promo.promo_spend, "units": promo.units,
                  "fitted": m.fitted, "residual": m.resid, "studentized": diag["studentized"],
                  "leverage": diag["leverage"], "cooks_d": diag["cooks"], "flag_cooks_4_over_n": diag["cooks"] > 4 / m.n,
                  "planted_outlier": promo.planted_outlier}).to_csv(out / "influence.csv", index=False, float_format="%.6g")
    pd.DataFrame([
        {"data": "promo, heteroscedastic + outliers", "bp_lm": diag["bp"]["lm"], "bp_p": diag["bp"]["p"],
         "slope_se_classical": diag["se"], "slope_se_hc1": diag["se_hc1"]},
        {"data": "promo, homoscedastic clean", "bp_lm": clean_diag["bp"]["lm"], "bp_p": clean_diag["bp"]["p"],
         "slope_se_classical": clean_diag["se"], "slope_se_hc1": clean_diag["se_hc1"]},
    ]).to_csv(out / "heteroscedasticity.csv", index=False, float_format="%.6g")
    vif.to_csv(out / "vif.csv", index=False, float_format="%.6g")
    rob.to_csv(out / "robust.csv", index=False, float_format="%.6g")
    pd.DataFrame([cov]).to_csv(out / "coverage.csv", index=False, float_format="%.6g")
    temp["bins"].to_csv(out / "binned_temperature.csv", index=False, float_format="%.6g")

    if figures:
        charts(Path(figures), df, promo, clean, simp, diag, temp)
    return {"anscombe": ans, "coef": coef, "forms": forms, "simpson": simp, "diag": diag, "clean_diag": clean_diag,
            "vif": vif, "robust": rob, "coverage": cov, "temp": temp}


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def charts(folder: Path, df, promo, clean, simp, diag, temp):
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Anscombe small multiples
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
    xs = np.array([2, 20])
    for ax, (k, (x, y)) in zip(axes.flat, synth.anscombe().items()):
        m = ols.fit({"x": x}, y)
        ax.plot(xs, m.beta[0] + m.beta[1] * xs, color=MUTED, lw=1.5)
        ax.scatter(x, y, s=34, color=S[0], edgecolor="white", lw=1, zorder=3)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.03, 0.95, f"r = {r:.2f}   ŷ = {m.beta[0]:.2f} + {m.beta[1]:.2f}x", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK)
        _style(ax, f"Set {k}")
    for ax in axes[1]:
        ax.set_xlabel("x", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("y", color=MUTED)
    fig.suptitle("Anscombe's quartet: same means, variances, r and fitted line, four different stories",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(folder / "anscombe.png", dpi=150)
    plt.close(fig)

    # 2. fit with confidence and prediction bands
    x, y = clean.promo_spend.to_numpy(float), clean.units.to_numpy(float)
    m = ols.fit({"promo": x}, y)
    g = np.linspace(0, 820, 200)
    pr = m.predict(np.column_stack([np.ones_like(g), g]))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.fill_between(g, pr["pi_lo"], pr["pi_hi"], color=S[0], alpha=0.12, lw=0, label="95% prediction interval (one store-week)")
    ax.fill_between(g, pr["ci_lo"], pr["ci_hi"], color=S[0], alpha=0.35, lw=0, label="95% confidence interval (the mean)")
    ax.plot(g, pr["fit"], color=S[0], lw=2, label=f"OLS fit: {m.beta[0]:.0f} + {m.beta[1]:.3f}·promo")
    t = synth.PROMO_TRUE
    ax.plot(g, t["intercept"] + t["slope"] * g, color=INK, lw=1.2, ls="--", label="true line: 120 + 0.350·promo")
    ax.scatter(x, y, s=22, color=INK, alpha=0.75, zorder=3, lw=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, f"Units vs promo spend, 60 stores: the mean is pinned down, single weeks are not (R² = {m.r2:.2f})",
           "Promo spend this week ($)", "Units sold")
    fig.tight_layout()
    fig.savefig(folder / "fit_bands.png", dpi=150)
    plt.close(fig)

    # 3. diagnostics
    m = diag["model"]
    out = promo.planted_outlier.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=1)
    ax.scatter(m.fitted[~out], m.resid[~out], s=24, color=S[0], lw=0, label="store")
    ax.scatter(m.fitted[out], m.resid[out], s=40, color=S[1], marker="D", lw=0, label="planted mis-keyed store")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    bp = diag["bp"]
    _style(ax, f"Residuals fan out as fitted grows\nBreusch-Pagan LM = {bp['lm']:.1f}, p = {bp['p']:.1g}",
           "Fitted units", "Residual")
    ax = axes[1]
    cd = diag["cooks"]
    idx = promo.store.to_numpy()
    ax.vlines(idx[~out], 0, cd[~out], color=S[0], lw=1.5)
    ax.vlines(idx[out], 0, cd[out], color=S[1], lw=2.5)
    ax.scatter(idx[out], cd[out], color=S[1], s=24, marker="D", zorder=3)
    ax.axhline(4 / m.n, color=MUTED, ls="--", lw=1)
    ax.text(2, 4 / m.n, " 4/n rule of thumb", va="bottom", fontsize=8, color=MUTED)
    _style(ax, "Cook's distance (influence on the fit)\nThe three mis-keyed stores dominate", "Store (sorted by promo spend)", "Cook's D")
    fig.tight_layout()
    fig.savefig(folder / "diagnostics.png", dpi=150)
    plt.close(fig)

    # 4. robust fits
    x, y = promo.promo_spend.to_numpy(float), promo.units.to_numpy(float)
    g = np.array([0, 820])
    mo = ols.fit({"x": x}, y)
    ts = robust.theil_sen(x, y)
    hb = robust.huber(x, y)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(x[~out], y[~out], s=22, color=MUTED, lw=0, alpha=0.8, label="store")
    ax.scatter(x[out], y[out], s=40, color=INK, marker="D", lw=0, label="planted mis-keyed store")
    ax.plot(g, mo.beta[0] + mo.beta[1] * g, color=S[1], lw=2, label=f"OLS slope {mo.beta[1]:.3f}")
    ax.plot(g, hb[0] + hb[1] * g, color=S[2], lw=2, label=f"Huber slope {hb[1]:.3f}")
    ax.plot(g, ts[0] + ts[1] * g, color=S[0], lw=2, label=f"Theil-Sen slope {ts[1]:.3f}")
    ax.plot(g, 120 + 0.35 * g, color=INK, lw=1.2, ls="--", label="true slope 0.350")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Three bad rows drag OLS down; the median-based Theil-Sen line barely moves",
           "Promo spend this week ($)", "Units sold")
    fig.tight_layout()
    fig.savefig(folder / "robust.png", dpi=150)
    plt.close(fig)

    # 5. Simpson's paradox
    lp, lu = np.log(df.price.to_numpy()), np.log(df.units.to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for i, f in enumerate(synth.FORMATS):
        k = df.format.to_numpy() == f
        ax.scatter(lp[k], lu[k], s=5, color=S[i], alpha=0.35, lw=0)
        mf = ols.fit({"x": lp[k]}, lu[k])
        gx = np.quantile(lp[k], [0.02, 0.98])
        ax.plot(gx, mf.beta[0] + mf.beta[1] * gx, color=S[i], lw=2.2, label=f"{f}: slope {mf.beta[1]:.2f}")
    mp = ols.fit({"x": lp}, lu)
    gx = np.quantile(lp, [0.01, 0.99])
    ax.plot(gx, mp.beta[0] + mp.beta[1] * gx, color=INK, lw=2, ls="--", label=f"pooled: slope {mp.beta[1]:+.2f}")
    ax.set_ylim(None, 7.45)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, title="store format", title_fontsize=8, alignment="left")
    _style(ax, "Pooled, higher prices 'sell more'; within each format they sell less",
           "ln price", "ln weekly units")
    ax = axes[1]
    rx, ry = simp["partial_x"], simp["partial_y"]
    ax.scatter(rx, ry, s=5, color=S[0], alpha=0.3, lw=0)
    gx = np.quantile(rx, [0.01, 0.99])
    b = simp["partial_fwl"]
    ax.plot(gx, b * gx, color=INK, lw=2, label=f"slope {b:.2f} (true −1.40)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax, "Partial regression: price and units after removing store size",
           "ln price, residual on ln size", "ln units, residual on ln size")
    fig.tight_layout()
    fig.savefig(folder / "simpson.png", dpi=150)
    plt.close(fig)

    # 6. smoothing a big scatter
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(temp["t"], temp["y"], s=4, color="#a3a29c", alpha=0.35, lw=0, label=f"store-week (n = {temp['t'].size:,})")
    lb = temp["linear"]
    gx = temp["grid"]
    ax.plot(gx, lb[0] + lb[1] * gx, color=S[1], lw=1.8, label=f"straight line (r = {temp['r']:.2f})")
    ax.plot(gx, temp["truth"], color=INK, lw=1.2, ls="--", label="planted curve")
    ax.plot(gx, temp["smooth"], color=S[0], lw=2.2, label="local linear smoother (25% span)")
    bn = temp["bins"]
    ax.scatter(bn.x_mean, bn.y_mean, s=40, color=S[0], edgecolor="white", lw=1, zorder=4, label="binned means (20 equal-count bins)")
    ax.set_ylim(-2.3, 0.85)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    leg.legend_handles[0].set_sizes([20])
    leg.legend_handles[0].set_alpha(0.8)
    _style(ax, "Sales peak around 22 °C: a binned scatter and smoother show it, r does not",
           "Temperature (°C)", "ln units vs the store's average week")
    fig.tight_layout()
    fig.savefig(folder / "smoothing.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="regkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every analysis, write CSVs and figures")
    r.add_argument("--out", default="sample_output")
    r.add_argument("--figures", default=None)
    r.add_argument("--sims", type=int, default=4000)
    a = ap.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None, a.sims)
    an = s["anscombe"]
    print(f"Anscombe: r = {', '.join(f'{v:.3f}' for v in an.pearson_r)} · Spearman = "
          f"{', '.join(f'{v:.2f}' for v in an.spearman_rho)}")
    c = s["coef"].set_index("term")
    lo, hi = c.loc["ln_price", ["ci_lo", "ci_hi"]]
    print(f"elasticity (log-log, controls): {c.loc['ln_price', 'coef']:.3f} [95% CI {lo:.3f}, {hi:.3f}] · true -1.40 · "
          f"all truths in CI: {bool(c.true_in_ci.all())}")
    sp = s["simpson"]
    print(f"Simpson: pooled slope {sp['pooled']:+.2f} → controlling for size {sp['controlled']:+.2f} · "
          f"FWL {sp['partial_fwl']:+.2f}")
    d, cd = s["diag"], s["clean_diag"]
    print(f"Breusch-Pagan p: hetero {d['bp']['p']:.2g} vs clean {cd['bp']['p']:.2f} · slope SE classical "
          f"{d['se']:.4f} vs HC1 {d['se_hc1']:.4f}")
    print(f"Cook's D top 3 stores: {[i + 1 for i in d['top5'][:3]]} (planted {[i + 1 for i in synth.OUTLIER_IDX]})")
    v = s["vif"].set_index("term")
    print(f"VIF ln_size {v.loc['ln_size', 'vif_with_traffic']:.0f}, ln_traffic {v.loc['ln_traffic', 'vif_with_traffic']:.0f}")
    rb = s["robust"]
    w = rb[rb.data == "with outliers"].set_index("method").slope
    print(f"slope with outliers: OLS {w['OLS']:.3f} · Huber {w['Huber']:.3f} · Theil-Sen {w['Theil-Sen']:.3f} · true 0.350")
    cv = s["coverage"]
    print(f"coverage over {cv['sims']} sims: PI {cv['pi_coverage']:.3f} · CI(mean) {cv['ci_mean_coverage']:.3f} · "
          f"slope CI {cv['slope_ci_coverage']:.3f}")
    t = s["temp"]
    print(f"temperature: r = {t['r']:.2f}; RMSE vs planted curve: smoother {t['rmse_smooth']:.3f}, line {t['rmse_linear']:.3f}")


if __name__ == "__main__":
    main()
