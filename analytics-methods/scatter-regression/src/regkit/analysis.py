"""The worked analyses behind the README: each returns plain numbers / DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import corr, diagnostics, ols, robust, smooth, synth


def anscombe_table() -> pd.DataFrame:
    rows = []
    for k, (x, y) in synth.anscombe().items():
        s = corr.summary(x, y)
        m = ols.fit({"x": x}, y)
        rows.append({"set": k, **{c: s[c] for c in ("mean_x", "mean_y", "var_x", "var_y", "pearson_r",
                                                    "spearman_rho", "kendall_tau")},
                     "intercept": m.beta[0], "slope": m.beta[1], "r2": m.r2})
    return pd.DataFrame(rows)


def _cols(df: pd.DataFrame) -> dict[str, np.ndarray]:
    t = synth.TRUE
    return {"ln_price": np.log(df.price.to_numpy()), "ln_size": np.log(df.size_index.to_numpy()),
            "promo": df.promo_spend.to_numpy(float),
            "temp_sq": (df.temperature.to_numpy() - t["temp_peak"]) ** 2}


def elasticity_model(df: pd.DataFrame) -> ols.OLS:
    """ln units = a + e·ln price + b·ln size + c·promo + d·(temp − 22)²."""
    return ols.fit(_cols(df), np.log(df.units.to_numpy()))


def coefficients(df: pd.DataFrame) -> pd.DataFrame:
    m = elasticity_model(df)
    tab = m.table()
    truth = {"const": synth.TRUE["intercept"], **{k: synth.TRUE[k] for k in ("ln_price", "ln_size", "promo", "temp_sq")}}
    tab["true"] = tab.term.map(truth)
    tab["true_in_ci"] = (tab.true >= tab.ci_lo) & (tab.true <= tab.ci_hi)
    return tab


def functional_forms(df: pd.DataFrame) -> pd.DataFrame:
    """Same data, four specifications; how to read the price coefficient in each.

    Controls (ln size, promo, temp²) are kept in every model so only the price term changes form.
    """
    c = _cols(df)
    controls = {k: c[k] for k in ("ln_size", "promo", "temp_sq")}
    units, price = df.units.to_numpy(float), df.price.to_numpy(float)
    mean_u, mean_p = units.mean(), price.mean()
    specs = [
        ("level-level", units, price, "units = a + b·price",
         lambda b: f"+$0.10 in price → {b * 0.1:+.1f} units/week"),
        ("log-level", np.log(units), price, "ln units = a + b·price",
         lambda b: f"+$0.10 in price → {100 * (np.exp(b * 0.1) - 1):+.1f}% units"),
        ("level-log", units, np.log(price), "units = a + b·ln price",
         lambda b: f"+1% price → {b * np.log(1.01):+.2f} units/week"),
        ("log-log", np.log(units), np.log(price), "ln units = a + b·ln price",
         lambda b: f"+1% price → {100 * (1.01 ** b - 1):+.2f}% units (elasticity {b:.2f})"),
    ]
    rows = []
    for name, yv, xv, form, read in specs:
        m = ols.fit({"price_term": xv, **controls}, yv)
        b = m.beta[1]
        if name == "level-level":
            el = b * mean_p / mean_u
        elif name == "log-level":
            el = b * mean_p
        elif name == "level-log":
            el = b / mean_u
        else:
            el = b
        rows.append({"form": name, "model": form + " + controls", "price_coef": b, "se": m.se[1],
                     "r2": m.r2, "elasticity_at_mean": el, "reading": read(b),
                     "breusch_pagan_p": diagnostics.breusch_pagan(m)["p"]})
    return pd.DataFrame(rows)


def simpson(df: pd.DataFrame) -> dict:
    """Pooled vs controlled vs within-format price slopes on the log-log scale."""
    c = _cols(df)
    y = np.log(df.units.to_numpy())
    pooled = ols.fit({"ln_price": c["ln_price"]}, y)
    controlled = ols.fit({"ln_price": c["ln_price"], "ln_size": c["ln_size"]}, y)
    # format fixed effects: dummies for 3 of the 4 formats
    dummies = {f"fmt_{f}": (df.format == f).to_numpy(float) for f in synth.FORMATS[1:]}
    fe = ols.fit({"ln_price": c["ln_price"], **dummies}, y)
    within = {}
    for f in synth.FORMATS:
        m = df.format.to_numpy() == f
        within[f] = float(ols.fit({"ln_price": c["ln_price"][m]}, y[m]).beta[1])
    # Frisch-Waugh-Lovell: residualise both y and ln price on ln size, then regress residual on residual
    Z = {"ln_size": c["ln_size"]}
    ry = ols.fit(Z, y).resid
    rx = ols.fit(Z, c["ln_price"]).resid
    partial = ols.OLS(rx[:, None], ry, ["ln_price_resid"])
    return {"pooled": float(pooled.beta[1]), "pooled_ci": pooled.conf_int()[1].tolist(),
            "controlled": float(controlled.beta[1]), "controlled_ci": controlled.conf_int()[1].tolist(),
            "format_fe": float(fe.beta[1]), "within": within, "partial_fwl": float(partial.beta[0]),
            "corr_price_size": corr.pearson(c["ln_price"], c["ln_size"]), "partial_x": rx, "partial_y": ry}


def promo_diagnostics(promo: pd.DataFrame) -> dict:
    x, y = promo.promo_spend.to_numpy(float), promo.units.to_numpy(float)
    m = ols.fit({"promo": x}, y)
    cd = diagnostics.cooks_distance(m)
    bp = diagnostics.breusch_pagan(m)
    top = np.argsort(cd)[::-1][:5]
    return {"model": m, "cooks": cd, "leverage": m.hat, "bp": bp, "se": m.se[1], "se_hc1": m.hc1_se()[1],
            "flagged": np.flatnonzero(cd > 4 / m.n).tolist(), "top5": top.tolist(),
            "studentized": diagnostics.studentized(m)}


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    c = _cols(df)
    cols = {"ln_price": c["ln_price"], "ln_size": c["ln_size"], "ln_traffic": np.log(df.foot_traffic.to_numpy()),
            "promo": c["promo"], "temp_sq": c["temp_sq"]}
    with_traffic = diagnostics.vif(cols)
    without = diagnostics.vif({k: v for k, v in cols.items() if k != "ln_traffic"})
    return pd.DataFrame({"term": list(cols), "vif_with_traffic": [with_traffic[k] for k in cols],
                         "vif_without_traffic": [without.get(k, np.nan) for k in cols]})


def robust_compare(promo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = synth.PROMO_TRUE
    for label, d in [("with outliers", promo), ("outliers removed", promo[~promo.planted_outlier])]:
        x, y = d.promo_spend.to_numpy(float), d.units.to_numpy(float)
        m = ols.fit({"x": x}, y)
        fits = {"OLS": (m.beta[0], m.beta[1]), "Theil-Sen": robust.theil_sen(x, y), "Huber": robust.huber(x, y)}
        for name, (a, b) in fits.items():
            rows.append({"data": label, "method": name, "intercept": a, "slope": b, "true_slope": t["slope"],
                         "slope_error": b - t["slope"], "pearson_r": corr.pearson(x, y), "spearman_rho": corr.spearman(x, y)})
    return pd.DataFrame(rows)


def coverage(sims: int = 4000, n: int = 30, x0: float = 8.0, seed: int = 123) -> dict:
    """Repeat: draw data, fit, build 95% intervals; count how often they contain the truth / a new draw."""
    rng = np.random.default_rng(seed)
    beta = (2.0, 0.5)
    hit_pi = hit_ci_mean = hit_slope = hit_slope_hc1 = 0
    for _ in range(sims):
        x = rng.uniform(0, 10, n)
        y = beta[0] + beta[1] * x + rng.normal(0, 1, n)
        m = ols.fit({"x": x}, y)
        pr = m.predict([1, x0])
        y_new = beta[0] + beta[1] * x0 + rng.normal()
        hit_pi += pr["pi_lo"][0] <= y_new <= pr["pi_hi"][0]
        hit_ci_mean += pr["ci_lo"][0] <= beta[0] + beta[1] * x0 <= pr["ci_hi"][0]
        lo, hi = m.conf_int()[1]
        hit_slope += lo <= beta[1] <= hi
        lo, hi = m.conf_int(robust=True)[1]
        hit_slope_hc1 += lo <= beta[1] <= hi
    return {"sims": sims, "n": n, "pi_coverage": hit_pi / sims, "ci_mean_coverage": hit_ci_mean / sims,
            "slope_ci_coverage": hit_slope / sims, "slope_ci_hc1_coverage": hit_slope_hc1 / sims}


def bp_rejection_rate(sims: int = 300, hetero: bool = False, seed: int = 5) -> float:
    rej = 0
    for s in range(sims):
        d = synth.make_promo(seed=seed * 1000 + s, hetero=hetero, outliers=False)
        m = ols.fit({"x": d.promo_spend.to_numpy(float)}, d.units.to_numpy(float))
        rej += diagnostics.breusch_pagan(m)["p"] < 0.05
    return rej / sims


def temperature_shape(df: pd.DataFrame) -> dict:
    """ln units relative to each store's own average week, against temperature."""
    ly = np.log(df.units.to_numpy())
    ly = ly - pd.Series(ly).groupby(df.store.to_numpy()).transform("mean").to_numpy()
    t = df.temperature.to_numpy()
    grid = np.linspace(np.quantile(t, 0.005), np.quantile(t, 0.995), 60)
    gx, gy = smooth.local_linear(t, ly, grid, frac=0.25)
    tq = synth.TRUE["temp_sq"] * (t - synth.TRUE["temp_peak"]) ** 2
    truth = synth.TRUE["temp_sq"] * (grid - synth.TRUE["temp_peak"]) ** 2 - tq.mean()
    lin = ols.fit({"t": t}, ly)
    bins = smooth.binned(t, ly, 20)
    return {"t": t, "y": ly, "grid": gx, "smooth": gy, "truth": truth, "linear": lin.beta,
            "r": corr.pearson(t, ly), "spearman": corr.spearman(t, ly), "bins": bins,
            "rmse_smooth": float(np.sqrt(np.mean((gy - truth) ** 2))),
            "rmse_linear": float(np.sqrt(np.mean((lin.beta[0] + lin.beta[1] * grid - truth) ** 2)))}
