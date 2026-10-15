"""Price elasticity per (segment, tier) with sample-size rules and shrinkage.

Model (per cell, ordinary least squares on logs):

    log(units) = a + b * log(price / median_price) + c1*sin(2*pi*woy/52) + c2*cos(2*pi*woy/52)

elasticity = -b. Seasonality is controlled for, otherwise a summer price rise would look like
inelastic demand.

Rules carried over from the handover:
  * a cell needs >= MIN_OBS weeks (30) to stand on its own
  * the estimate must have the right sign and a 95% interval that excludes zero
Cells that fail are not dropped: they are SHRUNK toward the pooled estimate for their tier
(empirical Bayes: weight = pooled variance / (pooled variance + cell variance)), and flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBS = 30


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else 0.0
    return beta, np.sqrt(np.diag(cov)), r2


def _design(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    g = g[g.units > 0]
    ang = 2 * np.pi * g.week_of_year.to_numpy() / 52
    X = np.column_stack([np.ones(len(g)), np.log(g.price / g.price.median()), np.sin(ang), np.cos(ang)])
    return X, np.log(g.units.to_numpy())


def estimate(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seg, tier), g in history.groupby(["segment", "tier"]):
        X, y = _design(g)
        if len(y) < X.shape[1] + 2:
            rows.append({"segment": seg, "tier": tier, "n_obs": len(y), "raw_elasticity": np.nan, "raw_se": np.nan, "r2": np.nan})
            continue
        beta, se, r2 = _ols(X, y)
        rows.append({"segment": seg, "tier": tier, "n_obs": len(y), "raw_elasticity": -beta[1], "raw_se": se[1], "r2": r2})
    cells = pd.DataFrame(rows)

    pooled = []
    for tier, g in history.groupby("tier"):
        # pooled per grade, with a separate intercept per segment
        parts = [(_design(gg), seg) for seg, gg in g.groupby("segment")]
        segs = [s for _, s in parts]
        X = np.vstack([np.column_stack([x[:, 1:], np.eye(len(segs))[[segs.index(seg)] * len(x)]]) for (x, _), seg in parts])
        y = np.concatenate([yy for (_, yy), _ in parts])
        beta, se, _ = _ols(X, y)
        pooled.append({"tier": tier, "pooled_elasticity": -beta[0], "pooled_se": se[0]})
    cells = cells.merge(pd.DataFrame(pooled), on="tier")

    lo = cells.raw_elasticity - 1.96 * cells.raw_se
    cells["passes_rules"] = (cells.n_obs >= MIN_OBS) & (cells.raw_elasticity > 0) & (lo > 0)
    # between-cell spread of true elasticities around the pooled value (method of moments, floored)
    spread = max(float(np.nanvar(cells.loc[cells.passes_rules, "raw_elasticity"] - cells.loc[cells.passes_rules, "pooled_elasticity"])
                       - np.nanmean(cells.loc[cells.passes_rules, "raw_se"] ** 2)), 0.02)
    w = spread / (spread + cells.raw_se.fillna(np.inf) ** 2)
    cells["weight_on_cell"] = np.where(cells.passes_rules, 1.0, w.fillna(0)).round(3)
    cells["elasticity"] = (cells.weight_on_cell * cells.raw_elasticity.fillna(0)
                           + (1 - cells.weight_on_cell) * cells.pooled_elasticity).round(3)
    cells["method"] = np.where(cells.passes_rules, "cell estimate", "shrunk to grade pool")
    return cells


def backtest(history: pd.DataFrame, holdout_weeks: int = 26) -> pd.DataFrame:
    """Fit on all but the last `holdout_weeks`, predict units at the prices actually charged in the holdout.
    Compared with a naive forecast (training mean for the same cell) - the model has to beat it."""
    cut = history.week.max() - holdout_weeks
    train, test = history[history.week <= cut], history[history.week > cut]
    est = estimate(train).set_index(["segment", "tier"])
    out = []
    for (seg, tier), g in test.groupby(["segment", "tier"]):
        tr = train[(train.segment == seg) & (train.tier == tier) & (train.units > 0)]
        if len(tr) < 8:
            continue
        e = est.loc[(seg, tier), "elasticity"]
        ang_tr = 2 * np.pi * tr.week_of_year / 52
        # re-fit intercept and seasonality with the chosen elasticity held fixed
        Xr = np.column_stack([np.ones(len(tr)), np.sin(ang_tr), np.cos(ang_tr)])
        yr = np.log(tr.units) + e * np.log(tr.price / tr.price.median())
        b, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        ang = 2 * np.pi * g.week_of_year / 52
        pred = np.exp(b[0] + b[1] * np.sin(ang) + b[2] * np.cos(ang) - e * np.log(g.price / tr.price.median()))
        naive = np.full(len(g), tr.units.mean())
        act = g.units.to_numpy()
        out.append({"segment": seg, "tier": tier, "n": len(g),
                    "wape_model": float(np.abs(pred - act).sum() / act.sum()),
                    "wape_naive": float(np.abs(naive - act).sum() / act.sum())})
    return pd.DataFrame(out)
