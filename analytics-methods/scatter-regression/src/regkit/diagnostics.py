"""Regression diagnostics: heteroscedasticity, leverage, influence, multicollinearity."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .ols import OLS


def breusch_pagan(model: OLS) -> dict:
    """Koenker's studentised Breusch-Pagan test.

    Regress e² on the same X; LM = n·R² of that auxiliary regression ~ χ²(p − 1) under constant variance.
    """
    aux = OLS(model.X, model.resid ** 2)
    lm = model.n * aux.r2
    df = model.p - 1
    return {"lm": float(lm), "df": df, "p": float(stats.chi2.sf(lm, df))}


def leverage(model: OLS) -> np.ndarray:
    """h_ii, the diagonal of the hat matrix H = X(XᵀX)⁻¹Xᵀ. Σ h_ii = p; 'high' is often h > 2p/n."""
    return model.hat


def cooks_distance(model: OLS) -> np.ndarray:
    """D_i = e_i² / (p·s²) · h_ii / (1 − h_ii)². Rule of thumb: look at D_i > 4/n."""
    h = model.hat
    return model.resid ** 2 / (model.p * model.sigma2) * h / (1 - h) ** 2


def studentized(model: OLS) -> np.ndarray:
    """Internally studentised residuals e_i / (s·√(1 − h_ii))."""
    return model.resid / np.sqrt(model.sigma2 * (1 - model.hat))


def vif(cols: dict[str, np.ndarray]) -> dict[str, float]:
    """VIF_j = 1 / (1 − R²_j), R²_j from regressing x_j on the other predictors (with a constant)."""
    out = {}
    names = list(cols)
    for k in names:
        others = [np.asarray(cols[o], float) for o in names if o != k]
        X = np.column_stack([np.ones(len(cols[k]))] + others)
        out[k] = float(1 / (1 - OLS(X, np.asarray(cols[k], float)).r2))
    return out
