"""Ordinary least squares from the linear algebra up.

β = (XᵀX)⁻¹Xᵀy, computed by default through the QR decomposition X = QR (β = R⁻¹Qᵀy),
which gives the same answer without forming XᵀX and squaring the condition number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def design(cols: dict[str, np.ndarray], const: bool = True) -> tuple[np.ndarray, list[str]]:
    names = (["const"] if const else []) + list(cols)
    n = len(next(iter(cols.values())))
    parts = ([np.ones(n)] if const else []) + [np.asarray(v, float) for v in cols.values()]
    return np.column_stack(parts), names


@dataclass
class OLS:
    X: np.ndarray
    y: np.ndarray
    names: list[str] = field(default_factory=list)
    method: str = "qr"

    def __post_init__(self):
        X, y = np.asarray(self.X, float), np.asarray(self.y, float)
        if X.ndim == 1:
            X = X[:, None]
        self.X, self.y = X, y
        self.n, self.p = X.shape
        if not self.names:
            self.names = [f"x{i}" for i in range(self.p)]
        if self.method == "normal":
            XtX = X.T @ X
            self.beta = np.linalg.solve(XtX, X.T @ y)
            self.XtX_inv = np.linalg.inv(XtX)
            self.hat = np.einsum("ij,jk,ik->i", X, self.XtX_inv, X)
        else:
            Q, R = np.linalg.qr(X)
            self.beta = np.linalg.solve(R, Q.T @ y)
            Rinv = np.linalg.solve(R, np.eye(self.p))
            self.XtX_inv = Rinv @ Rinv.T
            self.hat = (Q ** 2).sum(axis=1)  # h_ii = diag(X(XᵀX)⁻¹Xᵀ) = row sums of Q²
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.df_resid = self.n - self.p
        self.sse = float(self.resid @ self.resid)
        self.sigma2 = self.sse / self.df_resid
        has_const = np.any(np.all(X == X[0], axis=0))
        sst = float(((y - y.mean()) ** 2).sum()) if has_const else float(y @ y)
        self.r2 = 1 - self.sse / sst
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - (1 if has_const else 0)) / self.df_resid
        self.cov = self.sigma2 * self.XtX_inv
        self.se = np.sqrt(np.diag(self.cov))
        self.t = self.beta / self.se
        self.pvalues = 2 * stats.t.sf(np.abs(self.t), self.df_resid)

    # ---- inference ------------------------------------------------------
    def tcrit(self, alpha: float = 0.05) -> float:
        return float(stats.t.ppf(1 - alpha / 2, self.df_resid))

    def conf_int(self, alpha: float = 0.05, robust: bool = False) -> np.ndarray:
        se = self.hc1_se() if robust else self.se
        c = self.tcrit(alpha)
        return np.column_stack([self.beta - c * se, self.beta + c * se])

    def hc1_cov(self) -> np.ndarray:
        """White/HC1 sandwich: n/(n−p) · (XᵀX)⁻¹ Xᵀ diag(e²) X (XᵀX)⁻¹."""
        meat = (self.X * self.resid[:, None] ** 2).T @ self.X
        return self.n / self.df_resid * self.XtX_inv @ meat @ self.XtX_inv

    def hc1_se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.hc1_cov()))

    def predict(self, x0, alpha: float = 0.05) -> dict:
        """Point prediction with a confidence interval (for the mean) and a prediction interval (for one new y).

        CI: ŷ ± t·s·√(x₀ᵀ(XᵀX)⁻¹x₀)      PI: ŷ ± t·s·√(1 + x₀ᵀ(XᵀX)⁻¹x₀)
        """
        x0 = np.atleast_2d(np.asarray(x0, float))
        yhat = x0 @ self.beta
        q = np.einsum("ij,jk,ik->i", x0, self.XtX_inv, x0)
        c, s = self.tcrit(alpha), np.sqrt(self.sigma2)
        ci, pi = c * s * np.sqrt(q), c * s * np.sqrt(1 + q)
        return {"fit": yhat, "ci_lo": yhat - ci, "ci_hi": yhat + ci, "pi_lo": yhat - pi, "pi_hi": yhat + pi}

    def table(self, robust: bool = True):
        import pandas as pd
        ci = self.conf_int()
        d = {"term": self.names, "coef": self.beta, "se": self.se, "t": self.t, "p": self.pvalues,
             "ci_lo": ci[:, 0], "ci_hi": ci[:, 1]}
        if robust:
            d["se_hc1"] = self.hc1_se()
        return pd.DataFrame(d)


def fit(cols: dict[str, np.ndarray], y, const: bool = True, method: str = "qr") -> OLS:
    X, names = design(cols, const)
    return OLS(X, np.asarray(y, float), names, method)
