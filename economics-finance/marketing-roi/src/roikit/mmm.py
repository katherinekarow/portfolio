"""Media-mix response curves: geometric adstock + Hill saturation, synthetic history and fitting.

weekly revenue_t = baseline_t + sum_c beta_c * hill(adstock_c,t; K_c, s_c) + noise
adstock_t        = spend_t + lam * adstock_{t-1}
hill(a; K, s)    = a^s / (a^s + K^s)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PARAMS = ("lam", "K", "s", "beta")


def adstock(x, lam):
    """Geometric carry-over along the last axis; lam is a scalar or has shape x.shape[:-1]."""
    x = np.asarray(x, dtype=float)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), x.shape[:-1])
    out = np.empty_like(x)
    carry = np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        carry = x[..., t] + lam * carry
        out[..., t] = carry
    return out


def hill(a, K, s):
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    return a ** s / (a ** s + K ** s)


def steady_response(weekly_spend, p):
    """Weekly revenue once adstock has settled at a constant weekly spend: a = x / (1 - lam)."""
    return p["beta"] * hill(np.asarray(weekly_spend) / (1 - p["lam"]), p["K"], p["s"])


def make_history(cfg: dict, rng: np.random.Generator):
    """Synthetic weekly history from the true parameters in cfg. Returns (DataFrame, truth dict)."""
    h, ch = cfg["history"], cfg["channels"]
    n = h["weeks"]
    media = cfg["budget"]["total"] - sum(v["amount"] for v in cfg["budget"]["fixed"].values())
    weekly = media / cfg["budget"]["weeks"]
    df = pd.DataFrame({"week": np.arange(n)})
    season = 1 + h["season_amp"] * np.sin(2 * np.pi * df.week / 52)
    rev = h["baseline"] * season
    truth = {}
    for c, v in ch.items():
        on = np.repeat(rng.random(n // 4 + 1) < 0.7, 4)[:n]          # flighting: 4-week bursts and dark spells
        spend = on * rng.uniform(0, h["spend_scale"] * weekly * v["max_share"], n)
        df[f"spend_{c}"] = spend.round(0)
        truth[c] = {k: v[k] for k in PARAMS}
        rev = rev + v["beta"] * hill(adstock(df[f"spend_{c}"].to_numpy(), v["lam"]), v["K"], v["s"])
    df["revenue"] = rev + rng.normal(0, h["noise_sd"], n)
    return df, truth


# ---------------------------------------------------------------- fitting
# Fitted vector per channel: lam, log K, s, log beta. Baseline: intercept, sin, cos.
_LO = np.array([0.0, np.log(50), 0.5, np.log(50)])
_HI = np.array([0.95, np.log(1e5), 3.5, np.log(1e5)])


def _unpack(theta, channels):
    p = {}
    for i, c in enumerate(channels):
        lam, lk, s, lb = theta[4 * i: 4 * i + 4]
        p[c] = {"lam": lam, "K": np.exp(lk), "s": s, "beta": np.exp(lb)}
    return p, theta[4 * len(channels):]


def predict(theta, df, channels):
    p, base = _unpack(theta, channels)
    w = df.week.to_numpy()
    y = base[0] + base[1] * np.sin(2 * np.pi * w / 52) + base[2] * np.cos(2 * np.pi * w / 52)
    for c in channels:
        y = y + p[c]["beta"] * hill(adstock(df[f"spend_{c}"].to_numpy(), p[c]["lam"]), p[c]["K"], p[c]["s"])
    return y


def fit(df: pd.DataFrame, channels, rng: np.random.Generator, starts: int = 8):
    """Nonlinear least squares with bounds and random restarts (Hill curves have local minima).

    Returns params dict, covariance of the transformed vector, residual sd, and the raw theta.
    """
    y = df.revenue.to_numpy()
    k = len(channels)
    lo = np.r_[np.tile(_LO, k), -1e6, -1e6, -1e6]
    hi = np.r_[np.tile(_HI, k), 1e6, 1e6, 1e6]
    best = None
    for _ in range(starts):
        x0 = np.r_[np.concatenate([[rng.uniform(0.1, 0.8), np.log(rng.uniform(500, 8000)),
                                    rng.uniform(0.8, 2.0), np.log(rng.uniform(1000, 8000))] for _ in channels]),
                   y.mean() * 0.8, 0.0, 0.0]
        r = least_squares(lambda t: predict(t, df, channels) - y, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    dof = len(y) - len(best.x)
    sigma = float(np.sqrt(2 * best.cost / dof))
    J = best.jac
    cov = sigma ** 2 * np.linalg.pinv(J.T @ J)
    params, base = _unpack(best.x, channels)
    return {"params": params, "theta": best.x, "cov": cov, "sigma": sigma, "base": base}


def draw_params(fitres, channels, n, rng):
    """Draw n parameter sets from the fit's asymptotic normal (in transformed space), clipped to bounds."""
    k = len(channels)
    th = rng.multivariate_normal(fitres["theta"], fitres["cov"], size=n, method="eigh")[:, : 4 * k]
    th = np.clip(th, np.tile(_LO, k), np.tile(_HI, k))
    out = {}
    for i, c in enumerate(channels):
        out[c] = {"lam": th[:, 4 * i], "K": np.exp(th[:, 4 * i + 1]), "s": th[:, 4 * i + 2], "beta": np.exp(th[:, 4 * i + 3])}
    return out


def param_table(fitres, truth, channels) -> pd.DataFrame:
    k = len(channels)
    se = np.sqrt(np.clip(np.diag(fitres["cov"])[: 4 * k], 0, None))
    rows = []
    for i, c in enumerate(channels):
        f, t = fitres["params"][c], truth[c]
        for j, name in enumerate(PARAMS):
            val = f[name]
            s = se[4 * i + j] * (val if name in ("K", "beta") else 1)      # delta method for the log params
            rows.append({"channel": c, "param": name, "true": t[name], "fitted": round(float(val), 4),
                         "se": round(float(s), 4), "abs_pct_error": round(abs(val - t[name]) / t[name] * 100, 1)})
    return pd.DataFrame(rows)
