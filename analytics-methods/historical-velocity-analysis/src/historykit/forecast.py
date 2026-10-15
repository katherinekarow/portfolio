"""Forecasting baselines written out as recursions, plus a rolling-origin backtest.

Holt-Winters (additive seasonality), one step ahead:
    forecast  f_t = l_{t-1} + b_{t-1} + s_{t-m}
    level     l_t = a (y_t - s_{t-m}) + (1 - a)(l_{t-1} + b_{t-1})
    trend     b_t = b (l_t - l_{t-1}) + (1 - b) b_{t-1}
    season    s_t = g (y_t - l_t) + (1 - g) s_{t-m}
Multiplicative seasonality replaces "- s" by "/ s" and "+ s" by "x s".
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import metrics


def naive(y, h: int) -> np.ndarray:
    return np.full(h, float(np.asarray(y)[-1]))


def seasonal_naive(y, h: int, m: int) -> np.ndarray:
    y = np.asarray(y, float)
    return np.array([y[len(y) - m + (i % m)] for i in range(h)])


def ses(y, alpha, h: int = 1):
    """Simple exponential smoothing: l_t = a y_t + (1 - a) l_{t-1}, l_0 = y_0.

    `alpha` may be an array (one smoothing run per value). Returns (fitted, forecast).
    """
    y = np.asarray(y, float)
    a = np.atleast_1d(np.asarray(alpha, float))
    level = np.full(a.shape, y[0])
    fitted = np.empty((len(y),) + a.shape)
    for t, yt in enumerate(y):
        fitted[t] = level
        level = a * yt + (1 - a) * level
    fc = np.repeat(level[None, :], h, axis=0)
    if np.ndim(alpha) == 0:
        return fitted[:, 0], fc[:, 0]
    return fitted, fc


def hw_init(y, m: int, seasonal: str = "additive"):
    """Initial states from the first two seasons.

    trend b0 = (mean of season 2 - mean of season 1) / m. The mean of season 1 sits at the middle of
    that season, t = (m - 1) / 2, so the straight line through it is s1 + b0 (t - (m - 1) / 2).
    Season = each value of season 1 relative to that line (difference or ratio). The recursion
    starts at t = 0, so the starting level is the line at t = -1. A clean linear trend plus a fixed
    pattern is then reproduced exactly.
    """
    y = np.asarray(y, float)
    s1, s2 = y[:m].mean(), y[m: 2 * m].mean()
    trend = (s2 - s1) / m
    line = s1 + trend * (np.arange(m) - (m - 1) / 2)
    season = y[:m] - line if seasonal == "additive" else y[:m] / line
    level = s1 - trend * (m + 1) / 2
    return level, trend, season


def holt_winters(y, m: int, alpha, beta, gamma, h: int = 1, seasonal: str = "additive", init=None):
    """Run the Holt-Winters recursions. Parameters may be equal-length arrays (vectorised grid).

    Returns dict with fitted one-step forecasts, final states and the h-step forecast.
    """
    y = np.asarray(y, float)
    a, b, g = (np.atleast_1d(np.asarray(v, float)) for v in (alpha, beta, gamma))
    l0, b0, s0 = init if init is not None else hw_init(y, m, seasonal)
    k = len(a)
    level = np.full(k, float(l0))
    trend = np.full(k, float(b0))
    season = np.tile(np.asarray(s0, float), (k, 1))       # season[:, p] for position p = t mod m
    add = seasonal == "additive"
    fitted = np.empty((len(y), k))
    for t, yt in enumerate(y):
        p = t % m
        s_old = season[:, p]
        fitted[t] = level + trend + s_old if add else (level + trend) * s_old
        new_level = a * (yt - s_old if add else yt / s_old) + (1 - a) * (level + trend)
        trend = b * (new_level - level) + (1 - b) * trend
        level = new_level
        season[:, p] = g * (yt - level if add else yt / level) + (1 - g) * s_old
    n = len(y)
    steps = np.arange(1, h + 1)
    pos = (n + steps - 1) % m
    if add:
        fc = level[None, :] + steps[:, None] * trend[None, :] + season[:, pos].T
    else:
        fc = (level[None, :] + steps[:, None] * trend[None, :]) * season[:, pos].T
    out = {"fitted": fitted, "level": level, "trend": trend, "season": season, "forecast": fc}
    if np.ndim(alpha) == 0:
        out = {"fitted": fitted[:, 0], "level": level[0], "trend": trend[0], "season": season[0],
               "forecast": fc[:, 0]}
    return out


ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.5)
BETAS = (0.0, 0.01, 0.05)
GAMMAS = (0.05, 0.1, 0.2, 0.3)


def fit_ses(y, h: int, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    fitted, fc = ses(y, np.array(grid), h)
    sse = ((fitted[1:] - np.asarray(y, float)[1:, None]) ** 2).sum(0)
    i = int(np.argmin(sse))
    return fc[:, i], {"alpha": grid[i]}


def fit_holt_winters(y, m: int, h: int, seasonal: str = "additive"):
    """Choose (alpha, beta, gamma) on a grid by in-sample one-step SSE (after the first season)."""
    grid = np.array(list(itertools.product(ALPHAS, BETAS, GAMMAS)))
    res = holt_winters(y, m, grid[:, 0], grid[:, 1], grid[:, 2], h, seasonal)
    y = np.asarray(y, float)
    sse = ((res["fitted"][m:] - y[m:, None]) ** 2).sum(0)
    i = int(np.argmin(sse))
    return res["forecast"][:, i], dict(zip(("alpha", "beta", "gamma"), grid[i]))


METHODS = ("naive", "seasonal naive (7)", "same weekday last year", "SES", "Holt-Winters add.", "Holt-Winters mult.")


def forecast(method: str, y, h: int) -> np.ndarray:
    if method == "naive":
        return naive(y, h)
    if method == "seasonal naive (7)":
        return seasonal_naive(y, h, 7)
    if method == "same weekday last year":
        y = np.asarray(y, float)
        return y[len(y) - 364: len(y) - 364 + h]
    if method == "SES":
        return fit_ses(y, h)[0]
    if method == "Holt-Winters add.":
        return fit_holt_winters(y, 7, h, "additive")[0]
    if method == "Holt-Winters mult.":
        return fit_holt_winters(y, 7, h, "multiplicative")[0]
    raise ValueError(method)


def rolling_origin(series: pd.Series, n_origins: int = 13, step: int = 28, h: int = 14,
                   train_days: int = 728, methods=METHODS) -> pd.DataFrame:
    """Forecast h days from each origin; the origin moves forward `step` days each time.

    Training uses the `train_days` before each origin only, so no future data leaks in.
    """
    y = series.to_numpy(float)
    n = len(y)
    first = n - h - step * (n_origins - 1)
    rows = []
    for o in range(n_origins):
        origin = first + step * o
        train = y[max(0, origin - train_days): origin]
        actual = y[origin: origin + h]
        for method in methods:
            fc = forecast(method, train, h)
            for i in range(h):
                rows.append({"origin": series.index[origin], "date": series.index[origin + i], "step": i + 1,
                             "method": method, "actual": actual[i], "forecast": fc[i],
                             "scale": metrics.mase_scale(train, 7)})
    return pd.DataFrame(rows)


def accuracy(bt: pd.DataFrame) -> pd.DataFrame:
    """MAPE, WAPE, bias and MASE per method, pooled over every origin and step."""
    out = []
    for method, g in bt.groupby("method", sort=False):
        out.append({"method": method,
                    "MAPE_%": 100 * metrics.mape(g.actual, g.forecast),
                    "WAPE_%": 100 * metrics.wape(g.actual, g.forecast),
                    "bias_%": 100 * metrics.bias(g.actual, g.forecast),
                    "MASE": float(np.mean(np.abs(g.actual - g.forecast) / g.scale))})
    return pd.DataFrame(out)
