"""Forecast accuracy metrics. e_t = forecast_t - actual_t."""
from __future__ import annotations

import numpy as np


def _arr(a, f):
    return np.asarray(a, float), np.asarray(f, float)


def mape(actual, forecast) -> float:
    """mean(|e_t| / |y_t|) over periods with y_t != 0."""
    y, f = _arr(actual, forecast)
    nz = y != 0
    return float(np.mean(np.abs(f[nz] - y[nz]) / np.abs(y[nz])))


def wape(actual, forecast) -> float:
    """sum|e_t| / sum|y_t|: absolute error as a share of total volume."""
    y, f = _arr(actual, forecast)
    return float(np.abs(f - y).sum() / np.abs(y).sum())


def bias(actual, forecast) -> float:
    """sum(e_t) / sum(y_t): positive = over-forecast."""
    y, f = _arr(actual, forecast)
    return float((f - y).sum() / y.sum())


def mase_scale(train, m: int = 1) -> float:
    """In-sample MAE of the seasonal naive forecast: mean|y_t - y_{t-m}|."""
    y = np.asarray(train, float)
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def mase(actual, forecast, train, m: int = 1) -> float:
    """mean|e_t| / mase_scale(train, m). Below 1 = better than seasonal naive in-sample."""
    y, f = _arr(actual, forecast)
    return float(np.mean(np.abs(f - y)) / mase_scale(train, m))
