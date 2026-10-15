"""Classical seasonal decomposition (centred moving-average trend + seasonal indices)."""
from __future__ import annotations

import numpy as np


def centred_ma(y: np.ndarray, m: int) -> np.ndarray:
    """Centred moving average of order m. Even m uses the 2 x m MA (end weights 1/2m)."""
    y = np.asarray(y, float)
    if m % 2:
        w = np.full(m, 1 / m)
    else:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    half = len(w) // 2
    out = np.full(len(y), np.nan)
    out[half: len(y) - half] = np.convolve(y, w, mode="valid")
    return out


def decompose(y, m: int, model: str = "multiplicative") -> dict:
    """y = T + S + R (additive) or y = T x S x R (multiplicative).

    Seasonal indices are the average detrended value at each position (t mod m), then
    normalised: additive indices sum to 0, multiplicative indices average 1.
    """
    y = np.asarray(y, float)
    trend = centred_ma(y, m)
    detr = y - trend if model == "additive" else y / trend
    pos = np.arange(len(y)) % m
    raw = np.array([np.nanmean(detr[pos == p]) for p in range(m)])
    idx = raw - raw.mean() if model == "additive" else raw / raw.mean()
    seasonal = idx[pos]
    resid = y - trend - seasonal if model == "additive" else y / (trend * seasonal)
    return {"observed": y, "trend": trend, "seasonal": seasonal, "resid": resid, "indices": idx,
            "model": model, "m": m}


def weekday_indices(daily, model: str = "multiplicative") -> dict:
    """Weekday indices from a daily series (pandas Series with a DatetimeIndex)."""
    dec = decompose(daily.to_numpy(), 7, model)
    first = daily.index[0].weekday()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {names[(first + p) % 7]: dec["indices"][p] for p in range(7)}
