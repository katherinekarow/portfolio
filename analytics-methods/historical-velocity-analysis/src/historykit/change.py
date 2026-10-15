"""Change detection (CUSUM) and z-score anomaly flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import history


def cusum(z, k: float = 0.5, h: float = 5.0) -> dict:
    """Two-sided tabular CUSUM on standardised values z (NaN = skipped day).

    S+_t = max(0, S+_{t-1} + z_t - k)     S-_t = max(0, S-_{t-1} - z_t - k)
    Alarm at the first t where S+ > h or S- > h.
    """
    z = np.asarray(z, float)
    sp, sn = np.zeros(len(z)), np.zeros(len(z))
    up = dn = 0.0
    alarm_up = alarm_dn = None
    for t, v in enumerate(z):
        if not np.isnan(v):
            up = max(0.0, up + v - k)
            dn = max(0.0, dn - v - k)
        sp[t], sn[t] = up, dn
        if alarm_up is None and up > h:
            alarm_up = t
        if alarm_dn is None and dn > h:
            alarm_dn = t
    return {"s_pos": sp, "s_neg": sn, "alarm_up": alarm_up, "alarm_down": alarm_dn}


def yoy_residual(daily: pd.Series, holidays=()) -> pd.Series:
    """Weekday-aligned log YoY ratio scaled by its Poisson noise: log(y_t / y_{t-364}) / sqrt(1/y_t + 1/y_{t-364}).

    Days where either side is a holiday are dropped (NaN).
    """
    ly = history.last_year(daily, "weekday")
    r = np.log(daily / ly) / np.sqrt(1 / daily + 1 / ly)
    hol = pd.DatetimeIndex(holidays)
    bad = daily.index.isin(hol) | (daily.index - pd.Timedelta(days=364)).isin(hol)
    return r.where(~bad)


def standardise(x: pd.Series, ref_start, ref_end) -> pd.Series:
    """z = (x - mean_ref) / sd_ref using an in-control reference window."""
    ref = x.loc[ref_start:ref_end].dropna()
    return (x - ref.mean()) / ref.std(ddof=1)


def detect_shift(daily: pd.Series, holidays, ref_days: int = 182, k: float = 0.5, h: float = 5.0) -> dict:
    r = yoy_residual(daily, holidays)
    first = r.first_valid_index()
    ref_end = first + pd.Timedelta(days=ref_days - 1)
    z = standardise(r, first, ref_end)
    monitor = z.loc[ref_end + pd.Timedelta(days=1):]
    res = cusum(monitor.to_numpy(), k, h)
    alarm = None if res["alarm_up"] is None else monitor.index[res["alarm_up"]]
    alarm_dn = None if res["alarm_down"] is None else monitor.index[res["alarm_down"]]
    return {"z": z, "monitor": monitor, "s_pos": pd.Series(res["s_pos"], monitor.index),
            "s_neg": pd.Series(res["s_neg"], monitor.index), "alarm_up": alarm, "alarm_down": alarm_dn,
            "ref": (first, ref_end), "k": k, "h": h}


def robust_z(x) -> np.ndarray:
    """(x - median) / (1.4826 x MAD): a z-score that a few outliers cannot inflate."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad)


def anomaly_flags(resid, threshold: float = 3.5) -> np.ndarray:
    return np.abs(robust_z(resid)) > threshold
