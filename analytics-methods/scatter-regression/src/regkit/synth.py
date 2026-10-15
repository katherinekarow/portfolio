"""Synthetic farm co-op store data with known (planted) relationships.

Every generator is seeded, so the same call always returns the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- truth for the store-week panel -------------------------------------
FORMATS = ["kiosk", "market", "standard", "superstore"]
FORMAT_SIZE = {"kiosk": 0.4, "market": 0.8, "standard": 1.5, "superstore": 3.0}  # x median floor area
TRUE = {
    "intercept": 7.5,
    "ln_size": 1.0,         # bigger stores sell more
    "ln_price": -1.4,       # price elasticity (the number we want)
    "promo": 0.0004,        # +0.04% units per $ of promo spend, i.e. +4% per $100
    "temp_sq": -0.004,      # units peak at 22 degC and fall off either side
    "noise_sd": 0.12,       # multiplicative noise on units (log scale)
    "price_size": 0.5,      # bigger stores charge more: the confounder
    "temp_peak": 22.0,
}


def make_stores(n_stores: int = 60, n_weeks: int = 52, seed: int = 7) -> pd.DataFrame:
    """Weekly panel: one row per store and week.

    Planted structure:
      ln units = 7.5 + 1.0 ln size − 1.4 ln price + 0.0004 promo − 0.004 (temp − 22)² + N(0, 0.12)
      ln price = ln 3 + 0.5 ln size + store and week noise   (size drives both → confounder)
      ln traffic = ln 2000 + 0.95 ln size + small noise       (traffic ≈ size → collinear pair)
    """
    rng = np.random.default_rng(seed)
    fmt = np.array(FORMATS)[np.arange(n_stores) % 4]
    size = np.array([FORMAT_SIZE[f] for f in fmt])  # floor area is set by the format
    store_price = rng.normal(0, 0.03, n_stores)
    store_temp = rng.normal(0, 1.5, n_stores)

    s = np.repeat(np.arange(n_stores), n_weeks)
    w = np.tile(np.arange(1, n_weeks + 1), n_stores)
    ln_size = np.log(size[s])
    ln_price = np.log(3.0) + TRUE["price_size"] * ln_size + store_price[s] + rng.normal(0, 0.07, s.size)
    promo = rng.uniform(0, 600, s.size).round(0)
    temp = 20 + 10 * np.sin(2 * np.pi * (w - 13) / 52) + store_temp[s] + rng.normal(0, 3, s.size)
    ln_traffic = np.log(2000) + 0.95 * ln_size + rng.normal(0, 0.05, s.size)
    ln_units = (TRUE["intercept"] + TRUE["ln_size"] * ln_size + TRUE["ln_price"] * ln_price
                + TRUE["promo"] * promo + TRUE["temp_sq"] * (temp - TRUE["temp_peak"]) ** 2
                + rng.normal(0, TRUE["noise_sd"], s.size))
    return pd.DataFrame({
        "store": s + 1, "week": w, "format": fmt[s], "size_index": size[s].round(4),
        "price": np.exp(ln_price).round(3), "promo_spend": promo, "temperature": temp.round(2),
        "foot_traffic": np.exp(ln_traffic).round(0), "units": np.exp(ln_units).round(0).clip(1),
    })


# ---- promo cross-section: linear truth, optional heteroscedasticity and outliers ----
PROMO_TRUE = {"intercept": 120.0, "slope": 0.35}
OUTLIER_IDX = (57, 58, 59)


def make_promo(n: int = 60, seed: int = 11, hetero: bool = True, outliers: bool = True) -> pd.DataFrame:
    """One week, one row per store: units = 120 + 0.35·promo + noise.

    hetero=True:   noise sd = 2 + 0.12·promo (a funnel); otherwise sd = 25.
    outliers=True: the three highest-promo stores have units mis-keyed (logged in cases of 10),
                   giving three high-leverage, badly fitting points.
    """
    rng = np.random.default_rng(seed)
    promo = np.sort(rng.uniform(20, 800, n)).round(0)
    sd = 2 + 0.12 * promo if hetero else np.full(n, 25.0)
    units = PROMO_TRUE["intercept"] + PROMO_TRUE["slope"] * promo + rng.normal(0, 1, n) * sd
    flag = np.zeros(n, bool)
    if outliers:
        idx = list(OUTLIER_IDX)
        units[idx] = units[idx] / 10
        flag[idx] = True
    return pd.DataFrame({"store": np.arange(1, n + 1), "promo_spend": promo,
                         "units": units.round(1), "planted_outlier": flag})


# ---- Anscombe's quartet (Anscombe 1973, published values) ----------------
_AX = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (_AX, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (_AX, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (_AX, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}


def anscombe() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {k: (np.array(x, float), np.array(y, float)) for k, (x, y) in ANSCOMBE.items()}


def make_simple(n: int = 30, seed: int = 0, beta=(2.0, 0.5), sd: float = 1.0):
    """Clean simple-regression data for coverage simulations: y = 2 + 0.5x + N(0, sd), x ~ U(0, 10)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, beta[0] + beta[1] * x + rng.normal(0, sd, n)
