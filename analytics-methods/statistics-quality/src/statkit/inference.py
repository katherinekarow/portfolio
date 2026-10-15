"""Confidence intervals and tests. Statistics are computed here; scipy supplies only the t / χ² / normal tables."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st

from .descriptive import mean, variance


def t_ci_mean(x, conf: float = 0.95):
    """x̄ ± t(1−α/2, n−1)·s/√n."""
    x = np.asarray(x, float)
    n = x.size
    se = math.sqrt(variance(x) / n)
    t = _st.t.ppf(0.5 + conf / 2, n - 1)
    m = mean(x)
    return m - t * se, m + t * se


def wilson_ci(k: int, n: int, conf: float = 0.95):
    """Wilson score interval: (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)."""
    z = _st.norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (centre - half) / d, (centre + half) / d


def welch_t(x, y):
    """Welch two-sample t: t = (x̄ − ȳ)/√(s²ₓ/nₓ + s²ᵧ/nᵧ), Welch–Satterthwaite df, two-sided p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = variance(x) / x.size, variance(y) / y.size
    t = (mean(x) - mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1))
    p = 2 * _st.t.sf(abs(t), df)
    return {"t": t, "df": df, "p": float(p), "diff": mean(x) - mean(y)}


def chi2_independence(table):
    """χ² = Σ (O − E)² / E with E = row total × column total / grand total; df = (r−1)(c−1)."""
    o = np.asarray(table, float)
    e = o.sum(1, keepdims=True) * o.sum(0, keepdims=True) / o.sum()
    chi2 = float(((o - e) ** 2 / e).sum())
    dof = (o.shape[0] - 1) * (o.shape[1] - 1)
    return {"chi2": chi2, "dof": dof, "p": float(_st.chi2.sf(chi2, dof)), "expected": e}


def bootstrap_ci(x, stat=np.mean, B: int = 4000, conf: float = 0.95, rng=None):
    """Percentile bootstrap: resample with replacement B times, take the α/2 and 1−α/2 quantiles of the statistic."""
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(B, x.size))
    boot = stat(x[idx], axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def power_two_sample(delta: float, sigma: float, n: int, alpha: float = 0.05) -> float:
    """Normal-approximation power of a two-sided two-sample test with n per group:
    power = Φ(δ/(σ√(2/n)) − z(1−α/2)) + Φ(−δ/(σ√(2/n)) − z(1−α/2))."""
    z = _st.norm.ppf(1 - alpha / 2)
    ncp = delta / (sigma * math.sqrt(2 / n))
    return float(_st.norm.cdf(ncp - z) + _st.norm.cdf(-ncp - z))


def n_for_power(delta: float, sigma: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """n per group = 2·(z(1−α/2) + z(power))²·σ²/δ²."""
    z = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return math.ceil(2 * z * z * sigma ** 2 / delta ** 2)


def simulated_error_rates(rng, delta, sigma, n, alpha=0.05, reps=4000):
    """Type I rate (no true difference) and power (true difference δ) of the Welch test, by simulation."""
    def reject(shift):
        x = rng.normal(0, sigma, (reps, n))
        y = rng.normal(shift, sigma, (reps, n))
        vx, vy = x.var(1, ddof=1) / n, y.var(1, ddof=1) / n
        t = (y.mean(1) - x.mean(1)) / np.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (n - 1) + vy ** 2 / (n - 1))
        return float((2 * _st.t.sf(np.abs(t), df) < alpha).mean())
    type_i, power = reject(0.0), reject(delta)
    return {"type_I": type_i, "power": power, "type_II": 1 - power}
