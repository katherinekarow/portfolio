"""Descriptive statistics and the four workhorse distributions, written out from the formulas."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def mean(x) -> float:
    x = np.asarray(x, float)
    return float(x.sum() / x.size)


def median(x) -> float:
    s = np.sort(np.asarray(x, float))
    n = s.size
    mid = n // 2
    return float(s[mid]) if n % 2 else float((s[mid - 1] + s[mid]) / 2)


def mode(x):
    """Most frequent value (smallest one on ties)."""
    c = Counter(np.asarray(x).tolist())
    top = max(c.values())
    return min(v for v, k in c.items() if k == top)


def variance(x, ddof: int = 1) -> float:
    """Sample variance s² = Σ(x − x̄)² / (n − 1)."""
    x = np.asarray(x, float)
    m = mean(x)
    return float(((x - m) ** 2).sum() / (x.size - ddof))


def std(x, ddof: int = 1) -> float:
    return math.sqrt(variance(x, ddof))


def skewness(x, adjusted: bool = True) -> float:
    """g1 = m3 / m2^1.5 (moments about the mean); adjusted G1 = g1·√(n(n−1))/(n−2)."""
    x = np.asarray(x, float)
    n = x.size
    d = x - mean(x)
    m2, m3 = (d ** 2).mean(), (d ** 3).mean()
    g1 = m3 / m2 ** 1.5
    return float(g1 * math.sqrt(n * (n - 1)) / (n - 2)) if adjusted else float(g1)


def percentile(x, q: float) -> float:
    """Linear-interpolation percentile (the numpy/Excel PERCENTILE.INC default): rank h = (n−1)·q/100."""
    s = np.sort(np.asarray(x, float))
    h = (s.size - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, s.size - 1)
    return float(s[lo] + (h - lo) * (s[hi] - s[lo]))


def summary(x) -> dict:
    return {"n": int(np.size(x)), "mean": mean(x), "median": median(x), "std": std(x),
            "skew": skewness(x), "p05": percentile(x, 5), "p50": percentile(x, 50), "p95": percentile(x, 95)}


# --- distributions ---------------------------------------------------------------------------------
def normal_pdf(x, mu=0.0, sigma=1.0):
    z = (np.asarray(x, float) - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x, mu=0.0, sigma=1.0):
    """Φ(z) = ½·(1 + erf(z/√2))."""
    z = (np.asarray(x, float) - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + np.vectorize(math.erf)(z))


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(K = k) = C(n, k)·pᵏ·(1 − p)ⁿ⁻ᵏ."""
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def poisson_pmf(k: int, lam: float) -> float:
    """P(K = k) = e^(−λ)·λᵏ / k!  (via logs so large k doesn't overflow)."""
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def expon_pdf(x, rate: float):
    """f(x) = λ·e^(−λx), x ≥ 0."""
    x = np.asarray(x, float)
    return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)


def clt_demo(rng: np.random.Generator, sizes=(1, 2, 5, 10, 30, 100), reps: int = 20000, scale: float = 12.0):
    """Means of n exponential clinic wait times: the skew of the mean falls like 2/√n and its sd like σ/√n."""
    rows = []
    for n in sizes:
        means = rng.exponential(scale, size=(reps, n)).mean(axis=1)
        rows.append({"n": n, "mean_of_means": float(means.mean()), "sd_of_means": std(means),
                     "sd_theory": scale / math.sqrt(n), "skew_of_means": skewness(means),
                     "skew_theory": 2 / math.sqrt(n),
                     "share_within_1.96se": float((np.abs(means - scale) <= 1.96 * scale / math.sqrt(n)).mean())})
    return rows
