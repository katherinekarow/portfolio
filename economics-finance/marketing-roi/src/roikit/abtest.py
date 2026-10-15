"""A/B test design and analysis for conversion rates.

Sample size per arm, two-sided two-proportion z-test (pooled variance under H0, unpooled under H1):
    n = [ z_(1-a/2) sqrt(2 pbar (1-pbar)) + z_(1-b) sqrt(p1(1-p1) + p2(1-p2)) ]^2 / (p2 - p1)^2
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def sample_size(p1, p2, alpha=0.05, power=0.80):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(math.ceil(n))


def power_at(p1, p2, n, alpha=0.05):
    """Power of the two-sided test with n per arm (same approximation as sample_size)."""
    za = norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0, se1 = math.sqrt(2 * pbar * (1 - pbar) / n), math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return float(norm.cdf((abs(p2 - p1) - za * se0) / se1))


def mde(p1, n, alpha=0.05, power=0.80):
    """Smallest absolute lift detectable with n per arm: solve power_at(p1, p1 + d, n) = power."""
    return float(brentq(lambda d: power_at(p1, p1 + d, n, alpha) - power, 1e-9, 1 - p1 - 1e-9))


def analyse(conv_a, n_a, conv_b, n_b, alpha=0.05):
    """Lift of B over A: Wald CI (unpooled) and pooled two-sided z-test p-value."""
    pa, pb = conv_a / n_a, conv_b / n_b
    diff = pb - pa
    se = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = norm.ppf(1 - alpha / 2)
    pool = (conv_a + conv_b) / (n_a + n_b)
    z_stat = diff / math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    return {"rate_a": pa, "rate_b": pb, "abs_lift": diff, "rel_lift": diff / pa,
            "ci_low": diff - z * se, "ci_high": diff + z * se, "z": z_stat, "p_value": float(2 * norm.sf(abs(z_stat)))}


def ci_coverage(p1, p2, n, sims, rng, alpha=0.05):
    """Share of simulated tests whose Wald CI contains the true difference."""
    a, b = rng.binomial(n, p1, sims) / n, rng.binomial(n, p2, sims) / n
    se = np.sqrt(a * (1 - a) / n + b * (1 - b) / n)
    z = norm.ppf(1 - alpha / 2)
    d = b - a
    return float(np.mean((d - z * se <= p2 - p1) & (p2 - p1 <= d + z * se)))


def peeking_fpr(looks, n_per_arm, p, sims, rng, alpha=0.05):
    """A/A tests (no true difference) checked at `looks` equally spaced points; stop at the first
    'significant' look. Returns the false-positive rate for naive peeking and for a Bonferroni
    threshold alpha / looks."""
    step = n_per_arm // looks
    a = np.cumsum(rng.binomial(step, p, (sims, looks)), axis=1)
    b = np.cumsum(rng.binomial(step, p, (sims, looks)), axis=1)
    n = step * np.arange(1, looks + 1)
    pool = (a + b) / (2 * n)
    z = np.abs(b - a) / n / np.sqrt(np.maximum(pool * (1 - pool) * 2 / n, 1e-15))
    naive = np.mean((z > norm.ppf(1 - alpha / 2)).any(axis=1))
    bonf = np.mean((z > norm.ppf(1 - alpha / (2 * looks))).any(axis=1))
    return float(naive), float(bonf)


def plan(tests, alpha, power) -> pd.DataFrame:
    rows = []
    for t in tests:
        p1 = t["baseline"]
        p2 = p1 * (1 + t["target_rel_lift"])
        n = sample_size(p1, p2, alpha, power)
        w = t["weekly_traffic_per_arm"]
        rows.append({"test": t["test"], "metric": t["metric"], "baseline_rate": p1,
                     "target_rel_lift": t["target_rel_lift"], "n_per_arm": n, "weekly_traffic_per_arm": w,
                     "weeks_needed": math.ceil(n / w), "mde_rel_at_4_weeks": round(mde(p1, 4 * w, alpha, power) / p1, 4),
                     "alpha": alpha, "power": power})
    return pd.DataFrame(rows)
