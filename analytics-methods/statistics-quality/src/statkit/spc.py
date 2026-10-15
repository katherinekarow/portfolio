"""Statistical process control: X̄–R, I-MR, p-chart, Western Electric rules, CUSUM and EWMA."""
from __future__ import annotations

import math

import numpy as np

# Tabulated Shewhart constants by subgroup size n (ASTM / Montgomery tables).
CONSTANTS = {
    2: {"A2": 1.880, "D3": 0.0, "D4": 3.267, "d2": 1.128},
    3: {"A2": 1.023, "D3": 0.0, "D4": 2.574, "d2": 1.693},
    4: {"A2": 0.729, "D3": 0.0, "D4": 2.282, "d2": 2.059},
    5: {"A2": 0.577, "D3": 0.0, "D4": 2.114, "d2": 2.326},
    6: {"A2": 0.483, "D3": 0.0, "D4": 2.004, "d2": 2.534},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078},
}


def xbar_r(subgroups, baseline: int | None = None) -> dict:
    """X̄ chart: X̿ ± A2·R̄.  R chart: D3·R̄ … D4·R̄.  σ̂(within) = R̄/d2.
    Limits are set on the first `baseline` subgroups (phase I) and applied to all."""
    g = np.asarray(subgroups, float)
    n = g.shape[1]
    c = CONSTANTS[n]
    xbar, r = g.mean(1), g.max(1) - g.min(1)
    b = slice(0, baseline or len(g))
    xbb, rbar = xbar[b].mean(), r[b].mean()
    out = {"xbar": xbar, "range": r, "center": xbb, "rbar": rbar,
           "ucl_x": xbb + c["A2"] * rbar, "lcl_x": xbb - c["A2"] * rbar,
           "ucl_r": c["D4"] * rbar, "lcl_r": c["D3"] * rbar, "sigma_within": rbar / c["d2"], "n": n}
    out["flag_x"] = (xbar > out["ucl_x"]) | (xbar < out["lcl_x"])
    out["flag_r"] = (r > out["ucl_r"]) | (r < out["lcl_r"])
    return out


def imr(x, baseline: int | None = None) -> dict:
    """Individuals chart: X̄ ± 3·MR̄/d2 (= X̄ ± 2.66·MR̄), MR chart UCL = D4·MR̄ = 3.267·MR̄, with n = 2 constants."""
    x = np.asarray(x, float)
    b = slice(0, baseline or len(x))
    mr = np.abs(np.diff(x))
    mrbar = mr[: (baseline or len(x)) - 1].mean()
    c = CONSTANTS[2]
    centre = x[b].mean()
    sigma = mrbar / c["d2"]
    out = {"x": x, "mr": np.r_[np.nan, mr], "center": centre, "mrbar": mrbar, "sigma": sigma,
           "ucl": centre + 3 * sigma, "lcl": centre - 3 * sigma, "ucl_mr": c["D4"] * mrbar}
    out["flag"] = (x > out["ucl"]) | (x < out["lcl"])
    return out


def p_chart(defects, sizes, baseline: int | None = None) -> dict:
    """p̄ = Σdefects/Σinspected (phase I); limits p̄ ± 3·√(p̄(1−p̄)/nᵢ), clipped to [0, 1]."""
    d, n = np.asarray(defects, float), np.asarray(sizes, float)
    b = slice(0, baseline or len(d))
    pbar = d[b].sum() / n[b].sum()
    p = d / n
    se = np.sqrt(pbar * (1 - pbar) / n)
    ucl, lcl = np.minimum(pbar + 3 * se, 1), np.maximum(pbar - 3 * se, 0)
    return {"p": p, "pbar": pbar, "ucl": ucl, "lcl": lcl, "z": (p - pbar) / se, "flag": (p > ucl) | (p < lcl)}


def western_electric(x, center: float, sigma: float) -> list[tuple[int, int]]:
    """Western Electric zone rules. Returns (index, rule) at the point that completes each pattern.
    1: one point beyond 3σ.  2: 2 of 3 beyond 2σ, same side.  3: 4 of 5 beyond 1σ, same side.
    4: 8 in a row on the same side of the centre line."""
    z = (np.asarray(x, float) - center) / sigma
    hits = []
    for i in range(len(z)):
        if abs(z[i]) > 3:
            hits.append((i, 1))
        for side in (1, -1):
            s = side * z
            if i >= 2 and (s[i - 2:i + 1] > 2).sum() >= 2 and s[i] > 2:
                hits.append((i, 2))
            if i >= 4 and (s[i - 4:i + 1] > 1).sum() >= 4 and s[i] > 1:
                hits.append((i, 3))
            if i >= 7 and (s[i - 7:i + 1] > 0).all():
                hits.append((i, 4))
    return sorted(set(hits))


def cusum(x, target: float, sigma: float, k: float = 0.5, h: float = 5.0) -> dict:
    """Tabular CUSUM (in σ units):  C⁺ₜ = max(0, zₜ − k + C⁺ₜ₋₁),  C⁻ₜ = max(0, −zₜ − k + C⁻ₜ₋₁),
    signal when either exceeds h. k = 0.5 tunes it to a 1σ shift."""
    z = (np.asarray(x, float) - target) / sigma
    cp, cm = np.zeros(len(z)), np.zeros(len(z))
    a = b = 0.0
    for i, zi in enumerate(z):
        a = max(0.0, zi - k + a)
        b = max(0.0, -zi - k + b)
        cp[i], cm[i] = a, b
    sig = np.flatnonzero((cp > h) | (cm > h))
    return {"c_plus": cp, "c_minus": cm, "h": h, "k": k, "first_signal": int(sig[0]) if sig.size else None}


def cusum_run_lengths(rng, shift: float = 0.0, k: float = 0.5, h: float = 5.0, runs: int = 3000, max_len: int = 20000):
    """Simulated run length (samples until the first signal) of a two-sided CUSUM on N(shift, 1) data."""
    a = np.zeros(runs); b = np.zeros(runs)
    rl = np.zeros(runs, int); alive = np.ones(runs, bool)
    t = 0
    while alive.any() and t < max_len:
        t += 1
        z = rng.normal(shift, 1, alive.sum())
        a[alive] = np.maximum(0, z - k + a[alive])
        b[alive] = np.maximum(0, -z - k + b[alive])
        fired = alive.copy()
        fired[alive] = (a[alive] > h) | (b[alive] > h)
        rl[fired] = t
        alive &= ~fired
    rl[alive] = max_len
    return rl


def siegmund_arl(shift: float, k: float = 0.5, h: float = 5.0, two_sided: bool = True) -> float:
    """Siegmund's approximation: ARL = (e^(−2Δb) + 2Δb − 1)/(2Δ²), Δ = δ − k, b = h + 1.166.
    Two-sided: 1/ARL = 1/ARL⁺ + 1/ARL⁻."""
    def one(d):
        dd, bb = d - k, h + 1.166
        return bb * bb if abs(dd) < 1e-12 else (math.exp(-2 * dd * bb) + 2 * dd * bb - 1) / (2 * dd * dd)
    up = one(shift)
    return 1 / (1 / up + 1 / one(-shift)) if two_sided else up


def ewma(x, target: float, sigma: float, lam: float = 0.2, L: float = 3.0) -> dict:
    """zₜ = λxₜ + (1 − λ)zₜ₋₁, z₀ = target; limits target ± L·σ·√(λ/(2−λ)·(1 − (1−λ)^(2t)))."""
    x = np.asarray(x, float)
    z = np.empty(len(x)); prev = target
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(x) + 1)
    w = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    flag = (z > target + w) | (z < target - w)
    sig = np.flatnonzero(flag)
    return {"z": z, "ucl": target + w, "lcl": target - w, "flag": flag, "first_signal": int(sig[0]) if sig.size else None}
