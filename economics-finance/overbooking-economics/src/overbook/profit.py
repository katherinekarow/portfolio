"""Expected session economics for a booking limit, and three ways to find the best limit.

A session has C slots. We book n patients; S of them show. Each patient seen earns r; each patient who
shows when all C slots are full is "bumped" (rescheduled and compensated) at cost b:

    profit(n) = r * E[min(S, C)] - b * E[(S - C)+]           computed exactly from the pmf of S

Marginal rule (binomial). Booking patient n+1 adds a Bernoulli(p) show X independent of S_n:
    if X = 1 and S_n < C  -> one more patient seen:  +r
    if X = 1 and S_n >= C -> one more bump:          -b
so  Delta(n) = profit(n+1) - profit(n) = p * [ r - (r + b) * P(S_n >= C) ].
P(S_n >= C) rises with n, so Delta falls (profit is concave) and the optimum is the first n where
    P(S_n >= C) >= r / (r + b)                               (the newsvendor critical ratio)
Note p multiplies both terms, so it drops out of the stopping rule: a lower show rate changes the
optimum only through the distribution of S_n. For the beta-binomial the next patient's show is
correlated with S_n (P(X = 1 | S_n = s) = (a + s) / (a + b + n)), and the rule becomes
    Delta(n) = sum_s P(S_n = s) * (a + s)/(a + b + n) * [ r * 1(s < C) - b * 1(s >= C) ].

Normal approximation: S_n ~ N(n p, n p (1 - p)), with a continuity correction,
    P(S_n >= C) = r/(r+b)  <=>  n p + z sqrt(n p (1-p)) = C - 1/2,   z = Phi^-1(b / (r + b))
which is a quadratic in sqrt(n); the limit is the ceiling of its root.

Demand-constrained: requests D ~ Poisson(lam), so bookings are N = min(D, L) and
    E[profit | L] = sum_{n<L} P(D=n) profit(n) + P(D >= L) profit(L),
    E[profit | L+1] - E[profit | L] = P(D >= L+1) * Delta(L).
Demand scales the marginal value of a higher limit but never flips its sign, so the best limit is the
same as without the cap, except that once P(D >= L+1) is negligible a higher limit buys nothing; we
report the smallest limit within a billionth of the maximum.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize as so
from scipy import stats

from .showup import betabin_params, poisson_pmf, showup_pmf

REL_TOL = 1e-9


def default_lmax(C: int, p: float) -> int:
    """Largest limit we evaluate: comfortably past any sensible optimum."""
    return int(math.ceil(2 * C / max(p, 0.05))) + 10


def session_table(C: int, p: float, r: float, b: float, rho: float = 0.0, lmax: int | None = None) -> dict:
    """Exact per-session outcomes for n = 0..lmax booked patients."""
    lmax = default_lmax(C, p) if lmax is None else lmax
    out = {k: np.zeros(lmax + 1) for k in ("served", "bumps", "empty", "shows", "pbump", "profit")}
    for n in range(lmax + 1):
        f = showup_pmf(n, p, rho)
        s = np.arange(n + 1)
        out["served"][n] = f @ np.minimum(s, C)
        out["bumps"][n] = f @ np.maximum(s - C, 0)
        out["empty"][n] = f @ np.maximum(C - s, 0)
        out["shows"][n] = f @ s
        out["pbump"][n] = f[s > C].sum()
    out["profit"] = r * out["served"] - b * out["bumps"]
    return out


def profit_curve(C: int, p: float, r: float, b: float, rho: float = 0.0, lam: float | None = None,
                 lmax: int | None = None) -> pd.DataFrame:
    """Expected session outcomes for every booking limit L = 0..lmax (optionally demand-capped)."""
    t = session_table(C, p, r, b, rho, lmax)
    n = len(t["profit"])
    keys = ("served", "bumps", "empty", "shows", "pbump", "profit")
    if lam is None or lam <= 0:
        cols = {k: t[k] for k in keys}
        cols["booked"] = np.arange(n, dtype=float)
    else:
        pois = poisson_pmf(lam, n - 1)
        cdf_before = np.concatenate([[0.0], np.cumsum(pois)[:-1]])          # P(D < L)
        sf = np.maximum(0.0, 1.0 - cdf_before)                              # P(D >= L)
        cols = {}
        for k in keys + ("booked",):
            x = np.arange(n, dtype=float) if k == "booked" else t[k]
            before = np.concatenate([[0.0], np.cumsum(pois * x)[:-1]])
            cols[k] = before + sf * x
    return pd.DataFrame({"limit": np.arange(n), **cols})


def _argmax_smallest(profit: np.ndarray, scale: float) -> int:
    best = profit.max()
    return int(np.flatnonzero(profit >= best - REL_TOL * scale)[0])


def optimal_limit(C: int, p: float, r: float, b: float, rho: float = 0.0, lam: float | None = None,
                  lmax: int | None = None) -> dict:
    """Exact enumeration: the smallest limit whose expected profit is (within 1e-9 relative) the maximum."""
    cur = profit_curve(C, p, r, b, rho, lam, lmax)
    L = _argmax_smallest(cur.profit.to_numpy(), r * max(C, 1))
    row = cur.iloc[L]
    return {"limit": L, "profit": float(row["profit"]), "bumps": float(row["bumps"]),
            "empty": float(row["empty"]), "pbump": float(row["pbump"]), "at_edge": L == len(cur) - 1}


def marginal_rule_limit(C: int, p: float, r: float, b: float, rho: float = 0.0, lmax: int | None = None) -> int:
    """Book one more while it adds expected profit (see module docstring for the derivation)."""
    lmax = default_lmax(C, p) if lmax is None else lmax
    cr = r / (r + b)
    for n in range(C, lmax):
        if rho <= 0:
            if stats.binom.sf(C - 1, n, p) >= cr:          # P(S_n >= C) reached the critical ratio
                return n
        else:
            a, bb = betabin_params(p, rho)
            f = showup_pmf(n, p, rho)
            s = np.arange(n + 1)
            q = (a + s) / (a + bb + n)
            if f @ (q * np.where(s < C, r, -b)) <= 0:
                return n
    return lmax


def normal_limit(C: int, p: float, r: float, b: float, rho: float = 0.0) -> int:
    """Normal approximation of the critical-ratio rule (closed form for the binomial)."""
    z = stats.norm.ppf(b / (r + b))
    if rho <= 0:
        sg = math.sqrt(p * (1 - p))
        x = (-z * sg + math.sqrt(z * z * sg * sg + 4 * p * (C - 0.5))) / (2 * p)
        return max(C, math.ceil(x * x - 1e-12))

    def g(n):          # P(S_n >= C) - critical ratio, variance inflated by the design effect
        sd = math.sqrt(n * p * (1 - p) * (1 + (n - 1) * rho))
        return stats.norm.sf((C - 0.5 - n * p) / sd) - r / (r + b)
    lo, hi = float(C), float(default_lmax(C, p))
    if g(lo) >= 0:
        return C
    return max(C, math.ceil(so.brentq(g, lo, hi)))


def service_limit(C: int, p: float, rho: float = 0.0, alpha: float = 0.05, lmax: int | None = None) -> int:
    """Largest limit keeping P(at least one bump) <= alpha."""
    lmax = default_lmax(C, p) if lmax is None else lmax
    L = C
    for n in range(C, lmax + 1):
        f = showup_pmf(n, p, rho)
        if f[C + 1:].sum() > alpha:
            break
        L = n
    return L
