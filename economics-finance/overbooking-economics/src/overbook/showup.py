"""Show-up models: how many of n booked patients actually arrive.

Binomial: every patient shows independently with probability p.
Beta-binomial: the session's show rate itself varies (weather, a flu week, a late bus), drawn from
Beta(a, b) with mean p and intra-session correlation rho = 1 / (a + b + 1). Same mean as the binomial,
variance n p (1 - p) (1 + (n - 1) rho). rho -> 0 gives back the binomial.

Both pmfs are built from the ratio P(k+1) / P(k) in log space, then normalised. The JavaScript page
uses the identical recursion, so the two agree to floating-point rounding.
"""
from __future__ import annotations

import numpy as np

P_EPS = 1e-9


def betabin_params(p: float, rho: float) -> tuple[float, float]:
    """Beta(a, b) with mean p and intra-session correlation rho."""
    if not 0 < rho < 1:
        raise ValueError("rho must be in (0, 1)")
    k = (1 - rho) / rho
    return p * k, (1 - p) * k


def showup_pmf(n: int, p: float, rho: float = 0.0) -> np.ndarray:
    """P(S = k) for k = 0..n, S = patients who show out of n booked."""
    n = int(n)
    if n <= 0:
        return np.array([1.0])
    p = min(max(float(p), P_EPS), 1 - P_EPS)
    k = np.arange(n, dtype=float)                       # step k -> k+1
    if rho <= 0:
        step = np.log((n - k) / (k + 1)) + np.log(p / (1 - p))
    else:
        a, b = betabin_params(p, rho)
        step = np.log((n - k) / (k + 1)) + np.log((k + a) / (n - k - 1 + b))
    logp = np.concatenate([[0.0], np.cumsum(step)])
    w = np.exp(logp - logp.max())
    return w / w.sum()


def poisson_pmf(lam: float, nmax: int) -> np.ndarray:
    """P(D = n) for n = 0..nmax, D ~ Poisson(lam); log-space so large lam doesn't underflow."""
    n = np.arange(nmax + 1, dtype=float)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, nmax + 1, dtype=float)))])
    return np.exp(-lam + n * np.log(lam) - logfact)


def sample_shows(rng: np.random.Generator, n: np.ndarray, p: float, rho: float = 0.0) -> np.ndarray:
    """Draw show-ups for an array of booked counts (session rate ~ Beta first when rho > 0)."""
    n = np.asarray(n)
    if rho <= 0:
        return rng.binomial(n, p)
    a, b = betabin_params(p, rho)
    return rng.binomial(n, rng.beta(a, b, size=n.shape))
