"""Monte Carlo: simulate a year of sessions patient by patient and compare with the exact formula."""
from __future__ import annotations

import numpy as np

from .showup import sample_shows


def simulate(L: int, C: int, p: float, r: float, b: float, rho: float = 0.0, lam: float | None = None,
             sessions: int = 500, seed: int = 0) -> dict:
    """Profit per session over `sessions` simulated sessions; mean and 95% interval."""
    rng = np.random.default_rng(seed)
    booked = np.full(sessions, L) if not lam else np.minimum(rng.poisson(lam, sessions), L)
    shows = sample_shows(rng, booked, p, rho)
    seen, bumped = np.minimum(shows, C), np.maximum(shows - C, 0)
    prof = r * seen - b * bumped
    se = prof.std(ddof=1) / np.sqrt(sessions)
    return {"mean": float(prof.mean()), "lo": float(prof.mean() - 1.96 * se), "hi": float(prof.mean() + 1.96 * se),
            "bumps": float(bumped.mean()), "empty": float(np.maximum(C - shows, 0).mean()),
            "sessions_with_bump": float((bumped > 0).mean())}


def simulate_years(L: int, C: int, p: float, r: float, b: float, rho: float = 0.0, lam: float | None = None,
                   years: int = 200, sessions: int = 500, seed: int = 0) -> dict:
    """Many independent simulated years: per-year 95% CIs and the pooled estimate."""
    yrs = [simulate(L, C, p, r, b, rho, lam, sessions, seed=seed * 100003 + y) for y in range(years)]
    means = np.array([y["mean"] for y in yrs])
    half = np.array([y["hi"] - y["mean"] for y in yrs])
    pooled_se = means.std(ddof=1) / np.sqrt(years)
    return {"years": yrs, "means": means, "half_width": half, "pooled_mean": float(means.mean()),
            "pooled_lo": float(means.mean() - 1.96 * pooled_se), "pooled_hi": float(means.mean() + 1.96 * pooled_se)}
