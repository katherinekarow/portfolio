"""Queueing formulas (Little's law, M/M/1, M/M/c Erlang C) and a discrete-event clinic simulation."""
from __future__ import annotations

import heapq
import math

import numpy as np


def mm1(lam: float, mu: float) -> dict:
    """ρ = λ/μ, L = ρ/(1−ρ), Lq = ρ²/(1−ρ), W = 1/(μ−λ), Wq = ρ/(μ−λ)."""
    rho = lam / mu
    if rho >= 1:
        raise ValueError("unstable: λ ≥ μ")
    return {"rho": rho, "L": rho / (1 - rho), "Lq": rho ** 2 / (1 - rho), "W": 1 / (mu - lam), "Wq": rho / (mu - lam)}


def erlang_c(c: int, a: float) -> float:
    """P(wait) with offered load a = λ/μ and c servers:
    C = [aᶜ/c! · c/(c−a)] / [Σₖ₌₀^(c−1) aᵏ/k! + aᶜ/c! · c/(c−a)]."""
    if a >= c:
        raise ValueError("unstable: a ≥ c")
    term = a ** c / math.factorial(c) * c / (c - a)
    return term / (sum(a ** k / math.factorial(k) for k in range(c)) + term)


def mmc(lam: float, mu: float, c: int) -> dict:
    """Wq = C(c, a)/(cμ − λ), W = Wq + 1/μ, Lq = λ·Wq, L = λ·W (Little), ρ = λ/(cμ)."""
    a = lam / mu
    pw = erlang_c(c, a)
    wq = pw / (c * mu - lam)
    return {"rho": lam / (c * mu), "P_wait": pw, "Wq": wq, "W": wq + 1 / mu, "Lq": lam * wq, "L": lam * (wq + 1 / mu)}


def simulate(lam: float, mu: float, c: int, n: int = 50000, rng=None, warmup: int = 2000) -> dict:
    """FCFS c-server queue with Poisson arrivals and exponential service. Each patient takes the server that
    frees up first; wait = max(0, server free time − arrival). L is measured independently by sweeping the
    number in the system over time, so Little's law L = λW is a real check."""
    rng = rng if rng is not None else np.random.default_rng(0)
    arr = np.cumsum(rng.exponential(1 / lam, n))
    svc = rng.exponential(1 / mu, n)
    free = [0.0] * c
    start = np.empty(n)
    for i in range(n):
        f = heapq.heappop(free)
        start[i] = max(arr[i], f)
        heapq.heappush(free, start[i] + svc[i])
    dep = start + svc
    wq, w = start - arr, dep - arr
    # time-average number in system on [t0, t1] by event sweep
    t0, t1 = arr[warmup], arr[-1]
    times = np.r_[arr, dep]; steps = np.r_[np.ones(n), -np.ones(n)]
    o = np.argsort(times, kind="stable"); times, steps = times[o], steps[o]
    count = np.cumsum(steps)
    seg_end = np.r_[times[1:], times[-1]]
    lo, hi = np.clip(times, t0, t1), np.clip(seg_end, t0, t1)
    L_time = float((count * (hi - lo)).sum() / (t1 - t0))
    keep = slice(warmup, None)
    lam_hat = (n - warmup) / (arr[-1] - arr[warmup - 1])
    return {"Wq": float(wq[keep].mean()), "W": float(w[keep].mean()), "P_wait": float((wq[keep] > 1e-12).mean()),
            "L_time_avg": L_time, "lambda_hat": float(lam_hat), "lambda_W": float(lam_hat * w[keep].mean()),
            "Wq_p90": float(np.quantile(wq[keep], 0.9))}
