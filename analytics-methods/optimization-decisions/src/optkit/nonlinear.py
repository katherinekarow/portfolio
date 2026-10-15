"""Non-linear optimisation: a pricing problem and a budget split.

Profit from a price p with exponential demand  D(p) = a·e^(−b·p)  and unit cost c:

    π(p)   = (p − c)·a·e^(−b·p)
    π'(p)  = a·e^(−b·p)·(1 − b·(p − c))           → zero at p* = c + 1/b
    π''(p) = a·b·e^(−b·p)·(b·(p − c) − 2)          (< 0 at p*, so it is a maximum)
"""
from __future__ import annotations

import math

import numpy as np


class ExpDemandProfit:
    def __init__(self, a=1000.0, b=0.25, cost=2.0):
        self.a, self.b, self.c = a, b, cost

    def demand(self, p):
        return self.a * np.exp(-self.b * p)

    def profit(self, p):
        return (p - self.c) * self.demand(p)

    def grad(self, p):
        return self.demand(p) * (1 - self.b * (p - self.c))

    def hess(self, p):
        return self.demand(p) * self.b * (self.b * (p - self.c) - 2)

    @property
    def optimum(self):
        return self.c + 1 / self.b


def gradient_ascent(f_grad, x0, step, n_iter=60):
    """x_{k+1} = x_k + α·f'(x_k). Near the optimum the error shrinks by |1 + α·f''(x*)| each step: linear."""
    xs = [float(x0)]
    for _ in range(n_iter):
        xs.append(xs[-1] + step * f_grad(xs[-1]))
    return np.array(xs)


def newton(f_grad, f_hess, x0, n_iter=8):
    """x_{k+1} = x_k − f'(x_k)/f''(x_k). Near the optimum the error is roughly squared each step: quadratic."""
    xs = [float(x0)]
    for _ in range(n_iter):
        h = f_hess(xs[-1])
        if h == 0:
            break
        xs.append(xs[-1] - f_grad(xs[-1]) / h)
    return np.array(xs)


INV_PHI = (math.sqrt(5) - 1) / 2   # 0.618…


def golden_section(f, lo, hi, tol=1e-8, maximize=True):
    """Bracket-shrinking search for a unimodal function; needs no derivative.

    Each step keeps 61.8% of the interval and reuses one of the two interior points.
    Returns (x, list of interval widths).
    """
    sgn = -1.0 if maximize else 1.0
    g = lambda x: sgn * f(x)
    x1 = hi - INV_PHI * (hi - lo)
    x2 = lo + INV_PHI * (hi - lo)
    f1, f2 = g(x1), g(x2)
    widths = [hi - lo]
    while hi - lo > tol:
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - INV_PHI * (hi - lo)
            f1 = g(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + INV_PHI * (hi - lo)
            f2 = g(x2)
        widths.append(hi - lo)
    return (lo + hi) / 2, widths


# ---------- budget allocation with a Lagrange multiplier ----------
def log_response(x, a, s):
    """Return from spending x on a channel with diminishing returns: a·ln(1 + x/s)."""
    return a * np.log1p(np.asarray(x, float) / s)


def marginal_return(x, a, s):
    return a / (s + np.asarray(x, float))


def lagrange_allocation(a, s, budget):
    """Maximise Σ a_i·ln(1 + x_i/s_i)  s.t.  Σ x_i = B, x_i >= 0.

    Lagrangian L = Σ a_i ln(1 + x_i/s_i) − λ(Σ x_i − B). Setting ∂L/∂x_i = 0:
        a_i/(s_i + x_i) = λ          (equal marginal return on every funded channel)
        x_i = a_i/λ − s_i,  λ = Σ_active a_i / (B + Σ_active s_i)
    A channel whose first dollar returns less than λ (a_i/s_i <= λ) gets nothing (KKT);
    drop it and re-solve until every funded channel is positive.
    λ is the value of one more unit of budget.
    """
    a = np.asarray(a, float)
    s = np.asarray(s, float)
    active = np.ones(len(a), bool)
    while True:
        lam = a[active].sum() / (budget + s[active].sum())
        x = np.where(active, a / lam - s, 0.0)
        if np.all(x >= -1e-12):
            return np.maximum(x, 0.0), float(lam)
        active &= x > 0
