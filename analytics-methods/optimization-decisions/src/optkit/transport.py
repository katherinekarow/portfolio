"""Transportation problem: move goods from supply points to demand points at least cost.

    minimise   Σ_ij c_ij x_ij
    subject to Σ_j x_ij = s_i  (each farm sends out what it has)
               Σ_i x_ij = d_j  (each market gets what it ordered)
               x_ij >= 0

It needs Σ s = Σ d. If supply is larger, a zero-cost dummy market absorbs the surplus
(the produce stays on the farm); if demand is larger, a dummy farm records unmet demand.
"""
from __future__ import annotations

import numpy as np

from .lp import simplex


def balance(cost, supply, demand, shortage_cost=0.0):
    cost = np.asarray(cost, float)
    supply = np.asarray(supply, float)
    demand = np.asarray(demand, float)
    gap = supply.sum() - demand.sum()
    if gap > 1e-9:
        cost = np.hstack([cost, np.zeros((len(supply), 1))])
        demand = np.append(demand, gap)
    elif gap < -1e-9:
        cost = np.vstack([cost, np.full((1, len(demand)), shortage_cost)])
        supply = np.append(supply, -gap)
    return cost, supply, demand


def lp_matrices(cost, supply, demand):
    m, n = cost.shape
    A = np.zeros((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A[m + j, j::n] = 1.0
    return cost.ravel(), A, np.concatenate([supply, demand])


def solve(cost, supply, demand):
    """Returns (plan m×n, total cost, u, v, reduced cost matrix) on the balanced problem.

    u_i + v_j is the marginal cost of delivering one more unit from farm i to market j.
    The duals are only unique up to a constant (one constraint is redundant), but the
    reduced costs c_ij − u_i − v_j are unique: how much a route's cost must fall before it is used.
    """
    cost, supply, demand = balance(cost, supply, demand)
    m, n = cost.shape
    c, A, b = lp_matrices(cost, supply, demand)
    r = simplex(c, A_eq=A, b_eq=b)
    if r.status != "optimal":
        raise ValueError(r.status)
    plan = r.x.reshape(m, n)
    u, v = r.shadow_eq[:m], r.shadow_eq[m:]
    # normalise so the first farm's dual is 0 (the usual MODI convention)
    v = v + u[0]
    u = u - u[0]
    reduced = cost - u[:, None] - v[None, :]
    return plan, float(r.objective), u, v, reduced


def northwest_corner(supply, demand):
    """The textbook starting plan: fill cells from the top-left, ignoring cost. Used as a baseline."""
    s, d = list(map(float, supply)), list(map(float, demand))
    plan = np.zeros((len(s), len(d)))
    i = j = 0
    while i < len(s) and j < len(d):
        q = min(s[i], d[j])
        plan[i, j] = q
        s[i] -= q
        d[j] -= q
        if s[i] <= 1e-12:
            i += 1
        else:
            j += 1
    return plan
