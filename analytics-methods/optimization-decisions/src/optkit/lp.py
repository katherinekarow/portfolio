"""Linear programming from scratch: a dense-tableau, two-phase simplex with Bland's rule.

Problem form (all variables x >= 0):

    maximise or minimise   c·x
    subject to             A_ub x <= b_ub
                           A_eq x  = b_eq

Besides the optimum the solver returns what a planner actually asks about:
shadow prices (value of one more unit of each resource), slacks, reduced costs,
and the ranges over which those numbers stay valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

TOL = 1e-9


@dataclass
class LPResult:
    status: str                      # "optimal", "infeasible" or "unbounded"
    x: np.ndarray | None = None      # decision variables
    objective: float | None = None   # in the user's sense (max or min)
    shadow_ub: np.ndarray | None = None   # d(objective)/d(b_ub)
    shadow_eq: np.ndarray | None = None   # d(objective)/d(b_eq)
    slack_ub: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None  # c_j - a_j·y (user sense)
    rhs_range_ub: np.ndarray | None = None   # (low, high) b_ub over which shadow prices hold
    rhs_range_eq: np.ndarray | None = None
    cost_range: np.ndarray | None = None     # (low, high) c_j over which the plan x stays optimal
    basis: list = field(default_factory=list)
    path: list = field(default_factory=list)  # x after each pivot (phase 2 only)
    iterations: int = 0


def _pivot(T: np.ndarray, r: int, k: int) -> None:
    T[r] /= T[r, k]
    for i in range(T.shape[0]):
        if i != r and T[i, k] != 0.0:
            T[i] -= T[i, k] * T[r]


def _simplex_phase(T, basis, cost, allowed, n_orig, path, max_iter=5000):
    """Minimise cost·z on the tableau T (rows = B^-1 [A | b]) with Bland's rule.

    Bland's rule: enter the lowest-index column with a negative reduced cost; among
    rows tied on the minimum ratio, leave by the lowest-index basic variable. It
    cannot cycle, which matters on degenerate problems such as transportation.
    """
    m = T.shape[0]
    it = 0
    while it < max_iter:
        cb = cost[basis]
        red = cost - cb @ T[:, :-1]
        entering = next((j for j in allowed if red[j] < -TOL), None)
        if entering is None:
            return "optimal", it
        col = T[:, entering]
        rows = [i for i in range(m) if col[i] > TOL]
        if not rows:
            return "unbounded", it
        ratios = np.array([T[i, -1] / col[i] for i in rows])
        best = ratios.min()
        tied = [rows[t] for t in range(len(rows)) if ratios[t] <= best + TOL]
        leave = min(tied, key=lambda i: basis[i])
        _pivot(T, leave, entering)
        basis[leave] = entering
        it += 1
        if path is not None:
            path.append(_extract_x(T, basis, n_orig))
    raise RuntimeError("simplex iteration limit reached")


def _extract_x(T, basis, n):
    x = np.zeros(n)
    for i, j in enumerate(basis):
        if j < n:
            x[j] = T[i, -1]
    return x


def simplex(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, maximize=False) -> LPResult:
    c = np.asarray(c, float)
    n = len(c)
    A_ub = np.zeros((0, n)) if A_ub is None else np.atleast_2d(np.asarray(A_ub, float))
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, float)
    A_eq = np.zeros((0, n)) if A_eq is None else np.atleast_2d(np.asarray(A_eq, float))
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, float)
    m_ub, m_eq = len(b_ub), len(b_eq)
    m = m_ub + m_eq

    # Equality form: [A_ub I; A_eq 0] [x; s] = b, then make every right-hand side >= 0.
    A = np.zeros((m, n + m_ub))
    A[:m_ub, :n] = A_ub
    A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = A_eq
    b = np.concatenate([b_ub, b_eq])
    sign = np.where(b < 0, -1.0, 1.0)
    A *= sign[:, None]
    b = b * sign

    # Starting basis: the slack where it has +1, an artificial variable elsewhere.
    n_real = n + m_ub
    art_rows = [i for i in range(m) if not (i < m_ub and sign[i] > 0)]
    n_art = len(art_rows)
    T = np.zeros((m, n_real + n_art + 1))
    T[:, :n_real] = A
    T[:, -1] = b
    basis = [0] * m
    for i in range(m_ub):
        if sign[i] > 0:
            basis[i] = n + i
    for k, i in enumerate(art_rows):
        T[i, n_real + k] = 1.0
        basis[i] = n_real + k

    total_iter = 0
    keep_rows = list(range(m))
    if n_art:
        cost1 = np.zeros(n_real + n_art)
        cost1[n_real:] = 1.0
        status, it = _simplex_phase(T, basis, cost1, range(n_real + n_art), n, None)
        total_iter += it
        if cost1[basis] @ T[:, -1] > 1e-7:
            return LPResult("infeasible", iterations=total_iter)
        # Drive zero-valued artificials out of the basis; rows where that is impossible are redundant.
        drop = []
        for i, j in enumerate(basis):
            if j >= n_real:
                cand = [k for k in range(n_real) if abs(T[i, k]) > 1e-9]
                if cand:
                    _pivot(T, i, cand[0])
                    basis[i] = cand[0]
                else:
                    drop.append(i)
        keep_rows = [i for i in range(m) if i not in drop]
        T = T[keep_rows]
        basis = [basis[i] for i in keep_rows]
    T = np.delete(T, np.s_[n_real:n_real + n_art], axis=1)

    cost = np.zeros(n_real)
    cost[:n] = -c if maximize else c
    path = [_extract_x(T, basis, n)]
    status, it = _simplex_phase(T, basis, cost, range(n_real), n, path)
    total_iter += it
    if status != "optimal":
        return LPResult(status, iterations=total_iter, path=path)

    x = _extract_x(T, basis, n)
    obj_min = float(cost[:n] @ x)
    # Duals of the (sign-flipped) equality system: y = B^-T c_B.
    Ak = A[keep_rows]
    B = Ak[:, basis]
    y_k = np.linalg.solve(B.T, cost[basis])
    y = np.zeros(m)
    y[keep_rows] = y_k
    y_min = y * sign                          # d(min objective)/d(original b)
    ysign = -1.0 if maximize else 1.0
    shadow = ysign * y_min                     # in the user's sense
    red_min = cost - Ak.T @ y_k               # >= 0 at a minimum
    reduced = ysign * red_min[:n]

    # Right-hand-side ranging: x_B(δ) = x_B + δ·B^-1 e_i must stay >= 0.
    Binv = np.linalg.inv(B)
    xB = T[:, -1]
    rhs_range = np.full((m, 2), np.nan)
    for i in range(m):
        if i not in keep_rows:
            continue
        d = sign[i] * Binv[:, keep_rows.index(i)]
        lo = max([-xB[r] / d[r] for r in range(len(d)) if d[r] > TOL], default=-np.inf)
        hi = min([-xB[r] / d[r] for r in range(len(d)) if d[r] < -TOL], default=np.inf)
        orig_b = np.concatenate([b_ub, b_eq])[i]
        rhs_range[i] = (orig_b + lo, orig_b + hi)

    # Objective-coefficient ranging (in min form, then mapped to the user's sense).
    nonbasic = [j for j in range(n_real) if j not in basis]
    cost_range = np.zeros((n, 2))
    for j in range(n):
        if j in basis:
            r = basis.index(j)
            ups = [red_min[l] / T[r, l] for l in nonbasic if T[r, l] > TOL]
            downs = [red_min[l] / T[r, l] for l in nonbasic if T[r, l] < -TOL]
            d_lo, d_hi = max(downs, default=-np.inf), min(ups, default=np.inf)
        else:
            d_lo, d_hi = -red_min[j], np.inf
        if maximize:   # c = -cost, so the interval flips
            cost_range[j] = (c[j] - d_hi, c[j] - d_lo)
        else:
            cost_range[j] = (c[j] + d_lo, c[j] + d_hi)

    return LPResult(
        "optimal", x=x, objective=-obj_min if maximize else obj_min,
        shadow_ub=shadow[:m_ub], shadow_eq=shadow[m_ub:],
        slack_ub=b_ub - A_ub @ x, reduced_costs=reduced,
        rhs_range_ub=rhs_range[:m_ub], rhs_range_eq=rhs_range[m_ub:],
        cost_range=cost_range, basis=list(basis), path=path, iterations=total_iter)


def vertices_2d(A_ub, b_ub):
    """Corner points of {x >= 0, A x <= b} in two variables (the graphical method)."""
    A = np.vstack([np.asarray(A_ub, float), -np.eye(2)])
    b = np.concatenate([np.asarray(b_ub, float), [0.0, 0.0]])
    pts = []
    for i, j in combinations(range(len(b)), 2):
        M = A[[i, j]]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, b[[i, j]])
        if np.all(A @ p <= b + 1e-9) and not any(np.allclose(p, q) for q in pts):
            pts.append(p)
    pts = np.array(pts)
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    return pts[order]
