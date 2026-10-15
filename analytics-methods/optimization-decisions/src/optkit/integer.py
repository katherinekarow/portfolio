"""Integer programming: knapsack (exact DP vs greedy) and branch-and-bound on LP relaxations."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .lp import simplex


# ---------- 0/1 knapsack ----------
def knapsack_dp(values, weights, capacity):
    """Exact 0/1 knapsack by dynamic programming, O(n·W) for integer weights.

    best[i][w] = max(best[i-1][w], best[i-1][w - w_i] + v_i)
    """
    v = list(values)
    w = [int(x) for x in weights]
    W = int(capacity)
    n = len(v)
    best = np.zeros((n + 1, W + 1))
    for i in range(1, n + 1):
        best[i] = best[i - 1]
        if w[i - 1] <= W:
            take = best[i - 1, :W + 1 - w[i - 1]] + v[i - 1]
            best[i, w[i - 1]:] = np.maximum(best[i - 1, w[i - 1]:], take)
    chosen, cap = [], W
    for i in range(n, 0, -1):
        if best[i, cap] != best[i - 1, cap]:
            chosen.append(i - 1)
            cap -= w[i - 1]
    return float(best[n, W]), sorted(chosen)


def knapsack_greedy(values, weights, capacity):
    """Take items in order of value per unit of weight while they fit. Fast, not always optimal."""
    order = sorted(range(len(values)), key=lambda i: (-values[i] / weights[i], i))
    left, chosen, total = capacity, [], 0.0
    for i in order:
        if weights[i] <= left:
            chosen.append(i)
            left -= weights[i]
            total += values[i]
    return float(total), sorted(chosen)


def knapsack_brute(values, weights, capacity):
    n = len(values)
    best, pick = 0.0, []
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if sum(weights[i] for i in idx) <= capacity:
            val = sum(values[i] for i in idx)
            if val > best + 1e-12:
                best, pick = float(val), idx
    return best, pick


def random_knapsack(rng, n=20):
    w = rng.integers(1, 31, n)
    v = np.round(w * rng.uniform(0.6, 1.6, n) + rng.uniform(0, 8, n), 1)
    cap = int(0.4 * w.sum())
    return v.tolist(), w.tolist(), cap


# ---------- branch and bound ----------
def branch_and_bound(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, maximize=False,
                     integer=None, tol=1e-6):
    """Depth-first branch-and-bound for a (mixed-)integer LP, using the from-scratch simplex.

    At each node, solve the LP relaxation. Prune if it is infeasible or its bound cannot beat
    the best integer solution found so far (the incumbent). Otherwise branch on the first
    fractional variable: x_j <= floor(v) on one side and x_j >= ceil(v) on the other.
    Returns (x, objective, nodes explored, log of nodes).
    """
    c = np.asarray(c, float)
    n = len(c)
    integer = list(range(n)) if integer is None else list(integer)
    A_ub = np.zeros((0, n)) if A_ub is None else np.asarray(A_ub, float)
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, float)
    sense = -1.0 if maximize else 1.0      # work as minimisation internally
    best_x, best_val = None, math.inf
    stack = [[]]                           # each node = list of (j, "le"/"ge", bound)
    nodes, log = 0, []
    while stack:
        cuts = stack.pop()
        nodes += 1
        rows, rhs = [A_ub], [b_ub]
        for j, kind, bound in cuts:
            e = np.zeros(n)
            e[j] = 1.0 if kind == "le" else -1.0
            rows.append(e[None, :])
            rhs.append([bound if kind == "le" else -bound])
        r = simplex(c, np.vstack(rows), np.concatenate(rhs), A_eq, b_eq, maximize=maximize)
        if r.status != "optimal":
            log.append((len(cuts), r.status, None))
            continue
        val = sense * r.objective
        if val >= best_val - tol:
            log.append((len(cuts), "pruned by bound", r.objective))
            continue
        frac = [j for j in integer if abs(r.x[j] - round(r.x[j])) > tol]
        if not frac:
            best_x, best_val = np.round(r.x, 9), val
            log.append((len(cuts), "new incumbent", r.objective))
            continue
        j = frac[0]
        log.append((len(cuts), f"branch on x{j}={r.x[j]:.3f}", r.objective))
        stack.append(cuts + [(j, "ge", math.ceil(r.x[j] - tol))])
        stack.append(cuts + [(j, "le", math.floor(r.x[j] + tol))])
    if best_x is None:
        return None, None, nodes, log
    return best_x, sense * best_val, nodes, log


def brute_force_ilp(c, A_ub, b_ub, upper, maximize=False):
    """Enumerate every integer point 0..upper_j (vectorised). Only for small problems."""
    c = np.asarray(c, float)
    A_ub = np.asarray(A_ub, float)
    b_ub = np.asarray(b_ub, float)
    grids = np.array(list(itertools.product(*[range(int(u) + 1) for u in upper])), float)
    ok = np.all(grids @ A_ub.T <= b_ub + 1e-9, axis=1)
    if not ok.any():
        return None, None
    vals = grids[ok] @ c
    k = int(np.argmax(vals) if maximize else np.argmin(vals))
    return grids[ok][k], float(vals[k])
