"""Assignment problem: pair n rows with n columns, one each, at least total cost.

Hungarian algorithm (shortest augmenting paths with potentials), O(n³). It keeps row
potentials u and column potentials v with u_i + v_j <= c_ij on every cell; edges where the
inequality is tight form the "equality graph", and each round grows the matching by one along
a cheapest augmenting path. At the end Σu + Σv equals the matching cost, which proves optimality.
"""
from __future__ import annotations

import itertools

import numpy as np


def hungarian(cost):
    """Return (assignment col for each row, total cost). Works for n rows <= m columns."""
    a = np.asarray(cost, float)
    n, m = a.shape
    if n > m:
        raise ValueError("need rows <= columns; transpose the matrix")
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)      # p[j] = row matched to column j (1-based, 0 = free)
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:              # flip the augmenting path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = np.zeros(n, int)
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    return assign, float(a[np.arange(n), assign].sum())


def hungarian_max(value):
    """Maximise total value by minimising (max − value)."""
    value = np.asarray(value, float)
    assign, _ = hungarian(value.max() - value)
    return assign, float(value[np.arange(len(assign)), assign].sum())


def brute_force(cost):
    a = np.asarray(cost, float)
    n = a.shape[0]
    best, arg = np.inf, None
    for perm in itertools.permutations(range(a.shape[1]), n):
        s = a[np.arange(n), perm].sum()
        if s < best:
            best, arg = s, perm
    return np.array(arg), float(best)


def greedy_max(value):
    """Repeatedly take the single highest remaining value. The intuitive approach, not optimal."""
    v = np.asarray(value, float).copy()
    n = v.shape[0]
    assign = np.full(n, -1)
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(v), v.shape)
        assign[i] = j
        v[i, :] = -np.inf
        v[:, j] = -np.inf
    val = np.asarray(value, float)
    return assign, float(val[np.arange(n), assign].sum())
