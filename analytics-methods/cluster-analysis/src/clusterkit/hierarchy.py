"""Agglomerative clustering with Ward linkage, via the Lance-Williams update."""
from __future__ import annotations

import numpy as np

from .distance import pairwise_sq_euclidean


def ward_linkage(X):
    """Return a linkage matrix in SciPy's format: row t = [id_a, id_b, height, size],
    where the new cluster gets id n + t.

    Work on squared distances. Start: d(i, j) = |x_i - x_j|^2, every point its own cluster.
    Repeatedly merge the closest pair (i, j); the distance from the merged cluster to any
    other cluster k is the Lance-Williams update for Ward:

        d(k, i+j) = [(n_i + n_k) d(k, i) + (n_j + n_k) d(k, j) - n_k d(i, j)] / (n_i + n_j + n_k)

    The reported height is sqrt(d(i, j)) - the convention SciPy uses - and equals
    sqrt(2 * increase in total within-cluster sum of squares caused by the merge).
    """
    X = np.asarray(X, float)
    n = len(X)
    D = pairwise_sq_euclidean(X)
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    ids = np.arange(n)                    # current cluster id held in each slot
    active = np.ones(n, bool)
    Z = np.zeros((n - 1, 4))
    for t in range(n - 1):
        flat = np.argmin(np.where(active[:, None] & active[None, :], D, np.inf))
        i, j = divmod(flat, n)
        if i > j:
            i, j = j, i
        dij = D[i, j]
        ni, nj = size[i], size[j]
        nk = size
        new = ((ni + nk) * D[i] + (nj + nk) * D[j] - nk * dij) / (ni + nj + nk)
        a, b = sorted((ids[i], ids[j]))
        Z[t] = [a, b, np.sqrt(dij), ni + nj]
        # merged cluster lives in slot i; slot j is retired
        D[i, :] = new
        D[:, i] = new
        D[i, i] = np.inf
        active[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] = ni + nj
        ids[i] = n + t
    return Z


def cut_tree(Z, k):
    """Flat labels 0..k-1 from a linkage matrix by undoing the last k-1 merges."""
    n = len(Z) + 1
    parent = np.arange(2 * n - 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in range(n - k):
        a, b = int(Z[t, 0]), int(Z[t, 1])
        parent[a] = n + t
        parent[b] = n + t
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels
