"""k-means (Lloyd's algorithm) with k-means++ seeding and restarts."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .distance import pairwise_sq_euclidean


def inertia(X, labels, centroids):
    """Within-cluster sum of squares: sum_k sum_{i in k} |x_i - mu_k|^2"""
    return float(((np.asarray(X) - np.asarray(centroids)[labels]) ** 2).sum())


def kmeans_pp_init(X, k, rng):
    """k-means++: first centre uniform at random; each next centre drawn with
    probability proportional to D(x)^2, the squared distance to the nearest centre so far."""
    n = len(X)
    centres = [X[rng.integers(n)]]
    d2 = pairwise_sq_euclidean(X, centres[0][None, :])[:, 0]
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        c = X[rng.choice(n, p=probs)]
        centres.append(c)
        d2 = np.minimum(d2, pairwise_sq_euclidean(X, c[None, :])[:, 0])
    return np.array(centres)


@dataclass
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iter: int
    history: list = field(default_factory=list)     # inertia after every assignment step


def lloyd(X, centroids, max_iter=300):
    """Alternate (1) assign each point to its nearest centroid, (2) move each centroid to
    the mean of its points. Each step can only lower (or keep) the inertia, and there are
    finitely many labelings, so it stops when the assignment no longer changes.
    `history` records the inertia after every half-step (assign, update, assign, ...)."""
    X = np.asarray(X, float)
    C = np.array(centroids, float)
    history = []
    labels = None
    it = 0
    for it in range(1, max_iter + 1):
        new = pairwise_sq_euclidean(X, C).argmin(1)
        history.append(inertia(X, new, C))
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(C)):
            members = X[labels == j]
            if len(members):                       # empty cluster: keep previous centre
                C[j] = members.mean(0)
        history.append(inertia(X, labels, C))
    return KMeansResult(labels, C, inertia(X, labels, C), it, history)


def kmeans(X, k, n_init=10, seed=0, init="k-means++", max_iter=300):
    """Best of `n_init` runs (lowest inertia). init='random' picks k distinct points."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        if init == "k-means++":
            C0 = kmeans_pp_init(X, k, rng)
        else:
            C0 = X[rng.choice(len(X), k, replace=False)]
        res = lloyd(X, C0, max_iter=max_iter)
        if best is None or res.inertia < best.inertia - 1e-12:
            best = res
    return best


def relabel_by(labels, centroids, key):
    """Renumber clusters 0..k-1 in descending order of `key` (one value per cluster), so
    cluster numbers are stable and meaningful across runs."""
    order = np.argsort(-np.asarray(key))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels], np.asarray(centroids)[order]
