"""Choosing k: elbow on inertia, silhouette, gap statistic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .distance import pairwise_euclidean
from .kmeans import kmeans


def silhouette_samples(X, labels):
    """s(i) = (b - a) / max(a, b)
    a = mean distance from i to the other members of its own cluster
    b = mean distance from i to the members of the nearest *other* cluster
    Points in a singleton cluster get s = 0 (the usual convention)."""
    labels = np.asarray(labels)
    D = pairwise_euclidean(X)
    ks = np.unique(labels)
    n = len(labels)
    # mean distance from every point to every cluster
    M = np.zeros((n, len(ks)))
    sizes = np.array([(labels == k).sum() for k in ks])
    for j, k in enumerate(ks):
        M[:, j] = D[:, labels == k].sum(1)
    own = np.searchsorted(ks, labels)
    own_size = sizes[own]
    a = np.where(own_size > 1, M[np.arange(n), own] / np.maximum(own_size - 1, 1), 0.0)
    M = M / sizes
    M[np.arange(n), own] = np.inf
    b = M.min(1)
    s = np.where(own_size > 1, (b - a) / np.maximum(a, b), 0.0)
    return s


def silhouette_score(X, labels):
    return float(silhouette_samples(X, labels).mean())


def elbow_k(ks, inertias):
    """Knee of the inertia curve: the point farthest from the straight line joining the
    first and last points, after scaling both axes to [0, 1]."""
    ks, w = np.asarray(ks, float), np.asarray(inertias, float)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (w - w[-1]) / (w[0] - w[-1])
    # line from (0, 1) to (1, 0): x + y - 1 = 0 -> distance |x + y - 1| / sqrt(2)
    dist = np.abs(x + y - 1) / np.sqrt(2)
    return int(ks[dist.argmax()])


def gap_statistic(X, ks, B=20, seed=0, n_init=5, reference="pca"):
    """Tibshirani, Walther & Hastie (2001).
    Gap(k) = E*[log W_k] - log W_k, with E* the mean over B reference data sets drawn
    uniformly (no clusters) and W_k the within-cluster sum of squares.
    s_k = sd(log W*_k) * sqrt(1 + 1/B). Choose the smallest k with Gap(k) >= Gap(k+1) - s_{k+1}.

    reference="box": uniform over the bounding box of the features.
    reference="pca": uniform over the box of the data's principal components (the paper's
    preferred option - the box follows the shape of the data, so correlated features do
    not make the reference look artificially spread out)."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    if reference == "pca":
        X = X - X.mean(0)
        X = X @ np.linalg.svd(X, full_matrices=False)[2].T     # rotate onto components
    lo, hi = X.min(0), X.max(0)
    refs = [rng.uniform(lo, hi, size=X.shape) for _ in range(B)]
    rows = []
    for k in ks:
        logw = np.log(kmeans(X, k, n_init=n_init, seed=seed).inertia)
        logw_ref = np.array([np.log(kmeans(R, k, n_init=n_init, seed=seed + b + 1).inertia)
                             for b, R in enumerate(refs)])
        rows.append({"k": k, "log_w": logw, "log_w_ref": logw_ref.mean(),
                     "gap": logw_ref.mean() - logw, "s_k": logw_ref.std() * np.sqrt(1 + 1 / B)})
    return pd.DataFrame(rows)


def gap_choose_k(gap_df):
    g, s, ks = gap_df.gap.to_numpy(), gap_df.s_k.to_numpy(), gap_df.k.to_numpy()
    for i in range(len(ks) - 1):
        if g[i] >= g[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def choose_k(X, k_max=8, seed=0, n_init=10, B=20, reference="pca"):
    """Run all three criteria over k = 1..k_max. Returns (table, picks)."""
    ks = list(range(1, k_max + 1))
    rows = []
    for k in ks:
        res = kmeans(X, k, n_init=n_init, seed=seed)
        rows.append({"k": k, "inertia": res.inertia,
                     "silhouette": silhouette_score(X, res.labels) if k > 1 else np.nan})
    table = pd.DataFrame(rows)
    gap = gap_statistic(X, ks, B=B, seed=seed, reference=reference)
    table = table.merge(gap[["k", "gap", "s_k"]], on="k")
    picks = {"elbow": elbow_k(table.k, table.inertia),
             "silhouette": int(table.k[table.silhouette.idxmax()]),
             "gap": gap_choose_k(gap)}
    return table, picks
