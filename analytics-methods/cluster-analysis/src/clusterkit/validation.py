"""Validating clusters: adjusted Rand index, confusion matrix, bootstrap stability."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from .kmeans import kmeans


def contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1), int)
    np.add.at(M, (ai, bi), 1)
    return M


def _comb2(x):
    x = np.asarray(x, float)
    return x * (x - 1) / 2


def adjusted_rand_index(a, b):
    """ARI = (Index - Expected) / (Max - Expected), counting pairs of points:
        Index    = sum_ij C(n_ij, 2)
        Expected = sum_i C(a_i, 2) * sum_j C(b_j, 2) / C(n, 2)
        Max      = [sum_i C(a_i, 2) + sum_j C(b_j, 2)] / 2
    n_ij = contingency counts, a_i / b_j = row / column totals.
    1 = identical partitions (up to renaming), about 0 = no better than chance."""
    M = contingency(a, b)
    n = M.sum()
    index = _comb2(M).sum()
    sa, sb = _comb2(M.sum(1)).sum(), _comb2(M.sum(0)).sum()
    expected = sa * sb / _comb2(n)
    mx = (sa + sb) / 2
    if mx == expected:
        return 1.0
    return float((index - expected) / (mx - expected))


def match_labels(pred, truth):
    """Rename predicted clusters to the truth labels that maximise agreement (brute force
    over permutations - fine for k <= 7)."""
    pk, tk = np.unique(pred), np.unique(truth)
    M = contingency(pred, truth)
    best, best_perm = -1, None
    for perm in permutations(range(len(tk)), len(pk)):
        score = M[np.arange(len(pk)), perm].sum()
        if score > best:
            best, best_perm = score, perm
    mapping = {pk[i]: tk[best_perm[i]] for i in range(len(pk))}
    return np.array([mapping[p] for p in pred])


def confusion(pred, truth, names=None):
    M = contingency(truth, pred)
    tk, pk = np.unique(truth), np.unique(pred)
    idx = [names.get(t, t) if names else t for t in tk]
    return pd.DataFrame(M, index=pd.Index(idx, name="true segment"),
                        columns=pd.Index([f"cluster {p}" for p in pk], name="found"))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def bootstrap_stability(X, labels, k, B=50, seed=0, n_init=5):
    """Hennig's clusterboot idea: resample rows with replacement, recluster, and for each
    original cluster record the best Jaccard overlap |A n B| / |A u B| with any bootstrap
    cluster (both restricted to the resampled points). Mean Jaccard > 0.85 = stable,
    < 0.6 = the cluster is an artefact of this sample."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(X)
    scores = np.zeros((B, k))
    for b in range(B):
        idx = np.unique(rng.integers(0, n, n))
        res = kmeans(X[idx], k, n_init=n_init, seed=seed + b + 1)
        for c in range(k):
            orig = idx[labels[idx] == c]
            scores[b, c] = max(jaccard(orig, idx[res.labels == j]) for j in range(k))
    return scores.mean(0), scores
