"""Principal component analysis via the SVD (with an eigendecomposition cross-check)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    components: np.ndarray        # rows = principal directions (orthonormal)
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    scores: np.ndarray            # data projected onto the components
    mean: np.ndarray


def pca(X, n_components=None):
    """Centre X, then X_c = U S V'. Rows of V' are the directions; variance along
    component j is S_j^2 / (n - 1); explained-variance ratio = S_j^2 / sum S^2.
    Scores = X_c V = U S. Signs are fixed so each component's largest loading is positive."""
    X = np.asarray(X, float)
    mean = X.mean(0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    flip = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
    Vt, U = Vt * flip[:, None], U * flip[None, :]
    var = S ** 2 / (len(X) - 1)
    ratio = S ** 2 / (S ** 2).sum()
    m = n_components or len(S)
    return PCAResult(Vt[:m], var[:m], ratio[:m], (U * S)[:, :m], mean)


def pca_eig(X):
    """Same answer from the covariance matrix: C = X_c'X_c/(n-1) = V diag(lambda) V'."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    vals, vecs = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order].T
