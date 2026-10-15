"""Distance measures, written out so the formulas are visible."""
from __future__ import annotations

import numpy as np


def euclidean(a, b):
    """sqrt(sum (a_i - b_i)^2)"""
    return float(np.sqrt(np.sum((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def manhattan(a, b):
    """sum |a_i - b_i|"""
    return float(np.sum(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def cosine(a, b):
    """1 - a.b / (|a| |b|)  - ignores magnitude, compares direction only."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def mahalanobis(a, b, cov):
    """sqrt((a - b)' S^-1 (a - b))  - rescales by spread AND removes correlation."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(diff @ np.linalg.solve(np.asarray(cov, float), diff)))


def pairwise_sq_euclidean(X, Y=None):
    """Matrix of squared Euclidean distances via |x|^2 + |y|^2 - 2 x.y (clipped at 0)."""
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    d = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.maximum(d, 0.0)


def pairwise_euclidean(X, Y=None):
    return np.sqrt(pairwise_sq_euclidean(X, Y))


def nearest_neighbour_agreement(X, labels):
    """Share of points whose nearest other point has the same label (a quick check of
    whether a distance 'sees' the structure)."""
    D = pairwise_sq_euclidean(X)
    np.fill_diagonal(D, np.inf)
    labels = np.asarray(labels)
    return float(np.mean(labels[D.argmin(1)] == labels))
