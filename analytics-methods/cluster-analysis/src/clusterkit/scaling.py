"""Feature scaling and curve-shape features."""
from __future__ import annotations

import numpy as np


class Standardiser:
    """z = (x - mean) / sd, column by column (population sd, ddof=0).

    Kept as an object so centroids can be translated back: x = mean + sd * z.
    """

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)          # constant column -> left at 0, not NaN
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.sd_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return self.mean_ + np.asarray(Z, float) * self.sd_


def zscore(X):
    return Standardiser().fit_transform(X)


def minmax(X):
    """x' = (x - min) / (max - min), each column mapped onto [0, 1]."""
    X = np.asarray(X, float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / rng


def normalise_curve(cum):
    """Cumulative curve(s) divided by their final value (last column) -> shape only.

    Two sessions that book the same way but differ 3x in size get the same curve.
    """
    cum = np.asarray(cum, float)
    return cum / cum[..., -1:]


def distance_shares(a, b):
    """Share of the squared Euclidean distance between rows a and b owed to each feature."""
    sq = (np.asarray(a, float) - np.asarray(b, float)) ** 2
    return sq / sq.sum()
