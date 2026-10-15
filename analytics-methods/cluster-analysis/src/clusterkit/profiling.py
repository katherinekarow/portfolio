"""Turning clusters into segments people can use."""
from __future__ import annotations

import numpy as np
import pandas as pd


def centroids_original_units(centroids_z, scaler, features):
    """x = mean + sd * z, so a centroid reads as 'books 62% of demand by 60 days out'."""
    return pd.DataFrame(scaler.inverse_transform(centroids_z), columns=features)


def average_curves(curves, labels, quantiles=(0.25, 0.5, 0.75)):
    """Mean and quantile band of the normalised cumulative curve within each cluster -
    the segment benchmark curve."""
    cols = [c for c in curves.columns if c.startswith("d")]
    out = []
    for c in np.unique(labels):
        sub = curves.loc[labels == c, cols]
        row = {"cluster": c, "stat": "mean", **sub.mean().to_dict()}
        out.append(row)
        for q in quantiles:
            out.append({"cluster": c, "stat": f"p{int(q * 100)}", **sub.quantile(q).to_dict()})
    return pd.DataFrame(out)


def robust_z(x, ref=None):
    """z = (x - median) / (1.4826 * MAD), median and MAD taken from `ref` (default x).
    1.4826 makes the MAD match the sd for normal data; unlike mean/sd, a few huge values
    cannot inflate the yardstick that is meant to catch them."""
    x = np.asarray(x, float)
    ref = x if ref is None else np.asarray(ref, float)
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    return (x - med) / (mad if mad > 0 else 1.0)


def knn_screen(X, k=5, z_cut=5.0):
    """Pre-clustering screen that needs no k for the clustering: distance from each point
    to its k-th nearest neighbour, as a robust z. Isolated points (and isolated pairs) score
    high. Used only to stop a handful of odd rows forming their own clusters while k is
    being chosen - they are scored again against the final segments afterwards."""
    from .distance import pairwise_euclidean
    D = pairwise_euclidean(X)
    np.fill_diagonal(D, np.inf)
    dk = np.sort(D, axis=1)[:, k - 1]
    z = robust_z(dk)
    return dk, z, z > z_cut


def distance_outliers(X, labels, centroids, z_cut=3.5, ref_mask=None):
    """Distance of each point to its own centroid, d = |x - mu_c|, turned into a robust z
    within its cluster (median / MAD of the distances of the cluster's reference members,
    `ref_mask`, default all). Flag z > z_cut."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    ref_mask = np.ones(len(X), bool) if ref_mask is None else np.asarray(ref_mask, bool)
    d = np.sqrt(((X - np.asarray(centroids)[labels]) ** 2).sum(1))
    z = np.zeros_like(d)
    for c in np.unique(labels):
        m = labels == c
        z[m] = robust_z(d[m], d[m & ref_mask])
    return d, z, z > z_cut
