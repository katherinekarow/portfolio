"""The full analysis, end to end, returning everything the CLI writes and charts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import distance, hierarchy, kmeans as km, pca, profiling, scaling, selection, synth, validation

SEED = 7


def scaling_demo(sessions):
    """Why scaling matters: add slot capacity (40x bigger numbers) to the behaviour features
    and compare raw vs z-scored distances."""
    cols = ["capacity"] + synth.FEATURES
    raw = sessions[cols].to_numpy(float)
    inl = ~sessions.is_outlier.to_numpy()
    truth = sessions.true_segment.to_numpy()
    rows = []
    for name, M in [("raw units", raw), ("min-max [0,1]", scaling.minmax(raw)), ("z-score", scaling.zscore(raw))]:
        r = km.kmeans(M[inl], 4, n_init=10, seed=SEED)
        # average share of squared distance owed to capacity, over all pairs of sessions
        diff2 = (M[inl][:, None, :] - M[inl][None, :, :]) ** 2
        tot = diff2.sum(-1)
        mask = tot > 0
        cap_share = float((diff2[..., 0][mask] / tot[mask]).mean())
        rows.append({"scaling": name, "capacity_share_of_sq_distance": cap_share,
                     "nearest_neighbour_same_segment": distance.nearest_neighbour_agreement(M[inl], truth[inl]),
                     "kmeans_ari_vs_truth": validation.adjusted_rand_index(r.labels, truth[inl])})
    return pd.DataFrame(rows)


def distance_demo(sessions, Z, cov):
    """The four distances for three illustrative pairs (in z-score units)."""
    ids = sessions.session_id.to_numpy()
    seg = sessions.true_segment.to_numpy()
    first = {s: np.nonzero(seg == s)[0][:2] for s in range(4)}
    pairs = [("same segment (steady)", first[1][0], first[1][1]),
             ("steady vs under-filled", first[1][0], first[3][0]),
             ("early vs late rush", first[0][0], first[2][0])]
    out = []
    for label, i, j in pairs:
        out.append({"pair": label, "a": ids[i], "b": ids[j],
                    "euclidean": distance.euclidean(Z[i], Z[j]),
                    "manhattan": distance.manhattan(Z[i], Z[j]),
                    "cosine": distance.cosine(Z[i], Z[j]),
                    "mahalanobis": distance.mahalanobis(Z[i], Z[j], cov)})
    return pd.DataFrame(out)


def run_analysis(seed: int = SEED, bootstrap_B: int = 50, gap_B: int = 20):
    sessions, curves = synth.make_sessions(seed=seed)
    feats = synth.FEATURES
    X = sessions[feats].to_numpy(float)
    truth = sessions.true_segment.to_numpy()
    planted = sessions.is_outlier.to_numpy()

    # 1. choose k on everything (to show what outliers do to the criteria)
    Z_all = scaling.zscore(X)
    k_table_all, picks_all = selection.choose_k(Z_all, seed=seed, B=gap_B)

    # 2. screen isolated rows, refit scaling on the rest, choose k again
    _, knn_z, screened = profiling.knn_screen(Z_all)
    keep = ~screened
    scaler = scaling.Standardiser().fit(X[keep])
    Z = scaler.transform(X)                      # every row, in the clean rows' z units
    Zk = Z[keep]
    k_table, picks = selection.choose_k(Zk, seed=seed, B=gap_B)
    k = picks["silhouette"]

    # 3. k-means++ with restarts on the kept rows; number clusters by early booking
    res = km.kmeans(Zk, k, n_init=10, seed=seed)
    cents_orig = scaler.inverse_transform(res.centroids)
    lab_k, cents = km.relabel_by(res.labels, res.centroids, cents_orig[:, feats.index("share_60")])
    labels = km.pairwise_sq_euclidean(Z, cents).argmin(1)        # assign every row
    assert np.array_equal(labels[keep], lab_k)

    # 4. outliers: distance to own centroid, robust z against the kept members
    dist, dz, flagged = profiling.distance_outliers(Z, labels, cents, ref_mask=keep)

    # 5. Ward hierarchy on the kept rows
    Zlink = hierarchy.ward_linkage(Zk)
    ward_labels = hierarchy.cut_tree(Zlink, k)

    # 6. PCA for the 2-D view
    p = pca.pca(Zk)
    scores_all = (Z - p.mean) @ p.components[:2].T

    # 7. validation
    inl = ~planted
    ari = {
        "kmeans_vs_truth": validation.adjusted_rand_index(labels[inl], truth[inl]),
        "ward_vs_truth": validation.adjusted_rand_index(ward_labels[~planted[keep]], truth[keep][~planted[keep]]),
        "kmeans_vs_ward": validation.adjusted_rand_index(lab_k, ward_labels),
        "service_line_vs_truth": validation.adjusted_rand_index(sessions.service_line[inl], truth[inl]),
    }
    conf = validation.confusion(labels[inl], truth[inl], names={s: v["name"] for s, v in synth.SEGMENTS.items()})
    stab_mean, _ = validation.bootstrap_stability(Zk, lab_k, k, B=bootstrap_B, seed=seed)

    # 8. profiles in original units
    prof = profiling.centroids_original_units(cents, scaler, feats)
    prof.insert(0, "cluster", range(k))
    prof["sessions"] = np.bincount(labels[~flagged], minlength=k)
    prof["bootstrap_jaccard"] = stab_mean
    prof["mean_capacity"] = [sessions.capacity[(labels == c) & ~flagged].mean() for c in range(k)]
    prof["top_service_line_share"] = [sessions.service_line[(labels == c) & ~flagged].value_counts(normalize=True).iloc[0]
                                      for c in range(k)]
    seg_of = conf.to_numpy().argmax(0)
    prof["matches_true_segment"] = [conf.index[i] for i in seg_of]
    avg_curves = profiling.average_curves(curves[~flagged].reset_index(drop=True), labels[~flagged])

    cov = np.cov(Zk, rowvar=False)
    out_sessions = sessions.assign(cluster=labels, screened_before_k=screened, knn_z=knn_z,
                                   dist_to_centroid=dist, dist_z=dz, flagged_outlier=flagged,
                                   pc1=scores_all[:, 0], pc2=scores_all[:, 1])
    return {
        "sessions": out_sessions, "curves": curves, "labels": labels, "flagged": flagged,
        "k": k, "picks": picks, "picks_all": picks_all, "k_table": k_table, "k_table_all": k_table_all,
        "kmeans": res, "centroids": cents, "scaler": scaler, "Z": Z, "keep": keep,
        "linkage": Zlink, "ward_labels": ward_labels, "pca": p, "ari": ari, "confusion": conf,
        "profiles": prof, "avg_curves": avg_curves, "stability": stab_mean,
        "scaling_demo": scaling_demo(sessions), "distance_demo": distance_demo(sessions, Z, cov),
    }
