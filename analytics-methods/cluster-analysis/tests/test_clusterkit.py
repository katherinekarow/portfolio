import numpy as np
import pandas as pd
import pytest

from clusterkit import distance, hierarchy, kmeans as km, pca, profiling, scaling, selection, synth, validation
from clusterkit.cli import main
from clusterkit.pipeline import run_analysis


@pytest.fixture(scope="module")
def analysis():
    return run_analysis(bootstrap_B=10, gap_B=8)


def blobs(seed=0, n=40, sep=6.0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [sep, 0], [0, sep], [sep, sep]], float)
    X = np.vstack([c + rng.normal(0, 1, (n, 2)) for c in centres])
    y = np.repeat(np.arange(4), n)
    return X, y


# ---------- scaling ----------
def test_standardise_mean0_sd1_and_inverse():
    rng = np.random.default_rng(1)
    X = rng.normal([5, 200, -3], [1, 50, 0.1], size=(500, 3))
    s = scaling.Standardiser().fit(X)
    Z = s.transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-12)
    assert np.allclose(Z.std(0), 1, atol=1e-12)
    assert np.allclose(s.inverse_transform(Z), X)
    M = scaling.minmax(X)
    assert np.allclose(M.min(0), 0) and np.allclose(M.max(0), 1)


def test_unscaled_distance_dominated_by_capacity(analysis):
    demo = analysis["scaling_demo"].set_index("scaling")
    assert demo.loc["raw units", "capacity_share_of_sq_distance"] > 0.95
    assert demo.loc["z-score", "capacity_share_of_sq_distance"] < 0.3
    assert demo.loc["raw units", "kmeans_ari_vs_truth"] < 0.1
    assert demo.loc["z-score", "kmeans_ari_vs_truth"] > 0.9


# ---------- distances ----------
def test_distance_hand_values():
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert distance.euclidean(a, b) == pytest.approx(5.0)
    assert distance.manhattan(a, b) == pytest.approx(7.0)
    assert distance.cosine([1, 0], [0, 1]) == pytest.approx(1.0)
    assert distance.cosine([1, 2], [2, 4]) == pytest.approx(0.0, abs=1e-12)
    # identity covariance -> Mahalanobis = Euclidean; diagonal cov rescales each axis
    assert distance.mahalanobis(a, b, np.eye(2)) == pytest.approx(5.0)
    assert distance.mahalanobis(a, b, np.diag([9.0, 16.0])) == pytest.approx(np.sqrt(2))
    X = np.random.default_rng(0).normal(size=(6, 3))
    D = distance.pairwise_euclidean(X)
    assert D[1, 4] == pytest.approx(distance.euclidean(X[1], X[4]))


# ---------- k-means ----------
def test_inertia_non_increasing_across_iterations():
    X, _ = blobs(sep=3.0)
    rng = np.random.default_rng(3)
    for _ in range(5):
        C0 = X[rng.choice(len(X), 4, replace=False)]
        res = km.lloyd(X, C0)
        h = np.array(res.history)
        assert np.all(np.diff(h) <= 1e-9)
        assert res.inertia == pytest.approx(h[-1])


def test_kmeans_recovers_planted_clusters(analysis):
    assert analysis["ari"]["kmeans_vs_truth"] > 0.9
    X, y = blobs()
    assert validation.adjusted_rand_index(km.kmeans(X, 4, seed=1).labels, y) > 0.95


def test_kmeans_pp_restarts_never_worse_than_single_run():
    X, _ = blobs(sep=3.5, seed=5)
    single = km.kmeans(X, 4, n_init=1, seed=11).inertia
    best = km.kmeans(X, 4, n_init=10, seed=11).inertia
    assert best <= single + 1e-9


# ---------- choosing k ----------
def test_silhouette_hand_example():
    # 1-D: cluster A = {0, 2}, cluster B = {10}. Point 0: a = 2, b = 10 -> 0.8.
    # Point 2: a = 2, b = 8 -> 0.75. Singleton: 0 by convention. Mean = 1.55 / 3.
    X = np.array([[0.0], [2.0], [10.0]])
    s = selection.silhouette_samples(X, [0, 0, 1])
    assert np.allclose(s, [0.8, 0.75, 0.0])
    assert selection.silhouette_score(X, [0, 0, 1]) == pytest.approx(1.55 / 3)


def test_silhouette_matches_sklearn_if_available():
    metrics = pytest.importorskip("sklearn.metrics")
    X, y = blobs(sep=3.0)
    assert selection.silhouette_score(X, y) == pytest.approx(metrics.silhouette_score(X, y), abs=1e-10)


def test_chosen_k_equals_planted_k(analysis):
    assert analysis["k"] == 4
    assert analysis["picks"]["silhouette"] == 4
    assert analysis["picks"]["gap"] == 4


# ---------- validation ----------
def test_ari_identical_relabelled_and_random():
    a = np.repeat([0, 1, 2], 50)
    assert validation.adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert validation.adjusted_rand_index(a, (a + 1) % 3) == pytest.approx(1.0)   # names don't matter
    rng = np.random.default_rng(0)
    vals = [validation.adjusted_rand_index(rng.integers(0, 4, 400), rng.integers(0, 4, 400)) for _ in range(50)]
    assert abs(np.mean(vals)) < 0.01 and max(abs(v) for v in vals) < 0.03


def test_ari_hand_case_and_sklearn():
    # contingency [[2,0],[1,1]]: index=1, sa=1+1=2, sb=C(3,2)+0=3, expected=2*3/6=1, max=2.5 -> ARI=0
    assert validation.adjusted_rand_index([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.0)
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 5, 200), rng.integers(0, 3, 200)
    b[:120] = a[:120] % 3
    assert validation.adjusted_rand_index(a, b) == pytest.approx(metrics.adjusted_rand_score(a, b))


def test_confusion_and_bootstrap_stability(analysis):
    conf = analysis["confusion"]
    assert conf.to_numpy().sum() == 292
    M = conf.to_numpy()
    assert len(set(M.argmax(1))) == 4            # each true segment maps to its own cluster
    assert M.max(1).sum() >= 290
    assert np.all(analysis["stability"] > 0.9)


# ---------- Ward ----------
def test_ward_matches_scipy():
    sch = pytest.importorskip("scipy.cluster.hierarchy")
    X = np.random.default_rng(4).normal(size=(60, 3))
    Z = hierarchy.ward_linkage(X)
    Zs = sch.linkage(X, method="ward")
    assert np.allclose(Z[:, 2], Zs[:, 2])
    assert np.allclose(Z[:, 3], Zs[:, 3])
    for k in (2, 3, 5):
        ref = sch.fcluster(Zs, k, criterion="maxclust")
        assert validation.adjusted_rand_index(hierarchy.cut_tree(Z, k), ref) == pytest.approx(1.0)


def test_ward_hand_case():
    # points 0, 1, 5 on a line: merge {0,1} at sqrt(1)=1; then d^2({0,1},5) by
    # Lance-Williams = [(1+1)*25 + (1+1)*16 - 1*1]/3 = 81/3 = 27 -> height sqrt(27).
    # Check: ESS increase = (2*1/3)*(0.5-5)^2 = 13.5 and sqrt(2*13.5) = sqrt(27).
    Z = hierarchy.ward_linkage(np.array([[0.0], [1.0], [5.0]]))
    assert Z[0, :2].tolist() == [0, 1] and Z[0, 2] == pytest.approx(1.0)
    assert Z[1, :2].tolist() == [2, 3] and Z[1, 2] == pytest.approx(np.sqrt(27))
    assert Z[1, 3] == 3


# ---------- PCA ----------
def test_pca_variance_ratio_and_orthonormal():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    p = pca.pca(X)
    assert p.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert np.all(np.diff(p.explained_variance_ratio) <= 0)
    assert np.allclose(p.components @ p.components.T, np.eye(5), atol=1e-10)
    vals, vecs = pca.pca_eig(X)
    assert np.allclose(p.explained_variance, vals)
    assert np.allclose(np.abs(np.sum(vecs * p.components, 1)), 1.0)
    assert np.allclose(p.scores, (X - X.mean(0)) @ p.components.T)
    # total variance is preserved
    assert p.explained_variance.sum() == pytest.approx(X.var(0, ddof=1).sum())


# ---------- profiling / outliers ----------
def test_planted_outliers_flagged(analysis):
    s = analysis["sessions"]
    planted = s.is_outlier.to_numpy()
    flagged = analysis["flagged"]
    assert flagged[planted].all()
    assert flagged[~planted].mean() < 0.05


def test_robust_z_and_profiles_in_original_units(analysis):
    assert profiling.robust_z([1, 2, 3, 4, 100])[2] == pytest.approx(0.0)
    prof = analysis["profiles"].set_index("matches_true_segment")
    assert set(prof.index) == {v["name"] for v in synth.SEGMENTS.values()}
    # planted fill rates and shapes come back in original units
    assert prof.loc["early planners", "fill_rate"] == pytest.approx(0.96, abs=0.04)
    assert prof.loc["under-filled, erratic", "fill_rate"] == pytest.approx(0.52, abs=0.04)
    assert prof.loc["late rush", "share_30"] < prof.loc["steady", "share_30"] < prof.loc["early planners", "share_30"]


# ---------- determinism / CLI ----------
def test_determinism():
    a, ca = synth.make_sessions(seed=3)
    b, cb = synth.make_sessions(seed=3)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ca, cb)
    X = scaling.zscore(a[synth.FEATURES].to_numpy())
    r1, r2 = km.kmeans(X, 4, seed=9), km.kmeans(X, 4, seed=9)
    assert np.array_equal(r1.labels, r2.labels) and r1.inertia == r2.inertia
    assert len(a) == 300 and a.is_outlier.sum() == 8


def test_cli_writes_files(tmp_path, capsys):
    main(["run", "--out", str(tmp_path / "out"), "--figures", str(tmp_path / "fig"),
          "--bootstrap", "3", "--gap-refs", "3"])
    for f in ["sessions_clustered.csv", "choose_k.csv", "segment_profiles.csv", "segment_curves.csv",
              "confusion_matrix.csv", "scaling_demo.csv", "distance_demo.csv", "ward_linkage.csv", "summary.csv"]:
        assert (tmp_path / "out" / f).stat().st_size > 0
    for f in ["choose_k.png", "pca_clusters.png", "segment_curves.png", "dendrogram.png"]:
        assert (tmp_path / "fig" / f).stat().st_size > 10_000
    assert '"k": 4' in capsys.readouterr().out
