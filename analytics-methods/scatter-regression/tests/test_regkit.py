import numpy as np
import pandas as pd
import pytest
from scipy import stats

from regkit import analysis as A
from regkit import corr, diagnostics, ols, robust, smooth, synth


@pytest.fixture(scope="module")
def stores():
    return synth.make_stores()


# ---------- correlation ----------
def test_pearson_matches_hand_calculation_and_numpy():
    x, y = np.array([1, 2, 3, 4, 5.0]), np.array([2, 4, 5, 4, 5.0])
    # Σdxdy = 6, Σdx² = 10, Σdy² = 6  ->  r = 6/√60
    assert corr.pearson(x, y) == pytest.approx(6 / np.sqrt(60))
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert corr.pearson(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])
    r, p = stats.pearsonr(a, b)
    assert corr.pearson_p(corr.pearson(a, b), 200) == pytest.approx(p)


def test_spearman_is_pearson_on_ranks_with_ties():
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3.0])
    y = np.array([2, 7, 1, 8, 2, 8, 1, 8, 2, 8.0])
    assert corr.rank([10, 20, 20, 30]).tolist() == [1, 2.5, 2.5, 4]
    assert corr.spearman(x, y) == pytest.approx(corr.pearson(corr.rank(x), corr.rank(y)))
    assert corr.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    # without ties, the classic 1 − 6Σd²/(n(n²−1)) shortcut agrees
    u, v = np.array([1, 2, 3, 4, 5.0]), np.array([2, 1, 4, 3, 5.0])
    d = corr.rank(u) - corr.rank(v)
    assert corr.spearman(u, v) == pytest.approx(1 - 6 * (d @ d) / (5 * 24))


def test_kendall_matches_scipy_tau_b():
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3.0])
    y = np.array([2, 7, 1, 8, 2, 8, 1, 8, 2, 8.0])
    assert corr.kendall(x, y) == pytest.approx(stats.kendalltau(x, y).statistic)


def test_anscombe_same_summary_stats_different_shapes():
    t = A.anscombe_table()
    for col, target in [("mean_x", 9.0), ("mean_y", 7.50), ("var_x", 11.0), ("pearson_r", 0.82),
                        ("intercept", 3.00), ("slope", 0.50)]:
        assert (t[col].round(2) == target).all(), col
    assert t.var_y.round(1).nunique() == 1  # 4.12-4.13
    # ...but rank-based measures and shapes differ
    assert t.spearman_rho.max() - t.spearman_rho.min() > 0.4
    sets = synth.anscombe()
    x2, y2 = sets["II"]
    quad = ols.fit({"x": x2, "x2": x2 ** 2}, y2)
    assert quad.r2 > 0.999                        # set II is an exact parabola
    assert np.unique(sets["IV"][0]).size == 2     # set IV: one point sets the slope
    m4 = ols.fit({"x": sets["IV"][0]}, sets["IV"][1])
    assert diagnostics.leverage(m4).max() == pytest.approx(1.0)


# ---------- OLS ----------
def test_ols_matches_lstsq_both_methods(stores):
    m = A.elasticity_model(stores)
    ref = np.linalg.lstsq(m.X, m.y, rcond=None)[0]
    assert np.allclose(m.beta, ref, atol=1e-9)
    mn = ols.OLS(m.X, m.y, m.names, method="normal")
    assert np.allclose(mn.beta, ref, atol=1e-8)
    assert np.allclose(mn.hat, m.hat, atol=1e-10)


def test_planted_truth_inside_95pct_ci(stores):
    tab = A.coefficients(stores)
    assert tab.true_in_ci.all()
    assert tab.set_index("term").loc["ln_price", "coef"] == pytest.approx(-1.4, abs=0.05)


def test_r2_equals_r_squared_and_adjusted_r2():
    x, y = synth.make_simple(50, seed=3)
    m = ols.fit({"x": x}, y)
    assert m.r2 == pytest.approx(corr.pearson(x, y) ** 2)
    assert m.adj_r2 == pytest.approx(1 - (1 - m.r2) * 49 / 48)
    res = stats.linregress(x, y)
    assert m.beta[1] == pytest.approx(res.slope)
    assert m.se[1] == pytest.approx(res.stderr)
    assert m.pvalues[1] == pytest.approx(res.pvalue, rel=1e-6, abs=1e-300)


def test_interval_coverage_by_simulation():
    c = A.coverage(sims=3000, seed=99)
    for k in ("pi_coverage", "ci_mean_coverage", "slope_ci_coverage"):
        assert 0.935 <= c[k] <= 0.965, (k, c[k])


def test_prediction_interval_wider_than_confidence_interval():
    x, y = synth.make_simple(40, seed=1)
    m = ols.fit({"x": x}, y)
    p = m.predict([[1, 2.0], [1, 5.0], [1, 12.0]])
    ci_w, pi_w = p["ci_hi"] - p["ci_lo"], p["pi_hi"] - p["pi_lo"]
    assert (pi_w > ci_w).all()
    assert ci_w[2] > ci_w[1]     # extrapolating widens the CI


# ---------- log-log & functional forms ----------
def test_log_log_recovers_planted_elasticity(stores):
    f = A.functional_forms(stores).set_index("form")
    assert f.loc["log-log", "price_coef"] == pytest.approx(synth.TRUE["ln_price"], abs=0.06)
    # the level-log model, converted at the mean, gives the same order of magnitude
    assert f.loc["level-log", "elasticity_at_mean"] == pytest.approx(-1.4, abs=0.1)
    # logs remove the multiplicative-noise heteroscedasticity
    assert f.loc["level-level", "breusch_pagan_p"] < 1e-6
    assert f.loc["log-log", "breusch_pagan_p"] > 0.01


# ---------- diagnostics ----------
def test_leverage_sums_to_p_and_matches_hat_matrix(stores):
    m = A.elasticity_model(stores.head(300))
    assert m.hat.sum() == pytest.approx(m.p)
    H = m.X @ np.linalg.inv(m.X.T @ m.X) @ m.X.T
    assert np.allclose(np.diag(H), m.hat)


def test_cooks_distance_flags_planted_points():
    d = A.promo_diagnostics(synth.make_promo())
    assert set(d["top5"][:3]) == set(synth.OUTLIER_IDX)
    assert set(synth.OUTLIER_IDX) <= set(d["flagged"])
    # cross-check one value by deletion: D_i = Σ(ŷ − ŷ₍ᵢ₎)² / (p s²)
    m, i = d["model"], synth.OUTLIER_IDX[0]
    keep = np.arange(m.n) != i
    b_i = np.linalg.lstsq(m.X[keep], m.y[keep], rcond=None)[0]
    manual = ((m.fitted - m.X @ b_i) ** 2).sum() / (m.p * m.sigma2)
    assert d["cooks"][i] == pytest.approx(manual)


def test_vif_high_for_collinear_pair(stores):
    v = A.vif_table(stores).set_index("term")
    assert v.loc["ln_size", "vif_with_traffic"] > 50
    assert v.loc["ln_traffic", "vif_with_traffic"] > 50
    assert v.loc["promo", "vif_with_traffic"] < 1.1
    assert v.loc["ln_size", "vif_without_traffic"] < v.loc["ln_size", "vif_with_traffic"] / 5


def test_breusch_pagan_detects_hetero_not_homo():
    assert A.bp_rejection_rate(200, hetero=True) > 0.9
    assert A.bp_rejection_rate(200, hetero=False) < 0.09   # close to the nominal 5%
    # matches the textbook n·R² of e² on X
    d = synth.make_promo(hetero=True, outliers=False)
    m = ols.fit({"x": d.promo_spend.to_numpy(float)}, d.units.to_numpy(float))
    e2 = m.resid ** 2
    aux_r2 = np.corrcoef(d.promo_spend, e2)[0, 1] ** 2
    assert diagnostics.breusch_pagan(m)["lm"] == pytest.approx(m.n * aux_r2)


def test_hc1_se_exceeds_classical_under_heteroscedasticity():
    d = synth.make_promo(hetero=True, outliers=False)
    m = ols.fit({"x": d.promo_spend.to_numpy(float)}, d.units.to_numpy(float))
    assert m.hc1_se()[1] > m.se[1]
    # HC1 = n/(n−p) · Σ(xᵢ−x̄)²eᵢ² / (Σ(xᵢ−x̄)²)² for the slope
    x = d.promo_spend.to_numpy(float)
    dx = x - x.mean()
    manual = np.sqrt(m.n / (m.n - 2) * (dx ** 2 * m.resid ** 2).sum() / (dx @ dx) ** 2)
    assert m.hc1_se()[1] == pytest.approx(manual)


# ---------- confounding ----------
def test_simpsons_paradox_sign_flip_and_fwl(stores):
    s = A.simpson(stores)
    assert s["pooled"] > 0 and s["pooled_ci"][0] > 0
    assert s["controlled"] < 0 and s["controlled_ci"][1] < 0
    assert all(v < 0 for v in s["within"].values())
    assert s["partial_fwl"] == pytest.approx(s["controlled"], abs=1e-10)   # Frisch-Waugh-Lovell
    assert s["format_fe"] == pytest.approx(-1.4, abs=0.1)


# ---------- robust ----------
def test_theil_sen_beats_ols_with_outliers():
    r = A.robust_compare(synth.make_promo())
    w = r[r.data == "with outliers"].set_index("method").slope_error.abs()
    assert w["Theil-Sen"] < w["OLS"] / 3
    assert w["Huber"] < w["OLS"]
    # brute-force check of the Theil-Sen slope on a small set
    x, y = np.array([1, 2, 3, 4, 10.0]), np.array([1, 2.1, 2.9, 4.2, -5.0])
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(5) for j in range(i + 1, 5)]
    assert robust.theil_sen(x, y)[1] == pytest.approx(np.median(slopes))
    assert robust.theil_sen(x, y)[1] > 0 > ols.fit({"x": x}, y).beta[1]


# ---------- smoothing ----------
def test_local_linear_exact_on_lines_and_recovers_curve(stores):
    x = np.linspace(0, 10, 200)
    gx, gy = smooth.local_linear(x, 3 - 2 * x, frac=0.2)
    assert np.allclose(gy, 3 - 2 * gx)          # local linear reproduces straight lines exactly
    t = A.temperature_shape(stores)
    assert t["rmse_smooth"] < 0.05 < t["rmse_linear"]
    b = smooth.binned(x, x, 10)
    assert b.n.sum() == 200 and (b.n == 20).all()


# ---------- reproducibility & CLI ----------
def test_determinism():
    pd.testing.assert_frame_equal(synth.make_stores(), synth.make_stores())
    pd.testing.assert_frame_equal(synth.make_promo(), synth.make_promo())
    assert not synth.make_stores(seed=1).equals(synth.make_stores(seed=2))


def test_cli_writes_files(tmp_path):
    from regkit.cli import main
    main(["run", "--out", str(tmp_path / "out"), "--figures", str(tmp_path / "fig"), "--sims", "200"])
    csvs = {p.name for p in (tmp_path / "out").glob("*.csv")}
    pngs = {p.name for p in (tmp_path / "fig").glob("*.png")}
    assert {"anscombe.csv", "elasticity_model.csv", "simpson.csv", "influence.csv", "coverage.csv"} <= csvs
    assert {"anscombe.png", "fit_bands.png", "diagnostics.png", "simpson.png", "robust.png", "smoothing.png"} == pngs
    a = pd.read_csv(tmp_path / "out" / "anscombe.csv")
    assert len(a) == 4
