import numpy as np
import pandas as pd
import pytest

from pricekit import elasticity, optimize, synth, validation


@pytest.fixture(scope="module")
def model():
    hist, truth = synth.make_history()
    est = elasticity.estimate(hist).merge(truth, on=["segment", "tier"])
    rec = optimize.recommend(synth.make_lots(), est)
    return hist, est, rec


def test_well_sampled_cells_recover_the_true_elasticity(model):
    _, est, _ = model
    ok = est[est.method == "cell estimate"]
    z = (ok.raw_elasticity - ok.true_elasticity).abs() / ok.raw_se
    assert (z < 3).all() and (z < 2).mean() >= 0.85


def test_thin_cells_are_flagged_and_shrinkage_helps(model):
    _, est, _ = model
    thin = est[est.thin]
    assert (thin.n_obs < elasticity.MIN_OBS).all() and (thin.method == "shrunk to grade pool").all()
    raw_err = (thin.raw_elasticity - thin.true_elasticity).abs().mean()
    shrunk_err = (thin.elasticity - thin.true_elasticity).abs().mean()
    assert shrunk_err < raw_err


def test_backtest_beats_naive(model):
    hist, *_ = model
    bt = elasticity.backtest(hist)
    assert (bt.wape_model * bt.n).sum() < 0.7 * (bt.wape_naive * bt.n).sum()


def test_closed_form_matches_brute_force(model):
    _, est, _ = model
    d = synth.make_lots().merge(est[["segment", "tier", "elasticity"]], on=["segment", "tier"])
    for _, r in d.iterrows():
        p, _ = optimize.best_price(r)
        grid = np.linspace(r.current_price * 0.85, r.current_price * 1.15, 4001)
        best = grid[np.argmax(optimize.contribution(grid, r))]
        assert optimize.contribution(p, r) >= optimize.contribution(best, r) - 1e-6 * abs(optimize.contribution(best, r))


def test_pava_and_ladder():
    assert np.allclose(optimize._pava(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)), [1, 2.5, 2.5, 4])
    p = optimize.enforce_ladder(np.array([950.0, 900.0, 1500.0, 2600.0]), np.array([100, 100, 50, 10]))
    assert all(p[i + 1] >= p[i] * 1.05 - 1e-6 for i in range(3))
    assert 900 < p[0] < 950                                     # meets in the middle (weighted), not just one side


def test_recommendations_respect_ladder_guardrails_and_capacity(model):
    *_, rec = model
    assert rec.reason.str.contains("ladder").any()               # the demo actually exercises the ladder repair
    checks = validation.run(model[1], rec)
    assert checks.passed.all(), checks


def test_validation_queries_catch_bad_output(model):
    _, est, rec = model
    bad = rec.copy()
    i = bad.index[(bad.tier == "organic")][0]
    bad.loc[i, "recommended_price"] = bad.loc[i, "current_price"] * 1.4
    j = bad.index[(bad.tier == "select")][0]
    std = bad.index[(bad.tier == "standard") & (bad.lot_id == bad.loc[j, "lot_id"])][0]
    bad.loc[j, "recommended_price"] = bad.loc[std, "recommended_price"] - 50
    bad_est = est.copy(); bad_est.loc[bad_est.thin, "method"] = "cell estimate"
    res = validation.run(bad_est, bad).set_index("check").passed
    assert not res["price moves stay within guardrails (or are flagged)"]
    assert not res["grade ladder: each tier costs more than the one below"]
    assert not res["thin cells are never used unshrunk"]
    assert not res["revenue impact = demand at new price x new price - current revenue"]


def test_inelastic_cells_rise_and_capacity_floor_holds(model):
    *_, rec = model
    inel = rec[(rec.elasticity < 1) & ~rec.reason.str.contains("ladder")]
    assert (inel.price_change_pct > 0.14).all()
    # where capacity binds (and no guardrail floor or ladder repair moved the price) the lot just sells out
    cap = rec[rec.reason.str.startswith("capacity") & ~rec.reason.str.contains("ladder|floored")]
    assert len(cap) > 10
    assert (cap.units_new <= cap.capacity + 0.05).all() and (cap.units_new >= cap.capacity * 0.98).all()
