import numpy as np
import pandas as pd
import pytest

from roikit import abtest, allocate, finance, kpi, mmm, pipeline


# ---------------------------------------------------------------- response curves
def test_adstock_hand_case():
    # a_t = x_t + 0.5 a_{t-1}: 100, 50, 25, 50 + 12.5
    assert np.allclose(mmm.adstock([100, 0, 0, 50], 0.5), [100, 50, 25, 62.5])
    assert np.allclose(mmm.adstock([3, 4], 0.0), [3, 4])                 # no carry-over


def test_hill_properties():
    K, s = 1200.0, 1.6
    a = np.linspace(0, 20000, 500)
    h = mmm.hill(a, K, s)
    assert h[0] == 0 and np.all(np.diff(h) > 0) and np.all(h < 1)       # starts at 0, rises, saturates below 1
    assert mmm.hill(K, K, s) == pytest.approx(0.5)                      # half saturation at K


def test_parameter_recovery(result):
    media = result["media"]
    B = np.linspace(0.05, 0.6, 50) * media
    for c in result["chans"]:
        fit, true = result["params"][c], result["truth"][c]
        assert abs(fit["lam"] - true["lam"]) < 0.08, c
        g_fit = allocate.gross_profit(B, fit, 78, 0.55)
        g_true = allocate.gross_profit(B, true, 78, 0.55)
        assert np.max(np.abs(g_fit - g_true)) / g_true.max() < 0.10, c   # curve within 10% over the planning range


# ---------------------------------------------------------------- allocation
def test_allocation_budget_and_bounds(result):
    opt, lo, hi = result["opt"], result["lo"], result["hi"]
    assert opt.sum() == pytest.approx(result["media"], rel=1e-6)
    assert np.all(opt >= lo - 1e-6) and np.all(opt <= hi + 1e-6)


def test_kkt_equal_marginal_roi(result):
    k = result["kkt"]
    assert k["conditions_hold"]
    assert k["spread"] / k["mu"] < 0.01                                  # interior marginal ROIs equal within 1%
    # analytic derivative matches a finite difference
    p = result["params"]["paid_social"]
    B, h = 150000.0, 1.0
    fd = (allocate.gross_profit(B + h, p, 78, 0.55) - allocate.gross_profit(B - h, p, 78, 0.55)) / (2 * h)
    assert allocate.marginal_roi(B, p, 78, 0.55) == pytest.approx(fd, rel=1e-5)


def test_optimal_beats_naive_and_random(result):
    P, chans = result["params"], result["chans"]
    total = lambda x: sum(allocate.gross_profit(x[i], P[c], 78, 0.55) for i, c in enumerate(chans))
    best = total(result["opt"])
    assert best >= total(result["naive"])
    f = result["financials"]
    assert f["optimal (truth)"]["gross_profit"] > f["naive (truth)"]["gross_profit"]   # holds on the true curves too
    rng = np.random.default_rng(0)
    lo, hi, media = result["lo"], result["hi"], result["media"]
    n = 0
    while n < 500:                                                       # random feasible allocations never win
        x = lo + rng.dirichlet(np.ones(4)) * (media - lo.sum())
        if np.all(x <= hi):
            assert total(x) <= best + 1e-6
            n += 1


# ---------------------------------------------------------------- finance
def test_npv_irr_hand_case():
    cash = [-100, 60, 60]                                                # months 1..3
    assert finance.npv(cash, 0.10) == pytest.approx(-100 / 1.1 + 60 / 1.1 ** 2 + 60 / 1.1 ** 3)
    r = finance.irr(cash)
    assert r == pytest.approx(0.130662, abs=1e-6)                        # -100 + 60/(1+r) + 60/(1+r)^2 = 0 (per-period form)
    assert finance.npv(cash, r) == pytest.approx(0, abs=1e-8)
    assert finance.payback_month([-100, 30, 30, 50]) == 4
    assert finance.payback_month([-100, 10]) is None
    assert finance.monthly_rate(0.10) == pytest.approx(1.1 ** (1 / 12) - 1)


def test_plan_cash_flows_consistent(result, cfg):
    cf = result["cash_flows"]
    s = finance.summary(cf, cfg["finance"]["annual_discount"])
    assert cf.spend.sum() == pytest.approx(cfg["budget"]["total"])        # every dollar of the $500k is scheduled
    assert finance.npv(cf.net, s["irr_monthly"]) == pytest.approx(0, abs=1e-4)
    assert s["roi_multiple"] == pytest.approx(cf.gross_profit.sum() / cfg["budget"]["total"])


def test_monte_carlo_summary(result):
    m = result["mc_summary"].set_index(["allocation", "metric"])
    o = m.loc[("optimal", "roi_multiple")]
    assert o.p10 < o.p50 < o.p90 and 0 <= o.prob_above_breakeven <= 1 and o.draws == 5000
    true_roi = result["financials"]["optimal (truth)"]["roi_multiple"]
    assert o.p10 < true_roi < o.p90                                       # the truth sits inside the 80% band
    assert o.prob_above_breakeven > m.loc[("naive", "roi_multiple")].prob_above_breakeven


# ---------------------------------------------------------------- A/B testing
def test_sample_size_known_value():
    # classic textbook case: 10% -> 12%, alpha 0.05 two-sided, 80% power -> 3,841 per arm
    assert abtest.sample_size(0.10, 0.12, 0.05, 0.80) == 3841
    # and the MDE is its inverse
    d = abtest.mde(0.10, 3841)
    assert d == pytest.approx(0.02, abs=2e-4)


def test_ci_coverage_close_to_95():
    cov = abtest.ci_coverage(0.04, 0.046, 20000, 20000, np.random.default_rng(1))
    assert 0.94 < cov < 0.96


def test_peeking_inflates_false_positives():
    rng = np.random.default_rng(2)
    one, _ = abtest.peeking_fpr(1, 8000, 0.02, 6000, rng)
    many, bonf = abtest.peeking_fpr(10, 8000, 0.02, 6000, rng)
    assert abs(one - 0.05) < 0.012                                        # single look keeps alpha
    assert many > 0.12                                                    # ten looks: false positives at least double
    assert bonf <= 0.05 + 0.01                                            # correction brings it back under alpha


def test_ab_analysis_matches_hand_calc():
    r = abtest.analyse(200, 10000, 260, 10000)
    se = np.sqrt(0.02 * 0.98 / 10000 + 0.026 * 0.974 / 10000)
    assert r["abs_lift"] == pytest.approx(0.006)
    assert r["ci_low"] == pytest.approx(0.006 - 1.959964 * se, abs=1e-6)
    assert r["p_value"] < 0.01


# ---------------------------------------------------------------- KPI + determinism
def test_kpi_rules(result):
    assert kpi.classify(1.30) == "exceed" and kpi.classify(1.0) == "meet" and kpi.classify(0.70) == "below"
    k = result["kpi"]
    assert len(k) == 12 and set(k.status) <= {"exceed", "meet", "below"}
    assert (k[k.status == "exceed"].ratio_to_plan >= 1.15).all()


def test_deterministic(cfg, result):
    again = pipeline.run(cfg)
    pd.testing.assert_frame_equal(again["allocation"], result["allocation"])
    pd.testing.assert_frame_equal(again["mc_summary"], result["mc_summary"])
    assert again["ab_result"] == result["ab_result"]
