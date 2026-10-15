import numpy as np
import pandas as pd
import pytest

from optkit import cli, decisions as D, nonlinear as N
from optkit import scenarios as S


@pytest.fixture
def model():
    return N.ExpDemandProfit(**{"a": 1000.0, "b": 0.25, "cost": 2.0})


def test_newton_converges_quadratically(model):
    xs = N.newton(model.grad, model.hess, 3.0, n_iter=5)
    e = np.abs(xs - model.optimum)
    ratios = [e[k + 1] / e[k] ** 2 for k in range(1, 5)]
    assert e[5] < 1e-8
    # e_{k+1} ≈ C·e_k²; for this model the algebra gives C → b = 0.25 (the ratio settles, it doesn't shrink)
    assert all(0.1 < r < 0.3 for r in ratios)
    assert ratios[-1] == pytest.approx(model.b, abs=0.01)


def test_gradient_ascent_converges_linearly(model):
    step = 0.005
    xs = N.gradient_ascent(model.grad, 3.0, step, n_iter=80)
    e = np.abs(xs - model.optimum)
    ratios = e[41:80] / e[40:79]
    rate = abs(1 + step * model.hess(model.optimum))     # theory: |1 + α·f''(p*)|
    assert np.allclose(ratios, rate, atol=2e-3)
    assert 0 < rate < 1


def test_golden_section_finds_known_optimum(model):
    x, widths = N.golden_section(model.profit, 2.0, 20.0, tol=1e-9)
    assert x == pytest.approx(model.optimum, abs=1e-7)
    assert widths[1] / widths[0] == pytest.approx(N.INV_PHI)
    # x² − 4x minimised at 2
    assert N.golden_section(lambda t: t * t - 4 * t, -10, 10, maximize=False)[0] == pytest.approx(2, abs=1e-6)


def test_lagrange_equal_marginal_returns():
    ch = S.CHANNELS
    x, lam = N.lagrange_allocation(ch["a"], ch["s"], ch["budget"])
    mr = N.marginal_return(x, ch["a"], ch["s"])
    assert x.sum() == pytest.approx(ch["budget"])
    assert np.allclose(mr[x > 0], lam)                  # funded channels: equal marginal return = λ
    assert np.all(mr[x == 0] <= lam + 1e-12)            # unfunded: first dollar earns less than λ
    total = N.log_response(x, ch["a"], ch["s"]).sum()
    rng = np.random.default_rng(0)
    for _ in range(500):                                 # no random feasible split does better
        y = rng.dirichlet(np.ones(4)) * ch["budget"]
        assert N.log_response(y, ch["a"], ch["s"]).sum() <= total + 1e-9
    # λ is the value of one more unit of budget
    x2, _ = N.lagrange_allocation(ch["a"], ch["s"], ch["budget"] + 1e-4)
    gain = N.log_response(x2, ch["a"], ch["s"]).sum() - total
    assert gain / 1e-4 == pytest.approx(lam, rel=1e-3)


def test_decision_tree_rollback_hand_case():
    # invest: 0.6 × 100 + 0.4 × (−40) = 44; safe: 30 -> invest, EV 44
    tree = D.decide([("invest", D.chance([("good", 0.6, D.end(100)), ("bad", 0.4, D.end(-40))])),
                     ("safe", D.end(30))])
    assert D.rollback(tree) == pytest.approx(44)
    assert tree["best"] == "invest"


def test_evpi_hand_case_and_criteria():
    info = D.evpi(S.OFFERS, S.OFFER_PROBS)
    # EV: fixed 50, indexed 52.5, spot 56.5; with perfect info 0.3·50 + 0.5·60 + 0.2·110 = 67
    assert info["ev_best"] == pytest.approx(56.5) and info["evpi"] == pytest.approx(10.5)
    c = D.criteria(S.OFFERS, S.OFFER_PROBS)
    assert c.worst_case.idxmax() == "fixed contract"      # maximin
    assert c.max_regret.idxmin() == "price-indexed"        # minimax regret: max regrets 60 / 30 / 35
    rng = np.random.default_rng(1)
    for _ in range(100):                                   # EVPI >= 0 always
        P = pd.DataFrame(rng.normal(0, 10, (3, 4)))
        assert D.evpi(P, rng.dirichlet(np.ones(4)))["evpi"] >= -1e-12


def test_forecast_value_between_zero_and_evpi():
    tree = D.offer_tree(S.OFFERS, S.OFFER_PROBS, S.FORECAST, 0.0)
    ev = D.rollback(tree)
    evsi = ev - tree["options"][0][1]["ev"]
    assert 0 <= evsi <= D.evpi(S.OFFERS, S.OFFER_PROBS)["evpi"]
    perfect = D.offer_tree(S.OFFERS, S.OFFER_PROBS, np.eye(3), 0.0, signal_names=("l", "m", "h"))
    assert D.rollback(perfect) == pytest.approx(67.0)      # a perfect forecast recovers EV with PI


def test_certainty_equivalent_risk_aversion():
    v, p = S.OFFERS.loc["spot market"], S.OFFER_PROBS
    ces = [D.certainty_equivalent(v, p, R) for R in (5, 20, 100, 1e6)]
    assert ces == sorted(ces) and ces[-1] == pytest.approx(56.5, abs=1e-2)
    assert D.certainty_equivalent([7, 7], [0.5, 0.5], 3) == pytest.approx(7)


def test_newsvendor_critical_ratio_equals_brute_force():
    nv = S.NEWSVENDOR
    q, cr = D.newsvendor_poisson(**nv)
    assert cr == pytest.approx(3.5 / 5.5)
    qs = np.arange(20, 61)
    exact = D.newsvendor_exact_curve(nv["lam"], nv["price"], nv["cost"], nv["salvage"], qs)
    sim = D.newsvendor_simulated(nv["lam"], nv["price"], nv["cost"], nv["salvage"], qs)
    assert qs[np.argmax(exact)] == q
    assert qs[np.argmax(sim)] == q


def test_sensitivity_tables():
    tw = D.two_way(S.OFFERS, S.OFFER_PROBS, [0.0, 0.2, 0.5], [40, 50, 70], "fixed contract")
    best = tw.set_index(["p_high", "fixed_offer"]).best_option
    assert best[(0.2, 50)] == "spot market"
    assert best[(0.0, 70)] == "fixed contract"
    assert best[(0.0, 40)] == "price-indexed"
    assert np.allclose(D.with_p_high([0.3, 0.5, 0.2], 0.4), [0.225, 0.375, 0.4])


def test_cli_writes_files_and_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    s1 = cli.run(a, tmp_path / "fig")
    cli.run(b)
    files = sorted(p.name for p in a.glob("*.csv"))
    assert len(files) >= 15 and (a / "summary.json").exists()
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert len(list((tmp_path / "fig").glob("*.png"))) == 7
    assert s1["lp_profit"] == pytest.approx(44000)
