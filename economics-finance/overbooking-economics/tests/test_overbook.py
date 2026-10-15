"""The maths, the estimator, the simulator and the CLI."""
import json

import numpy as np
import pytest
from scipy import stats

from overbook import (marginal_rule_limit, normal_limit, optimal_limit, profit_curve, service_limit,
                      session_table, showup_pmf)
from overbook import cli
from overbook import estimate as est
from overbook.simulate import simulate, simulate_years
from overbook.showup import poisson_pmf

C, P, R, B = 20, 0.85, 150.0, 300.0


@pytest.mark.parametrize("n,p,rho", [(0, .8, 0), (1, .5, 0), (22, .85, 0), (300, .97, 0), (22, .85, .05), (150, .6, .2)])
def test_pmf_sums_to_one_and_has_right_moments(n, p, rho):
    f = showup_pmf(n, p, rho)
    k = np.arange(n + 1)
    assert f.sum() == pytest.approx(1, abs=1e-12) and (f >= 0).all()
    assert f @ k == pytest.approx(n * p, rel=1e-9, abs=1e-12)
    var = n * p * (1 - p) * (1 + (n - 1) * rho)
    assert f @ (k - n * p) ** 2 == pytest.approx(var, rel=1e-8, abs=1e-12)


def test_binomial_pmf_matches_scipy_and_betabinomial_reduces_to_binomial():
    assert np.allclose(showup_pmf(40, .8), stats.binom.pmf(np.arange(41), 40, .8), atol=1e-14)
    a, b = .8 * (1 - .1) / .1, .2 * (1 - .1) / .1
    assert np.allclose(showup_pmf(40, .8, .1), stats.betabinom.pmf(np.arange(41), 40, a, b), atol=1e-13)
    for rho in (1e-4, 1e-6, 1e-8):
        gap = np.abs(showup_pmf(40, .8, rho) - showup_pmf(40, .8)).max()
        assert gap < 50 * rho
    assert optimal_limit(C, P, R, B, 1e-7)["limit"] == optimal_limit(C, P, R, B)["limit"]


def test_expected_profit_formula_matches_brute_force_and_monte_carlo():
    t = session_table(C, P, R, B, 0.0, 30)
    f = stats.binom.pmf(np.arange(25), 24, P)
    brute = sum(fk * (R * min(k, C) - B * max(k - C, 0)) for k, fk in enumerate(f))
    assert t["profit"][24] == pytest.approx(brute, rel=1e-12)
    for rho, lam in ((0.0, None), (0.05, None), (0.0, 22.0)):
        exact = profit_curve(C, P, R, B, rho, lam, 40).profit[23]
        mc = simulate(23, C, P, R, B, rho, lam, sessions=200_000, seed=3)
        assert mc["lo"] <= exact <= mc["hi"]


def test_year_simulation_ci_coverage_near_95_percent():
    exact = profit_curve(C, P, R, B).profit[22]
    sy = simulate_years(22, C, P, R, B, years=300, sessions=500, seed=11)
    cover = (np.abs(sy["means"] - exact) <= sy["half_width"]).mean()
    assert 0.91 <= cover <= 0.98
    assert sy["pooled_lo"] <= exact <= sy["pooled_hi"]


@pytest.mark.parametrize("Cg", [5, 20, 60])
@pytest.mark.parametrize("p", [0.6, 0.85, 0.97])
@pytest.mark.parametrize("ratio", [0.2, 1.0, 4.0])
@pytest.mark.parametrize("rho", [0.0, 0.04])
def test_enumeration_equals_marginal_rule(Cg, p, ratio, rho):
    assert optimal_limit(Cg, p, R, R * ratio, rho)["limit"] == marginal_rule_limit(Cg, p, R, R * ratio, rho)


@pytest.mark.parametrize("Cg", [100, 250, 500])
@pytest.mark.parametrize("p,ratio", [(0.75, 0.5), (0.85, 2.0), (0.95, 5.0), (0.9, 1.0)])
def test_normal_approximation_agrees_for_large_capacity(Cg, p, ratio):
    exact = optimal_limit(Cg, p, R, R * ratio)["limit"]
    assert abs(normal_limit(Cg, p, R, R * ratio) - exact) <= 1
    # beta-binomial is skewed, so the normal shortcut is only close in relative terms there
    od = optimal_limit(Cg, p, R, R * ratio, 0.01)["limit"]
    assert abs(normal_limit(Cg, p, R, R * ratio, 0.01) - od) <= max(2, 0.01 * od)


def test_huge_bump_cost_means_no_overbooking():
    for p in (0.6, 0.85, 0.99):
        assert optimal_limit(C, p, R, 1e9)["limit"] == C


def test_optimal_limit_rises_as_show_rate_falls_and_falls_as_bumps_get_dearer():
    limits = [optimal_limit(C, p, R, B)["limit"] for p in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6)]
    assert all(a <= b for a, b in zip(limits, limits[1:])) and limits[-1] > limits[0]
    by_b = [optimal_limit(C, P, R, b)["limit"] for b in (30, 100, 300, 1000, 5000)]
    assert all(a >= b for a, b in zip(by_b, by_b[1:]))


def test_demand_constrained_profit_and_limit_never_exceed_unconstrained():
    unc = profit_curve(C, P, R, B, lmax=45)
    u = optimal_limit(C, P, R, B, lmax=45)
    for lam in (5, 12, 20, 22, 26, 40):
        con = profit_curve(C, P, R, B, lam=lam, lmax=45)
        # up to the unconstrained optimum, a demand cap can only cost money (beyond it, it saves bumps)
        assert (con.profit[: u["limit"] + 1] <= unc.profit[: u["limit"] + 1] + 1e-9).all()
        o = optimal_limit(C, P, R, B, lam=lam, lmax=45)
        assert o["limit"] <= u["limit"] and o["profit"] <= u["profit"]
        assert con.booked.iloc[30] <= lam + 1e-9
    assert poisson_pmf(22.0, 200).sum() == pytest.approx(1, abs=1e-12)


def test_service_level_limit():
    L = service_limit(C, P, alpha=0.05)
    assert showup_pmf(L, P)[C + 1:].sum() <= 0.05 < showup_pmf(L + 1, P)[C + 1:].sum()


def test_shrinkage_pulls_small_clinics_toward_the_pool():
    hist, truth = est.make_history(seed=7)
    e = est.estimate(hist)
    for _, g in e.groupby("type"):
        m = g.prior_mean.iloc[0]
        assert ((g.shrunk_p - m).abs() <= (g.raw_p - m).abs() + 1e-12).all()
        small, big = g.nsmallest(3, "sessions"), g.nlargest(3, "sessions")
        assert small.weight_on_own_data.max() < big.weight_on_own_data.min()
        assert 0.01 < g.rho_hat.iloc[0] < 0.06            # true within-session correlation is 0.03
    # across several histories shrinkage is closer to the truth on average
    mse = []
    for s in range(8):
        h, t = est.make_history(seed=50 + s)
        x = est.estimate(h).merge(t, on=["clinic", "type"])
        mse.append((((x.raw_p - x.true_p) ** 2).mean(), ((x.shrunk_p - x.true_p) ** 2).mean()))
    raw, shr = np.mean(mse, axis=0)
    assert shr < raw


def test_regret_nonnegative_and_zero_at_true_rate():
    ph = np.round(np.arange(0.70, 0.981, 0.01), 2)
    rc = est.regret_curve(C, P, R, B, ph)
    assert (rc.regret >= -1e-9).all()
    assert rc.loc[np.isclose(rc.p_hat, P), "regret"].iloc[0] == pytest.approx(0, abs=1e-9)
    assert rc.regret.iloc[0] > 0 and rc.regret.iloc[-1] > 0


def test_deterministic():
    a, _ = est.make_history(seed=3)
    b, _ = est.make_history(seed=3)
    assert a.equals(b)
    assert simulate(22, C, P, R, B, 0.05, 22, 1000, seed=9) == simulate(22, C, P, R, B, 0.05, 22, 1000, seed=9)


def test_cli_run_and_export_write_files(tmp_path):
    cli.main(["run", "--out", str(tmp_path / "out"), "--figures", str(tmp_path / "fig"), "--replications", "2"])
    for f in ("profit_curve.csv", "method_agreement.csv", "sensitivity.csv", "demand_constrained.csv",
              "simulation_check.csv", "show_rate_estimates.csv", "regret_by_clinic.csv", "regret_curve.csv"):
        assert (tmp_path / "out" / f).stat().st_size > 0
    assert len(list((tmp_path / "fig").glob("*.png"))) == 4
    cli.main(["export", "--out", str(tmp_path / "data.json")])
    d = json.loads((tmp_path / "data.json").read_text())
    assert d["defaults"]["C"] == 20 and len(d["presets"]) >= 4
    for pr in d["presets"]:
        assert pr["python"]["limit"] == optimal_limit(pr["C"], pr["p"], pr["r"], pr["b"], pr["rho"], pr["lam"] or None)["limit"]
