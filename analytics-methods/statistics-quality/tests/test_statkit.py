import math

import numpy as np
import pytest
from scipy import stats

from statkit import capability, descriptive as ds, inference as inf, montecarlo as mc, pareto, queueing as q, spc, synth
from statkit.cli import main, pipeline_analysis, packing_analysis


# ---------------------------------------------------------------- descriptive & distributions
def test_variance_skew_percentile_match_numpy_scipy():
    x = np.random.default_rng(0).gamma(2.0, 3.0, 257)
    assert ds.variance(x) == pytest.approx(np.var(x, ddof=1), rel=1e-12)
    assert ds.std(x) == pytest.approx(np.std(x, ddof=1), rel=1e-12)
    assert ds.skewness(x) == pytest.approx(stats.skew(x, bias=False), rel=1e-10)
    assert ds.skewness(x, adjusted=False) == pytest.approx(stats.skew(x), rel=1e-10)
    for qq in (0, 5, 37.5, 50, 95, 100):
        assert ds.percentile(x, qq) == pytest.approx(np.percentile(x, qq))
    assert ds.median([3, 1, 2, 10]) == 2.5 and ds.mode([4, 2, 2, 4, 7]) == 2


def test_pmfs_and_pdfs_match_scipy():
    assert ds.binom_pmf(3, 20, 0.12) == pytest.approx(stats.binom.pmf(3, 20, 0.12))
    assert ds.poisson_pmf(7, 4.2) == pytest.approx(stats.poisson.pmf(7, 4.2))
    assert ds.poisson_pmf(400, 380.0) == pytest.approx(stats.poisson.pmf(400, 380.0))
    assert float(ds.expon_pdf(2.5, 0.4)) == pytest.approx(stats.expon.pdf(2.5, scale=1 / 0.4))
    assert float(ds.normal_pdf(5.1, 5.0, 0.04)) == pytest.approx(stats.norm.pdf(5.1, 5.0, 0.04))
    assert float(ds.normal_cdf(-1.7)) == pytest.approx(stats.norm.cdf(-1.7), abs=1e-12)


def test_clt_skew_and_sd_shrink_like_theory():
    rows = ds.clt_demo(np.random.default_rng(3), sizes=(1, 30), reps=20000)
    one, thirty = rows
    assert one["skew_of_means"] == pytest.approx(2.0, abs=0.15)
    assert thirty["skew_of_means"] == pytest.approx(2 / math.sqrt(30), abs=0.08)
    assert thirty["sd_of_means"] == pytest.approx(thirty["sd_theory"], rel=0.03)
    assert thirty["share_within_1.96se"] == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------- inference
def test_wilson_hand_case():
    lo, hi = inf.wilson_ci(8, 40)        # p̂ = 0.2, z = 1.96: hand-worked (0.1050, 0.3476)
    assert lo == pytest.approx(0.10500, abs=2e-4) and hi == pytest.approx(0.34757, abs=2e-4)
    lo0, hi0 = inf.wilson_ci(0, 25)      # never below 0, unlike the Wald interval
    assert lo0 == pytest.approx(0, abs=1e-12) and 0 < hi0 < 0.15


def test_t_ci_welch_and_chi_square_match_scipy():
    a, b = synth.clinic_waits()
    assert inf.t_ci_mean(a) == pytest.approx(stats.t.interval(0.95, len(a) - 1, a.mean(), stats.sem(a)))
    w = inf.welch_t(b, a)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert w["t"] == pytest.approx(ref.statistic) and w["p"] == pytest.approx(ref.pvalue)
    table = synth.defects_by_shift().values
    c = inf.chi2_independence(table)
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    assert c["chi2"] == pytest.approx(chi2) and c["p"] == pytest.approx(p) and c["dof"] == dof
    np.testing.assert_allclose(c["expected"], exp)
    assert c["p"] < 0.05                 # the planted night-shift bruising effect is detected


def test_bootstrap_ci_covers_true_mean_about_95_percent():
    rng = np.random.default_rng(5)
    hits = 0
    reps = 400
    for _ in range(reps):
        x = rng.normal(20, 4, 50)
        lo, hi = inf.bootstrap_ci(x, np.mean, B=800, rng=rng)
        hits += lo <= 20 <= hi
    assert 0.90 <= hits / reps <= 0.98


def test_type_I_rate_and_power_match_formula():
    n = inf.n_for_power(3.0, 10.0, power=0.8)
    assert n == math.ceil(2 * (1.959964 + 0.841621) ** 2 * 100 / 9)
    sim = inf.simulated_error_rates(np.random.default_rng(9), 3.0, 10.0, n, reps=4000)
    assert sim["type_I"] == pytest.approx(0.05, abs=0.012)
    assert sim["power"] == pytest.approx(inf.power_two_sample(3.0, 10.0, n), abs=0.03)
    assert sim["type_II"] == pytest.approx(1 - sim["power"])


# ---------------------------------------------------------------- SPC
def test_xbar_r_limits_match_hand_computation():
    ch = spc.xbar_r([[1, 2, 3, 4], [2, 3, 4, 5], [3, 3, 3, 3]])
    # X̿ = 3, R̄ = 2, n = 4 → A2 = 0.729, D3 = 0, D4 = 2.282, d2 = 2.059
    assert ch["center"] == pytest.approx(3) and ch["rbar"] == pytest.approx(2)
    assert ch["ucl_x"] == pytest.approx(4.458) and ch["lcl_x"] == pytest.approx(1.542)
    assert ch["ucl_r"] == pytest.approx(4.564) and ch["lcl_r"] == 0
    assert ch["sigma_within"] == pytest.approx(2 / 2.059)
    # A2 is 3/(d2·√n) for every tabulated n
    for n, c in spc.CONSTANTS.items():
        assert c["A2"] == pytest.approx(3 / (c["d2"] * math.sqrt(n)), abs=2e-3)


def test_imr_and_p_chart_hand_cases():
    m = spc.imr([10, 12, 11, 15, 13])            # MR = 2, 1, 4, 2 → MR̄ = 2.25
    assert m["mrbar"] == pytest.approx(2.25) and m["center"] == pytest.approx(12.2)
    assert m["ucl"] == pytest.approx(12.2 + 2.66 * 2.25, abs=0.01)
    assert m["ucl_mr"] == pytest.approx(3.267 * 2.25)
    p = spc.p_chart([4, 6], [100, 200])
    pbar = 10 / 300
    assert p["pbar"] == pytest.approx(pbar)
    assert p["ucl"][0] == pytest.approx(pbar + 3 * math.sqrt(pbar * (1 - pbar) / 100))
    assert p["lcl"][0] == 0                      # clipped


def test_individuals_chart_false_alarm_rate_near_0_0027():
    x = np.random.default_rng(2).normal(100, 5, 400_000)
    rate = spc.imr(x)["flag"].mean()
    assert rate == pytest.approx(2 * stats.norm.sf(3), abs=0.0004)


def test_western_electric_rules_fire_on_their_patterns():
    rules = lambda z: {r for _, r in spc.western_electric(z, 0, 1)}
    assert rules([0, 0, 3.4]) == {1}
    assert 2 in rules([2.5, 0.0, 2.4])
    assert 3 in rules([1.5, 1.4, 0.2, 1.6, 1.3])
    assert 4 in rules([0.3] * 8)
    assert spc.western_electric([0.3] * 7 + [-0.3], 0, 1) == []


def test_cusum_arl_matches_siegmund_in_control_and_after_shift():
    rng = np.random.default_rng(4)
    arl0 = spc.cusum_run_lengths(rng, 0.0, runs=3000).mean()
    arl1 = spc.cusum_run_lengths(rng, 1.0, runs=3000).mean()
    assert arl0 == pytest.approx(spc.siegmund_arl(0.0), rel=0.12)     # ~469 samples between false alarms
    assert arl1 == pytest.approx(spc.siegmund_arl(1.0), rel=0.10)     # ~10 samples to catch a 1σ shift
    assert spc.siegmund_arl(0.0) > 400 and spc.siegmund_arl(1.0) < 12


def test_cusum_detects_planted_shifts_within_k_samples():
    w, ch, cu, ew, we, *_ = packing_analysis()
    assert not ch["flag_x"].any()                                   # 3σ X̄ chart misses the 0.6σ drift
    assert synth.SHIFT_AT <= cu["first_signal"] <= synth.SHIFT_AT + 8
    assert synth.SHIFT_AT <= ew["first_signal"] <= synth.SHIFT_AT + 8
    assert all(i >= synth.SHIFT_AT for i, _ in we)
    pipe, pch, pcu, detect = pipeline_analysis()
    assert synth.DROP_AT <= detect["cusum"] <= synth.DROP_AT + 7
    assert detect["i_chart"] is None or detect["i_chart"] > detect["cusum"] + 7
    assert bool(pipe.i_flag[synth.SPIKE_AT])                         # the one-day duplicate load is a 3σ point
    assert not (pcu["c_minus"][:synth.DROP_AT] > pcu["h"]).any()      # no false drop alarm before the drop


def test_ewma_limits_approach_asymptote():
    e = spc.ewma(np.zeros(200), 0.0, 1.0, lam=0.2, L=3)
    assert e["ucl"][-1] == pytest.approx(3 * math.sqrt(0.2 / 1.8))
    assert e["ucl"][0] == pytest.approx(3 * 0.2)                     # first point: L·σ·λ


# ---------------------------------------------------------------- capability
def test_capability_hand_case():
    c = capability.indices(10, 1, 1.25, lsl=7, usl=14)
    assert c["Cp"] == pytest.approx(7 / 6) and c["Cpk"] == pytest.approx(1.0)
    assert c["Pp"] == pytest.approx(7 / 7.5) and c["Ppk"] == pytest.approx(3 / 3.75)
    assert c["ppm_within"] == pytest.approx((stats.norm.cdf(-3) + stats.norm.sf(4)) * 1e6, rel=1e-9)
    assert c["ppm_within"] == pytest.approx(1381.6, abs=0.5)


# ---------------------------------------------------------------- Pareto / Gini
def test_gini_extremes_and_hand_case():
    assert pareto.gini(np.full(50, 7.0)) == pytest.approx(0)
    conc = np.zeros(1000); conc[-1] = 1
    assert pareto.gini(conc) == pytest.approx(0.999)
    assert pareto.gini([1, 2, 3, 4, 5]) == pytest.approx(20 / 75)
    x, y = pareto.lorenz([3, 9, 1, 30, 7, 2])
    assert pareto.gini([3, 9, 1, 30, 7, 2]) == pytest.approx(1 - 2 * np.trapezoid(y, x))


def test_abc_class_thresholds():
    t = pareto.pareto_table(list("vwxyz"), [4, 30, 10, 50, 6])
    assert t["item"].tolist() == ["y", "w", "x", "z", "v"]
    assert t.cum_pct.tolist() == pytest.approx([50, 80, 90, 96, 100])
    assert t.abc.tolist() == ["A", "A", "B", "B", "C"]     # x starts exactly at 80% → B; v starts at 96% → C


# ---------------------------------------------------------------- queueing
def test_mm1_and_erlang_c_closed_forms():
    m = q.mm1(3, 4)
    assert m["Wq"] == pytest.approx(0.75) and m["L"] == pytest.approx(3) and m["Lq"] == pytest.approx(2.25)
    assert q.mmc(3, 4, 1)["Wq"] == pytest.approx(m["Wq"])           # Erlang C with c = 1 is M/M/1
    assert q.erlang_c(2, 1.0) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        q.mm1(4, 4)


def test_littles_law_and_erlang_c_vs_simulation():
    sim = q.simulate(8.4, 4.0, 3, n=200000, rng=np.random.default_rng(8))
    th = q.mmc(8.4, 4.0, 3)
    assert sim["L_time_avg"] == pytest.approx(sim["lambda_W"], rel=0.01)   # L = λW, measured independently
    assert sim["Wq"] == pytest.approx(th["Wq"], rel=0.08)
    assert sim["P_wait"] == pytest.approx(th["P_wait"], abs=0.02)
    assert sim["L_time_avg"] == pytest.approx(th["L"], rel=0.05)


# ---------------------------------------------------------------- Monte Carlo
def test_monte_carlo_mean_and_tornado():
    sim = mc.simulate(n=200000, seed=3)
    E = {k: d.mean() for k, (_, d) in mc.INPUTS.items()}
    analytic = E["boxes_sold"] * (1 - E["spoilage"]) * E["price"] - E["boxes_sold"] * E["unit_cost"] - E["fixed"]
    assert sim.mean() == pytest.approx(analytic, rel=0.02)    # inputs independent → E[profit] is exact
    t = mc.tornado()
    swings = [r["swing"] for r in t]
    assert swings == sorted(swings, reverse=True)
    fixed = next(r for r in t if r["input"] == "fixed")
    assert fixed["swing"] == pytest.approx(40000)             # P10–P90 of U(200k, 250k) is 40k, passes 1:1


# ---------------------------------------------------------------- determinism & CLI
def test_determinism():
    np.testing.assert_array_equal(synth.packing_line(), synth.packing_line())
    assert synth.pipeline_rows().equals(synth.pipeline_rows())
    np.testing.assert_array_equal(mc.simulate(1000), mc.simulate(1000))
    assert inf.bootstrap_ci(np.arange(30.0), rng=np.random.default_rng(1)) == \
        inf.bootstrap_ci(np.arange(30.0), rng=np.random.default_rng(1))


def test_cli_writes_files(tmp_path, capsys):
    main(["run", "--out", str(tmp_path / "out"), "--figures", str(tmp_path / "fig")])
    for f in ("packing_xbar_r.csv", "pipeline_monitor.csv", "p_chart.csv", "capability.csv", "inference.csv",
              "pareto_defects.csv", "abc_skus.csv", "queue_validation.csv", "tornado.csv", "monte_carlo.csv"):
        assert (tmp_path / "out" / f).stat().st_size > 0
    for f in ("xbar_r_chart.png", "pipeline_cusum.png", "pareto_defects.png", "erlang_c_wait.png", "tornado.png", "p_chart.png"):
        assert (tmp_path / "fig" / f).stat().st_size > 10_000
    assert '"packing_first_cusum"' in capsys.readouterr().out
