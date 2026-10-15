import numpy as np
import pandas as pd
import pytest

from vintagekit import backtest, cli, curves, plan, project, reference, synth
from vintagekit.synth import TERM


@pytest.fixture(scope="module")
def book():
    return cli.analyse(seed=7)


def test_amortisation_schedule_sums_to_the_balance():
    s = curves.schedule(10_000, 0.14, 36)
    assert s.principal.sum() == pytest.approx(10_000, abs=1e-6)
    assert s.balance.iloc[-1] == pytest.approx(0, abs=1e-6)
    assert np.allclose(s.payment, s.interest + s.principal)
    # closed-form remaining balance agrees with the month-by-month table
    assert np.allclose(curves.remaining_balance(10_000, 0.14, 36, s.month), s.balance, atol=1e-6)


def test_cohort_balance_identity(book):
    """Original balance = defaults + prepayments + scheduled principal + what is still outstanding."""
    c = book["curves"]
    g = c.groupby(["segment", "vintage"]).agg(orig=("orig_balance", "first"), d=("default_bal", "sum"),
                                              p=("prepay_bal", "sum"), s=("sched_principal", "sum"),
                                              b=("balance", "last"))
    assert np.allclose(g.orig, g.d + g.p + g.s + g.b, rtol=1e-9)


def test_cumulative_curves_are_monotone(book):
    c = book["curves"].sort_values(["segment", "vintage", "mob"])
    g = c.groupby(["segment", "vintage"])
    assert (g.cum_default_pct.diff().dropna() >= 0).all()
    assert (g.cum_prepay_pct.diff().dropna() >= 0).all()
    assert (g.balance_pct.diff().dropna() <= 1e-9).all()


def test_reference_percentiles_are_ordered_and_monotone(book):
    ref = book["ref"]
    assert (ref.p25 <= ref.p50 + 1e-12).all() and (ref.p50 <= ref.p75 + 1e-12).all()
    for _, r in ref.groupby("segment"):
        for col in ("p25", "p50", "p75"):
            assert (np.diff(r[col]) >= 0).all()
    assert (ref.n_vintages == synth.N_VINTAGES - TERM + 1).all()


def test_credibility_weight_rules():
    k = 1_000.0
    assert project.credibility_weight(0, k) == 0
    e = np.array([1, 10, 100, 1e3, 1e4, 1e6, 1e9])
    z = project.credibility_weight(e, k)
    assert (np.diff(z) > 0).all() and z[-1] > 0.999999
    assert project.credibility_weight(k, k) == pytest.approx(0.5)      # half weight when exposure = k
    assert (project.credibility_weight(e, np.inf) == 0).all()           # no between-vintage signal -> no weight


def test_buhlmann_straub_recovers_known_k():
    """theta_i ~ N(1, tau^2), x_ij | theta_i ~ N(theta_i, sigma^2 / w_ij): k should come back as sigma^2 / tau^2."""
    rng = np.random.default_rng(0)
    tau, sigma = 0.15, 3.0
    w = rng.uniform(50, 400, (600, 24))
    theta = rng.normal(1, tau, 600)
    x = theta[:, None] + rng.normal(0, 1, w.shape) * sigma / np.sqrt(w)
    bs = project.buhlmann_straub(x, w)
    assert bs["k"] == pytest.approx(sigma ** 2 / tau ** 2, rel=0.15)
    assert bs["mu"] == pytest.approx(1, abs=0.02)


def test_chain_ladder_reproduces_a_known_triangle():
    """Every vintage follows the same development pattern up to scale, so factors and projections are exact."""
    g = np.cumsum(np.arange(1, 7, dtype=float))            # cumulative pattern at MOB 1..6
    scale = np.array([5.0, 3.0, 8.0, 2.0])
    tri = pd.DataFrame([[s * g[m] if m < 6 - i else np.nan for m in range(6)] for i, s in enumerate(scale)],
                       columns=range(1, 7))
    f = project.dev_factors(tri, term=6)
    assert np.allclose(f[1:6], g[1:] / g[:-1])
    for i, s in enumerate(scale):
        n = 6 - i
        assert project.chain_ladder(tri.loc[i, n], n, f, term=6) == pytest.approx(s * g[-1])


def test_projection_never_below_current_and_band_brackets_it(book):
    p = book["proj"]
    for col in ("proj_credibility", "proj_chain_ladder", "proj_reference", "proj_low", "proj_high"):
        assert (p[col] >= p.cum_default_pct - 1e-12).all(), col
    assert (p.proj_low <= p.proj_credibility + 1e-12).all() and (p.proj_credibility <= p.proj_high + 1e-12).all()
    seasoned = p[p.mob == TERM]
    assert np.allclose(seasoned.proj_credibility, seasoned.cum_default_pct)


def test_gap_decomposition_adds_up(book):
    p = book["proj"]
    assert np.allclose(p.pace_gap_pct + p.future_gap_pct, p.lifetime_gap_pct)
    el = book["el"]
    assert el.loc["Portfolio", "expected_loss"] == pytest.approx(el.drop("Portfolio").expected_loss.sum())
    assert np.allclose(p.expected_loss, p.pd_remaining * p.lgd * p.ead)


def test_planted_downturn_is_flagged_behind(book):
    p = book["proj"]
    shocked = p.vintage >= synth.N_VINTAGES - synth.SHOCK_VINTAGES
    assert (p[shocked & (p.mob >= 6)].status == "behind").all()
    assert (p[~shocked].status == "behind").mean() < 0.10


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_downturn_detection_holds_across_seeds(seed):
    p = cli.analyse(seed)["proj"]
    shocked = p.vintage >= synth.N_VINTAGES - synth.SHOCK_VINTAGES
    hit = (p[shocked & (p.mob >= 6)].status == "behind").mean()
    false = (p[~shocked].status == "behind").mean()
    assert hit >= 0.75 and hit > 5 * false


def test_backtest_beats_naive_and_credibility_beats_chain_ladder_early(book):
    _, s = backtest.run(book["curves"], cuts=[3, 6, 12, 24])
    m = s.pivot(index="cut_mob", columns="method", values="mape")
    assert (m["credibility blend"] < 0.5 * m["naive (no further defaults)"]).all()
    assert m.loc[3, "credibility blend"] < m.loc[3, "chain ladder"]
    assert (m["credibility blend"].diff().dropna() < 0).all()        # error falls as the vintage matures


def test_determinism_with_seed():
    a, b = synth.simulate(seed=11), synth.simulate(seed=11)
    pd.testing.assert_frame_equal(a, b)
    assert not synth.simulate(seed=12).defaults.equals(a.defaults)


def test_cli_writes_files(tmp_path):
    cli.main(["run", "--out", str(tmp_path / "out"), "--figures", str(tmp_path / "fig")])
    for f in ("vintage_projection.csv", "backtest.csv", "segment_reference.csv"):
        assert (tmp_path / "out" / f).stat().st_size > 0
    assert len(list((tmp_path / "fig").glob("*.png"))) == 4
    vp = pd.read_csv(tmp_path / "out" / "vintage_projection.csv")
    assert set(vp.status) <= {"on-pace", "watch", "behind"} and len(vp) == 4 * synth.N_VINTAGES
