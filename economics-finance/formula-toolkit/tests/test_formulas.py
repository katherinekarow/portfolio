"""Each formula is checked against a hand calculation, an independent method, or the property it must have."""
import math
import random

import numpy as np
import pytest

import formulakit.formulas as F
from formulakit import REGISTRY
from formulakit.cli import docs, export, main

# ------------------------------------------------------------------ finance


def test_compound_and_inverse():
    fv = F.future_value(1000, 0.05, 10)
    assert fv == pytest.approx(1628.894627, abs=1e-6)
    assert F.present_value(fv, 0.05, 10) == pytest.approx(1000)
    assert F.growth_rate(1000, fv, 10) == pytest.approx(0.05)


def test_npv_hand_case():
    # -1000 + 300/1.08 + 400/1.08^2 + 500/1.08^3 + 200/1.08^4
    hand = -1000 + 277.7778 + 342.9355 + 396.9161 + 147.0060
    assert F.npv(0.08, [-1000, 300, 400, 500, 200]) == pytest.approx(hand, abs=1e-3)


@pytest.mark.parametrize("flows", [[-1000, 300, 400, 500, 200], [-500, 0, 0, 800], [-100, 110], [-2000, 100, 100, 100, 2500]])
def test_irr_zeroes_npv(flows):
    r = F.irr(flows)
    assert abs(F.npv(r, flows)) < 1e-6
    if flows == [-100, 110]:
        assert r == pytest.approx(0.10)


def test_irr_none_without_sign_change():
    assert F.irr([100, 100, 100]) is None


def test_payback_interpolates():
    assert F.payback([-1000, 300, 400, 500, 200]) == pytest.approx(2.6)
    assert F.payback([-1000, 100, 100]) is None


def test_amortization_adds_up():
    rows = F.amortization(20000, 0.01, 36)
    assert sum(r["principal"] for r in rows) == pytest.approx(20000, abs=1e-6)
    assert rows[-1]["balance"] == pytest.approx(0, abs=1e-6)
    assert rows[0]["interest"] == pytest.approx(200)
    assert F.loan_payment(20000, 0.01, 36) == pytest.approx(664.2861, abs=1e-4)
    assert F.loan_payment(1200, 0, 12) == 100


def test_effective_rate():
    assert F.effective_rate(0.12, 12) == pytest.approx(0.126825, abs=1e-6)
    assert F.effective_rate(0.12, 1) == pytest.approx(0.12)

# ------------------------------------------------------------------ macro


def test_macro_identities():
    assert F.money_multiplier(0.1) == pytest.approx(10)
    assert F.reserve_ratio(F.money_multiplier(0.08)) == pytest.approx(0.08)
    real = F.real_rate(0.07, 0.03)
    assert real == pytest.approx(0.038835, abs=1e-6)
    assert F.nominal_rate(real, 0.03) == pytest.approx(0.07)
    assert abs(real - (0.07 - 0.03)) < 0.002          # the shortcut is close for small rates

# ------------------------------------------------------------------ demand


def test_arc_elasticity_is_symmetric():
    a = F.arc_elasticity(20, 1000, 22, 880)
    b = F.arc_elasticity(22, 880, 20, 1000)
    assert a == pytest.approx(b) and a == pytest.approx(-1.3404, abs=1e-4)


def test_fit_elasticity_recovers_truth():
    rng = random.Random(3)
    prices = [15 + i for i in range(12)]
    exact = [5000 * p ** -1.4 for p in prices]
    assert F.fit_elasticity(prices, exact) == pytest.approx(-1.4)
    noisy = [q * math.exp(rng.gauss(0, 0.02)) for q in exact]
    assert F.fit_elasticity(prices, noisy) == pytest.approx(-1.4, abs=0.1)
    assert F.fit_elasticity(prices, noisy) == pytest.approx(np.polyfit(np.log(prices), np.log(noisy), 1)[0])


def test_demand_and_revenue_change_agree():
    q1 = F.demand_at_price(1000, 20, 22, -1.3)
    rev = (22 * q1) / (20 * 1000) - 1
    assert F.revenue_change(0.10, -1.3) == pytest.approx(rev)
    assert F.revenue_change(0.10, -0.5) > 0 > F.revenue_change(0.10, -1.5)    # inelastic vs elastic


def test_optimal_price_beats_grid():
    c, e = 12, -2.5
    p_star = F.optimal_price(c, e)
    profit = lambda p: (p - c) * p ** e
    grid = np.linspace(12.01, 60, 20000)
    assert p_star == pytest.approx(grid[np.argmax([profit(p) for p in grid])], abs=0.01)
    assert (p_star - c) / p_star == pytest.approx(-1 / e)                     # Lerner condition
    assert F.optimal_price(12, -0.8) is None


def test_capacity_price_fills_capacity():
    p = F.capacity_price(1200, 20, 1000, -1.5)
    assert F.demand_at_price(1200, 20, p, -1.5) == pytest.approx(1000)


def test_cross_elasticity_sign():
    assert F.cross_elasticity(500, 540, 30, 33) > 0       # substitutes
    assert F.cross_elasticity(500, 460, 30, 33) < 0       # complements

# ------------------------------------------------------------------ curves


def test_normalisation_and_blend():
    assert F.fill_rate(450, 2000) == pytest.approx(0.225)
    assert F.normalized_position(0.225, 0.30) == pytest.approx(0.75)
    assert F.blend_curves([0.35, 0.28, 0.42], [1200, 600, 200]) == pytest.approx(0.336)


@pytest.mark.parametrize("q", [0, 0.1, 0.25, 0.5, 0.6, 0.75, 1])
def test_percentile_matches_numpy(q):
    v = [0.22, 0.24, 0.19, 0.28, 0.26, 0.31, 0.21]
    assert F.percentile(v, q) == pytest.approx(np.percentile(v, q * 100))


def test_monotone_and_velocity():
    m = F.monotone_curve([0.05, 0.12, 0.11, 0.25, 0.24, 0.40])
    assert m == sorted(m) and m[2] == 0.12
    v = F.velocity([40, 110, 260, 520, 800, 950, 1000])
    assert v[-1] == 1 and v == sorted(v)
    assert F.pace_projection(350, 0.35) == pytest.approx(1000)


@pytest.mark.parametrize("gap", [0, 1, 7, 300, 1234])
def test_residual_allocation_sums_exactly_and_follows_shape(gap):
    ref = [0.30, 0.34, 0.40, 0.48, 0.60, 0.76, 1.00]
    a = F.residual_allocation(gap, ref)
    assert sum(a) == gap and all(x >= 0 for x in a)
    raw = [gap * (b - c) / 0.70 for c, b in zip(ref[:-1], ref[1:])]
    assert all(abs(x - r) < 1 for x, r in zip(a, raw))                         # never more than 1 unit off the exact share
    if gap >= 300:
        assert a == sorted(a)                                                   # more demand arrives near the event

# ------------------------------------------------------------------ targets


def test_gross_net_round_trip():
    s = F.survival_rate(860, 1000)
    g = F.gross_target(1000, s)
    assert g == pytest.approx(1162.79, abs=0.01)
    assert F.expected_net(g, s) == pytest.approx(1000)
    need = F.gross_still_needed(1000, 740, 0.89, 0.93)
    assert 740 * 0.89 + need * 0.93 == pytest.approx(1000)
    assert F.gross_still_needed(1000, 2000, 0.89, 0.93) == 0

# ------------------------------------------------------------------ stats


def test_credibility_limits():
    assert F.credibility_weight(0, 50) == 0
    assert F.credibility_weight(1e9, 50) == pytest.approx(1)
    assert F.credibility_weight(50, 50) == 0.5
    assert F.credibility_blend(1150, 1000, 0) == 1000 and F.credibility_blend(1150, 1000, 1) == 1150


def test_buhlmann_recovers_k():
    rng = np.random.default_rng(5)
    tau, sigma, n = 2.0, 6.0, 30                       # true k = sigma^2 / tau^2 = 9
    groups = [list(rng.normal(rng.normal(50, tau), sigma, n)) for _ in range(400)]
    assert F.buhlmann_k(groups) == pytest.approx(9, rel=0.2)
    assert F.buhlmann_k([[1, 2, 3], [1, 2, 3]]) is None    # no spread between groups -> no credibility constant


def test_shrinkage_behaviour():
    small, big = F.shrink_rate(3, 4, 0.55, 20), F.shrink_rate(300, 400, 0.55, 20)
    assert abs(small - 0.55) < abs(0.75 - 0.55) and abs(big - 0.75) < 0.02


def test_gates_and_wape():
    assert F.variance_gate(12450, 12610, 0.02)["pass"]
    assert not F.variance_gate(12450, 13000, 0.02)["pass"]
    assert F.variance_gate(0, 0.5, 0.02)["change"] == 0.5                       # max(|old|, 1) guards divide-by-zero
    assert F.reconcile_match(1049.5, 1000, 0.05)["pass"] and not F.reconcile_match(1060, 1000, 0.05)["pass"]
    assert F.wape([120, 80, 15, 300], [110, 95, 5, 310]) == pytest.approx(45 / 515)

# ------------------------------------------------------------------ registry, docs, CLI


def test_every_formula_runs_its_example():
    assert len(REGISTRY) >= 35
    for f in REGISTRY.values():
        f.run(f.examples[0])
        assert f.title and f.formula and f.explain


def test_formulas_md_is_current():
    from pathlib import Path
    md = (Path(__file__).resolve().parents[1] / "FORMULAS.md").read_text(encoding="utf-8")
    assert md.strip() == docs().strip(), "regenerate with: python -m formulakit docs --out FORMULAS.md"


def test_cli(capsys):
    main(["npv", "--rate", "0.08", "--flows", "-1000,300,400,500,200"])
    assert "164.64" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["npv", "--bogus", "1"])
    assert len(export()["formulas"]) == len(REGISTRY)
