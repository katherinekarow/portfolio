import numpy as np
import pytest

from optkit import lp, transport
from optkit import scenarios as S

linprog = pytest.importorskip("scipy.optimize").linprog


def random_lp(rng, with_eq):
    m, n = rng.integers(2, 7), rng.integers(2, 7)
    A = rng.uniform(0, 10, (m, n))
    b = rng.uniform(10, 100, m)
    c = rng.uniform(-2, 10, n)
    # add a ">=" row (needs phase 1): Σx >= 1
    A = np.vstack([A, -np.ones((1, n))])
    b = np.append(b, -1.0)
    Ae = be = None
    if with_eq:
        Ae = rng.uniform(0, 5, (1, n))
        be = np.array([Ae.sum() * 0.5])
    return c, A, b, Ae, be


def test_farm_lp_hand_solution():
    f = S.FARM
    r = lp.simplex(f["profit"], f["A"], f["b"], maximize=True)
    assert np.allclose(r.x, [30, 70]) and r.objective == pytest.approx(44000)
    # land and labour bind: y_land + y_lab = 300, y_land + 3 y_lab = 500 -> (200, 100); water is slack
    assert np.allclose(r.shadow_ub, [200, 100, 0])
    # optimum stays put while 1/3 <= c1/500 <= 1 (slopes of labour and land lines)
    assert np.allclose(r.cost_range[0], [500 / 3, 500])
    # graphical method: best vertex = simplex optimum
    v = lp.vertices_2d(f["A"], f["b"])
    assert np.allclose(v[np.argmax(v @ f["profit"])], r.x)


@pytest.mark.parametrize("seed", range(4))
def test_simplex_matches_linprog_on_random_lps(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    for t in range(60):
        c, A, b, Ae, be = random_lp(rng, with_eq=t % 3 == 0)
        r = lp.simplex(c, A, b, Ae, be, maximize=True)
        s = linprog(-c, A, b, Ae, be, method="highs")
        assert (r.status == "optimal") == (s.status == 0)
        if s.status == 0:
            assert r.objective == pytest.approx(-s.fun, abs=1e-6)
            checked += 1
    assert checked > 30


def test_duals_complementary_slackness_and_strong_duality():
    rng = np.random.default_rng(42)
    for _ in range(80):
        c, A, b, Ae, be = random_lp(rng, with_eq=False)
        r = lp.simplex(c, A, b, maximize=True)
        y = r.shadow_ub
        assert np.all(y >= -1e-9)                                 # dual feasibility (sign)
        assert np.allclose(y * r.slack_ub, 0, atol=1e-7)          # binding or worthless
        assert np.allclose(r.x * r.reduced_costs, 0, atol=1e-7)   # used or not worth using
        assert np.all(r.reduced_costs <= 1e-9)                    # no variable would add profit
        assert y @ b == pytest.approx(r.objective, abs=1e-6)      # strong duality


def test_shadow_price_equals_finite_difference_within_range():
    f = S.FARM
    base = lp.simplex(f["profit"], f["A"], f["b"], maximize=True)
    for i in range(2):
        b = np.array(f["b"]); b[i] += 5
        assert b[i] <= base.rhs_range_ub[i, 1]
        bumped = lp.simplex(f["profit"], f["A"], b, maximize=True)
        assert bumped.objective - base.objective == pytest.approx(5 * base.shadow_ub[i])


def test_infeasible_and_unbounded_detected():
    assert lp.simplex([1, 1], [[1, 1], [-1, -1]], [1, -2]).status == "infeasible"
    assert lp.simplex([1, 1], [[1, -1]], [1], maximize=True).status == "unbounded"


def test_transportation_balanced_and_matches_linprog():
    t = S.TRANSPORT
    plan, total, u, v, red = transport.solve(t["cost"], t["supply"], t["demand"])
    assert np.allclose(plan.sum(axis=1), t["supply"]) and np.allclose(plan.sum(axis=0), t["demand"])
    c, A, b = transport.lp_matrices(np.array(t["cost"], float), np.array(t["supply"]), np.array(t["demand"]))
    assert total == pytest.approx(linprog(c, A_eq=A, b_eq=b, method="highs").fun)
    assert np.all(red >= -1e-9)                            # no cheaper route left
    assert np.allclose(red[plan > 1e-9], 0)                # used routes priced exactly
    nw = transport.northwest_corner(t["supply"], t["demand"])
    assert total <= (np.array(t["cost"]) * nw).sum()


def test_unbalanced_transport_gets_dummy():
    plan, total, *_ = transport.solve([[4, 6], [5, 3]], [50, 70], [40, 40])
    assert plan.shape == (2, 3) and plan[:, 2].sum() == pytest.approx(40)
    assert total == pytest.approx(40 * 4 + 40 * 3)
