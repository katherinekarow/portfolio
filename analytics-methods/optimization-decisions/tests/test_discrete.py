import numpy as np
import pytest

from optkit import analysis, assignment, integer, lp


def test_knapsack_dp_equals_brute_force_and_beats_greedy():
    rng = np.random.default_rng(3)
    for _ in range(150):
        n = int(rng.integers(3, 12))
        v = rng.integers(1, 60, n).astype(float).tolist()
        w = rng.integers(1, 20, n).tolist()
        cap = int(rng.integers(5, sum(w)))
        dp, pick = integer.knapsack_dp(v, w, cap)
        bf, _ = integer.knapsack_brute(v, w, cap)
        g, _ = integer.knapsack_greedy(v, w, cap)
        assert dp == pytest.approx(bf)
        assert sum(w[i] for i in pick) <= cap and sum(v[i] for i in pick) == pytest.approx(dp)
        assert dp >= g - 1e-9


def test_greedy_trap_textbook_case():
    assert integer.knapsack_dp([60, 100, 120], [10, 20, 30], 50)[0] == 220
    assert integer.knapsack_greedy([60, 100, 120], [10, 20, 30], 50)[0] == 160


def test_branch_and_bound_equals_brute_force():
    rng = np.random.default_rng(8)
    for t in range(40):
        n = 4
        A = rng.integers(1, 9, (3, n)).astype(float)
        b = rng.integers(10, 30, 3).astype(float)
        c = rng.integers(1, 15, n).astype(float)
        x, val, _, _ = integer.branch_and_bound(c, A, b, maximize=True)
        upper = [int(min(b / A[:, j])) for j in range(n)]
        bx, bv = integer.brute_force_ilp(c, A, b, upper, maximize=True)
        assert val == pytest.approx(bv)
        assert np.all(A @ x <= b + 1e-9) and np.allclose(x, np.round(x))


def test_staffing_rounding_story():
    table, s, _ = analysis.staffing()
    A = np.array(list(analysis.S.SHIFTS.values()), float).T
    d = np.array(analysis.S.STAFF_NEEDED, float)
    bx, bv = integer.brute_force_ilp(analysis.S.SHIFT_COST, -A, -d, [7] * A.shape[1])
    assert s["integer_cost"] == pytest.approx(bv)                       # B&B = brute force
    assert s["lp_cost"] <= s["integer_cost"] < s["round_up_cost"]       # LP bound; rounding up wastes money
    assert not s["round_down_feasible"]                                 # rounding down leaves gaps
    assert np.all(A @ table.branch_and_bound.to_numpy() >= d)


def test_hungarian_matches_brute_force_and_scipy():
    lsa = pytest.importorskip("scipy.optimize").linear_sum_assignment
    rng = np.random.default_rng(5)
    for n in range(1, 8):
        for _ in range(5):
            C = rng.integers(0, 100, (n, n)).astype(float)
            a, cost = assignment.hungarian(C)
            assert sorted(a) == list(range(n))
            assert cost == pytest.approx(assignment.brute_force(C)[1])
            r, c = lsa(C)
            assert cost == pytest.approx(C[r, c].sum())
    C = rng.uniform(0, 1, (40, 40))           # larger instance vs scipy only
    r, c = lsa(C)
    assert assignment.hungarian(C)[1] == pytest.approx(C[r, c].sum())
    R = rng.uniform(0, 1, (4, 6))             # rectangular
    r, c = lsa(R)
    assert assignment.hungarian(R)[1] == pytest.approx(R[r, c].sum())


def test_hungarian_max_beats_greedy_on_lot_bids():
    _, v_h, v_g, bids = analysis.assignment_example()
    assert v_h >= v_g
    assert v_h == pytest.approx(-assignment.brute_force(-bids)[1])
