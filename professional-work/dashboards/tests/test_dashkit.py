"""Rules each dashboard's numbers must satisfy."""
import math

import numpy as np
import pytest

from dashkit import builders as b


@pytest.fixture(scope="module")
def pp():
    return b.price_performance()


def test_price_performance_totals_match_movers(pp):
    for yr_key in ("this",):
        assert sum(m["sales"] for m in pp["movers"]) == pytest.approx(pp["totals"][yr_key]["sales"], rel=1e-6)
        assert sum(m["units"] for m in pp["movers"]) == pp["totals"][yr_key]["units"]


def test_price_performance_vs_ly_is_consistent(pp):
    # sales growth = (1 + price growth) * (1 + units growth) - 1
    for m in pp["movers"]:
        assert (1 + m["price_vs_ly"]) * (1 + m["units_vs_ly"]) - 1 == pytest.approx(m["sales_vs_ly"], abs=1e-3)  # avg price is rounded to cents


def test_price_performance_season_to_date_lengths(pp):
    for s in pp["series"]:
        assert len(s["sales"]) == (pp["weeks_done"] if s["year"] == "this" else 52)


def test_outreach_ci_and_flags():
    d = b.outreach_effectiveness()
    for c in d["campaigns"]:
        p_t, p_h = c["booked_treated"] / c["contacted"], c["booked_holdout"] / c["holdout"]
        se = math.sqrt(p_t * (1 - p_t) / c["contacted"] + p_h * (1 - p_h) / c["holdout"])
        assert c["lift_lo"] == pytest.approx(p_t - p_h - 1.96 * se, abs=1e-4)
        assert c["lift_lo"] <= c["lift"] <= c["lift_hi"]
        assert c["significant"] == (c["lift_lo"] > 0)
    assert any(not c["significant"] for c in d["campaigns"]), "demo should include a campaign that does not work"
    for w, c in zip(d["weekly"], d["campaigns"]):
        assert w["cumulative"] == sorted(w["cumulative"])
        assert w["cumulative"][-1] == pytest.approx(max(c["incremental"], 0), abs=1)


def test_poisson_sf_matches_scipy():
    from scipy.stats import poisson
    for lam in (0.3, 2.5, 11.0):
        for k in range(0, 25):
            assert b._poisson_sf(k, lam) == pytest.approx(poisson.sf(k - 1, lam), abs=1e-9)


def test_same_week_rows_are_sane():
    d = b.same_week_openings()
    for r in d["sessions"]:
        assert 0 <= r["booked"] <= r["capacity"] and r["open"] == r["capacity"] - r["booked"]
        assert 0 <= r["p_fill"] <= 1 and 0 <= r["days_left"] < 14
        if r["action"].startswith("Overbook"):
            assert r["open"] == 0
    keys = [(r["days_left"], -r["expected_empty"]) for r in d["sessions"]]
    assert keys == sorted(keys)


def _brute_displaced(price, lam, open_slots, slots, n=400_000, seed=0):
    """Monte Carlo: regular demand D ~ Poisson(lam) fills open - slots seats; revenue lost vs not quoting."""
    D = np.random.default_rng(seed).poisson(lam, n)
    return price * (np.minimum(D, open_slots) - np.minimum(D, open_slots - slots)).mean()


def test_quote_displacement_matches_simulation():
    p = b.group_quote_inputs()
    s = p["sites"][0]
    w = max(s["weeks"], key=lambda w: w["expected_regular"])
    open_slots = w["capacity"] - w["booked"]
    for slots in (1, open_slots // 2, open_slots):
        q = b.quote(p, s["site"], w["week"], slots)
        assert q["displaced"] == pytest.approx(_brute_displaced(s["regular_price"], w["expected_regular"], open_slots, slots), rel=0.02, abs=0.5)


def test_quote_rules():
    p = b.group_quote_inputs()
    s = p["sites"][1]
    w = s["weeks"][3]
    open_slots = w["capacity"] - w["booked"]
    assert not b.quote(p, s["site"], w["week"], open_slots + 1)["feasible"]
    prev = 0
    for k in range(1, open_slots + 1):
        q = b.quote(p, s["site"], w["week"], k)
        assert q["quote"] >= q["floor"] - 1e-9
        assert q["displaced"] >= prev - 1e-9          # more slots never displace less
        prev = q["displaced"]


@pytest.fixture(scope="module")
def comp():
    return b.comparables()


def test_comparables_curves(comp):
    n = len(comp["days_out"])
    assert comp["days_out"][0] == 364 and comp["days_out"][-1] == 0
    for s in comp["sessions"]:
        assert len(s["curve"]) == n and s["curve"][-1] == s["final_fill"]


def test_nearest_exact_match_first_and_sorted(comp):
    t = comp["sessions"][7]
    res = b.nearest(comp, t["reach"], t["season"], t["length"], t["capacity"], k=8)
    assert res[0]["reach"] == t["reach"] and res[0]["capacity"] == t["capacity"]
    w, caps = comp["weights"], [s["capacity"] for s in comp["sessions"]]
    span = max(caps) - min(caps)
    d = [w["reach"] * (s["reach"] != t["reach"]) + w["season"] * (s["season"] != t["season"])
         + w["length"] * (s["length"] != t["length"]) + w["capacity"] * abs(s["capacity"] - t["capacity"]) / span for s in res]
    assert d == sorted(d)


def test_build_writes_every_dashboard(tmp_path):
    from dashkit.cli import build
    sizes = build(tmp_path, only=["outreach-effectiveness", "group-quote"])
    assert set(sizes) == {"outreach-effectiveness", "group-quote"}
    assert (tmp_path / "group-quote" / "data.json").exists()
