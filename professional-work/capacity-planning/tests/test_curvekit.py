"""Invariant tests: these are the rules the real pipeline enforced before publishing."""
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curvekit import curves, gross_net, quality, strategies, synth  # noqa: E402


@pytest.fixture(scope="module")
def world():
    deps = synth.make_sessions(n_hist=120, n_future=16, seed=3)
    bk = synth.make_appointments(deps, seed=5)
    cv = curves.segment_curves(curves.historical_profiles(bk, deps))
    plan = curves.project_and_decompose(curves.current_position(bk, deps), cv)
    return deps, bk, cv, plan


def test_target_curves_never_decrease(world):
    _, _, cv, _ = world
    for _, g in cv.sort_values("days_out", ascending=False).groupby("segment"):
        assert g.target_fill.is_monotonic_increasing
        assert g.target_fill.between(0, 1).all()


def test_projection_respects_capacity_and_current_appointments(world):
    *_, plan = world
    assert (plan.projected_final <= plan.capacity).all()
    assert (plan.projected_final >= plan.net_appointments).all()
    assert (plan.projected_low <= plan.projected_final).all()
    assert (plan.projected_final <= plan.projected_high).all()


@pytest.mark.parametrize("strategy", ["baseline", "milestones"])
def test_allocations_reconcile_to_the_gap(world, strategy):
    _, _, cv, plan = world
    alloc = strategies.build_outreach_plan(plan, cv, strategy)
    assert strategies.reconcile(plan, alloc).ok.all()
    assert (alloc.outreach_units >= -1e-9).all()


def test_milestones_front_load_more_than_baseline(world):
    _, _, cv, plan = world
    row = plan.sort_values("gap_to_target").iloc[-1]
    if row.gap_to_target <= 0:
        pytest.skip("no gap in this sample")
    base = strategies.allocate(row, cv, "baseline")
    mile = strategies.allocate(row, cv, "milestones")
    first_quarter = lambda a: a.outreach_units.iloc[: max(1, len(a) // 4)].sum()  # noqa: E731
    assert first_quarter(mile) >= first_quarter(base) - 1e-6


def test_gross_to_net_is_never_more_than_gross(world):
    deps, bk, cv, _ = world
    gn = gross_net.gross_to_net(curves.current_position(bk, deps), gross_net.cancellation_profile(cv))
    assert (gn.expected_net_from_gross <= gn.gross_appointments).all()


def test_quality_checks_pass_on_clean_data(world):
    deps, bk, _, plan = world
    assert quality.run_checks(bk, deps, plan).passed.all()


def test_validation_gate_blocks_big_moves(world):
    *_, plan = world
    moved = plan.assign(projected_final=plan.projected_final * 1.2)
    ok, breaches = quality.validation_gate(moved, plan, "session_id", "projected_final", 0.05)
    assert not ok and len(breaches) == len(plan)


def test_snapshot_same_day_identical_rerun_keeps_one_copy(world):
    *_, plan = world
    conn = sqlite3.connect(":memory:")
    t = datetime(2026, 1, 5, 9, tzinfo=timezone.utc)
    quality.snapshot_first(conn, "PLAN", plan, t)
    msg = quality.snapshot_first(conn, "PLAN", plan, t.replace(hour=15))
    assert msg.startswith("replaced")
    assert conn.execute("SELECT COUNT(DISTINCT history_added_at) FROM HIST_PLAN").fetchone()[0] == 1
    quality.snapshot_first(conn, "PLAN", plan.assign(net_appointments=plan.net_appointments + 1), t.replace(hour=16))
    assert conn.execute("SELECT COUNT(DISTINCT history_added_at) FROM HIST_PLAN").fetchone()[0] == 2


def test_dashboard_export_is_consistent_with_the_model():
    import pathlib
    from curvekit import export
    data = export.build(str(pathlib.Path(__file__).resolve().parents[1] / "config.yaml"))
    assert all(data["reconciled"].values())
    for d in data["sessions"]:
        s = d["series"]
        assert len(s["days_out_hist"]) == len(s["net"]) == len(s["gross"])
        assert len(s["days_out_all"]) == len(s["target"]) == len(s["p10"]) == len(s["p90"])
        assert s["net"][-1] == d["net"] and s["days_out_hist"][-1] == d["days_out_today"]
        assert s["proj_mid"][-1] == d["projected"]["mid"] and s["glide"][-1] == d["target_final"]
        assert s["proj_low"][-1] <= s["proj_mid"][-1] <= s["proj_high"][-1] <= d["capacity"]
        for k, units in d["outreach"].items():
            if units:
                assert abs(sum(units) - d["gap_to_target"]) <= 0.5 * len(units)
