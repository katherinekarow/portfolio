"""Run every case on a small database: results must match (or differ exactly where expected),
and the plans must change the way each case claims. Timing is not asserted - it is measured
on the full 1M-row run and reported in RESULTS.md."""
import pytest

from tunekit import bench, data
from tunekit.cases import CASES


@pytest.fixture(scope="module")
def results():
    con = data.build(":memory:", n_appointments=30_000)
    return {r["case"]: r for r in bench.run_all(con, repeat=1)}


@pytest.mark.parametrize("case", [c for c in CASES], ids=[c.name for c in CASES])
def test_results_match_unless_the_old_query_was_wrong(results, case):
    r = results[case.name]
    assert r["rows"] > 0
    assert r["same_result"] == case.same_result


def test_fan_out_overstated_the_total(results):
    assert results["fan_out_join"]["overstatement_pct"] > 5


def test_sargable_rewrite_turns_scan_into_search(results):
    r = results["sargable_dates"]
    assert r["slow_plan"][0].startswith("SCAN") and r["fast_plan"][0].startswith("SEARCH")


def test_latest_status_index_removes_the_sort_and_table_lookups(results):
    r = results["latest_status"]
    assert any("TEMP B-TREE FOR ORDER BY" in p for p in r["slow_plan"])
    assert not any("TEMP B-TREE FOR ORDER BY" in p for p in r["fast_plan"])
    assert any("COVERING INDEX ix_events_appt_ts" in p for p in r["fast_plan"])
    assert r["alt"]["same_result"]


def test_conditional_aggregation_has_no_correlated_subqueries(results):
    r = results["conditional_aggregation"]
    assert sum("CORRELATED SCALAR SUBQUERY" in p for p in r["slow_plan"]) == 4
    assert not any("CORRELATED" in p for p in r["fast_plan"])


def test_covering_join_uses_both_new_indexes(results):
    plan = " ".join(results["covering_join"]["fast_plan"])
    assert "ix_sessions_date" in plan and "COVERING INDEX ix_appts_sess_ref_cover" in plan
    assert results["covering_join"]["slow_plan"][0].startswith("SCAN b")
