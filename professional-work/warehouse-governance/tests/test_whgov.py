import shutil
from pathlib import Path

import networkx as nx
import pytest

from whgov import contracts, lint, loader, project
from whgov.cli import MODELS, run_all


@pytest.fixture(scope="module")
def built():
    return run_all()


# ------------------------------------------------------------------ graph & build
def test_dependencies_are_read_from_sql_not_declared(built):
    _, models, g, *_ = built
    assert set(g.predecessors("mart.session_plan")) == {"core.sessions", "core.appointments_daily", "mart.segment_curves"}
    assert "ranked" not in g and "spine" not in g                     # CTEs are not dependencies


def test_build_order_respects_dependencies(built):
    _, _, g, log, *_ = built
    order = [r["model"] for r in log]
    for a, b in g.edges:
        if a in order and b in order:
            assert order.index(a) < order.index(b)


def test_every_contract_passes(built):
    *_, results, _ = built
    failed = [r for r in results if not r["passed"]]
    assert failed == []
    assert len(results) >= 30


def test_staging_rejects_exactly_the_injected_rows(built):
    con, *_, injected = built
    got = {}
    for key, reason in con.execute("SELECT source_key, reason_code FROM stg.rejected_rows"):
        got.setdefault(reason, set()).add(key)
    for reason in ("MISSING_SESSION", "INVALID_UNITS", "UNKNOWN_EVENT_TYPE"):
        assert got[reason] == injected[reason]
    clean = {r[0] for r in con.execute("SELECT event_id FROM stg.appointments_clean")}
    assert injected["messy_codes"] <= clean                          # fixable rows are fixed, not rejected
    raw_ids = con.execute("SELECT COUNT(DISTINCT event_id) FROM raw.appointments_feed").fetchone()[0]
    assert len(clean) + sum(len(v) for v in got.values()) == raw_ids


def test_targets_never_decrease_toward_session(built):
    con, *_ = built
    bad = con.execute("""SELECT COUNT(*) FROM mart.segment_curves a JOIN mart.segment_curves b
                         ON a.segment = b.segment AND b.days_out = a.days_out - 1 WHERE b.target_fill < a.target_fill""").fetchone()[0]
    assert bad == 0


def test_select_rebuilds_only_upstream(built):
    con, models, *_ = built
    log = project.build(con, models, select="+mart.segment_curves")
    assert [r["model"] for r in log] == ["core.sessions", "stg.appointments_clean", "core.appointments_daily", "mart.segment_curves"]


def test_cycle_is_rejected(tmp_path):
    for layer, name, src in [("mart", "a", "mart.b"), ("mart", "b", "mart.a")]:
        (tmp_path / layer).mkdir(exist_ok=True)
        (tmp_path / layer / f"{name}.sql").write_text(f"-- model: {layer}.{name}\nSELECT x FROM {src}\n")
    with pytest.raises(ValueError, match="cycle"):
        project.graph(project.load_models(tmp_path), set())


def test_unknown_reference_is_rejected(tmp_path):
    (tmp_path / "mart").mkdir()
    (tmp_path / "mart" / "a.sql").write_text("-- model: mart.a\nSELECT x FROM core.ghost\n")
    with pytest.raises(ValueError, match="unknown relation core.ghost"):
        project.graph(project.load_models(tmp_path), set())


# ------------------------------------------------------------------ contracts catch bad data
def test_contracts_catch_broken_data(built):
    con, models, *_ = built
    dup = con.execute("SELECT session_id FROM mart.session_plan LIMIT 1").fetchone()[0]
    con.execute("INSERT INTO mart.session_plan (session_id, segment, capacity, net_appointments, projected_final, gap_to_target, status) "
                "VALUES (?, 'x', 100, 50, 120, -3, 'late')", (dup,))
    res = {(r["model"], r["test"]): r for r in contracts.run_tests(con, {"mart.session_plan": models["mart.session_plan"]})}
    m = "mart.session_plan"
    assert not res[(m, "unique(session_id)")]["passed"]
    assert not res[(m, "accepted_values(status)")]["passed"]
    assert not res[(m, "expression(projected_final BETWEEN net_appointments AND capacity)")]["passed"]
    assert not res[(m, "expression(gap_to_target >= 0)")]["passed"]
    assert not res[(m, "assert_empty(custom)")]["passed"]            # row count no longer reconciles
    con.execute("DELETE FROM mart.session_plan WHERE status = 'late'")


def test_documentation_coverage_is_enforced(built, tmp_path):
    con, models, *_ = built
    m = models["rpt.segment_summary"]
    m2 = project.Model(m.name, m.path, m.sql, m.meta, {k: v for k, v in m.columns.items() if k != "behind_count"}, m.tests)
    r = [x for x in contracts.run_tests(con, {m.name: m2}) if x["test"] == "documentation"][0]
    assert not r["passed"] and "behind_count" in r["error"]


# ------------------------------------------------------------------ linter
def test_real_models_lint_clean():
    found = lint.lint(project.load_models(MODELS), MODELS)
    assert [str(f) for f in found] == []


BAD = """-- model: rpt.bad_report
-- description: deliberately broken
SELECT *, a.capacity * 2, b.status AS Status
from mart.session_plan a, core.sessions b
WHERE b.session_date > '2026-01-01'
ORDER BY 1
"""


def test_linter_finds_every_violation(tmp_path):
    (tmp_path / "rpt").mkdir()
    p = tmp_path / "rpt" / "bad_report.sql"
    p.write_text(BAD)
    codes = {(f.code, f.line) for f in lint.lint_model(project.parse_model(p), tmp_path)}
    assert ("WG001", 3) in codes                     # SELECT *
    assert ("WG002", 3) in codes                     # a.capacity * 2 without alias
    assert ("WG007", 3) in codes                     # alias Status not snake_case
    assert ("WG003", 4) in codes                     # comma join
    assert ("WG004", 4) in codes                     # rpt reading core
    assert ("WG005", 1) in codes                     # no grain / owner / unique test
    assert ("WG006", 5) in codes                     # hard-coded date
    assert ("WG008", 6) in codes                     # ORDER BY
    assert ("WG009", 4) in codes                     # lower-case 'from'


def test_linter_ignores_strings_comments_and_window_order_by(tmp_path):
    (tmp_path / "mart").mkdir()
    p = tmp_path / "mart" / "ok.sql"
    p.write_text("-- model: mart.ok\n-- description: d\n-- grain: g\n-- owner: o\n-- test: unique(k)\n"
                 "SELECT k,\n       'select *, from x, y' AS note,  -- SELECT * in a comment\n"
                 "       ROW_NUMBER() OVER (PARTITION BY k ORDER BY v) AS rn\nFROM core.t\n")
    assert lint.lint_model(project.parse_model(p), tmp_path) == []


def test_file_location_must_match_name(tmp_path):
    (tmp_path / "core").mkdir()
    p = tmp_path / "core" / "thing.sql"
    p.write_text("-- model: mart.thing\n-- description: d\n-- grain: g\n-- owner: o\n-- test: unique(k)\nSELECT k FROM core.t\n")
    assert any(f.code == "WG007" and "file path" in f.message for f in lint.lint_model(project.parse_model(p), tmp_path))


def test_impact_analysis(built):
    _, _, g, *_ = built
    assert nx.descendants(g, "core.appointments_daily") == {"mart.segment_curves", "mart.session_plan",
                                                        "rpt.pace_dashboard", "rpt.segment_summary"}
