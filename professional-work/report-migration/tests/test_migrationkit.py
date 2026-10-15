import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from migrationkit import columns, inventory, parity, reference, synth
from migrationkit.cli import ROOT, parity_demo, translate_all
from migrationkit.translate import Translator

LEGACY = ROOT / "legacy"
SF, SQ = Translator("snowflake"), Translator("sqlite")


def one(sql, t=SF):
    return " ".join(t.translate(sql).sql.split())


# ------------------------------------------------------------------ rules
@pytest.mark.parametrize("src,expected", [
    ("SELECT NVL(a, 0) AS x FROM t", "SELECT COALESCE(a, 0) AS x FROM t"),
    ("SELECT NVL2(a, 'y', 'n') AS x FROM t", "SELECT CASE WHEN a IS NOT NULL THEN 'y' ELSE 'n' END AS x FROM t"),
    ("SELECT DECODE(s, 'A', 1, 0) AS x FROM t", "SELECT CASE WHEN s = 'A' THEN 1 ELSE 0 END AS x FROM t"),
    ("SELECT DECODE(s, NULL, 'none', 'some') AS x FROM t", "SELECT CASE WHEN s IS NULL THEN 'none' ELSE 'some' END AS x FROM t"),
    ("SELECT TRUNC(order_date, 'MM') AS m FROM t", "SELECT DATE_TRUNC('MONTH', order_date) AS m FROM t"),
    ("SELECT ADD_MONTHS(start_date, -3) AS d FROM t", "SELECT DATEADD(MONTH, -3, start_date) AS d FROM t"),
    ("SELECT SYSDATE AS now FROM dual", "SELECT CURRENT_TIMESTAMP() AS now"),
    ("SELECT a FROM t MINUS SELECT a FROM u", "SELECT a FROM t EXCEPT SELECT a FROM u"),
    ("SELECT /*+ FULL(t) */ a FROM t", "SELECT a FROM t"),
    ("SELECT NVL(DECODE(s, 'A', NVL(f, 0)), -1) AS x FROM t",
     "SELECT COALESCE(CASE WHEN s = 'A' THEN COALESCE(f, 0) END, -1) AS x FROM t"),
])
def test_snowflake_rules(src, expected):
    assert one(src) == expected


def test_strings_and_comments_are_never_rewritten():
    sql = "SELECT 'NVL(x,1) and SYSDATE' AS note -- DECODE(a,b) in a comment\nFROM t"
    assert SF.translate(sql).sql == sql


def test_decode_with_column_search_value_is_null_safe():
    assert "(s = k OR (s IS NULL AND k IS NULL))" in one("SELECT DECODE(s, k, 1, 0) AS x FROM t")


def test_numeric_trunc_is_flagged_not_guessed():
    r = SF.translate("SELECT TRUNC(amount) AS a FROM t")
    assert "TRUNC(amount)" in r.sql and any("numeric" in w for w in r.warnings)


def test_outer_join_rewrite():
    out = one("SELECT a.x AS x, b.y AS y FROM t a, u b, v c WHERE a.id = b.id(+) AND c.id = a.id AND a.flag = 1")
    assert out == "SELECT a.x AS x, b.y AS y FROM t a LEFT JOIN u b ON a.id = b.id INNER JOIN v c ON c.id = a.id WHERE a.flag = 1"


def test_rownum_with_order_by_is_flagged():
    r = SF.translate((LEGACY / "recent_cancellations.sql").read_text())
    assert r.sql.rstrip().endswith("LIMIT 10") and "ROWNUM" not in r.sql.upper().split("--")[0]
    assert any("BEFORE sorting" in w for w in r.warnings)


def test_rownum_on_sorted_subquery_is_safe():
    r = SF.translate((LEGACY / "top_sessions.sql").read_text())
    assert r.sql.rstrip().endswith("LIMIT 5") and r.warnings == []


def test_hierarchical_query_needs_review():
    r = SF.translate((LEGACY / "referrer_hierarchy.sql").read_text())
    assert any("CONNECT BY" in w for w in r.warnings)


def test_change_log_points_at_code_lines_not_comments():
    r = SF.translate((LEGACY / "appointment_activity.sql").read_text())
    lines = {(c.rule, c.line) for c in r.changes}
    assert ("DECODE -> CASE (NULL-safe)", 5) in lines and ("DECODE -> CASE (NULL-safe)", 15) in lines
    assert ("(+) outer join -> LEFT JOIN", 11) in lines


# ------------------------------------------------------------------ semantics: run translated SQL, compare to Oracle semantics
@pytest.fixture(scope="module")
def data():
    t = synth.make_tables()
    return t, synth.to_sqlite(t)


def run_sqlite(con, name):
    sql = SQ.translate((LEGACY / f"{name}.sql").read_text()).sql
    sql, _ = columns.fix_aliases(sql)
    return pd.read_sql_query(sql, con)


def test_appointment_activity_matches_oracle_semantics(data):
    t, con = data
    r = parity.compare(reference.appointment_activity(t), run_sqlite(con, "appointment_activity"),
                       keys=["session_id", "appointment_month", "status_label", "channel"], tolerance={"revenue": 1e-6})
    assert r.verdict == "PASS", r.summary()
    assert {"Unknown", "Other", "Active", "Cancelled"} <= set(run_sqlite(con, "appointment_activity").status_label)


def test_capacity_position_keeps_sessions_without_appointments(data):
    t, con = data
    new = run_sqlite(con, "capacity_position")
    new.columns = ["session_id", "b_session_id", "capacity", "booked", "remaining", "referral_mix"]
    assert len(new) == len(t["sessions"])                         # the (+) join became a LEFT JOIN, not an INNER JOIN
    r = parity.compare(reference.capacity_position(t), new.drop(columns="b_session_id"), keys=["session_id"])
    assert r.verdict == "PASS", r.summary()


def test_top_sessions(data):
    t, con = data
    r = parity.compare(reference.top_sessions(t), run_sqlite(con, "top_sessions"), keys=["session_id"], tolerance={"revenue": 1e-6})
    assert r.verdict == "PASS"


def test_naive_decode_translation_would_be_wrong(data):
    """Why the NULL-safe rewrite matters: CASE x WHEN NULL never matches."""
    _, con = data
    naive = con.execute("SELECT COUNT(*) FROM appointments WHERE (CASE status WHEN NULL THEN 'Unknown' END) = 'Unknown'").fetchone()[0]
    ours = con.execute("SELECT COUNT(*) FROM appointments WHERE " + SQ.translate(
        "SELECT DECODE(status, NULL, 'Unknown') AS s FROM appointments").sql.split("SELECT ")[1].split(" AS s")[0] + " = 'Unknown'").fetchone()[0]
    assert naive == 0 and ours == int(data[0]["appointments"].status.isna().sum()) > 0


# ------------------------------------------------------------------ output columns
def test_column_audit_finds_the_merged_column_causes():
    cols = columns.output_columns((LEGACY / "capacity_position.sql").read_text())
    issues = {c.position: c.issue for c in cols if c.issue}
    assert "duplicate output name 'session_id'" in issues[2]
    assert "without alias" in issues[4] and "without alias" in issues[5]
    assert "implicit alias" in issues[6]


def test_fix_aliases_leaves_no_issues_and_runs(data):
    _, con = data
    sql, notes = columns.fix_aliases(SQ.translate((LEGACY / "capacity_position.sql").read_text()).sql)
    assert not [c for c in columns.output_columns(sql) if c.issue] and len(notes) == 3
    names = [d[0] for d in con.execute(sql).description]
    assert len(names) == len(set(names))


# ------------------------------------------------------------------ parity harness
def test_parity_demo_explains_every_difference():
    r = parity_demo()
    assert r.verdict == "PASS_WITH_EXPLAINED"
    cats = {d.category for d in r.diagnoses}
    assert cats == {"rounding", "null_vs_zero"}


def _base():
    return pd.DataFrame({"k": ["a", "b", "c", "d"], "amt": [10.0, 20.0, 30.0, 40.0], "day": pd.to_datetime(["2026-01-01"] * 4)})


def test_parity_flags_real_logic_changes_and_missing_rows():
    L, N = _base(), _base()
    N.loc[1, "amt"] = 27.0
    assert parity.compare(L, N, ["k"]).verdict == "FAIL"
    assert parity.compare(L, N.iloc[:3], ["k"]).only_in_legacy.k.tolist() == ["d"]
    assert parity.compare(L, pd.concat([N, N.iloc[[0]]]), ["k"]).duplicate_keys["new"] == 1


def test_parity_recognises_scale_and_date_shift():
    L, N = _base(), _base()
    N["amt"] = N.amt * 1.08
    N["day"] = N.day + pd.Timedelta(hours=5)
    cats = {(d.column, d.category) for d in parity.compare(L, N, ["k"]).diagnoses}
    assert cats == {("amt", "scale_factor"), ("day", "date_shift")}


# ------------------------------------------------------------------ inventory & batch
def test_migration_plan():
    df = inventory.plan(ROOT / "inventory.yaml", LEGACY)
    wave = dict(zip(df.name, df.wave))
    assert wave["charge_audit_2019"] == wave["legacy_referral_payouts"] == "retire (unused)"
    assert wave["appointment_activity"].startswith("1")
    assert df.set_index("name").loc["referrer_hierarchy", "manual_items"] == 2


def test_translate_all(tmp_path):
    log = {r["report"]: r for r in translate_all("snowflake", tmp_path)}
    assert log["referrer_hierarchy"]["status"] == "REVIEW" and log["appointment_activity"]["status"] == "auto"
    assert "-- REVIEW:" in (tmp_path / "recent_cancellations.sql").read_text()
