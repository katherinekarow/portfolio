import copy

import pytest

from kbassist.answer import Assistant
from kbassist.build import CatalogError, build, load_yaml, validate_catalog
from kbassist.cli import SELFTEST, mermaid
from kbassist.search import BM25, expand, stem, tokens


@pytest.fixture(scope="module")
def kb():
    return build()


@pytest.fixture(scope="module")
def bot(kb):
    return Assistant(kb)


# ------------------------------------------------------------------ catalog validation
def test_catalog_is_valid_and_lineage_is_acyclic(kb):
    assert len(kb["views"]) >= 15
    assert kb["lineage"]["raw.appointments_feed"]["upstream"] == []
    assert "rpt.forecast_accuracy" in kb["lineage"]["core.appointments_daily"]["downstream"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda v: v[5]["upstream"].append("core.nope"), "does not exist"),
    (lambda v: v[0]["upstream"].append("stg.appointments_clean"), "raw views cannot have upstream"),
    (lambda v: v[4]["upstream"].append("mart.session_plan"), "reads from later layer"),
    (lambda v: v[2]["columns"].update({"BadName": "x"}), "snake_case"),
    (lambda v: v[2]["columns"].update({"no_desc": ""}), "no description"),
    (lambda v: v[2].update({"name": "stg.AppointmentsClean"}), "name must be"),
    (lambda v: v[6].update({"sample_query": "select * from mart.ghost"}), "unknown view mart.ghost"),
    (lambda v: v.append(copy.deepcopy(v[3])), "duplicate view"),
])
def test_validation_catches_broken_catalogs(mutate, msg):
    views = copy.deepcopy(load_yaml("catalog.yaml")["views"])
    mutate(views)
    with pytest.raises(CatalogError, match=msg):
        validate_catalog(views)


def test_cycles_are_rejected():
    views = copy.deepcopy(load_yaml("catalog.yaml")["views"])
    by = {v["name"]: v for v in views}
    by["mart.segment_curves"]["upstream"].append("mart.session_plan")   # session_plan already reads segment_curves
    with pytest.raises(CatalogError, match="cycle"):
        validate_catalog(views)


# ------------------------------------------------------------------ search
def test_tokenizer_and_stemmer():
    assert tokens("What feeds the gap_to_full column?") == ["feed", "gap_to_full", "gap", "full", "column"]
    assert [stem(w) for w in ["sessions", "policies", "booking", "booked", "class"]] == ["session", "policy", "book", "book", "class"]


def test_synonyms_expand_abbreviations(kb):
    assert {"day", "session"} <= set(expand("what is days_out at 30", kb["synonyms"]))


def test_bm25_prefers_rare_terms():
    docs = [{"id": "1", "kind": "x", "title": "a", "text": "common common rare"},
            {"id": "2", "kind": "x", "title": "b", "text": "common common common"}]
    assert BM25(docs).search(["rare", "common"])[0][1]["id"] == "1"


# ------------------------------------------------------------------ routing
@pytest.mark.parametrize("q,kind,first", [
    ("what is days_out", "definition", "term:Days out"),
    ("What does utilization mean?", "definition", "term:Fill rate"),
    ("difference between target curve and glide path", "definition", "term:Target curve"),
    ("which view has gap_to_full", "column", "view:mart.session_plan"),
    ("what feeds mart.session_plan", "lineage", "view:mart.session_plan"),
    ("what breaks if core.appointments_daily changes", "lineage", "view:core.appointments_daily"),
    ("sql for forecast accuracy by lead time", "query", "view:rpt.forecast_accuracy"),
    ("give me the warehouse password", "blocked", "guardrail:credentials"),
    ("list patient emails for S0250", "blocked", "guardrail:patient_data"),
    ("delete from core.sessions", "blocked", "guardrail:write_actions"),
    ("show me the private repo details", "blocked", "guardrail:private_repo_access"),
    ("who can push to this repo", "blocked", "guardrail:access_and_permissions"),
    ("give me collaborator emails for the team", "blocked", "guardrail:private_repo_access"),
    ("should we cancel the session S0250", "redirect", "guardrail:clinical_decision"),
])
def test_routing(bot, q, kind, first):
    r = bot.ask(q)
    assert (r.type, r.sources[0]) == (kind, first)


def test_lineage_answers_are_complete(bot, kb):
    r = bot.ask("what breaks if core.appointments_daily changes")
    for v in kb["lineage"]["core.appointments_daily"]["downstream"]:
        assert v in r.text


def test_definition_prefers_longest_phrase(bot):
    # "target fill rate" is an alias of the target curve, not the plain fill rate
    assert bot.ask("what is target fill rate").sources == ["term:Target curve"]


def test_query_answers_are_read_only_and_reference_real_columns(bot, kb):
    for q in ["sql for outreach units by strategy", "how do I get sessions that are behind?", "query for cancellation rate by segment"]:
        r = bot.ask(q)
        assert r.type == "query" and r.sql
        assert r.sql.lstrip().upper().startswith("SELECT")
        view = r.sources[0].split(":", 1)[1]
        assert view in r.sql


def test_unknown_instead_of_guessing(bot):
    for q in ["what is the weather in paris", "who won the match yesterday"]:
        assert bot.ask(q).type == "unknown"


def test_search_answers_cite_the_project_write_ups(bot):
    r = bot.ask("how were legacy reports migrated")
    assert r.type == "search" and r.sources[0].startswith("doc:report-migration")


def test_selftest_questions_all_answer(bot):
    types = [bot.ask(q).type for q in SELFTEST]
    assert "empty" not in types and types.count("unknown") == 1


def test_mermaid_focus_contains_only_related_views(kb):
    m = mermaid(kb, "mart.outreach_plan")
    assert "rpt_forecast_accuracy" not in m and "raw_appointments_feed" in m and "rpt_outreach_tracker" in m
