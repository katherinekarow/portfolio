import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ratekit import index, matching, quality, synth
from ratekit.cli import run


@pytest.fixture(scope="module")
def r():
    return run()


def key(df):
    return set(zip(df.lot_id, df.tier, df.competitor, df.product_id))


@pytest.mark.parametrize("label,tier", [("Organic Premium", "organic"), ("Fancy Grade", "premium"), ("Choice Plus", "select"),
                                        ("Std Grade", "standard"), ("Extra Fncy", "premium"), ("Feild Run", "standard"),
                                        ("Grade TBA", None)])
def test_grade_labels(label, tier):
    assert matching.tier_from_label(label)[0] == tier


def test_every_true_match_is_found(r):
    t = r["world"]["truth"]
    truth = t["match"][~t["match"].product_id.isin(t["tba_products"])]
    assert key(truth) <= key(r["matches"])


def test_matching_is_one_to_one_and_within_rules(r):
    m = r["matches"]
    assert not m.duplicated(["competitor", "product_id"]).any()           # a competitor product stands in for one of ours
    assert not m.duplicated(["lot_id", "tier", "competitor"]).any()
    assert (m.date_gap_days <= matching.MAX_DATE_GAP).all() and (m.week_gap <= matching.MAX_WEEK_GAP).all()
    # anything matched beyond the planted truth must be a genuinely comparable listing (decoys far away are rejected)
    extra = m[~m.product_id.isin(r["world"]["truth"]["match"].product_id)]
    assert len(extra) <= 8 and (extra.date_gap_days <= 7).all()


def test_assignment_beats_greedy_nearest():
    ours = pd.DataFrame({"lot_id": ["D1", "D2"], "region": "north", "harvest_date": [dt.date(2026, 5, 1), dt.date(2026, 5, 4)],
                         "weeks": 7, "tier": "standard"})
    theirs = pd.DataFrame({"competitor": "Competitor A", "product_id": ["A-1-st", "A-2-st"], "region": "north",
                           "harvest_date": [dt.date(2026, 5, 3), dt.date(2026, 5, 9)], "weeks": 7, "grade_label": "Standard Grade"})
    m, _ = matching.match(ours, theirs)
    # greedy would give A-1 to both; the assignment gives each lot its own comparable
    assert dict(zip(m.lot_id, m.product_id)) == {"D1": "A-1-st", "D2": "A-2-st"}


def test_unconfident_labels_are_left_for_review(r):
    assert set(r["unmapped"].product_id) == r["world"]["truth"]["tba_products"]
    fuzzy = r["matches"][r["matches"].product_id.isin(r["world"]["truth"]["typo_products"])]
    assert fuzzy.tier_rule.str.startswith("similar").all()


def test_feed_quality_finds_exactly_the_injected_problems(r):
    t, rep, c = r["world"]["truth"], r["report"], r["clean"]
    assert rep["duplicates_removed"] == t["duplicates"]
    assert set(map(tuple, c[c.is_outlier][["quote_date", "product_id"]].to_numpy())) == t["outliers"]
    assert rep["stale_days"] == {t["stale"]["competitor"]: t["stale"]["days"]}
    assert rep["missing_days"] == {t["missing"]["competitor"]: t["missing"]["days"]}


def test_index_never_uses_stale_or_outlier_prices(r):
    w, c, idx = r["world"], r["clean"], r["index"]
    t = w["truth"]
    # rebuild the market median by hand on one stale day and check it excludes Competitor B
    day = t["stale"]["days"][-1]
    row = idx[(idx.date == day) & idx.price_index.notna()].iloc[0]
    m = r["matches"]
    prods = m[(m.lot_id == row.lot_id) & (m.tier == row.tier) & (m.competitor != "Competitor B")].product_id
    usable = c[c.usable & c.product_id.isin(prods)]
    latest = usable[usable.quote_date <= day].sort_values("quote_date").groupby("product_id").tail(1)
    latest = latest[(pd.to_datetime(day) - pd.to_datetime(latest.quote_date)).dt.days <= index.MAX_AGE]
    assert row.market_median == pytest.approx(latest.price_usd.median())


def test_cells_with_too_few_comparables_get_no_index(r):
    idx = r["index"]
    thin = idx[idx.competitors < index.MIN_COMPETITORS]
    assert len(thin) > 0 and thin.price_index.isna().all() and (thin.position == "not enough comparables").all()
