import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from orderrec import bid_optimizer as bo
from orderrec.cli import run as cli_run
from orderrec.ingest import UnknownHeaderError, combine_uploads, harmonize_headers, write_books
from orderrec.normalize import normalize_vendor, parse_amount, tier_from_label
from orderrec.reconcile import Tolerance, load_internal, run
from orderrec.synth import make_world
from orderrec.weekly import weekly_summary

RUN = dt.date(2026, 3, 2)


@pytest.fixture(scope="module")
def world():
    return make_world(run_date=RUN)


@pytest.fixture(scope="module")
def clean(world):
    return normalize_vendor(world.vendor)


@pytest.fixture(scope="module")
def recon(world, clean):
    return run(clean[0], world.internal_daily, RUN)


# ---------------------------------------------------------------- normalize
@pytest.mark.parametrize("label,tier", [("Prem.", "PREMIUM"), ("  choice grade ", "CHOICE"), ("Sel.", "SELECT"),
                                        ("Rsv. Grade", "RESERVE"), ("STD", "STANDARD"), ("Extra Fancy Plus", None)])
def test_tier_labels(label, tier):
    assert tier_from_label(label) == tier


@pytest.mark.parametrize("text,value", [("€1,234.5", 1234.5), (" 99.00 ", 99.0), ("$1,000.00", 1000.0), ("TBD", None)])
def test_parse_amount(text, value):
    assert parse_amount(text) == value


def test_vendor_rejects_and_duplicates(world, clean):
    df, rejects = clean
    assert dict(zip(rejects["Offer ID"], rejects.reason_code)) == world.truth["rejects"]
    assert df.attrs["duplicates_removed"] == len(world.truth["duplicates"])
    assert df.offer_id.is_unique
    # the re-sent (later) version is the one kept
    resent = world.vendor[world.vendor["Offer ID"].isin(world.truth["duplicates"])]
    latest = resent.groupby("Offer ID")["Updated At"].max()
    kept = df.set_index("offer_id").loc[latest.index, "updated_at"]
    assert (kept.values == latest.values).all()


# ---------------------------------------------------------------- reconcile
@pytest.mark.parametrize("cls,key", [("PRICE_MISMATCH", "price_mismatch"), ("MISSING_INTERNAL", "missing_internal"),
                                     ("MISSING_VENDOR", "missing_vendor"),
                                     ("CANCELLED_BUT_CHARGED", "cancelled_but_charged"), ("IN_TRANSIT", "in_transit")])
def test_reconciliation_finds_exactly_the_injected_problems(recon, world, cls, key):
    assert recon.ids(cls) == world.truth[key]


def test_rounding_noise_is_not_flagged(recon):
    matched = recon.detail[recon.detail.recon_class == "MATCHED"]
    assert (matched.diff_usd.abs() > 0).sum() > 10          # noise exists...
    assert matched.diff_usd.abs().max() < 0.25               # ...and stays inside tolerance


def test_zero_tolerance_would_create_false_alarms(world, clean):
    strict = run(clean[0], world.internal_daily, RUN, Tolerance(absolute_usd=0, relative=0))
    assert len(strict.ids("PRICE_MISMATCH")) > len(world.truth["price_mismatch"]) + 10


def test_control_totals(recon):
    c = recon.control
    assert c["rows_reconciled"] and c["money_reconciled"]
    assert sum(c["counts"].values()) == c["union_offers"]


def test_missing_monday_file_falls_back_with_warning(world):
    _, cutoff, warnings = load_internal(world.internal_daily, RUN)
    assert RUN not in world.internal_daily
    assert cutoff == dt.date(2026, 2, 27) and "not available" in warnings[0]
    _, cutoff, warnings = load_internal(world.internal_daily, dt.date(2026, 2, 27))
    assert cutoff == dt.date(2026, 2, 27) and warnings == []


# ---------------------------------------------------------------- ingest
@pytest.fixture(scope="module")
def grid(world, tmp_path_factory):
    return combine_uploads(write_books(world.upload_books, tmp_path_factory.mktemp("books")))


def test_upload_combination(world, grid):
    valid, rejects = grid
    total = sum(len(t) for tabs in world.upload_books.values() for t in tabs.values())
    assert len(valid) + len(rejects) == total
    assert sorted(rejects.reason_code) == ["MIN_ABOVE_MAX", "MIN_ABOVE_MAX", "MISSING_LOT", "MISSING_LOT"]
    assert set(rejects.source_row) == {5, 9}                 # Excel rows users can jump to
    assert (valid.min_bid_usd <= valid.max_bid_usd).all()
    assert set(valid.from_tier) <= {"STANDARD", "CHOICE", "SELECT", "PREMIUM"}


def test_unknown_header_fails_loudly():
    df = pd.DataFrame(columns=["Lot", "From", "To", "Min", "Max", "Notes"])
    with pytest.raises(UnknownHeaderError, match="Notes"):
        harmonize_headers(df, "book[tab]")


# ---------------------------------------------------------------- optimizer
def test_unit_value_matches_definition():
    lam, c = 2.3, 4
    k = np.arange(0, 60)
    assert bo._unit_value(1000, lam, c) == pytest.approx(1000 * (np.minimum(k, c) * poisson.pmf(k, lam)).sum())


def _toy():
    inv = pd.DataFrame([
        {"lot_id": "L1", "tier": t, "open_units": o, "spot_price_usd": f, "expected_demand": d}
        for t, o, f, d in [("STANDARD", 3, 900, 2.0), ("CHOICE", 2, 1100, 1.2), ("SELECT", 1, 1400, 0.6)]])
    bids = pd.DataFrame([
        ("A", "L1", "STANDARD", "CHOICE", 260.0), ("B", "L1", "STANDARD", "CHOICE", 120.0),
        ("C", "L1", "STANDARD", "CHOICE", 95.0), ("D", "L1", "CHOICE", "SELECT", 420.0),
        ("E", "L1", "STANDARD", "SELECT", 300.0), ("F", "L1", "CHOICE", "SELECT", 150.0),
        ("G", "L1", "STANDARD", "CHOICE", 410.0)],
        columns=["offer_id", "lot_id", "from_tier", "to_tier", "amount_usd"])
    return bids, inv


def _feasible(bids, x, inv):
    acc = bids[x.astype(bool)]
    for r in inv.itertuples():
        u = (acc.to_tier == r.tier).sum() - (acc.from_tier == r.tier).sum()
        if u > r.open_units:
            return False
    return True


def test_milp_matches_brute_force():
    bids, inv = _toy()
    best = max((bo.expected_revenue(bids, np.array(x), inv), x)
               for x in itertools.product([0, 1], repeat=len(bids)) if _feasible(bids, np.array(x), inv))
    x = bo.optimize(bids, inv)
    assert _feasible(bids, x, inv)
    assert bo.expected_revenue(bids, x, inv) == pytest.approx(best[0], abs=1e-6)


def test_plan_beats_rule_of_thumb_and_respects_capacity(world, clean, grid):
    p = bo.plan(clean[0], world.inventory, grid[0])
    s = p.summary.set_index("policy").expected_total_revenue_usd
    assert s["optimized"] >= s["baseline"] and s["optimized"] >= s["accept none"]
    d = p.decisions
    for col in ["accept_optimized", "accept_baseline"]:
        acc = d[d[col] == 1]
        use = (acc.groupby(["lot_id", "to_tier"]).size().rename_axis(["d", "t"])
               .sub(acc.groupby(["lot_id", "from_tier"]).size().rename_axis(["d", "t"]), fill_value=0))
        cap = world.inventory.set_index(["lot_id", "tier"]).open_units.rename_axis(["d", "t"])
        assert (use <= cap.reindex(use.index)).all(), col


# ---------------------------------------------------------------- weekly + end to end
def test_weekly_totals(clean, recon):
    wk = weekly_summary(clean[0], recon.detail, as_of=RUN)
    assert wk.accepted.sum() == (clean[0].status == "ACCEPTED").sum()
    assert not wk.complete_week.iloc[-1]
    assert np.isnan(wk.wow_revenue_change.iloc[-1])


def test_cli_end_to_end(tmp_path):
    s = cli_run(RUN, tmp_path)
    assert s["rows_reconciled"] and s["money_reconciled"]
    sheets = pd.ExcelFile(s["excel"]).sheet_names
    for name in ["Summary", "Exceptions", "Matched", "Combined upload", "Bid decisions", "Recommended floors"]:
        assert name in sheets
    ex = pd.read_excel(s["excel"], sheet_name="Exceptions")
    assert set(ex.recon_class) == {"PRICE_MISMATCH", "MISSING_INTERNAL", "MISSING_VENDOR", "CANCELLED_BUT_CHARGED"}
    assert len(list((tmp_path / "inputs").glob("internal_substitutions_*.csv"))) == 19
