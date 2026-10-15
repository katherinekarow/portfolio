"""Synthetic data for a farm cooperative's grade-substitution program.

Buyers with a contract for one produce grade can bid a premium to receive a higher grade
instead. Bids are placed on a third-party buyer marketplace; the co-op invoices them.

Three sources, deliberately messy in the ways real feeds are:

1. Marketplace export (CSV)          - every substitution offer the buyer marketplace knows about.
   * free-text grade labels ("Prem.", "premium grade", "PREMIUM") instead of codes
   * amounts in USD / EUR / GBP, some with thousands separators or currency symbols
   * offers re-sent with a later timestamp (duplicates), stray whitespace, mixed case
2. Internal processed-substitutions report (one file per business day, none on weekends)
   * the system of record: what was actually charged, always in USD
3. Upload workbooks (Excel, one tab per region, inconsistent headers)
   * the price grids that analysts upload to the vendor; must be combined and validated

Known problems are injected on purpose and recorded in `truth` so tests can prove the
reconciliation finds exactly them.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TIERS = ["STANDARD", "CHOICE", "SELECT", "PREMIUM", "RESERVE"]
VENDOR_LABELS = {
    "STANDARD": ["Standard", "STD", "standard grade", "Std."],
    "CHOICE": ["Choice", "CHC", "choice grade", "Chc."],
    "SELECT": ["Select", "SEL", "select grade", "Sel."],
    "PREMIUM": ["Premium", "PRM", "premium grade", "Prem."],
    "RESERVE": ["Reserve", "RSV", "reserve grade", "Rsv. Grade"],
}
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
BASE_BID = {("STANDARD", "CHOICE"): 90, ("CHOICE", "SELECT"): 140, ("SELECT", "PREMIUM"): 420,
            ("PREMIUM", "RESERVE"): 900, ("STANDARD", "SELECT"): 230}


@dataclass
class World:
    vendor: pd.DataFrame
    internal_daily: dict[dt.date, pd.DataFrame]
    upload_books: dict[str, dict[str, pd.DataFrame]]
    inventory: pd.DataFrame
    truth: dict = field(default_factory=dict)


def _money(rng, amount: float, currency: str) -> str:
    """Render an amount the way vendor exports do: sometimes '1,234.50', sometimes '€1234.5'."""
    style = rng.integers(0, 4)
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}[currency]
    if style == 0:
        return f"{amount:.2f}"
    if style == 1:
        return f"{amount:,.2f}"
    if style == 2:
        return f"{sym}{amount:.1f}"
    return f" {amount:.2f} "


def make_world(run_date: dt.date = dt.date(2026, 3, 2), n_offers: int = 1500, seed: int = 17,
               history_days: int = 28) -> World:
    rng = np.random.default_rng(seed)
    lots = [f"L{i:03d}" for i in range(60)]
    pairs = list(BASE_BID)
    rows = []
    for i in range(n_offers):
        frm, to = pairs[rng.integers(len(pairs))]
        cur = rng.choice(["USD", "USD", "USD", "EUR", "GBP"])
        usd = BASE_BID[(frm, to)] * rng.uniform(0.7, 1.6)
        status = rng.choice(["ACCEPTED", "ACCEPTED", "PENDING", "DECLINED", "CANCELLED"], p=[.45, .1, .25, .12, .08])
        processed = run_date - dt.timedelta(days=int(rng.integers(0, history_days)))
        rows.append({
            "Offer ID": f"OF-{100000 + i}",
            "Order Ref": f"OR{rng.integers(10**6, 10**7)}",
            "Lot": lots[rng.integers(len(lots))],
            "From Grade": rng.choice(VENDOR_LABELS[frm]),
            "To Grade": rng.choice(VENDOR_LABELS[to]),
            "Bid": usd / FX_TO_USD[cur],
            "Currency": cur,
            "Status": status if rng.random() > .1 else status.lower(),
            "Updated At": dt.datetime.combine(processed, dt.time(int(rng.integers(0, 23)), int(rng.integers(0, 59)))),
            "_tier_from": frm, "_tier_to": to, "_usd": round(usd, 2), "_processed": processed,
        })
    v = pd.DataFrame(rows)
    v["_report_date"] = v._processed.map(_next_business_day)
    visible = v._report_date < run_date          # problems can only be detected once their file has landed

    truth = {"price_mismatch": set(), "missing_internal": set(), "missing_vendor": set(),
             "cancelled_but_charged": set(), "duplicates": set(), "in_transit": set()}

    # ---- internal system of record: accepted offers become charges on business days
    acc = v[v.Status.str.upper() == "ACCEPTED"].copy()
    acc["charged_usd"] = acc._usd
    # a) some accepted offers never reached the internal system
    miss = acc[visible[acc.index]].sample(frac=0.03, random_state=seed).index
    truth["missing_internal"] = set(acc.loc[miss, "Offer ID"])
    acc = acc.drop(miss)
    # b) some were charged a different amount (wrong FX or manual override) beyond tolerance
    mm = acc[visible[acc.index]].sample(frac=0.04, random_state=seed + 1).index
    acc.loc[mm, "charged_usd"] = (acc.loc[mm, "charged_usd"] * rng.choice([0.8, 1.15, 1.3], len(mm))).round(2)
    truth["price_mismatch"] = set(acc.loc[mm, "Offer ID"])
    # c) small rounding differences that must NOT be flagged
    rd = acc.drop(mm).sample(frac=0.05, random_state=seed + 2).index
    acc.loc[rd, "charged_usd"] = acc.loc[rd, "charged_usd"] + rng.choice([-0.01, 0.01, 0.02], len(rd))
    # d) cancelled offers that were still charged
    canc = v[(v.Status.str.upper() == "CANCELLED") & visible].sample(8, random_state=seed + 3).copy()
    canc["charged_usd"] = canc._usd
    truth["cancelled_but_charged"] = set(canc["Offer ID"])
    # accepted offers whose internal file has not been produced yet are timing, not errors
    truth["in_transit"] = set(acc.loc[~visible[acc.index], "Offer ID"])
    internal = pd.concat([acc, canc])
    # e) charges the vendor has no record of
    ghosts = []
    for k in range(6):
        frm, to = pairs[rng.integers(len(pairs))]
        oid = f"OF-9{k:05d}"
        ghosts.append({"Offer ID": oid, "Order Ref": f"OR{rng.integers(10**6, 10**7)}", "Lot": lots[k],
                       "_tier_from": frm, "_tier_to": to, "charged_usd": float(BASE_BID[(frm, to)]),
                       "_processed": run_date - dt.timedelta(days=int(rng.integers(3, 9)))})
        truth["missing_vendor"].add(oid)
    internal = pd.concat([internal, pd.DataFrame(ghosts)], ignore_index=True)

    # internal report is produced per business day; weekend activity lands on the next Monday file,
    # and the Monday file for the run date is deliberately missing (job not run yet).
    internal["report_date"] = internal._processed.map(_next_business_day)
    daily = {}
    for day, g in internal.groupby("report_date"):
        if day == run_date:
            continue
        daily[day] = pd.DataFrame({
            "OFFER_ID": g["Offer ID"], "ORDER_REF": g["Order Ref"], "LOT_ID": g["Lot"],
            "FROM_TIER": g._tier_from, "TO_TIER": g._tier_to, "CHARGED_USD": g.charged_usd.round(2),
            "PROCESSED_DATE": g._processed,
        }).reset_index(drop=True)

    # ---- vendor duplicates: re-sent rows a few minutes later (latest wins)
    dup = v.sample(15, random_state=seed + 4).copy()
    dup["Updated At"] = dup["Updated At"] + pd.Timedelta(minutes=25)
    truth["duplicates"] = set(dup["Offer ID"])
    v = pd.concat([v, dup], ignore_index=True)
    vendor = v[["Offer ID", "Order Ref", "Lot", "From Grade", "To Grade", "Bid", "Currency", "Status", "Updated At"]].copy()
    vendor["Bid"] = [_money(rng, b, c) for b, c in zip(vendor.Bid, vendor.Currency)]
    vendor["Status"] = vendor.Status.map(lambda s: f"  {s} " if rng.random() < .05 else s)
    # rows the cleaning step must reject (never guess): unknown label, unparseable amount, unknown currency
    bad = vendor[(vendor.Status.str.strip().str.upper() == "DECLINED")
                 & ~vendor["Offer ID"].isin(truth["duplicates"])].head(3).index
    vendor.loc[bad[0], "To Grade"] = "Extra Fancy Plus"
    vendor.loc[bad[1], "Bid"] = "TBD"
    vendor.loc[bad[2], "Currency"] = "JPY"
    truth["rejects"] = dict(zip(vendor.loc[bad, "Offer ID"], ["UNMAPPED_GRADE", "UNPARSEABLE_AMOUNT", "UNKNOWN_CURRENCY"]))

    # ---- inventory of open units per lot and tier (for the bid optimizer)
    inv = pd.DataFrame([{"lot_id": d, "tier": t, "open_units": int(rng.poisson(lam))}
                        for d in lots for t, lam in zip(TIERS, [6, 5, 4, 2, 1])])
    inv["spot_price_usd"] = inv.tier.map({"STANDARD": 900, "CHOICE": 1100, "SELECT": 1400, "PREMIUM": 2600, "RESERVE": 4200})
    inv["sell_prob"] = inv.tier.map({"STANDARD": .55, "CHOICE": .45, "SELECT": .35, "PREMIUM": .2, "RESERVE": .1})
    # forecast of remaining spot-market demand per tier (Poisson mean) - drives the displacement cost.
    # Substitution programs mostly sell into units that are not expected to sell, so demand < open units.
    inv["expected_demand"] = (inv.open_units * rng.uniform(0.1, 0.8, len(inv))).round(2)

    return World(vendor=vendor, internal_daily=daily, upload_books=_upload_books(rng), inventory=inv, truth=truth)


def _next_business_day(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _upload_books(rng) -> dict[str, dict[str, pd.DataFrame]]:
    """Two workbooks x several regional tabs with drifting header names and a few bad rows."""
    header_styles = [
        {"lot": "Lot", "frm": "From Grade", "to": "To Grade", "min": "Min Bid USD", "max": "Max Bid USD"},
        {"lot": "lot_id", "frm": "from", "to": "to", "min": "min_bid", "max": "max_bid"},
        {"lot": "LOT NO", "frm": "FROM GRADE", "to": "TO GRADE", "min": "MIN", "max": "MAX"},
    ]
    books = {}
    for b, regions in enumerate([["Americas", "Europe"], ["Asia", "Caribbean", "Alaska"]]):
        tabs = {}
        for r_i, region in enumerate(regions):
            h = header_styles[(b + r_i) % len(header_styles)]
            rows = []
            for d in range(10):
                for (frm, to), base in BASE_BID.items():
                    lo = round(base * rng.uniform(0.6, 0.8))
                    hi = round(base * rng.uniform(1.4, 1.8))
                    rows.append({h["lot"]: f"L{(b * 30 + r_i * 10 + d):03d}", h["frm"]: frm.title(), h["to"]: to.lower(),
                                 h["min"]: lo, h["max"]: hi})
            df = pd.DataFrame(rows)
            if r_i == 0:   # a bad row: min above max, and a blank lot
                df.loc[3, h["min"]] = df.loc[3, h["max"]] + 50
                df.loc[7, h["lot"]] = None
            tabs[region] = df
        books[f"price_grid_book_{b + 1}.xlsx"] = tabs
    return books
