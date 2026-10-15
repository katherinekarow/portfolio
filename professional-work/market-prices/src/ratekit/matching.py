"""Match competitor products to ours.

1. Grade from a free-text label: keyword rules first, then fuzzy similarity to known
   labels (difflib). Low-confidence labels are reported, not silently mapped.
2. Listing match: same region, length within +/- 1 week, harvest date within +/- 7 days.
   Score = 1 - (|date gap| / 7) * 0.6 - |week gap| * 0.25  (closer is better).
3. One-to-one: a competitor listing can only stand in for one of our lots, so for each
   competitor and region we solve an ASSIGNMENT problem (Hungarian algorithm) over the
   score matrix instead of letting every lot grab its nearest candidate.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .synth import LABELS

KEYWORDS = [("organic", "organic"), ("certified", "organic"), ("bio", "organic"),
            ("premium", "premium"), ("fancy", "premium"),
            ("select", "select"), ("choice", "select"),
            ("standard", "standard"), ("field", "standard"), ("value", "standard"), ("std", "standard")]
_KNOWN = [(lbl.lower(), tier) for tier, lbls in LABELS.items() for lbl in lbls]
MAX_DATE_GAP, MAX_WEEK_GAP, MIN_SCORE = 7, 1, 0.3


def tier_from_label(label: str) -> tuple[str | None, float, str]:
    low = label.lower()
    for kw, tier in KEYWORDS:                                  # order matters: "Organic Premium" is organic, "Organic Select" too
        if re.search(rf"\b{kw}", low):
            return tier, 1.0, f"keyword '{kw}'"
    best = max(_KNOWN, key=lambda k: difflib.SequenceMatcher(None, low, k[0]).ratio())
    r = difflib.SequenceMatcher(None, low, best[0]).ratio()
    return (best[1], round(r, 2), f"similar to '{best[0]}'") if r >= 0.75 else (None, round(r, 2), "no confident match")


def match(ours: pd.DataFrame, theirs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = theirs.copy()
    mapped = t.grade_label.map(tier_from_label)
    t["tier"], t["tier_confidence"], t["tier_rule"] = zip(*mapped)
    unmapped = t[t.tier.isna()]
    t = t[t.tier.notna()]
    listings_t = t.drop_duplicates(["competitor", "product_id"]).assign(listing=lambda x: x.product_id.str.rsplit("-", n=1).str[0])
    listings_t = listings_t.drop_duplicates(["competitor", "listing"])[["competitor", "listing", "region", "harvest_date", "weeks"]]
    lots_o = ours.drop_duplicates("lot_id")[["lot_id", "region", "harvest_date", "weeks"]]

    pairs = []
    for (comp, region), cand in listings_t.groupby(["competitor", "region"]):
        mine = lots_o[lots_o.region == region].reset_index(drop=True)
        cand = cand.reset_index(drop=True)
        if mine.empty:
            continue
        gap_d = np.abs((pd.to_datetime(mine.harvest_date).to_numpy()[:, None] - pd.to_datetime(cand.harvest_date).to_numpy()[None, :])
                       / np.timedelta64(1, "D"))
        gap_n = np.abs(mine.weeks.to_numpy()[:, None] - cand.weeks.to_numpy()[None, :])
        score = 1 - gap_d / MAX_DATE_GAP * 0.6 - gap_n * 0.25
        feasible = (gap_d <= MAX_DATE_GAP) & (gap_n <= MAX_WEEK_GAP) & (score >= MIN_SCORE)
        cost = np.where(feasible, -score, 1e6)
        rows, cols = linear_sum_assignment(cost)
        for r_, c_ in zip(rows, cols):
            if feasible[r_, c_]:
                pairs.append({"lot_id": mine.lot_id[r_], "competitor": comp, "listing": cand.listing[c_],
                              "date_gap_days": int(gap_d[r_, c_]), "week_gap": int(gap_n[r_, c_]),
                              "match_score": round(float(score[r_, c_]), 3)})
    listings = pd.DataFrame(pairs)
    t["listing"] = t.product_id.str.rsplit("-", n=1).str[0]
    prod = listings.merge(t[["competitor", "listing", "product_id", "tier", "grade_label", "tier_confidence", "tier_rule"]],
                       on=["competitor", "listing"])
    return prod.sort_values(["lot_id", "tier", "competitor"]).reset_index(drop=True), unmapped
