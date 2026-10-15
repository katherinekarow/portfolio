"""Synthetic produce market price feed (a co-op and three competing suppliers).

Our side:   lots (region, harvest date, contract weeks) x 4 grades, with our price per day.
Their side: three competitors with their own lots (dates and lengths close to ours,
            but not identical), free-text grade labels, and a daily price feed that has
            the problems real feeds have:
  * one competitor's feed goes STALE (prices frozen) for several days
  * a second competitor has MISSING days
  * typos: a few prices x10 or /10
  * duplicates (same observation delivered twice) and mixed currencies

`truth` records the correct product matches and every injected problem.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TIERS = ["standard", "select", "premium", "organic"]
LABELS = {
    "standard": ["Standard Grade", "Field Run", "Value Pack", "Std Grade"],
    "select": ["Select Grade", "Choice Pack", "No.1 Select", "Choice Plus"],
    "premium": ["Extra Fancy", "Premium Pack", "Fancy Grade", "Top Premium"],
    "organic": ["Certified Organic", "Organic Select", "Bio Organic", "Organic Premium"],
}
BASE = {"standard": 900, "select": 1150, "premium": 1500, "organic": 2600}
FX = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
COMPETITORS = {"Competitor A": 0.95, "Competitor B": 1.05, "Competitor C": 1.00}   # price level vs ours


def make(seed: int = 43, n_lots: int = 30, days: int = 60, start: dt.date = dt.date(2026, 1, 5)) -> dict:
    rng = np.random.default_rng(seed)
    ours, theirs, truth_match = [], [], []
    regions = ["north", "south", "east", "west"]
    for i in range(n_lots):
        region = regions[i % 4]
        # lots in the same region are spaced >= 45 days apart, so a comparable can only belong to one of them
        d0 = start + dt.timedelta(days=60 + (i // 4) * 45 + int(rng.integers(0, 6)))
        weeks = int(rng.choice([4, 7, 10]))
        dep_id = f"L{i:03d}"
        for tier in TIERS:
            ours.append({"lot_id": dep_id, "region": region, "harvest_date": d0, "weeks": weeks, "tier": tier,
                         "base_price": BASE[tier] * weeks / 7 * rng.uniform(0.9, 1.1)})
        for comp in COMPETITORS:
            if rng.random() < 0.15:          # this competitor has no comparable product
                continue
            cd = d0 + dt.timedelta(days=int(rng.integers(-4, 5)))
            cn = weeks + int(rng.choice([0, 0, 0, 1, -1]))
            cid = f"{comp[-1]}-{i:03d}"
            for tier in TIERS:
                label = rng.choice(LABELS[tier])
                theirs.append({"competitor": comp, "product_id": f"{cid}-{tier[:2]}", "region": region,
                               "harvest_date": cd, "weeks": cn, "grade_label": label})
                truth_match.append({"lot_id": dep_id, "tier": tier, "competitor": comp,
                                    "product_id": f"{cid}-{tier[:2]}"})
        # a decoy: same competitor, same region, a different listing a few weeks away
        comp = rng.choice(list(COMPETITORS))
        decoy = d0 + dt.timedelta(days=int(rng.choice([-1, 1]) * rng.integers(15, 30)))
        for tier in TIERS:
            theirs.append({"competitor": comp, "product_id": f"X{comp[-1]}{i:03d}-{tier[:2]}", "region": region,
                           "harvest_date": decoy, "weeks": weeks, "grade_label": rng.choice(LABELS[tier])})
    ours, theirs = pd.DataFrame(ours), pd.DataFrame(theirs)
    # label noise: typos a fuzzy match should fix, and placeholders nobody should guess
    typo = {"Extra Fancy": "Extra Fncy", "Field Run": "Feild Run", "Choice Pack": "Choise Pack"}
    real = ~theirs.product_id.str.startswith("X")
    pick = theirs[real & theirs.grade_label.isin(list(typo))].sample(9, random_state=seed).index
    theirs.loc[pick, "grade_label"] = theirs.loc[pick, "grade_label"].map(typo)
    tba = theirs[real & ~theirs.index.isin(pick)].sample(4, random_state=seed + 1).index
    tier_before_tba = dict(zip(theirs.loc[tba, "product_id"], theirs.loc[tba, "grade_label"]))
    theirs.loc[tba, "grade_label"] = "Grade TBA"

    # ---- daily price observations
    dates = [start + dt.timedelta(days=k) for k in range(days)]
    our_prices = []
    for _, r in ours.iterrows():
        walk = r.base_price * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
        our_prices += [{"date": d, "lot_id": r.lot_id, "tier": r.tier, "price": round(p, 2)} for d, p in zip(dates, walk)]
    our_prices = pd.DataFrame(our_prices)

    obs = []
    def true_tier(pid):            # tier suffix in the id is the truth, whatever the label says
        return {"st": "standard", "se": "select", "pr": "premium", "or": "organic"}[pid.rsplit("-", 1)[1]]
    tier_of = {p: true_tier(p) for p in theirs.product_id}
    for _, r in theirs.iterrows():
        lvl = COMPETITORS[r.competitor]
        base = BASE[tier_of[r.product_id]] * r.weeks / 7 * lvl * rng.uniform(0.92, 1.08)
        walk = base * np.exp(np.cumsum(rng.normal(0, 0.012, days)))
        cur = {"Competitor A": "USD", "Competitor B": "EUR", "Competitor C": rng.choice(["USD", "GBP"])}[r.competitor]
        for d, p in zip(dates, walk):
            obs.append({"quote_date": d, "competitor": r.competitor, "product_id": r.product_id,
                        "price": round(p / FX[cur], 2), "currency": cur})
    obs = pd.DataFrame(obs)

    truth = {"match": pd.DataFrame(truth_match), "typo_products": set(theirs.loc[pick, "product_id"]),
             "tba_products": set(tier_before_tba)}
    # stale: Competitor B prices frozen for 6 days (feed resent yesterday's file)
    s0 = dates[35]
    stale_days = [s0 + dt.timedelta(days=k) for k in range(6)]
    b = obs.competitor == "Competitor B"
    frozen = obs[b & (obs.quote_date == s0 - dt.timedelta(days=1))].set_index("product_id").price
    m = b & obs.quote_date.isin(stale_days)
    obs.loc[m, "price"] = obs.loc[m, "product_id"].map(frozen)
    truth["stale"] = {"competitor": "Competitor B", "days": stale_days}
    # missing days: Competitor C delivered nothing on 3 days
    missing = [dates[12], dates[13], dates[40]]
    obs = obs[~((obs.competitor == "Competitor C") & obs.quote_date.isin(missing))]
    truth["missing"] = {"competitor": "Competitor C", "days": missing}
    # typos
    idx = rng.choice(obs.index, 12, replace=False)
    obs.loc[idx[:8], "price"] = obs.loc[idx[:8], "price"] * 10
    obs.loc[idx[8:], "price"] = (obs.loc[idx[8:], "price"] / 10).round(2)
    truth["outliers"] = set(map(tuple, obs.loc[idx, ["quote_date", "product_id"]].to_numpy()))
    # duplicates
    dup = obs.sample(40, random_state=seed)
    obs = pd.concat([obs, dup], ignore_index=True)
    truth["duplicates"] = len(dup)
    return {"ours": ours, "our_prices": our_prices, "theirs": theirs, "observations": obs.reset_index(drop=True),
            "dates": dates, "truth": truth}
