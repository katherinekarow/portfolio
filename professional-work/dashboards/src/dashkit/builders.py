"""Data behind each dashboard in the gallery (docs/dashboards/).

Every builder returns a plain dict that is written to docs/dashboards/<slug>/data.json.
All numbers are computed here - the pages only draw them - so each dashboard's logic is
testable in Python (tests/test_dashkit.py).

Two builders reuse sibling projects so the dashboards and the models can't drift:
  fill_pace, comparables   -> ../capacity-planning (curvekit)
  market_review            -> ../market-prices (ratekit)
The others generate their own synthetic data from documented rules.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PW = HERE.parents[3]                                   # professional-work/
for sib in ("capacity-planning", "market-prices"):
    p = str(PW / sib / "src")
    if p not in sys.path:
        sys.path.insert(0, p)

AS_OF = dt.date(2026, 3, 2)


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


# --------------------------------------------------------------------------- 1. fill pace
def fill_pace() -> dict:
    from curvekit import export
    return export.build(str(PW / "capacity-planning" / "config.yaml"))


# --------------------------------------------------------------------------- 2. price & performance
GRADES = ["Standard", "Select", "Premium", "Organic"]
REGIONS = ["North", "South", "East", "West"]


def price_performance(seed: int = 61, weeks: int = 52) -> dict:
    """Weekly co-op sales by grade and region, this season vs last season.

    units = base * season(w) * (price / ref) ** (-e) * noise   (same week last year as comparison)
    """
    rng = np.random.default_rng(seed)
    ref = {"Standard": 22.0, "Select": 27.0, "Premium": 34.0, "Organic": 41.0}         # $ per case
    elast = {"Standard": 1.6, "Select": 1.3, "Premium": 1.0, "Organic": 0.7}
    base = {"Standard": 900, "Select": 600, "Premium": 380, "Organic": 240}
    rows = []
    for year, shift in (("last", 0.0), ("this", 0.04)):
        for g in GRADES:
            for reg in REGIONS:
                rmult = {"North": 1.1, "South": 0.9, "East": 1.0, "West": 0.85}[reg]
                drift = np.cumsum(rng.normal(0, 0.012, weeks))
                for w in range(weeks):
                    season = 1 + 0.45 * math.sin(2 * math.pi * (w - 10) / 52)
                    price = ref[g] * (1 + shift) * math.exp(drift[w]) * (1 + rng.normal(0, 0.02))
                    units = base[g] * rmult * season * (price / ref[g]) ** (-elast[g]) * math.exp(rng.normal(0, 0.06))
                    if year == "this" and g == "Organic":
                        units *= 1.12                                   # organic demand growth this season
                    rows.append({"year": year, "grade": g, "region": reg, "week": w + 1,
                                 "price": round(price, 2), "units": int(units)})
    df = pd.DataFrame(rows)
    df["sales"] = (df.price * df.units).round(2)
    weeks_done = 26                                                        # season to date
    cur = df[(df.year == "this") & (df.week <= weeks_done)]
    prev = df[(df.year == "last") & (df.week <= weeks_done)]

    def agg(x):
        s, u = x.sales.sum(), x.units.sum()
        return {"sales": _r(s, 0), "units": int(u), "avg_price": _r(s / u if u else 0)}

    series = []
    for (yr, g, reg), x in df.groupby(["year", "grade", "region"]):
        x = x.sort_values("week")
        if yr == "this":
            x = x[x.week <= weeks_done]
        series.append({"year": yr, "grade": g, "region": reg, "sales": x.sales.round(0).tolist(),
                       "units": x.units.tolist(), "price": x.price.tolist()})
    movers = []
    for (g, reg), x in cur.groupby(["grade", "region"]):
        y = prev[(prev.grade == g) & (prev.region == reg)]
        a, b = agg(x), agg(y)
        movers.append({"grade": g, "region": reg, **a, "sales_vs_ly": _r(a["sales"] / b["sales"] - 1, 4),
                       "price_vs_ly": _r(a["avg_price"] / b["avg_price"] - 1, 4), "units_vs_ly": _r(a["units"] / b["units"] - 1, 4)})
    return {"weeks_done": weeks_done, "grades": GRADES, "regions": REGIONS,
            "totals": {"this": agg(cur), "last": agg(prev)}, "series": series, "movers": movers}


# --------------------------------------------------------------------------- 3. outreach effectiveness
CAMPAIGNS = [("SMS reminder", 0.028, 0.35), ("Phone call", 0.061, 4.20), ("Partner referral", 0.044, 2.10),
             ("Waitlist callback", 0.095, 3.10), ("Email newsletter", 0.006, 0.05)]


def outreach_effectiveness(seed: int = 62, weeks: int = 16) -> dict:
    """Randomised holdout per campaign: lift = treated rate - holdout rate, with a 95% CI
    (difference of two proportions), incremental appointments and cost per incremental one."""
    rng = np.random.default_rng(seed)
    out, weekly = [], []
    for name, true_lift, cost in CAMPAIGNS:
        base_rate = rng.uniform(0.08, 0.14)
        n_t = int(rng.integers(6000, 14000)); n_h = int(n_t * 0.2)
        conv_t = int(rng.binomial(n_t, base_rate + true_lift)); conv_h = int(rng.binomial(n_h, base_rate))
        p_t, p_h = conv_t / n_t, conv_h / n_h
        lift = p_t - p_h
        se = math.sqrt(p_t * (1 - p_t) / n_t + p_h * (1 - p_h) / n_h)
        incr = lift * n_t
        spend = cost * n_t
        out.append({"campaign": name, "contacted": n_t, "holdout": n_h, "booked_treated": conv_t, "booked_holdout": conv_h,
                    "rate_treated": _r(p_t, 4), "rate_holdout": _r(p_h, 4), "lift": _r(lift, 4),
                    "lift_lo": _r(lift - 1.96 * se, 4), "lift_hi": _r(lift + 1.96 * se, 4),
                    "significant": bool(lift - 1.96 * se > 0), "incremental": _r(incr, 0), "spend": _r(spend, 0),
                    "cost_per_incremental": _r(spend / incr, 2) if incr > 0 else None})
        shares = rng.dirichlet(np.ones(weeks) * 3)
        weekly.append({"campaign": name, "cumulative": np.round(np.cumsum(shares) * max(incr, 0)).tolist()})
    return {"weeks": weeks, "campaigns": out, "weekly": weekly}


# --------------------------------------------------------------------------- 4. market review
def market_review() -> dict:
    from ratekit.cli import run
    r = run()
    idx = r["index"]
    last = idx[idx.date == idx.date.max()]
    cells = [{"lot": a.lot_id, "grade": a.tier, "index": _r(a.price_index, 3), "position": a.position,
              "competitors": int(a.competitors), "price": _r(a.price), "market": _r(a.market_median)}
             for a in last.itertuples()]
    trend = (idx.dropna(subset=["price_index"]).groupby(["date", "tier"]).price_index.median().unstack())
    rep = r["report"]
    return {"as_of": str(idx.date.max()), "grades": list(trend.columns), "dates": [str(d) for d in trend.index],
            "trend": {g: [_r(v, 3) for v in trend[g]] for g in trend.columns},
            "cells": cells,
            "feed": {"stale": {k: [str(d) for d in v] for k, v in rep["stale_days"].items()},
                     "missing": {k: [str(d) for d in v] for k, v in rep["missing_days"].items()},
                     "outliers": rep["outliers"], "duplicates": rep["duplicates_removed"]}}


# --------------------------------------------------------------------------- 5. same-week openings
def _poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for i in range(k):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1 - cdf)


def same_week_openings(seed: int = 63, n: int = 36) -> dict:
    """Sessions in the next 14 days with open slots. Expected walk-in/late demand ~ Poisson; expected
    no-shows ~ Binomial(booked, p). Each session gets a fill probability and a recommended action."""
    rng = np.random.default_rng(seed)
    sites = ["Riverside", "Hillcrest", "Lakeview", "Northgate", "Old Town", "Westfield"]
    kinds = ["Mammography", "Diabetes screening", "Flu clinic", "Cardio check", "Hearing test"]
    rows = []
    for i in range(n):
        day = AS_OF + dt.timedelta(days=int(rng.integers(0, 14)))
        cap = int(rng.choice([24, 32, 40, 60]))
        booked = min(cap, int(cap * rng.uniform(0.55, 1.12)))
        p_noshow = rng.uniform(0.05, 0.16)
        exp_noshow = booked * p_noshow
        days_left = (day - AS_OF).days
        late_lam = (cap - booked) * min(1.2, rng.uniform(0.3, 0.9) * (days_left + 3) / 7)
        waitlist = int(rng.poisson(3)) if rng.random() < 0.6 else 0
        open_slots = cap - booked
        p_fill = _poisson_sf(open_slots, late_lam + waitlist)
        expected_empty = max(open_slots - late_lam - waitlist, 0) + exp_noshow
        if open_slots == 0 and exp_noshow >= 2:
            action = "Overbook by %d" % int(exp_noshow * 0.6)
        elif waitlist >= open_slots > 0:
            action = "Call waitlist"
        elif p_fill < 0.3 and days_left <= 5:
            action = "Partner referral push"
        elif p_fill < 0.6:
            action = "SMS to nearby patients"
        else:
            action = "On track"
        rows.append({"session": f"S{3000 + i}", "date": str(day), "days_left": days_left, "site": rng.choice(sites),
                     "type": rng.choice(kinds), "capacity": cap, "booked": booked, "open": open_slots,
                     "waitlist": waitlist, "exp_no_shows": _r(exp_noshow, 1), "p_fill": _r(p_fill, 3),
                     "expected_empty": _r(expected_empty, 1), "action": action})
    rows.sort(key=lambda r: (r["days_left"], -r["expected_empty"]))
    return {"as_of": str(AS_OF), "sessions": rows}


# --------------------------------------------------------------------------- 6. group quote
def group_quote_inputs(seed: int = 64) -> dict:
    """Parameters the in-browser quote calculator uses. The calculator prices a block of slots for an
    employer screening day: floor = sum of displaced expected regular revenue + delivery cost + margin.
    The same formula is implemented in Python (quote()) so the browser can be tested against it."""
    rng = np.random.default_rng(seed)
    sites = []
    for name in ["Riverside", "Hillcrest", "Lakeview", "Northgate"]:
        weeks = []
        for w in range(12):
            cap = int(rng.choice([40, 60, 80]))
            booked = int(cap * rng.uniform(0.2, 0.85))
            lam = (cap - booked) * rng.uniform(0.2, 0.9)
            weeks.append({"week": w + 1, "capacity": cap, "booked": booked, "expected_regular": _r(lam, 2)})
        sites.append({"site": name, "regular_price": _r(rng.uniform(85, 120), 2), "weeks": weeks})
    return {"sites": sites, "delivery_cost_per_slot": 18.0, "margin": 0.12, "min_discount_for_size": {"50": 0.05, "100": 0.1}}


def quote(params: dict, site: str, week: int, slots: int) -> dict:
    s = next(x for x in params["sites"] if x["site"] == site)
    w = next(x for x in s["weeks"] if x["week"] == week)
    open_slots = w["capacity"] - w["booked"]
    if slots > open_slots:
        return {"feasible": False, "open": open_slots}
    lam, price = w["expected_regular"], s["regular_price"]
    # the k-th slot given away loses price * P(regular demand >= open - k + 1)
    displaced = sum(price * _poisson_sf(open_slots - k + 1, lam) for k in range(1, slots + 1))
    cost = params["delivery_cost_per_slot"] * slots
    floor = (displaced + cost) * (1 + params["margin"])
    disc = 0.0
    for size, d in sorted(((int(k), v) for k, v in params["min_discount_for_size"].items())):
        if slots >= size:
            disc = d
    list_price = price * slots * (1 - disc)
    return {"feasible": True, "open": open_slots, "displaced": round(displaced, 2), "delivery_cost": round(cost, 2),
            "floor": round(floor, 2), "list_price": round(list_price, 2), "quote": round(max(floor, list_price), 2),
            "per_slot": round(max(floor, list_price) / slots, 2), "binding": "floor" if floor > list_price else "list"}


# --------------------------------------------------------------------------- 7. comparable finder
def comparables() -> dict:
    """Completed sessions with their attributes and weekly fill curves; the page finds the k nearest
    to a hypothetical new session (weighted distance on attributes) and averages their curves -
    the 'proxy curve' used when a new session has no history of its own."""
    from curvekit import synth
    import yaml
    cfg = yaml.safe_load((PW / "capacity-planning" / "config.yaml").read_text())["data"]
    s = synth.make_sessions(cfg["historical_sessions"], cfg["future_sessions"], cfg["seed"])
    a = synth.make_appointments(s, cfg["seed"] + 4)
    hist = s[~s.is_future]
    pts = list(range(364, -1, -7)) + [0]
    pts = sorted(set(pts), reverse=True)
    piv = a[a.days_out.isin(pts)].pivot_table(index="session_id", columns="days_out", values="net_appointments")
    out = []
    for r in hist.itertuples():
        reach, season, length = r.segment.split(" / ")
        curve = piv.loc[r.session_id, pts].to_numpy() / r.capacity
        out.append({"id": r.session_id, "reach": reach, "season": season, "length": length, "capacity": int(r.capacity),
                    "final_fill": _r(curve[-1], 3), "curve": [_r(v, 3) for v in curve]})
    return {"days_out": pts, "sessions": out, "weights": {"reach": 1.0, "season": 1.0, "length": 0.8, "capacity": 0.5}}


def nearest(data: dict, reach: str, season: str, length: str, capacity: int, k: int = 5) -> list[dict]:
    """Python reference for the page's nearest-neighbour search."""
    w = data["weights"]
    caps = [s["capacity"] for s in data["sessions"]]
    span = max(caps) - min(caps) or 1

    def dist(s):
        return (w["reach"] * (s["reach"] != reach) + w["season"] * (s["season"] != season)
                + w["length"] * (s["length"] != length) + w["capacity"] * abs(s["capacity"] - capacity) / span)
    return sorted(data["sessions"], key=lambda s: (dist(s), s["id"]))[:k]


BUILDERS = {"fill-pace": fill_pace, "price-performance": price_performance, "outreach-effectiveness": outreach_effectiveness,
            "market-review": market_review, "same-week-openings": same_week_openings, "group-quote": group_quote_inputs,
            "comparables": comparables}
