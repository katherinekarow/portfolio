"""Decide which pending substitution bids to accept.

Accepting a bid gives a buyer grade B produce from a lot instead of the grade A they contracted:
it uses one unit of B and frees one unit of A. Whether that is worth it depends on how likely
the B unit would otherwise sell on the spot market - and that changes with every bid accepted.

Model (per lot d and tier t, remaining spot-market demand D ~ Poisson(lambda)):

    expected spot-market revenue with c units left   R(c) = spot price * E[min(D, c)]
    marginal value of the c-th unit                R(c) - R(c-1) = spot price * P(D >= c)

Net use u = (bids into t) - (bids out of t) leaves c = open - u units, so the displacement
cost C(u) = R(open) - R(open - u) is convex and piecewise linear in u. We maximise

    sum(bid_i * x_i) - sum_{d,t} C_dt(u_dt)
    s.t. u_dt <= open_dt, x_i binary

as a mixed-integer program (scipy.optimize.milp) using the epigraph of each convex C.
The business-as-usual baseline - "accept any bid at or above the grid minimum while a
unit is free, highest bid first" - is scored with the same expected-revenue function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import poisson


@dataclass
class Plan:
    decisions: pd.DataFrame          # one row per pending bid with accept flags
    summary: pd.DataFrame            # optimizer vs baseline


def _unit_value(price: float, lam: float, c: int) -> float:
    """price * E[min(D, c)] for D ~ Poisson(lam)."""
    if c <= 0:
        return 0.0
    k = np.arange(c)
    return float(price * poisson.sf(k, lam).sum())     # E[min(D,c)] = sum_{k=0}^{c-1} P(D > k)


def expected_revenue(bids: pd.DataFrame, accept: np.ndarray, inv: pd.DataFrame) -> float:
    """Accepted bids + expected spot-market sales of whatever units are left."""
    acc = bids[accept.astype(bool)]
    use = (acc.groupby(["lot_id", "to_tier"]).size().rename_axis(["lot_id", "tier"])
           .sub(acc.groupby(["lot_id", "from_tier"]).size().rename_axis(["lot_id", "tier"]), fill_value=0))
    total = float(acc.amount_usd.sum())
    for r in inv.itertuples():
        u = int(use.get((r.lot_id, r.tier), 0))
        total += _unit_value(r.spot_price_usd, r.expected_demand, r.open_units - u)
    return total


def baseline(bids: pd.DataFrame, inv: pd.DataFrame, floors: pd.Series) -> np.ndarray:
    """Rule of thumb: highest bid first, accept if >= grid minimum and a target unit is free."""
    left = inv.set_index(["lot_id", "tier"]).open_units.to_dict()
    accept = np.zeros(len(bids), dtype=int)
    for pos in np.argsort(-bids.amount_usd.to_numpy(), kind="stable"):
        b = bids.iloc[pos]
        floor = floors.get((b.lot_id, b.from_tier, b.to_tier), np.nan)
        if np.isnan(floor) or b.amount_usd < floor:
            continue
        if left.get((b.lot_id, b.to_tier), 0) > 0:
            accept[pos] = 1
            left[(b.lot_id, b.to_tier)] -= 1
            left[(b.lot_id, b.from_tier)] = left.get((b.lot_id, b.from_tier), 0) + 1
    return accept


def optimize(bids: pd.DataFrame, inv: pd.DataFrame, time_limit: float = 30) -> np.ndarray:
    bids = bids.reset_index(drop=True)
    n = len(bids)
    inv_i = inv.set_index(["lot_id", "tier"])
    cells = sorted(set(zip(bids.lot_id, bids.to_tier)) | set(zip(bids.lot_id, bids.from_tier)))
    cells = [c for c in cells if c in inv_i.index]
    m = len(cells)
    # variables: x_0..x_{n-1} (binary accept), z_0..z_{m-1} (displacement cost per cell, continuous)
    c_obj = np.concatenate([-bids.amount_usd.to_numpy(), np.ones(m)])     # milp minimises
    rows, lo, hi = [], [], []

    for j, (dep, tier) in enumerate(cells):
        into = ((bids.lot_id == dep) & (bids.to_tier == tier)).to_numpy().astype(float)
        out = ((bids.lot_id == dep) & (bids.from_tier == tier)).to_numpy().astype(float)
        use = into - out                                                   # u = use @ x
        r = inv_i.loc[(dep, tier)]
        cap = int(r.open_units)
        # capacity: u <= open units
        rows.append(np.concatenate([use, np.zeros(m)])); lo.append(-np.inf); hi.append(cap)
        # epigraph of convex C(u) on integer points u in [-out, min(cap, into)]
        base = _unit_value(r.spot_price_usd, r.expected_demand, cap)
        u_min, u_max = -int(out.sum()), min(cap, int(into.sum()))
        cost = {u: base - _unit_value(r.spot_price_usd, r.expected_demand, cap - u) for u in range(u_min, u_max + 1)}
        for u in range(u_min, u_max):
            slope = cost[u + 1] - cost[u]
            # z >= cost[u] + slope * (use@x - u)   ->   slope*use@x - z <= slope*u - cost[u]
            z = np.zeros(m); z[j] = -1
            rows.append(np.concatenate([slope * use, z])); lo.append(-np.inf); hi.append(slope * u - cost[u])
        if u_min == u_max:                                                  # no freedom: fixed cost
            z = np.zeros(m); z[j] = 1
            rows.append(np.concatenate([np.zeros(n), z])); lo.append(cost[u_min]); hi.append(np.inf)

    integrality = np.concatenate([np.ones(n), np.zeros(m)])
    bounds = Bounds(np.concatenate([np.zeros(n), np.full(m, -np.inf)]), np.concatenate([np.ones(n), np.full(m, np.inf)]))
    res = milp(c_obj, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"optimizer failed: {res.message}")
    return np.round(res.x[:n]).astype(int)


def recommended_floors(inv: pd.DataFrame, paths: list[tuple[str, str]], quantile: float = 0.75) -> pd.DataFrame:
    """Minimum bid worth accepting for the FIRST substitution on each path, per lot,
    = value of the target unit given up - value of the source unit freed."""
    inv_i = inv.set_index(["lot_id", "tier"])
    rows = []
    for dep in inv.lot_id.unique():
        for frm, to in paths:
            t, f = inv_i.loc[(dep, to)], inv_i.loc[(dep, frm)]
            if t.open_units <= 0:
                continue
            give = t.spot_price_usd * poisson.sf(t.open_units - 1, t.expected_demand)     # P(D >= open)
            gain = f.spot_price_usd * poisson.sf(f.open_units, f.expected_demand)        # P(D >= open+1)
            rows.append({"lot_id": dep, "from_tier": frm, "to_tier": to, "break_even_usd": max(give - gain, 0)})
    be = pd.DataFrame(rows)
    out = be.groupby(["from_tier", "to_tier"]).break_even_usd.agg(
        median_break_even_usd="median", recommended_floor_usd=lambda s: s.quantile(quantile))
    return out.round(0).reset_index()


def plan(vendor_clean: pd.DataFrame, inv: pd.DataFrame, grid: pd.DataFrame) -> Plan:
    bids = vendor_clean[vendor_clean.status == "PENDING"][
        ["offer_id", "lot_id", "from_tier", "to_tier", "amount_usd"]].reset_index(drop=True)
    floors = grid.set_index(["lot_id", "from_tier", "to_tier"]).min_bid_usd
    # lots without an uploaded grid fall back to the path median (and are flagged)
    path_median = grid.groupby(["from_tier", "to_tier"]).min_bid_usd.median()
    has_grid = [k in floors.index for k in zip(bids.lot_id, bids.from_tier, bids.to_tier)]
    filled = pd.Series([floors.get(k, path_median.get(k[1:], np.nan)) for k in
                        zip(bids.lot_id, bids.from_tier, bids.to_tier)],
                       index=pd.MultiIndex.from_frame(bids[["lot_id", "from_tier", "to_tier"]]))
    filled = filled[~filled.index.duplicated()]

    base = baseline(bids, inv, filled)
    opt = optimize(bids, inv)
    bids["grid_floor_usd"] = [filled.get(k) for k in zip(bids.lot_id, bids.from_tier, bids.to_tier)]
    bids["grid_present"] = has_grid
    bids["accept_baseline"] = base
    bids["accept_optimized"] = opt

    rev = {"baseline": expected_revenue(bids, base, inv), "optimized": expected_revenue(bids, opt, inv),
           "accept none": expected_revenue(bids, np.zeros(len(bids)), inv)}
    summary = pd.DataFrame([
        {"policy": k, "bids_accepted": int({"baseline": base, "optimized": opt}.get(k, np.zeros(1)).sum()),
         "substitution_revenue_usd": round(float(bids.amount_usd[{"baseline": base, "optimized": opt}.get(k, np.zeros(len(bids))).astype(bool)].sum()), 2),
         "expected_total_revenue_usd": round(v, 2)}
        for k, v in rev.items()])
    summary["lift_vs_accept_none_usd"] = (summary.expected_total_revenue_usd - rev["accept none"]).round(2)
    return Plan(decisions=bids, summary=summary)
