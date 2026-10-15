"""Recommended prices per lot and tier.

Demand at price p, from the current price p0 and forecast q0 with elasticity e:
    q(p) = q0 * (p / p0) ** (-e)
Contribution (what we maximise) with unit cost c and capacity K:
    max (p - c) * min(q(p), K)

Closed form, then rules:
  1. unconstrained optimum      p* = c * e / (e - 1)              (only exists when e > 1;
                                                                      inelastic cells want the highest allowed price)
  2. capacity                    never price so low that demand exceeds capacity: p >= p_K where q(p_K) = K
                                 (selling the extra demand is impossible, so the lower price only gives money away)
  3. guardrails                  at most +/- MAX_MOVE from today's price in one cycle
  4. grade ladder                 within a lot, each tier must cost at least LADDER_GAP more than the tier below;
                                 violations are repaired with the pool-adjacent-violators algorithm (closest
                                 monotone prices, weighted by demand), then re-clipped to the guardrails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import TIERS

MAX_MOVE = 0.15
LADDER_GAP = 0.05


def demand(p, p0, q0, e):
    return q0 * (np.asarray(p, dtype=float) / p0) ** (-e)


def contribution(p, row) -> np.ndarray:
    return (np.asarray(p, dtype=float) - row.unit_cost) * np.minimum(demand(p, row.current_price, row.forecast_units, row.elasticity), row.capacity)


def best_price(row) -> tuple[float, str]:
    p0, q0, e, c, K = row.current_price, row.forecast_units, row.elasticity, row.unit_cost, row.capacity
    lo, hi = p0 * (1 - MAX_MOVE), p0 * (1 + MAX_MOVE)
    p_cap = p0 * (q0 / K) ** (1 / e)                       # price at which demand equals capacity
    if e > 1:
        p, why = c * e / (e - 1), "elastic optimum"
    else:
        p, why = hi, "inelastic: raise to guardrail"
    if p < p_cap:
        p, why = p_cap, "capacity binds: no lower than sell-out price"
    if p > hi:
        p, why = hi, why + " (capped at +15%)"
    if p < lo:
        p, why = lo, why + " (floored at -15%)"
    return float(p), why


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression."""
    blocks = [[float(y[i]), float(w[i]), 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks.pop(i + 1)
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, _, n in blocks])


def enforce_ladder(prices: np.ndarray, weights: np.ndarray, gap: float = LADDER_GAP) -> np.ndarray:
    """Smallest weighted change (in log price) so that each tier >= previous tier * (1 + gap)."""
    k = np.arange(len(prices)) * np.log1p(gap)
    adj = _pava(np.log(prices) - k, weights) + k             # remove the required steps, make monotone, add back
    return np.exp(adj)


def recommend(lots: pd.DataFrame, elasticities: pd.DataFrame) -> pd.DataFrame:
    d = lots.merge(elasticities[["segment", "tier", "elasticity", "method"]], on=["segment", "tier"], how="left")
    rec = d.apply(best_price, axis=1, result_type="expand")
    d["price_raw"], d["reason"] = rec[0], rec[1]
    d["tier_rank"] = d.tier.map({t: i for i, t in enumerate(TIERS)})
    d = d.sort_values(["lot_id", "tier_rank"]).reset_index(drop=True)
    out = []
    for _, g in d.groupby("lot_id", sort=False):
        p = g.price_raw.to_numpy()
        steps_ok = np.all(p[1:] >= p[:-1] * (1 + LADDER_GAP) - 1e-9)
        if not steps_ok:
            p = enforce_ladder(p, g.forecast_units.to_numpy())
            g = g.assign(reason=np.where(np.abs(p - g.price_raw) > 0.01, g.reason + " + ladder fix", g.reason))
        out.append(g.assign(recommended_price=p))
    d = pd.concat(out)
    lo, hi = d.current_price * (1 - MAX_MOVE), d.current_price * (1 + MAX_MOVE)
    d["guardrail_conflict"] = (d.recommended_price < lo - 0.01) | (d.recommended_price > hi + 0.01)
    d["recommended_price"] = d.recommended_price.round(0)
    d["units_now"] = np.minimum(d.forecast_units, d.capacity)
    d["units_new"] = np.minimum(demand(d.recommended_price, d.current_price, d.forecast_units, d.elasticity), d.capacity).round(1)
    d["revenue_now"] = (d.current_price * d.units_now).round(2)
    d["revenue_new"] = (d.recommended_price * d.units_new).round(2)
    d["revenue_impact"] = (d.revenue_new - d.revenue_now).round(2)
    d["contribution_impact"] = ((d.recommended_price - d.unit_cost) * d.units_new - (d.current_price - d.unit_cost) * d.units_now).round(2)
    d["price_change_pct"] = (d.recommended_price / d.current_price - 1).round(4)
    return d.drop(columns=["tier_rank"])
