"""Synthetic price / demand history for a produce co-op: market segments x produce grades.

Each (segment, grade) cell has a TRUE constant price elasticity. Weekly observations record
the price that was charged and the units sold:

    units = base * (price / ref_price) ** (-elasticity) * season(week) * lead_effect * noise

Some cells are deliberately thin (few weeks of history) so the estimator's sample-size
rules and shrinkage can be tested. The truth is returned so tests can check recovery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ["retail-east", "retail-west", "wholesale-east", "wholesale-west"]
TIERS = ["standard", "select", "premium", "organic"]
REF_PRICE = {"standard": 900, "select": 1000, "premium": 1500, "organic": 2600}
TRUE_ELASTICITY = {"standard": 1.8, "select": 1.4, "premium": 1.1, "organic": 0.7}
SEG_SHIFT = {"retail-east": -0.15, "retail-west": 0.2, "wholesale-east": 0.0, "wholesale-west": 0.3}


def make_history(seed: int = 33, weeks: int = 156, thin_cells: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    cells = [(s, t) for s in SEGMENTS for t in TIERS]
    thin = {cells[i] for i in rng.choice(len(cells), thin_cells, replace=False)}
    for seg, tier in cells:
        e = TRUE_ELASTICITY[tier] + SEG_SHIFT[seg] * (1 if tier != "organic" else 0.3)
        base = rng.uniform(80, 160) if tier != "organic" else rng.uniform(15, 30)
        ref = REF_PRICE[tier]
        n_weeks = int(rng.integers(12, 22)) if (seg, tier) in thin else weeks
        truth.append({"segment": seg, "tier": tier, "true_elasticity": round(e, 3), "thin": (seg, tier) in thin})
        for w in range(weeks - n_weeks, weeks):
            price = ref * float(np.exp(rng.normal(0, 0.12)))            # price tests and promotions
            season = 1 + 0.25 * np.sin(2 * np.pi * (w % 52) / 52)
            lead = rng.choice([1.0, 0.85, 1.15])
            units = base * (price / ref) ** (-e) * season * lead * float(np.exp(rng.normal(0, 0.10)))
            rows.append({"segment": seg, "tier": tier, "week": w, "week_of_year": w % 52, "price": round(price, 2),
                         "units": int(rng.poisson(units))})
    return pd.DataFrame(rows), pd.DataFrame(truth)


def make_lots(seed: int = 34, n: int = 24) -> pd.DataFrame:
    """Upcoming lots with current price, current demand forecast at that price, capacity and unit cost."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        seg = SEGMENTS[i % len(SEGMENTS)]
        for tier in TIERS:
            price = REF_PRICE[tier] * rng.uniform(0.9, 1.1)
            demand = (rng.uniform(70, 150) if tier != "organic" else rng.uniform(12, 28)) * rng.uniform(0.7, 1.3)
            cap = int(demand * rng.uniform(0.75, 1.35))
            rows.append({"lot_id": f"L{i:03d}", "segment": seg, "tier": tier, "current_price": round(price, 2),
                         "forecast_units": round(demand, 1), "capacity": max(cap, 3),
                         "unit_cost": round(REF_PRICE[tier] * 0.35, 2)})
    return pd.DataFrame(rows)
