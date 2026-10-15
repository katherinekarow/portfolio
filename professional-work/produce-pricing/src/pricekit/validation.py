"""Validation queries from the handover, run against the model outputs.

Each query returns the rows that BREAK a rule; an empty result is a pass. They are plain
SQL on purpose: the next owner can run them in the warehouse without the Python code.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from .optimize import LADDER_GAP, MAX_MOVE
from .synth import TIERS

CHECKS = {
    "thin cells are never used unshrunk": """
        SELECT segment, tier, n_obs, method
        FROM elasticity
        WHERE n_obs < 30 AND method = 'cell estimate'""",
    "elasticities are positive and plausible (0.2 - 4)": """
        SELECT segment, tier, elasticity FROM elasticity
        WHERE elasticity IS NULL OR elasticity < 0.2 OR elasticity > 4""",
    "price moves stay within guardrails (or are flagged)": f"""
        SELECT lot_id, tier, current_price, recommended_price
        FROM recommendations
        WHERE ABS(recommended_price / current_price - 1) > {MAX_MOVE} + 0.005
          AND guardrail_conflict = 0""",
    "grade ladder: each tier costs more than the one below": f"""
        SELECT lo.lot_id, lo.tier AS lower_tier, hi.tier AS higher_tier, lo.recommended_price, hi.recommended_price
        FROM recommendations AS lo
        INNER JOIN tier_order AS a ON a.tier = lo.tier
        INNER JOIN tier_order AS b ON b.rank = a.rank + 1
        INNER JOIN recommendations AS hi ON hi.lot_id = lo.lot_id AND hi.tier = b.tier
        WHERE hi.recommended_price < lo.recommended_price * (1 + {LADDER_GAP}) - 1""",
    "never sell more than capacity": """
        SELECT lot_id, tier, units_new, capacity FROM recommendations WHERE units_new > capacity + 0.05""",
    "revenue impact = demand at new price x new price - current revenue": """
        SELECT lot_id, tier, revenue_impact,
               units_new * recommended_price - units_now * current_price AS recomputed
        FROM recommendations
        WHERE ABS(revenue_impact - (units_new * recommended_price - units_now * current_price)) > 0.05""",
}


def run(elasticity: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    con = sqlite3.connect(":memory:")
    elasticity.to_sql("elasticity", con, index=False)
    recommendations.assign(guardrail_conflict=recommendations.guardrail_conflict.astype(int)).to_sql("recommendations", con, index=False)
    pd.DataFrame({"tier": TIERS, "rank": range(len(TIERS))}).to_sql("tier_order", con, index=False)
    rows = []
    for name, sql in CHECKS.items():
        bad = pd.read_sql_query(sql, con)
        rows.append({"check": name, "failing_rows": len(bad), "passed": bad.empty})
    return pd.DataFrame(rows)
