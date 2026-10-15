"""Report inventory -> migration waves.

Each legacy report gets an effort score from what the translator found in its SQL
(automatic rewrites are cheap, manual items and output-column problems are not) and a
value score from the inventory (how many people use it, how often, how critical).
Waves: quick wins first (high value, low effort), then high value / high effort,
then the rest; reports nobody has opened in 90 days are proposed for retirement instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .columns import output_columns
from .translate import Translator

CRIT = {"high": 3, "medium": 2, "low": 1}
FREQ = {"daily": 3, "weekly": 2, "monthly": 1}


def assess(sql: str) -> dict:
    r = Translator("snowflake").translate(sql)
    cols = output_columns(sql)
    col_issues = sum(1 for c in cols if c.issue)
    effort = 1 + 0.25 * len(r.changes) + 3 * len(r.warnings) + 1.5 * col_issues
    return {"auto_rewrites": len(r.changes), "manual_items": len(r.warnings), "column_issues": col_issues,
            "effort": round(effort, 2)}


def plan(inventory_path: Path, legacy_dir: Path) -> pd.DataFrame:
    inv = yaml.safe_load(Path(inventory_path).read_text())["reports"]
    rows = []
    for rep in inv:
        sql_path = Path(legacy_dir) / f"{rep['name']}.sql"
        a = assess(sql_path.read_text()) if sql_path.exists() else {"auto_rewrites": None, "manual_items": None,
                                                                      "column_issues": None, "effort": rep.get("effort_estimate", 3)}
        value = CRIT[rep["criticality"]] * FREQ[rep["frequency"]] * (1 + rep["users"] / 10)
        rows.append({**rep, **a, "value": round(value, 2)})
    df = pd.DataFrame(rows)
    hi_v, hi_e = df.value.median(), df.effort.median()
    df["wave"] = "3 - remaining"
    df.loc[(df.value >= hi_v) & (df.effort <= hi_e), "wave"] = "1 - quick wins"
    df.loc[(df.value >= hi_v) & (df.effort > hi_e), "wave"] = "2 - high value, needs work"
    df.loc[df.days_since_last_view > 90, "wave"] = "retire (unused)"
    return df.sort_values(["wave", "value"], ascending=[True, False]).reset_index(drop=True)
