"""90-day KPI review: compare each market x channel's observed ROI with the plan and assign an action.

Each test market is treated as a one-third-scale copy of the national response curve, so at one third
of the national spend it has the national ROI times the market's own responsiveness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .finance import weekly_gp

RULES = {"exceed": "scale: move budget in (+20%) and widen to the next market",
         "meet": "optimise: keep budget, iterate creative and targeting via A/B tests",
         "below": "pivot: cut to the channel minimum, test a new format or audience"}


def classify(ratio, exceed=1.15, below=0.85):
    return "exceed" if ratio >= exceed else ("below" if ratio < below else "meet")


def review(alloc, params_fit, params_true, cfg, rng) -> pd.DataFrame:
    b, f, k = cfg["budget"], cfg["finance"], cfg["kpi"]
    weeks = int(round(k["days"] / 7))
    rows = []
    for c, B in alloc.items():
        spend = B / b["weeks"] * weeks
        plan = weekly_gp({c: B}, params_fit, b["weeks"], weeks, f["gross_margin"]).sum() / spend
        true = weekly_gp({c: B}, params_true, b["weeks"], weeks, f["gross_margin"]).sum() / spend
        for m, resp in k["markets"].items():
            obs = true * resp * np.exp(rng.normal(-0.5 * k["shock_sd"] ** 2, k["shock_sd"]))
            ratio = obs / plan
            status = classify(ratio, k["exceed_ratio"], k["below_ratio"])
            rows.append({"market": m, "channel": c, "spend_90d": round(spend / len(k["markets"]), 0),
                         "planned_roi": round(float(plan), 3), "observed_roi": round(float(obs), 3),
                         "ratio_to_plan": round(float(ratio), 3), "status": status, "action": RULES[status]})
    return pd.DataFrame(rows)
