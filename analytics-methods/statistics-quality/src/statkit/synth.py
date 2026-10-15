"""Seeded synthetic data with planted structure: a produce packing line, a clinic, a data pipeline and a SKU list."""
from __future__ import annotations

import numpy as np
import pandas as pd

# packing line: 5 kg boxes, filler drifts up by 0.025 kg (0.6 within-sigma) at subgroup SHIFT_AT
BOX_TARGET, BOX_SIGMA, BOX_SHIFT, SHIFT_AT, N_SUBGROUPS, SUBGROUP_N = 5.03, 0.04, 0.025, 28, 40, 5
LSL, USL = 4.90, 5.15
# pipeline: silent 1% drop in daily rows from DROP_AT; a duplicate load on SPIKE_AT
ROWS_BASE, ROWS_NOISE, ROWS_DROP, DROP_AT, SPIKE_AT, N_DAYS, BASELINE_DAYS = 1_200_000, 0.008, 0.010, 84, 40, 120, 56
# inspection: defect rate steps from 3.5% to 6% on day P_SHIFT_AT
P0, P1, P_SHIFT_AT, P_DAYS = 0.035, 0.06, 42, 60
DEFECT_REASONS = {"Bruising": 412, "Underweight": 236, "Mislabel": 121, "Soft fruit": 74, "Wrong grade": 52,
                  "Torn box": 31, "Foreign matter": 18, "Missing seal": 12, "Wrong count": 9, "Other": 6}


def packing_line(seed: int = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(BOX_TARGET, BOX_SIGMA, (N_SUBGROUPS, SUBGROUP_N))
    w[SHIFT_AT:] += BOX_SHIFT
    return w.round(3)


def pipeline_rows(seed: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = np.arange(N_DAYS)
    dow = day % 7
    factor = np.array([1.00, 1.04, 1.03, 1.02, 1.06, 0.72, 0.65])[dow]
    rows = ROWS_BASE * factor * (1 + rng.normal(0, ROWS_NOISE, N_DAYS))
    rows[DROP_AT:] *= 1 - ROWS_DROP
    rows[SPIKE_AT] *= 1.06
    return pd.DataFrame({"day": day, "weekday": dow, "rows": rows.round().astype(int)})


def inspections(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rng.integers(180, 221, P_DAYS)
    p = np.where(np.arange(P_DAYS) >= P_SHIFT_AT, P1, P0)
    return pd.DataFrame({"day": np.arange(P_DAYS), "inspected": n, "defects": rng.binomial(n, p)})


def defects_by_shift(seed: int = 14) -> pd.DataFrame:
    """Defect reason × shift counts. Night shift bruises more (planted dependence)."""
    rng = np.random.default_rng(seed)
    probs = {"Day": [0.45, 0.35, 0.20], "Night": [0.58, 0.27, 0.15]}
    cols = ["Bruising", "Underweight", "Mislabel"]
    return pd.DataFrame({s: rng.multinomial(400, p) for s, p in probs.items()}, index=cols).T


def clinic_waits(seed: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Minutes from check-in to being seen at two clinic sites (right-skewed); site B is ~4 minutes slower."""
    rng = np.random.default_rng(seed)
    a = rng.lognormal(np.log(14), 0.55, 160)
    b = rng.lognormal(np.log(17.5), 0.55, 140)
    return a.round(1), b.round(1)


def sku_revenue(seed: int = 16, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sku": [f"SKU-{i:03d}" for i in range(1, n + 1)],
                         "revenue": rng.lognormal(9, 1.3, n).round(0)})
