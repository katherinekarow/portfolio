"""Synthetic consumer-loan book: monthly vintages simulated from segment-specific hazards.

Each vintage is a cohort of equal-sized, fully amortising 36-month loans. Every month on book (MOB) the
surviving loans face a default hazard and then a prepayment hazard; loan counts are drawn binomially, so
small cohorts are noisy in the same way real ones are. Balances follow the standard amortisation formula.

Planted effects (known, so the analysis can be checked against them):
* vintage-to-vintage quality noise (lognormal multiplier on the default hazard)
* calendar seasonality in defaults (higher in winter) and prepayments (higher in summer)
* a downturn: vintages originated in the last ``SHOCK_VINTAGES`` months default at ``SHOCK_MULT`` x the usual rate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curves import remaining_balance

TERM = 36            # loan term and the "lifetime" horizon, in months
N_VINTAGES = 48      # originated in months 0..47; data is observed to the end of month 47
SHOCK_VINTAGES = 8   # the last 8 vintages were written into a downturn
SHOCK_MULT = 1.8
START = pd.Period("2022-01", "M")

# segment: annual rate, loan size, loans per vintage, default-hazard scale, peak MOB, monthly prepay rate, LGD
SEGMENTS = {
    "Personal A": dict(rate=0.09, size=12_000, n=5_000, base=0.0016, peak=10, smm=0.024, lgd=0.70),
    "Personal B": dict(rate=0.14, size=10_000, n=4_000, base=0.0032, peak=9, smm=0.018, lgd=0.74),
    "Personal C": dict(rate=0.21, size=8_000, n=3_000, base=0.0060, peak=8, smm=0.012, lgd=0.80),
    "Auto B": dict(rate=0.075, size=24_000, n=2_500, base=0.0024, peak=11, smm=0.016, lgd=0.45),
}


def default_shape(t):
    """Hump-shaped default hazard: rises from 0, peaks at ``peak`` MOB, decays slowly."""
    return t * np.exp(1 - t)


def simulate(seed: int = 7, n_vintages: int = N_VINTAGES, shock_vintages: int = SHOCK_VINTAGES,
             shock_mult: float = SHOCK_MULT, full: bool = False) -> pd.DataFrame:
    """Monthly cohort panel. ``full=True`` keeps every MOB to the term (the hidden truth for immature vintages);
    otherwise rows after the observation date are dropped, as they would be in real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, (seg, p) in enumerate(SEGMENTS.items()):
        t = np.arange(1, TERM + 1)
        bal = remaining_balance(p["size"], p["rate"], TERM, np.arange(0, TERM + 1))  # per loan, MOB 0..36
        for v in range(n_vintages):
            cal = (v + t - 1) % 12                                  # calendar month of each MOB, 0 = January
            season_d = 1 + 0.12 * np.cos(2 * np.pi * cal / 12)
            season_p = 1 + 0.15 * np.cos(2 * np.pi * (cal - 6) / 12)
            quality = np.exp(rng.normal(0, 0.12))
            shock = shock_mult if v >= n_vintages - shock_vintages else 1.0
            h_d = p["base"] * default_shape(t / p["peak"]) * season_d * quality * shock
            h_p = p["smm"] * np.minimum(1, t / 12) * season_p * np.exp(rng.normal(0, 0.08))
            n0 = int(p["n"] * (1 + 0.15 * np.sin(2 * np.pi * (v % 12 - 2) / 12)) * np.exp(rng.normal(0, 0.05)))
            alive = n0
            for m in t:
                d = rng.binomial(alive, min(h_d[m - 1], 1))
                pp = rng.binomial(alive - d, min(h_p[m - 1], 1))
                start_bal = bal[m - 1]
                alive -= d + pp
                lgd = float(np.clip(rng.normal(p["lgd"], 0.04), 0, 1)) if d else 0.0
                rows.append((seg, v, m, n0, n0 * p["size"], d, pp, d * start_bal, pp * start_bal,
                             alive * (start_bal - bal[m]), alive * bal[m], d * start_bal * lgd, alive))
    df = pd.DataFrame(rows, columns=["segment", "vintage", "mob", "n_loans", "orig_balance", "defaults", "prepays",
                                     "default_bal", "prepay_bal", "sched_principal", "balance", "net_loss", "alive"])
    df["vintage_month"] = (START + df.vintage).astype(str)
    df["observed"] = df.vintage + df.mob <= n_vintages
    df["shocked"] = df.vintage >= n_vintages - shock_vintages
    return df if full else df[df.observed].reset_index(drop=True)
