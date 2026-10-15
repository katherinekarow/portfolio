"""Monte Carlo uncertainty propagation and one-at-a-time (tornado) sensitivity for a packing-season profit model."""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

# Each input: (label, distribution sampler, P10, base, P90)
INPUTS = {
    "boxes_sold": ("Boxes sold", _st.norm(52000, 6000)),
    "price": ("Price per box ($)", _st.triang(c=0.4, loc=17.0, scale=6.0)),     # 17 … 23, mode 19.4
    "unit_cost": ("Variable cost per box ($)", _st.norm(12.5, 0.9)),
    "spoilage": ("Spoilage share", _st.beta(4, 60)),                          # mean ≈ 6.3%
    "fixed": ("Fixed season cost ($)", _st.uniform(loc=200000, scale=50000)),
}


def profit(boxes_sold, price, unit_cost, spoilage, fixed):
    """Profit = boxes · (1 − spoilage) · price − boxes · unit cost − fixed (spoiled boxes still cost to pack)."""
    return boxes_sold * (1 - spoilage) * price - boxes_sold * unit_cost - fixed


def base_case() -> dict:
    return {k: float(d.median()) for k, (_, d) in INPUTS.items()}


def simulate(n: int = 100000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = {k: d.rvs(size=n, random_state=rng) for k, (_, d) in INPUTS.items()}
    return profit(**draws)


def summary(sim: np.ndarray) -> dict:
    return {"mean": float(sim.mean()), "p05": float(np.quantile(sim, 0.05)), "p50": float(np.median(sim)),
            "p95": float(np.quantile(sim, 0.95)), "prob_loss": float((sim < 0).mean()),
            "base_case_profit": float(profit(**base_case()))}


def tornado(low_q: float = 0.10, high_q: float = 0.90):
    """Move one input to its P10 and then its P90, others at base (median). Sorted by swing, largest first."""
    base = base_case()
    b = profit(**base)
    rows = []
    for k, (label, d) in INPUTS.items():
        lo, hi = dict(base), dict(base)
        lo[k], hi[k] = float(d.ppf(low_q)), float(d.ppf(high_q))
        p_lo, p_hi = profit(**lo), profit(**hi)
        rows.append({"input": k, "label": label, "value_p10": lo[k], "value_p90": hi[k],
                     "profit_at_p10": p_lo, "profit_at_p90": p_hi, "swing": abs(p_hi - p_lo), "base_profit": b})
    return sorted(rows, key=lambda r: -r["swing"])
