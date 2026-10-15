"""Budget allocation across channels under a total budget and per-channel min/max.

Each channel's 18-month gross profit for a budget B spent evenly over W weeks (steady state):
    f(B) = margin * W * beta * hill(B / W / (1 - lam); K, s)
Optimum (KKT): every channel strictly between its bounds has the same marginal ROI f'(B) = mu;
a channel at its minimum has f' <= mu, a channel at its maximum has f' >= mu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .mmm import hill


def gross_profit(B, p, weeks, margin):
    return margin * weeks * p["beta"] * hill(np.asarray(B, float) / weeks / (1 - p["lam"]), p["K"], p["s"])


def marginal_roi(B, p, weeks, margin):
    """d gross_profit / dB, analytic."""
    a = np.asarray(B, float) / weeks / (1 - p["lam"])
    K, s = p["K"], p["s"]
    dh = s * K ** s * np.maximum(a, 1e-12) ** (s - 1) / (a ** s + K ** s) ** 2
    return margin * p["beta"] * dh / (1 - p["lam"])


def _total(x, params, chans, weeks, margin):
    return sum(gross_profit(x[i], params[c], weeks, margin) for i, c in enumerate(chans))


def greedy(params, chans, budget, lo, hi, weeks, margin, step=500.0):
    """Hand out the budget in small steps, each to the channel with the largest gain from that step."""
    x = lo.astype(float).copy()
    left = budget - x.sum()
    while left > 1e-9:
        d = min(step, left)
        gain = [gross_profit(x[i] + d, params[c], weeks, margin) - gross_profit(x[i], params[c], weeks, margin)
                if x[i] + d <= hi[i] + 1e-9 else -np.inf for i, c in enumerate(chans)]
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            break
        x[i] += d
        left -= d
    return x


def optimise(params, chans, budget, lo, hi, weeks, margin, starts=()):
    """SLSQP from several feasible starts (greedy, equal split, any extra), keep the best."""
    cons = [{"type": "eq", "fun": lambda x: x.sum() / budget - 1}]
    bounds = list(zip(lo, hi))
    x_greedy = greedy(params, chans, budget, lo, hi, weeks, margin)
    cands = [x_greedy, np.clip(np.full(len(chans), budget / len(chans)), lo, hi), *starts]
    best, best_val = None, -np.inf
    for x0 in cands:
        r = minimize(lambda x: -_total(x, params, chans, weeks, margin) / budget, x0, method="SLSQP",
                     bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
        x = np.clip(r.x, lo, hi)
        x *= budget / x.sum() if abs(x.sum() - budget) > 1e-6 else 1
        v = _total(x, params, chans, weeks, margin)
        if v > best_val:
            best, best_val = x, v
    return best, x_greedy


def kkt(x, params, chans, lo, hi, weeks, margin, tol=1.0):
    """Marginal ROI per channel and where it sits. tol is in dollars of distance from a bound."""
    m = np.array([marginal_roi(x[i], params[c], weeks, margin) for i, c in enumerate(chans)])
    status = np.where(x <= lo + tol, "at min", np.where(x >= hi - tol, "at max", "interior"))
    inner = status == "interior"
    mu = float(m[inner].mean()) if inner.any() else float("nan")
    spread = float(m[inner].max() - m[inner].min()) if inner.sum() > 1 else 0.0
    ok = inner.any() and np.all(m[status == "at min"] <= mu + 1e-6) and np.all(m[status == "at max"] >= mu - 1e-6)
    return {"marginal": m, "status": status, "mu": mu, "spread": spread, "conditions_hold": bool(ok)}


def table(chans, labels, naive, opt, true_opt, params_fit, params_true, lo, hi, weeks, margin) -> pd.DataFrame:
    k = kkt(opt, params_fit, chans, lo, hi, weeks, margin)
    rows = []
    for i, c in enumerate(chans):
        rows.append({
            "channel": c, "label": labels[c], "min": lo[i], "max": hi[i],
            "naive_spend": round(naive[i], 0), "optimal_spend": round(opt[i], 0),
            "optimal_share": round(opt[i] / opt.sum(), 4),
            "naive_gp_true": round(float(gross_profit(naive[i], params_true[c], weeks, margin)), 0),
            "optimal_gp_true": round(float(gross_profit(opt[i], params_true[c], weeks, margin)), 0),
            "optimal_gp_model": round(float(gross_profit(opt[i], params_fit[c], weeks, margin)), 0),
            "marginal_roi_naive": round(float(marginal_roi(naive[i], params_fit[c], weeks, margin)), 3),
            "marginal_roi_optimal": round(float(k["marginal"][i]), 3),
            "bound_status": k["status"][i],
            "true_optimal_spend": round(true_opt[i], 0),
        })
    return pd.DataFrame(rows)
