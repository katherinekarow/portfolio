"""Monthly cash flows of the plan -> NPV, IRR, payback, ROI multiple; Monte Carlo over fit uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .mmm import hill

WEEKS_PER_MONTH = 52 / 12


def monthly_rate(annual):
    return (1 + annual) ** (1 / 12) - 1


def npv(cash, rate):
    """Cash flows at the end of months 1..n (cash[0] is month 1), discounted at a monthly rate."""
    cash = np.asarray(cash, float)
    t = np.arange(1, cash.shape[-1] + 1)
    return (cash / (1 + rate) ** t).sum(axis=-1)


def irr(cash):
    """Monthly IRR by root-finding NPV(r) = 0; nan if NPV has no sign change on (-0.99, 10)."""
    f = lambda r: npv(cash, r)
    lo, hi = -0.99, 10.0
    if np.sign(f(lo)) == np.sign(f(hi)):
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-12))


def payback_month(cash):
    """First month in which cumulative (undiscounted) cash flow turns non-negative; None if never."""
    c = np.cumsum(cash)
    idx = np.flatnonzero(c >= 0)
    return int(idx[0] + 1) if idx.size else None


def weekly_gp(alloc, params, weeks_spend, horizon_weeks, margin):
    """Weekly gross profit of a plan spending alloc[c] evenly over weeks_spend, with carry-over afterwards.

    params values may be arrays of draws (shape (n,)); returns (n, horizon_weeks) or (horizon_weeks,).
    Closed form for constant spend x: adstock_t = x (1 - lam^(t+1)) / (1 - lam) while spending,
    then decays by lam each week.
    """
    t = np.arange(horizon_weeks)
    total = 0.0
    for c, B in alloc.items():
        p = params[c]
        lam = np.asarray(p["lam"], float)[..., None]
        x = B / weeks_spend
        ramp = x * (1 - lam ** (np.minimum(t, weeks_spend - 1) + 1)) / (1 - lam)
        decay = lam ** np.maximum(t - (weeks_spend - 1), 0)
        a = ramp * decay
        total = total + margin * np.asarray(p["beta"], float)[..., None] * hill(a, np.asarray(p["K"], float)[..., None],
                                                                              np.asarray(p["s"], float)[..., None])
    return total


def to_months(weekly, horizon_months):
    """Sum weekly values into calendar months (week w belongs to month floor(w / 4.33))."""
    m = np.minimum((np.arange(weekly.shape[-1]) / WEEKS_PER_MONTH).astype(int), horizon_months - 1)
    out = np.zeros(weekly.shape[:-1] + (horizon_months,))
    for j in range(horizon_months):
        out[..., j] = weekly[..., m == j].sum(axis=-1)
    return out


def spend_schedule(alloc, fixed, weeks_spend, horizon_months):
    """Monthly outflows: media spent evenly per week (bucketed into months like the gross profit);
    fixed lines by their timing rule."""
    spend_months = int(round(weeks_spend / WEEKS_PER_MONTH))
    wk = np.zeros(int(np.ceil(horizon_months * WEEKS_PER_MONTH)))
    wk[:weeks_spend] = sum(alloc.values()) / weeks_spend
    s = to_months(wk, horizon_months)
    for v in fixed.values():
        a, timing = v["amount"], v["timing"]
        if timing == "upfront":
            s[0] += a
        elif timing == "monthly":
            s[:spend_months] += a / spend_months
        elif timing == "upfront_then_monthly":          # 40% set-up in month 1, the rest spread
            s[0] += 0.4 * a
            s[:spend_months] += 0.6 * a / spend_months
        else:
            raise ValueError(timing)
    return s


def cash_flows(alloc, params, cfg):
    b, f = cfg["budget"], cfg["finance"]
    H = f["horizon_months"]
    gp = to_months(weekly_gp(alloc, params, b["weeks"], int(np.ceil(H * WEEKS_PER_MONTH)), f["gross_margin"]), H)
    out = spend_schedule(alloc, b["fixed"], b["weeks"], H)
    return pd.DataFrame({"month": np.arange(1, H + 1), "gross_profit": gp, "spend": out, "net": gp - out})


def summary(cf: pd.DataFrame, annual):
    r = monthly_rate(annual)
    m_irr = irr(cf.net.to_numpy())
    return {"npv": float(npv(cf.net, r)), "irr_monthly": m_irr, "irr_annual": (1 + m_irr) ** 12 - 1,
            "payback_month": payback_month(cf.net.to_numpy()), "roi_multiple": float(cf.gross_profit.sum() / cf.spend.sum()),
            "gross_profit": float(cf.gross_profit.sum()), "spend": float(cf.spend.sum())}


def monte_carlo(alloc, draws, cfg, rng):
    """ROI multiple and NPV for each parameter draw, times a lognormal 'transfer' factor per draw."""
    b, f = cfg["budget"], cfg["finance"]
    H = f["horizon_months"]
    n = len(next(iter(draws.values()))["lam"])
    gp_w = weekly_gp(alloc, draws, b["weeks"], int(np.ceil(H * WEEKS_PER_MONTH)), f["gross_margin"])
    transfer = np.exp(rng.normal(-0.5 * f["transfer_sd"] ** 2, f["transfer_sd"], n))   # mean 1
    gp = to_months(gp_w, H) * transfer[:, None]
    spend = spend_schedule(alloc, b["fixed"], b["weeks"], H)
    roi = gp.sum(axis=1) / spend.sum()
    v = npv(gp - spend, monthly_rate(f["annual_discount"]))
    return pd.DataFrame({"roi_multiple": roi, "npv": v})


def mc_summary(mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("roi_multiple", "npv"):
        x = mc[col]
        rows.append({"metric": col, "mean": x.mean(), "p10": x.quantile(0.10), "p50": x.quantile(0.50),
                     "p90": x.quantile(0.90), "prob_above_breakeven": float((x > (1 if col == "roi_multiple" else 0)).mean()),
                     "draws": len(x)})
    return pd.DataFrame(rows).round(4)
