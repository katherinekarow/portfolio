"""Amortisation maths and cumulative vintage curves (% of original balance by months on book)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def payment(principal: float, annual_rate: float, term: int) -> float:
    """Level monthly payment: P * r / (1 - (1 + r)^-n)."""
    r = annual_rate / 12
    return principal / term if r == 0 else principal * r / (1 - (1 + r) ** -term)


def remaining_balance(principal: float, annual_rate: float, term: int, t):
    """Scheduled balance after t payments: P * ((1+r)^n - (1+r)^t) / ((1+r)^n - 1)."""
    r = annual_rate / 12
    t = np.asarray(t, dtype=float)
    if r == 0:
        return principal * (1 - t / term)
    g = 1 + r
    return principal * (g ** term - g ** t) / (g ** term - 1)


def schedule(principal: float, annual_rate: float, term: int) -> pd.DataFrame:
    """Month-by-month amortisation table."""
    pay, r = payment(principal, annual_rate, term), annual_rate / 12
    bal, rows = principal, []
    for m in range(1, term + 1):
        interest = bal * r
        princ = pay - interest
        bal -= princ
        rows.append((m, pay, interest, princ, max(bal, 0.0)))
    return pd.DataFrame(rows, columns=["month", "payment", "interest", "principal", "balance"])


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative default, prepayment and net loss, and balance runoff, as % of original balance."""
    df = panel.sort_values(["segment", "vintage", "mob"]).copy()
    g = df.groupby(["segment", "vintage"])
    for src, dst in [("default_bal", "cum_default"), ("prepay_bal", "cum_prepay"), ("net_loss", "cum_net_loss")]:
        df[dst + "_bal"] = g[src].cumsum()
        df[dst + "_pct"] = 100 * df[dst + "_bal"] / df.orig_balance
    df["balance_pct"] = 100 * df.balance / df.orig_balance
    return df.reset_index(drop=True)


def triangle(curves: pd.DataFrame, value: str = "cum_default_pct") -> pd.DataFrame:
    """Vintage x MOB table (NaN where not yet observed)."""
    return curves.pivot(index="vintage", columns="mob", values=value)
