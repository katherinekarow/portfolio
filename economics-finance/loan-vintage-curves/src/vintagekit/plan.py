"""Loss vs plan per vintage and expected loss (PD x LGD x EAD) on the outstanding book.

Plan: each vintage was planned to follow its segment's seasoned P50 default curve. Status uses the credibility
projection, so a vintage two months old is not flagged on a handful of early defaults:

    ratio = projected lifetime default / planned lifetime default
    on-pace  ratio <= 1.10      watch  1.10 < ratio <= 1.25      behind  ratio > 1.25 (worse than plan)

The gap to plan is split into the part already on the books and the part still to come:
    pace gap today        = C(n) - plan(n)
    projected lifetime gap = projected C(T) - plan(T)      (future gap = lifetime gap - pace gap)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import TERM

WATCH, BEHIND = 1.10, 1.25


def status(ratio):
    r = np.asarray(ratio)
    return np.where(r > BEHIND, "behind", np.where(r > WATCH, "watch", "on-pace"))


def compare(proj: pd.DataFrame, models: dict, lgd: pd.Series) -> pd.DataFrame:
    df = proj.copy()
    df["plan_to_date_pct"] = [models[s].r(m) for s, m in zip(df.segment, df.mob)]
    df["plan_lifetime_pct"] = [models[s].r(TERM) for s in df.segment]
    df["pace_gap_pct"] = df.cum_default_pct - df.plan_to_date_pct
    df["lifetime_gap_pct"] = df.proj_credibility - df.plan_lifetime_pct
    df["future_gap_pct"] = df.lifetime_gap_pct - df.pace_gap_pct
    df["plan_ratio"] = df.proj_credibility / df.plan_lifetime_pct
    df["status"] = status(df.plan_ratio)
    df["lgd"] = df.segment.map(lgd)
    df["lifetime_gap_net_loss"] = df.lifetime_gap_pct / 100 * df.orig_balance * df.lgd
    # expected loss on what is still outstanding: future defaulted $ spread over today's balance
    future_default = (df.proj_credibility - df.cum_default_pct) / 100 * df.orig_balance
    df["ead"] = df.balance
    df["pd_remaining"] = np.where(df.ead > 0, future_default / df.ead.where(df.ead > 0, 1), 0.0)
    df["expected_loss"] = df.pd_remaining * df.lgd * df.ead
    return df


def portfolio_el(df: pd.DataFrame) -> pd.DataFrame:
    """Expected loss by segment and in total: EL = sum PD x LGD x EAD, with EAD-weighted PD and LGD."""
    live = df[df.ead > 0]
    g = live.groupby("segment").apply(
        lambda x: pd.Series({"ead": x.ead.sum(), "pd": (x.pd_remaining * x.ead).sum() / x.ead.sum(),
                             "lgd": (x.lgd * x.pd_remaining * x.ead).sum() / (x.pd_remaining * x.ead).sum(),
                             "expected_loss": x.expected_loss.sum()}), include_groups=False)
    tot = pd.Series({"ead": g.ead.sum(), "pd": (g.pd * g.ead).sum() / g.ead.sum(),
                     "lgd": (g.lgd * g.pd * g.ead).sum() / (g.pd * g.ead).sum(),
                     "expected_loss": g.expected_loss.sum()}, name="Portfolio")
    out = pd.concat([g, tot.to_frame().T])
    out["el_pct_of_ead"] = 100 * out.expected_loss / out.ead
    return out
