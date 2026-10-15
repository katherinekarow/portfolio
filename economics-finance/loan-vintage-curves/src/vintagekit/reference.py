"""Segment reference curves from seasoned vintages: P25 / P50 / P75 of the cumulative curve at each MOB."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import TERM


def seasoned(curves: pd.DataFrame, term: int = TERM) -> pd.Series:
    """Vintages observed for the full term, per segment (a boolean per row)."""
    max_mob = curves.groupby(["segment", "vintage"]).mob.transform("max")
    return max_mob >= term


def reference(curves: pd.DataFrame, exclude: set | None = None, term: int = TERM) -> pd.DataFrame:
    """Percentile bands per segment and MOB, from seasoned vintages (optionally leaving some out).

    Cumulative curves can only rise, but percentiles taken MOB by MOB across a small set of vintages can dip.
    Each band is therefore forced monotone with a running maximum. Because P25 <= P50 <= P75 holds at every MOB
    before the running maximum, it still holds after it. Balance runoff gets a running minimum instead.
    """
    c = curves[seasoned(curves, term) & (curves.mob <= term)]
    if exclude:
        c = c[~c.set_index(["segment", "vintage"]).index.isin(list(exclude))]
    out = []
    for seg, g in c.groupby("segment"):
        tri = g.pivot(index="vintage", columns="mob", values="cum_default_pct")
        q = np.nanpercentile(tri.to_numpy(), [25, 50, 75], axis=0)
        q = np.maximum.accumulate(q, axis=1)
        prep = np.maximum.accumulate(g.groupby("mob").cum_prepay_pct.median().to_numpy())
        bal = np.minimum.accumulate(g.groupby("mob").balance_pct.median().to_numpy())
        loss = np.maximum.accumulate(g.groupby("mob").cum_net_loss_pct.median().to_numpy())
        df = pd.DataFrame({"segment": seg, "mob": tri.columns.to_numpy(), "p25": q[0], "p50": q[1], "p75": q[2],
                           "prepay_p50": prep, "balance_p50": bal, "net_loss_p50": loss,
                           "n_vintages": len(tri)})
        zero = pd.DataFrame({"segment": [seg], "mob": [0], "p25": [0.0], "p50": [0.0], "p75": [0.0],
                             "prepay_p50": [0.0], "balance_p50": [100.0], "net_loss_p50": [0.0],
                             "n_vintages": [len(tri)]})
        out.append(pd.concat([zero, df], ignore_index=True))
    return pd.concat(out, ignore_index=True)


def lgd(curves: pd.DataFrame, term: int = TERM) -> pd.Series:
    """Loss given default per segment, from seasoned vintages: net loss / defaulted balance."""
    c = curves[seasoned(curves, term)]
    s = c.groupby("segment")[["net_loss", "default_bal"]].sum()
    return (s.net_loss / s.default_bal).rename("lgd")
