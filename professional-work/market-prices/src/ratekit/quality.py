"""Feed quality: clean the competitor observations before anyone compares prices with them.

  * currency      every price converted to USD
  * duplicates    identical deliveries collapsed
  * outliers      a price must be BOTH statistically extreme for that product (robust z > 6 against its
                  rolling median) AND economically implausible (more than ~65% away from it). The second test
                  stops quiet products - whose spread is tiny - from flagging ordinary moves.
  * stale feed    a competitor whose prices do not change AT ALL for >= 3 consecutive quote days
                  while the products normally move daily: the vendor is re-sending an old file
  * missing days  quote days with no delivery from a competitor
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import FX

STALE_MIN_DAYS = 3
OUTLIER_Z = 6.0
OUTLIER_MIN_LOG_JUMP = 0.5          # |log(price / rolling median)| > 0.5  ~  more than +65% / -40%


def clean(obs: pd.DataFrame, all_days: list) -> tuple[pd.DataFrame, dict]:
    o = obs.copy()
    o["price_usd"] = (o.price * o.currency.map(FX)).round(2)
    n0 = len(o)
    o = o.drop_duplicates(["quote_date", "competitor", "product_id", "price", "currency"])
    report = {"duplicates_removed": n0 - len(o)}

    # outliers vs each product's centred rolling median (on logs, scaled by the product's robust spread)
    o = o.sort_values(["product_id", "quote_date"])
    lp = np.log(o.price_usd)
    med = lp.groupby(o.product_id).transform(lambda s: s.rolling(9, center=True, min_periods=3).median())
    dev = lp - med
    mad = dev.abs().groupby(o.product_id).transform("median").clip(lower=0.005)
    o["robust_z"] = (dev / (1.4826 * mad)).round(2)
    o["is_outlier"] = (o.robust_z.abs() > OUTLIER_Z) & (dev.abs() > OUTLIER_MIN_LOG_JUMP)

    # stale runs: whole-competitor snapshot identical to the previous quote day
    stale_rows, missing = [], {}
    for comp, g in o[~o.is_outlier].groupby("competitor"):
        snap = g.pivot_table(index="quote_date", columns="product_id", values="price_usd", aggfunc="first").sort_index()
        days = list(snap.index)
        missing[comp] = [d for d in all_days if d not in set(days)]
        unchanged = (snap.diff().abs().fillna(1) < 1e-9).mean(axis=1) > 0.95     # >95% of products identical to yesterday
        run = []
        for d, flag in zip(days, unchanged):
            if flag:
                run.append(d)
            else:
                if len(run) >= STALE_MIN_DAYS:
                    stale_rows += [{"competitor": comp, "quote_date": x} for x in run]
                run = []
        if len(run) >= STALE_MIN_DAYS:
            stale_rows += [{"competitor": comp, "quote_date": x} for x in run]
    stale = pd.DataFrame(stale_rows, columns=["competitor", "quote_date"])
    o = o.merge(stale.assign(is_stale=True), on=["competitor", "quote_date"], how="left")
    o["is_stale"] = o.is_stale.astype("boolean").fillna(False).astype(bool)
    o["usable"] = ~o.is_outlier & ~o.is_stale
    report.update({"outliers": int(o.is_outlier.sum()), "stale_days": stale.groupby("competitor").quote_date.apply(list).to_dict(),
                   "missing_days": {k: v for k, v in missing.items() if v}})
    return o.reset_index(drop=True), report
