"""Price index: our price relative to the market, per lot and tier, per day.

    index = our price / median of matched competitors' usable prices (same quote day)

Only usable observations count (no outliers, no stale days). A competitor whose latest
usable price is older than MAX_AGE days is dropped from that day's comparison rather than
carried forward indefinitely. A cell needs at least MIN_COMPETITORS to get an index at all.
Positioning: below market (< 0.95), at market, above market (> 1.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_AGE = 2
MIN_COMPETITORS = 2


def daily_index(our_prices: pd.DataFrame, matches: pd.DataFrame, clean_obs: pd.DataFrame, days: list) -> pd.DataFrame:
    usable = clean_obs[clean_obs.usable][["quote_date", "product_id", "price_usd"]]
    m = matches[["lot_id", "tier", "competitor", "product_id"]]
    # forward-fill each product's last usable price for at most MAX_AGE days
    full = (usable.pivot_table(index="quote_date", columns="product_id", values="price_usd", aggfunc="first")
            .reindex(days).ffill(limit=MAX_AGE))
    comp = full.stack().rename("price_usd").reset_index().rename(columns={"level_0": "quote_date"})
    comp.columns = ["date", "product_id", "price_usd"]
    comp = comp.merge(m, on="product_id")
    agg = comp.groupby(["date", "lot_id", "tier"]).agg(market_median=("price_usd", "median"),
                                                              competitors=("competitor", "nunique"),
                                                              market_min=("price_usd", "min"),
                                                              market_max=("price_usd", "max")).reset_index()
    out = our_prices.merge(agg, on=["date", "lot_id", "tier"], how="left")
    out["competitors"] = out.competitors.fillna(0).astype(int)
    ok = out.competitors >= MIN_COMPETITORS
    out["price_index"] = np.where(ok, (out.price / out.market_median).round(3), np.nan)
    out["position"] = np.select([~ok, out.price_index < 0.95, out.price_index > 1.05],
                                ["not enough comparables", "below market", "above market"], "at market")
    return out


def latest_positioning(idx: pd.DataFrame) -> pd.DataFrame:
    last = idx[idx.date == idx.date.max()]
    return last.pivot_table(index="tier", columns="position", values="lot_id", aggfunc="count", fill_value=0)
