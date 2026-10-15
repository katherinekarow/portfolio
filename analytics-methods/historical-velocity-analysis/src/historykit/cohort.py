"""Cohort / vintage views of the order book.

Service-week triangle: rows = service week (the cohort), columns = weeks out (9 .. 0),
cell = orders received by the end of the week that many weeks before. Weeks whose order
cut-off is after the as-of date are unknown, which gives the triangle shape.

Order-week cohorts: rows = the week an order was placed, columns = how many weeks later
the visit is. Reading it along a row shows where one week's orders land.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import Book


def _week_cells(book: Book, net: bool):
    origin = book.dates[0] - pd.Timedelta(days=book.dates[0].weekday())   # Monday
    sday = (book.dates - origin).days.to_numpy()
    lead = np.arange(book.gross.shape[1])
    sweek = sday // 7
    bweek = (sday[:, None] - lead[None, :]) // 7
    vals = book.gross - (book.cancels if net else 0)
    asof_week = (book.asof - origin).days // 7
    return origin, sweek, bweek, vals, asof_week


def service_week_triangle(book: Book, net: bool = False) -> pd.DataFrame:
    origin, sweek, bweek, vals, asof_week = _week_cells(book, net)
    wo = sweek[:, None] - bweek                       # weeks out when the order arrived
    n_w, n_j = sweek.max() + 1, wo.max() + 1
    inc = np.zeros((n_w, n_j))
    np.add.at(inc, (np.broadcast_to(sweek[:, None], wo.shape), wo), vals)
    cum = np.cumsum(inc[:, ::-1], axis=1)[:, ::-1]    # cum[w, j] = received with >= j weeks out
    known = (np.arange(n_w)[:, None] - np.arange(n_j)[None, :]) <= asof_week
    tri = np.where(known, cum, np.nan)
    first_full = int(sweek[0]) if book.dates[0].weekday() == 0 else int(sweek[0]) + 1
    idx = origin + pd.to_timedelta(np.arange(n_w) * 7, unit="D")
    df = pd.DataFrame(tri[:, ::-1], index=idx, columns=[f"{j}w" for j in range(n_j - 1, -1, -1)])
    df.index.name = "service_week"
    last_full = (book.dates[-1] - origin).days // 7
    if book.dates[-1].weekday() != 6:
        last_full -= 1
    return df.iloc[first_full: last_full + 1]


def order_week_cohorts(book: Book, net: bool = False) -> pd.DataFrame:
    origin, sweek, bweek, vals, asof_week = _week_cells(book, net)
    lag = sweek[:, None] - bweek
    shift = -bweek.min()
    n_b, n_j = bweek.max() + shift + 1, lag.max() + 1
    tab = np.zeros((n_b, n_j))
    np.add.at(tab, (bweek + shift, lag), vals)
    idx = origin + pd.to_timedelta((np.arange(n_b) - shift) * 7, unit="D")
    df = pd.DataFrame(tab, index=idx, columns=[f"+{j}w" for j in range(n_j)])
    df.index.name = "order_week"
    keep = (df.index >= book.dates[0] + pd.Timedelta(weeks=9)) & (df.index <= book.asof)
    return df.loc[keep]
