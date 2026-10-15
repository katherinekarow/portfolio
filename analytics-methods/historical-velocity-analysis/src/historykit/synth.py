"""Synthetic appointment book for a network of clinics.

Every service date gets an expected demand built from known parts:

    expected = base x trend x weekday x annual x holiday x level_shift x day_noise

The orders for a date are then spread over lead times (days between the order and
the visit) from a two-part mix: short-notice orders and planned orders. A
share of orders cancel later, before the visit. Because every part is planted,
the methods in this package can be checked against the truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

START = "2023-01-02"          # a Monday
ASOF = "2025-12-28"           # a Sunday: 156 complete weeks of history
SHIFT_DATE = "2025-04-14"     # a new clinic joins the network
SHIFT_SIZE = 0.12             # +12% level shift
LEADS = 60                    # lead times 0..59 days
BASE = 220.0
TREND = 0.06                  # +6% a year
WEEKDAY = np.array([1.25, 1.15, 1.10, 1.05, 0.95, 0.60, 0.35])  # Mon..Sun
HOLIDAY_FACTOR = 0.25
CANCEL_RATE = 0.07


def holidays(years) -> list[pd.Timestamp]:
    """Fixed-date and moving public holidays (generic, not tied to a country)."""
    out = []
    for y in years:
        out += [pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26)]
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31")
        out.append(may[may.weekday == 0][-1])                  # last Monday of May
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30")
        out.append(sep[sep.weekday == 0][0])                   # first Monday of September
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        out.append(nov[nov.weekday == 3][3])                   # fourth Thursday of November
    return sorted(out)


def lead_pmfs(n: int = LEADS):
    """Short-notice (geometric, mean ~2.5 days) and planned (negative binomial, mean ~21 days)."""
    lead = np.arange(n)
    p = 1 / 3.5
    short = (1 - p) ** lead * p
    r, mean = 4.0, 21.0
    q = r / (r + mean)
    planned = np.array([math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                                 + r * math.log(q) + k * math.log(1 - q)) for k in lead])
    return short / short.sum(), planned / planned.sum()


@dataclass
class Book:
    dates: pd.DatetimeIndex        # service dates (history + the next LEADS days)
    gross: np.ndarray              # (n_dates, LEADS) orders received with lead l
    cancels: np.ndarray            # (n_dates, LEADS) cancellations made l days out
    expected: np.ndarray           # planted expected gross demand
    asof: pd.Timestamp
    shift_date: pd.Timestamp
    holidays: list = field(default_factory=list)

    @property
    def n_hist(self) -> int:
        """Number of service dates that are complete at the as-of date."""
        return int((self.dates <= self.asof).sum())

    def observed_mask(self) -> np.ndarray:
        """True where a (service date, lead) cell was received on or before the as-of date."""
        days_out = (self.dates - self.asof).days.to_numpy()[:, None]
        return np.arange(self.gross.shape[1])[None, :] >= days_out

    def net_final(self) -> np.ndarray:
        return self.gross.sum(1) - self.cancels.sum(1)

    def daily(self) -> pd.Series:
        """Net orders (after cancellations) for complete service dates."""
        n = self.n_hist
        return pd.Series(self.net_final()[:n], index=self.dates[:n], name="orders")


def make_book(seed: int = 7, start: str = START, asof: str = ASOF, shift_date: str = SHIFT_DATE,
              shift_size: float = SHIFT_SIZE, leads: int = LEADS) -> Book:
    rng = np.random.default_rng(seed)
    asof_ts = pd.Timestamp(asof)
    dates = pd.date_range(start, asof_ts + pd.Timedelta(days=leads - 1), freq="D")
    n = len(dates)
    t_years = (dates - pd.Timestamp(start)).days.to_numpy() / 365.25
    doy = dates.dayofyear.to_numpy()
    trend = (1 + TREND) ** t_years
    annual = 1 + 0.12 * np.cos(2 * np.pi * (doy - 20) / 365.25) + 0.04 * np.cos(4 * np.pi * (doy - 20) / 365.25)
    weekday = WEEKDAY[dates.weekday.to_numpy()]
    hol = holidays(sorted(set(dates.year)))
    is_hol = dates.isin(hol)
    holiday = np.where(is_hol, HOLIDAY_FACTOR, 1.0)
    shift = np.where(dates >= pd.Timestamp(shift_date), 1 + shift_size, 1.0)
    noise = np.exp(rng.normal(0, 0.04, n))
    expected = BASE * trend * annual * weekday * holiday * shift
    total = rng.poisson(expected * noise)

    short_pmf, planned_pmf = lead_pmfs(leads)
    w_short = rng.beta(9, 21, n)                       # share of short-notice orders varies by day
    n_short = rng.binomial(total, w_short)
    gross = rng.multinomial(n_short, short_pmf) + rng.multinomial(total - n_short, planned_pmf)

    cancels = np.zeros_like(gross)
    for lead in range(leads):                         # an order placed l days out cancels 0..l days out
        nc = rng.binomial(gross[:, lead], CANCEL_RATE)
        cancels[:, : lead + 1] += rng.multinomial(nc, np.full(lead + 1, 1 / (lead + 1)))
    return Book(dates, gross, cancels, expected, asof_ts, pd.Timestamp(shift_date), hol)
