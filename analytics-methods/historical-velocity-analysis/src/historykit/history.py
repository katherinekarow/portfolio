"""Historical comparisons: last-year alignment, YoY, rolling sums, moving averages, indexing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def last_year(series: pd.Series, method: str = "weekday") -> pd.Series:
    """The comparable value from last year for every date.

    weekday:  the date 364 days earlier (52 weeks), so a Monday is compared with a Monday.
    calendar: the same calendar date one year earlier (29 Feb has no match -> NaN).
    """
    s = series.asfreq("D")
    if method == "weekday":
        return s.shift(364)
    if method == "calendar":
        prev = s.index - pd.DateOffset(years=1)
        out = s.reindex(prev).to_numpy()
        leap = (s.index.month == 2) & (s.index.day == 29)
        out = np.where(leap, np.nan, out)
        return pd.Series(out, index=s.index, name=series.name)
    raise ValueError(method)


def yoy(series: pd.Series, method: str = "weekday") -> pd.Series:
    """Year-over-year change: x_t / x_last_year - 1."""
    return series.asfreq("D") / last_year(series, method) - 1


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Trailing sum of the last `window` days (NaN until the window is full)."""
    return series.rolling(window, min_periods=window).sum()


def moving_average(series: pd.Series, window: int, centred: bool = False) -> pd.Series:
    """Trailing (or centred) simple moving average."""
    return series.rolling(window, min_periods=window, center=centred).mean()


def index_to_base(series: pd.Series, start, end) -> pd.Series:
    """Rescale so the average over [start, end] equals 100."""
    base = series.loc[start:end].mean()
    return series / base * 100


def comparison_table(daily: pd.Series) -> pd.DataFrame:
    """Daily table with every comparison used in the README."""
    df = pd.DataFrame({"orders": daily})
    df["weekday"] = df.index.day_name().str[:3]
    df["roll7"] = rolling_sum(daily, 7)
    df["roll28"] = rolling_sum(daily, 28)
    df["ma7"] = moving_average(daily, 7)
    df["ly_weekday"] = last_year(daily, "weekday")
    df["ly_calendar"] = last_year(daily, "calendar")
    df["yoy_weekday"] = df["orders"] / df["ly_weekday"] - 1
    df["yoy_calendar"] = df["orders"] / df["ly_calendar"] - 1
    df["yoy_roll28"] = df["roll28"] / df["roll28"].shift(364) - 1
    first_year = daily.index[0] + pd.Timedelta(days=363)
    df["index_ma28"] = index_to_base(moving_average(daily, 28), daily.index[0], first_year)
    return df
