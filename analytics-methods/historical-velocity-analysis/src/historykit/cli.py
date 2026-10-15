"""python -m historykit run --out sample_output --figures figures"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import change, cohort, decompose, forecast, history, synth, velocity


def current_pace(book: synth.Book, weeks: int = 52) -> pd.DataFrame:
    """Pace and projection for every future service date on the books at the as-of date."""
    C = velocity.cumulative(book.gross)
    V = velocity.velocity(C)
    n = book.n_hist
    rows = []
    for d in range(n, len(book.dates)):
        k = (book.dates[d] - book.asof).days
        ref = velocity.reference_rows(book.dates, book.dates[d], n, weeks)
        pr = velocity.pace_projection(C[d, k], V[ref], k)
        rows.append({"service_date": book.dates[d].date(), "weekday": book.dates[d].day_name()[:3],
                     "days_out": k, "orders_so_far": int(C[d, k]),
                     "pace_vs_history": velocity.pace_vs_history(C[d, k], C[ref], k),
                     "proj_low80": pr["low80"], "proj_low50": pr["low"], "projection": pr["point"],
                     "proj_high50": pr["high"], "proj_high80": pr["high80"],
                     "actual_final_hidden": int(C[d, 0])})
    return pd.DataFrame(rows)


def run(out: Path, figures: Path | None = None, seed: int = 7) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    book = synth.make_book(seed)
    daily = book.daily()
    n = book.n_hist

    # 1. historical comparisons
    tab = history.comparison_table(daily)
    tab.round(4).to_csv(out / "daily_comparisons.csv", index_label="date")
    yearly = daily.groupby(daily.index.year).agg(["sum", "count"])

    # 2. velocity
    C = velocity.cumulative(book.gross)
    V = velocity.velocity(C)
    ref_all = np.arange(n - 364, n)
    bands_all = velocity.percentiles(V[ref_all])
    pd.DataFrame({"days_out": np.arange(V.shape[1]), "P25": bands_all[0], "P50": bands_all[1],
                  "P75": bands_all[2]}).round(4).to_csv(out / "velocity_percentiles.csv", index=False)
    pace = current_pace(book)
    pace.round(1).to_csv(out / "current_pace.csv", index=False)
    pbt = velocity.backtest_projection(book.gross, book.dates, range(n - 364, n))
    pbt_sum = pbt.groupby("days_out").agg(mean_error=("rel_error", "mean"), median_error=("rel_error", "median"),
                                          mean_abs_error=("rel_error", lambda e: e.abs().mean()),
                                          coverage_50=("inside", "mean"),
                                          coverage_80=("inside80", "mean"), n=("rel_error", "size"))
    pbt_sum.round(4).to_csv(out / "projection_backtest.csv")

    target_date = book.asof + pd.Timedelta(days=21)
    while target_date.weekday() != 1:
        target_date += pd.Timedelta(days=1)
    d = int(np.where(book.dates == target_date)[0][0])
    k = (target_date - book.asof).days
    ref = velocity.reference_rows(book.dates, target_date, n)
    bands = velocity.percentiles(V[ref])
    pr = velocity.pace_projection(C[d, k], V[ref], k)
    target = {"date": target_date, "k": k, "weekday": target_date.day_name(), "observed_curve": C[d],
              "hist_median": np.median(C[ref], axis=0), "actual": C[d, 0],
              "pace": velocity.pace_vs_history(C[d, k], C[ref], k), **pr}

    # 3. decomposition
    weekly = daily.resample("W-SUN").sum()
    dec = decompose.decompose(weekly.to_numpy(), 52, "multiplicative")
    pd.DataFrame({k_: dec[k_] for k_ in ("observed", "trend", "seasonal", "resid")}, index=weekly.index) \
        .round(4).to_csv(out / "decomposition_weekly.csv", index_label="week_ending")
    wk_est = decompose.weekday_indices(daily)
    wk_true = dict(zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], synth.WEEKDAY / synth.WEEKDAY.mean()))
    pd.DataFrame({"estimated": wk_est, "planted": wk_true}).round(4).to_csv(out / "weekday_indices.csv",
                                                                             index_label="weekday")
    dec7 = decompose.decompose(daily.to_numpy(), 7, "multiplicative")
    flags = change.anomaly_flags(np.log(dec7["resid"]))
    anomalies = pd.DataFrame({"date": daily.index[flags].date, "orders": daily.to_numpy()[flags],
                              "robust_z": change.robust_z(np.log(dec7["resid"]))[flags].round(2),
                              "is_holiday": daily.index[flags].isin(book.holidays)})
    anomalies.to_csv(out / "anomalies.csv", index=False)

    # 4. forecasting backtest
    fbt = forecast.rolling_origin(daily)
    acc = forecast.accuracy(fbt)
    acc.round(3).to_csv(out / "forecast_accuracy.csv", index=False)

    # 5. change detection
    det = change.detect_shift(daily, book.holidays)

    # 6. cohorts
    tri = cohort.service_week_triangle(book)
    tri.to_csv(out / "service_week_triangle.csv")
    cohort.order_week_cohorts(book).to_csv(out / "order_week_cohorts.csv")

    hol_in = pd.DatetimeIndex(book.holidays)
    hol_in = hol_in[(hol_in >= daily.index[0]) & (hol_in <= daily.index[-1])]
    summary = {
        "history": f"{daily.index[0].date()} to {daily.index[-1].date()} ({n} days)",
        "orders_by_year": {int(y): int(v) for y, v in yearly["sum"].items()},
        "yoy_variance_weekday": float(tab.yoy_weekday.var()), "yoy_variance_calendar": float(tab.yoy_calendar.var()),
        "yoy_roll28_before_shift": float(tab.yoy_roll28.loc["2024-06-01":"2025-04-13"].mean()),
        "yoy_roll28_after_shift": float(tab.yoy_roll28.loc["2025-05-12":].mean()),
        "velocity_P50_at_7_14_28": [float(bands_all[1][j]) for j in (7, 14, 28)],
        "current_target": {"date": str(target_date.date()), "days_out": k, "orders_so_far": int(C[d, k]),
                           "pace_vs_history": target["pace"], "projection": pr["point"], "low": pr["low"],
                           "high": pr["high"], "low80": pr["low80"], "high80": pr["high80"],
                           "actual": int(C[d, 0])},
        "projection_backtest": pbt_sum.round(4).reset_index().to_dict("records"),
        "forecast_accuracy": acc.round(3).to_dict("records"),
        "cusum_alarm": str(det["alarm_up"].date()) if det["alarm_up"] is not None else None,
        "cusum_delay_days": (det["alarm_up"] - book.shift_date).days if det["alarm_up"] is not None else None,
        "shift_date": str(book.shift_date.date()),
        "anomalies": int(flags.sum()), "anomalies_holidays": int(anomalies.is_holiday.sum()),
        "holidays_in_history": len(hol_in),
        "weekday_indices": {k_: round(float(v), 3) for k_, v in wk_est.items()},
        "cancel_rate": float(book.cancels[:n].sum() / book.gross[:n].sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if figures is not None:
        from . import plots
        figures = Path(figures)
        figures.mkdir(parents=True, exist_ok=True)
        plots.velocity_figure(V[ref], bands, target, figures / "velocity_curves.png")
        plots.decomposition_figure(weekly.index, dec, figures / "decomposition.png", book.shift_date)
        plots.weekday_figure(wk_est, wk_true, figures / "weekday_indices.png")
        plots.backtest_figure(acc, figures / "backtest_accuracy.png")
        plots.yoy_figure(tab, det, book.shift_date, figures / "yoy_level_shift.png")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="historykit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate data, run every analysis, write CSVs and figures")
    r.add_argument("--out", default="sample_output")
    r.add_argument("--figures", default=None)
    r.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    s = run(Path(a.out), Path(a.figures) if a.figures else None, a.seed)
    print(json.dumps({k: s[k] for k in ("history", "cusum_alarm", "cusum_delay_days", "current_target")},
                     indent=2, default=str))
