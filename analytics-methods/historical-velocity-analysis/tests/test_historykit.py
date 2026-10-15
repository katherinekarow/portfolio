import json

import numpy as np
import pandas as pd
import pytest

from historykit import change, cli, cohort, decompose, forecast, history, metrics, synth, velocity


# ---------- velocity ----------
def test_velocity_monotone_and_ends_at_one(book):
    V = velocity.velocity(velocity.cumulative(book.gross))[: book.n_hist]
    assert np.allclose(V[:, 0], 1.0)
    assert np.all(np.diff(V, axis=1) <= 1e-12)          # share falls as days-out grows
    assert V.min() >= 0


def test_percentiles_ordered_and_monotone(book):
    V = velocity.velocity(velocity.cumulative(book.gross))[: book.n_hist]
    p = velocity.percentiles(V, (10, 25, 50, 75, 90))
    assert np.all(np.diff(p, axis=0) >= -1e-12)          # P10 <= P25 <= ... <= P90 at every lead
    assert np.all(np.diff(p, axis=1) <= 1e-12)           # each percentile curve is monotone too


def test_cumulative_and_pickup_add_up(book):
    C = velocity.cumulative(book.gross)
    assert np.array_equal(C[:, 0], book.gross.sum(1))
    assert np.array_equal(C[:, 10], book.gross[:, 10:].sum(1))
    net = velocity.pickup(book.gross, book.cancels)
    assert np.array_equal(net.sum(1), book.net_final())


def test_pace_projection_exact_on_a_typical_curve():
    V_ref = np.array([[1, 0.6, 0.2], [1, 0.5, 0.1], [1, 0.4, 0.05]])
    pr = velocity.pace_projection(50, V_ref, 1)          # median share at k=1 is 0.5
    assert pr["point"] == pytest.approx(100)
    assert pr["low"] <= pr["point"] <= pr["high"]
    assert pr["low80"] <= pr["low"] and pr["high"] <= pr["high80"]
    C_ref = V_ref * np.array([[200], [100], [50]])
    assert velocity.pace_vs_history(60, C_ref, 1) == pytest.approx(60 / 50)


def test_pace_projection_unbiased_on_held_out_dates(book):
    n = book.n_hist
    bt = velocity.backtest_projection(book.gross, book.dates, range(n - 364, n), days_out=(7, 14, 28))
    for k, g in bt.groupby("days_out"):
        assert abs(g.rel_error.mean()) < 0.02, k
        assert abs(g.rel_error.median()) < 0.02, k
        assert 0.42 <= g.inside.mean() <= 0.58, k        # P25-P75 interval should hold ~50%
        assert 0.72 <= g.inside80.mean() <= 0.88, k


# ---------- historical comparisons ----------
def test_weekday_aligned_yoy_removes_weekday_artefact(daily):
    wk = history.yoy(daily, "weekday").dropna()
    cal = history.yoy(daily, "calendar").dropna()
    assert wk.var() < cal.var() / 5
    # 364 days back is always the same weekday; the calendar date one year back is not
    ly = history.last_year(daily, "weekday").dropna()
    assert (ly.index.weekday == (ly.index - pd.Timedelta(days=364)).weekday).all()


def test_rolling_moving_average_and_index_by_hand():
    s = pd.Series([1.0, 2, 3, 4, 5, 6], index=pd.date_range("2024-01-01", periods=6))
    assert history.rolling_sum(s, 3).tolist()[2:] == [6, 9, 12, 15]
    assert history.moving_average(s, 3).iloc[-1] == 5
    assert history.moving_average(s, 3, centred=True).iloc[1] == 2
    idx = history.index_to_base(s, "2024-01-01", "2024-01-02")    # base mean 1.5
    assert idx.iloc[0] == pytest.approx(100 / 1.5) and idx.iloc[2] == pytest.approx(200)
    assert history.yoy(pd.Series(np.arange(1.0, 400), index=pd.date_range("2024-01-01", periods=399))).iloc[364] \
        == pytest.approx(365 / 1 - 1)


def test_yoy_recovers_planted_growth(daily):
    tab = history.comparison_table(daily)
    before = tab.yoy_roll28.loc["2024-06-01":"2025-04-13"].mean()
    after = tab.yoy_roll28.loc["2025-05-12":].mean()
    assert before == pytest.approx(synth.TREND, abs=0.02)
    assert after == pytest.approx((1 + synth.TREND) * (1 + synth.SHIFT_SIZE) - 1, abs=0.03)


# ---------- decomposition ----------
def test_centred_moving_average_by_hand():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    ma4 = decompose.centred_ma(y, 4)                     # 2x4 MA reproduces a straight line
    assert np.isnan(ma4[:2]).all() and np.isnan(ma4[-2:]).all()
    assert np.allclose(ma4[2:-2], y[2:-2])
    y2 = np.array([4.0, 0, 8, 4, 0, 8, 4])
    assert decompose.centred_ma(y2, 3)[1] == pytest.approx(4)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
def test_decomposition_normalised_and_recombines(daily, model):
    dec = decompose.decompose(daily.to_numpy(), 7, model)
    if model == "additive":
        assert dec["indices"].sum() == pytest.approx(0, abs=1e-9)
        rebuilt = dec["trend"] + dec["seasonal"] + dec["resid"]
    else:
        assert dec["indices"].mean() == pytest.approx(1, abs=1e-12)
        rebuilt = dec["trend"] * dec["seasonal"] * dec["resid"]
    ok = ~np.isnan(rebuilt)
    assert ok.sum() == len(daily) - 6
    assert np.allclose(rebuilt[ok], dec["observed"][ok])


def test_weekday_indices_match_planted(daily):
    est = decompose.weekday_indices(daily)
    truth = synth.WEEKDAY / synth.WEEKDAY.mean()
    for name, t in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], truth):
        assert est[name] == pytest.approx(t, rel=0.05), name


# ---------- forecasting ----------
def test_holt_winters_matches_hand_computed_example():
    # m = 2, alpha = beta = gamma = 0.5, l0 = 10, b0 = 1, s = [-1, 1]; worked by hand:
    # t0: f = 10+1-1 = 10;           l = .5(10+1)+.5(11) = 11;          b = 1;        s0 = .5(10-11)+.5(-1) = -1
    # t1: f = 11+1+1 = 13;           l = .5(14-1)+.5(12) = 12.5;        b = 1.25;     s1 = .5(14-12.5)+.5(1) = 1.25
    # t2: f = 12.5+1.25-1 = 12.75;   l = .5(11+1)+.5(13.75) = 12.875;   b = 0.8125;   s0 = -1.4375
    # t3: f = 12.875+.8125+1.25 = 14.9375; l = 14.21875;                b = 1.078125; s1 = 1.515625
    # t4: f = 14.21875+1.078125-1.4375 = 13.859375; l = 14.8671875;     b = 0.86328125; s0 = -1.65234375
    # next: 14.8671875 + 0.86328125 + 1.515625 = 17.24609375
    y = [10, 14, 11, 16, 13]
    r = forecast.holt_winters(y, 2, 0.5, 0.5, 0.5, h=2, init=(10, 1, [-1, 1]))
    assert np.allclose(r["fitted"], [10, 13, 12.75, 14.9375, 13.859375])
    assert r["level"] == pytest.approx(14.8671875)
    assert r["trend"] == pytest.approx(0.86328125)
    assert np.allclose(r["season"], [-1.65234375, 1.515625])
    assert r["forecast"][0] == pytest.approx(17.24609375)
    assert r["forecast"][1] == pytest.approx(14.8671875 + 2 * 0.86328125 - 1.65234375)


def test_holt_winters_vectorised_grid_equals_scalar_runs(daily):
    y = daily.to_numpy()[:200]
    grid = np.array([[0.1, 0.0, 0.1], [0.3, 0.05, 0.2]])
    for seasonal in ("additive", "multiplicative"):
        vec = forecast.holt_winters(y, 7, grid[:, 0], grid[:, 1], grid[:, 2], 7, seasonal)
        for j, (a, b, g) in enumerate(grid):
            one = forecast.holt_winters(y, 7, a, b, g, 7, seasonal)
            assert np.allclose(vec["forecast"][:, j], one["forecast"])


def test_holt_winters_learns_a_clean_seasonal_trend():
    t = np.arange(140)
    y = 100 + 0.5 * t + np.tile([10, 5, 0, -5, -10, 0, 0], 20)
    fc = forecast.holt_winters(y, 7, 0.3, 0.1, 0.2, h=7)["forecast"]
    t_new = np.arange(140, 147)
    assert np.allclose(fc, 100 + 0.5 * t_new + np.array([10, 5, 0, -5, -10, 0, 0]), atol=1e-6)


def test_ses_and_naive_by_hand():
    fitted, fc = forecast.ses([10, 20, 30], 0.5, 2)
    assert np.allclose(fitted, [10, 10, 15]) and np.allclose(fc, [22.5, 22.5])
    assert forecast.naive([1, 2, 3], 2).tolist() == [3, 3]
    assert forecast.seasonal_naive([1, 2, 3, 4, 5], 4, 2).tolist() == [4, 5, 4, 5]


def test_metrics_by_hand():
    y, f = [100, 200, 50], [110, 180, 50]
    assert metrics.mape(y, f) == pytest.approx((0.1 + 0.1 + 0) / 3)
    assert metrics.wape(y, f) == pytest.approx(30 / 350)
    assert metrics.bias(y, f) == pytest.approx(-10 / 350)
    train = [10, 12, 11, 15]                             # |12-10|, |11-12|, |15-11| -> mean 7/3
    assert metrics.mase_scale(train, 1) == pytest.approx(7 / 3)
    assert metrics.mase(y, f, train, 1) == pytest.approx((30 / 3) / (7 / 3))
    assert metrics.mape([0, 10], [5, 12]) == pytest.approx(0.2)   # zero actuals are skipped


def test_seasonal_naive_has_in_sample_mase_of_one(daily):
    y = daily.to_numpy()
    fitted = y[:-7]                                      # in-sample seasonal naive: y_hat_t = y_{t-7}
    assert metrics.mase(y[7:], fitted, y, 7) == pytest.approx(1.0)


def test_rolling_origin_uses_no_future_data(daily):
    bt1 = forecast.rolling_origin(daily, n_origins=2, step=28, h=7, methods=("Holt-Winters mult.",))
    tampered = daily.copy()
    first_origin = bt1.origin.min()
    tampered.loc[first_origin + pd.Timedelta(days=7):] *= 3     # change only data after the first window
    bt2 = forecast.rolling_origin(tampered, n_origins=2, step=28, h=7, methods=("Holt-Winters mult.",))
    a = bt1[bt1.origin == first_origin].forecast.to_numpy()
    b = bt2[bt2.origin == first_origin].forecast.to_numpy()
    assert np.allclose(a, b)


def test_backtest_ranks_seasonal_methods_above_naive(daily):
    acc = forecast.accuracy(forecast.rolling_origin(daily)).set_index("method")
    assert acc.loc["Holt-Winters mult.", "WAPE_%"] < acc.loc["seasonal naive (7)", "WAPE_%"] < acc.loc["naive", "WAPE_%"]
    assert abs(acc.loc["Holt-Winters mult.", "bias_%"]) < 3


# ---------- change detection ----------
def test_cusum_by_hand():
    r = change.cusum([0, 2, 2, 2, np.nan, -3], k=0.5, h=3)
    assert np.allclose(r["s_pos"], [0, 1.5, 3, 4.5, 4.5, 1])
    assert np.allclose(r["s_neg"], [0, 0, 0, 0, 0, 2.5])
    assert r["alarm_up"] == 3 and r["alarm_down"] is None


def test_cusum_detects_planted_shift_quickly(book, daily):
    det = change.detect_shift(daily, book.holidays)
    assert det["alarm_up"] is not None
    delay = (det["alarm_up"] - book.shift_date).days
    assert 0 <= delay <= 14                              # no alarm before the shift, found within 2 weeks
    flat = synth.make_book(seed=7, shift_size=0.0).daily()
    assert change.detect_shift(flat, book.holidays)["alarm_up"] is None


def test_robust_z_flags_holidays(book, daily):
    dec = decompose.decompose(daily.to_numpy(), 7, "multiplicative")
    flags = change.anomaly_flags(np.log(dec["resid"]))
    hol = daily.index.isin(book.holidays)
    assert flags[hol].mean() >= 0.9                      # holidays (factor 0.25) are found
    assert flags[~hol].mean() < 0.02
    assert change.robust_z([1, 2, 3, 4, 100])[2] == 0


# ---------- cohorts ----------
def test_service_week_triangle(book):
    tri = cohort.service_week_triangle(book)
    weekly_gross = pd.Series(book.gross.sum(1), index=book.dates).resample("W-MON", label="left",
                                                                             closed="left").sum()
    complete = tri.index[tri.index + pd.Timedelta(days=6) <= book.asof]
    assert np.allclose(tri.loc[complete, "0w"], weekly_gross.loc[complete])
    vals = tri.to_numpy()
    known = ~np.isnan(vals)
    assert np.all(np.diff(vals, axis=1)[known[:, 1:] & known[:, :-1]] >= 0)     # cumulative grows to 0w
    future = tri.index[tri.index > book.asof]
    assert tri.loc[future, "0w"].isna().all() and tri.loc[future, "9w"].notna().all()


def test_order_week_cohorts_sum_to_orders_taken(book):
    coh = cohort.order_week_cohorts(book)
    order_date = book.dates.to_numpy()[:, None] - np.arange(book.gross.shape[1])[None, :].astype("timedelta64[D]")
    wk = coh.index[5]
    mask = (order_date >= wk.to_datetime64()) & (order_date < (wk + pd.Timedelta(days=7)).to_datetime64())
    assert coh.loc[wk].sum() == book.gross[mask].sum()


# ---------- data, determinism, CLI ----------
def test_synthetic_book_is_deterministic():
    a, b, c = synth.make_book(3), synth.make_book(3), synth.make_book(4)
    assert np.array_equal(a.gross, b.gross) and np.array_equal(a.cancels, b.cancels)
    assert not np.array_equal(a.gross, c.gross)
    assert (a.cancels.sum() / a.gross.sum()) == pytest.approx(synth.CANCEL_RATE, abs=0.005)


def test_cli_writes_files(tmp_path):
    out, figs = tmp_path / "out", tmp_path / "figs"
    cli.main(["run", "--out", str(out), "--figures", str(figs)])
    for f in ["daily_comparisons.csv", "velocity_percentiles.csv", "current_pace.csv", "projection_backtest.csv",
              "decomposition_weekly.csv", "weekday_indices.csv", "forecast_accuracy.csv", "anomalies.csv",
              "service_week_triangle.csv", "order_week_cohorts.csv", "summary.json"]:
        assert (out / f).stat().st_size > 0, f
    for f in ["velocity_curves.png", "decomposition.png", "weekday_indices.png", "backtest_accuracy.png",
              "yoy_level_shift.png"]:
        assert (figs / f).stat().st_size > 10_000, f
    s = json.loads((out / "summary.json").read_text())
    assert s["cusum_delay_days"] <= 14
