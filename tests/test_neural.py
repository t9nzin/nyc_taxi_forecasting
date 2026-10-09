import numpy as np
import pandas as pd
import pytest

from taxi.models import neural


def _features(zones=((161, "Manhattan"), (132, "Queens")), hours=72):
    """gold.features-shaped pandas rows with a daily cycle."""
    ts = pd.date_range("2024-01-01", periods=hours, freq="h")
    rows = []
    for zone_id, borough in zones:
        level = 30 if borough == "Manhattan" else 10
        rows.append(pd.DataFrame({
            "zone_id": zone_id, "hour_ts": ts, "borough": borough,
            "pickups": np.round(level * (1.5 + np.sin(2 * np.pi * ts.hour / 24))).astype(int),
            "hour": ts.hour, "day_of_week": ts.dayofweek,
            "is_weekend": (ts.dayofweek >= 5).astype(int), "is_holiday": 0,
            "temp_f": 40.0, "precip_in": 0.0, "is_raining": 0, "wind_mph": 5.0,
            "humidity_pct": 50.0, "visibility_mi": 10.0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_to_long_encodes_calendar_and_borough():
    pdf = _features()
    df, static_df = neural.to_long(pdf.sample(frac=1, random_state=0), neural.fit_scales(pdf, "2024-01-03"))
    assert list(df.columns[:3]) == ["unique_id", "ds", "y"]
    assert set(neural.FUTR_EXOG) <= set(df.columns)
    assert df.groupby("unique_id")["ds"].is_monotonic_increasing.all()
    # Hour 0 and hour 23 are neighbours on the circle.
    h0 = df[df.ds.dt.hour == 0].iloc[0]
    h23 = df[df.ds.dt.hour == 23].iloc[0]
    assert np.hypot(h0.hour_sin - h23.hour_sin, h0.hour_cos - h23.hour_cos) < 0.3
    s = static_df.set_index("unique_id")
    assert s.loc[161, "borough_manhattan"] == 1 and s.loc[161, "borough_queens"] == 0
    assert s.loc[132, "borough_queens"] == 1
    assert (s.sum(axis=1) == 1).all()


def test_scales_use_train_hours_only_and_round_trip():
    pdf = _features()
    late = pdf.hour_ts >= "2024-01-02"
    pdf.loc[late, "pickups"] *= 100  # a level shift after train_end mustn't leak in
    pdf.loc[late, "temp_f"] = 90.0
    scales = neural.fit_scales(pdf, "2024-01-02")
    assert scales["zone"][161] == pytest.approx(pdf[~late & (pdf.zone_id == 161)].pickups.mean())
    assert scales["mean"]["temp_f"] == 40.0 and scales["std"]["temp_f"] == 1.0  # constant -> std 1

    df, _ = neural.to_long(pdf, scales)
    early = df[df.ds < "2024-01-02"]
    assert early.groupby("unique_id").y.mean().round(6).eq(1.0).all()  # y in units of the zone mean
    with pytest.raises(ValueError, match="not seen in training"):
        neural.to_long(pdf.assign(zone_id=999), scales)


def test_to_output_rescales_and_clips():
    fc = pd.DataFrame({
        "unique_id": [161], "ds": [pd.Timestamp("2024-01-02")],
        "nn-median": [1.5], "nn-lo-80": [-0.5], "nn-hi-80": [3.0],
    })
    out = neural._to_output(fc, {"zone": {161: 2.0}})
    assert list(out.columns) == neural.OUTPUT_COLUMNS
    assert out.iloc[0][["prediction", "p10", "p90"]].tolist() == [3.0, 0.0, 6.0]


@pytest.fixture(scope="module", params=list(neural.SPECS))
def tiny_backtest(request):
    pytest.importorskip("neuralforecast")
    kind = request.param
    pdf = _features(hours=24 * 12)
    params = {**neural.SPECS[kind]["params"], "input_size": 48, "max_steps": 5, "val_check_steps": 5,
              "batch_size": 2, "windows_batch_size": 64, "accelerator": "cpu"}
    if kind == "tft":
        params["hidden_size"] = 8
    model, fc = neural.backtest(pdf, "2024-01-10", "2024-01-11", "2024-01-12", kind, params)
    return pdf, model, fc


def test_backtest_forecasts_each_test_hour_once(tiny_backtest):
    pdf, _, fc = tiny_backtest
    test_hours = pd.date_range("2024-01-11", "2024-01-11 23:00", freq="h")
    assert len(fc) == 2 * len(test_hours)
    assert not fc.duplicated(["zone_id", "hour_ts"]).any()
    assert set(fc.hour_ts) == set(test_hours)
    assert (fc.p10 <= fc.prediction).all() and (fc.prediction <= fc.p90).all()


def test_predict_next_hour_matches_backtest(tiny_backtest, tmp_path):
    """The registered pyfunc path (save -> load -> predict_next_hour) must give
    the backtest's forecast for the same hour: same weights, same scaling."""
    pdf, model, fc = tiny_backtest
    model.save(str(tmp_path))
    loaded = neural.Forecaster.load(str(tmp_path))
    target = pd.Timestamp("2024-01-11 00:00")
    window = pdf[pdf.hour_ts <= target].copy()
    window["pickups"] = window["pickups"].astype(float).where(window.hour_ts < target)
    out = loaded.predict_next_hour(window).sort_values("zone_id", ignore_index=True)
    expected = fc[fc.hour_ts == target].sort_values("zone_id", ignore_index=True)
    assert out.hour_ts.eq(target).all()
    np.testing.assert_allclose(out.prediction, expected.prediction, rtol=1e-4)
