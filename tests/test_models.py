from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from taxi import aggregate, features
from taxi.models import chronos, glm, naive


@pytest.fixture(scope="module")
def gold(spark):
    """Three weeks of two zones with a daily cycle and a 3x level difference."""
    zones = spark.createDataFrame(
        [(161, "Manhattan", "Midtown Center"), (132, "Queens", "JFK Airport")],
        "zone_id INT, borough STRING, zone_name STRING",
    )
    rows = []
    for h in range(21 * 24):
        ts = datetime(2024, 1, 1) + timedelta(hours=h)
        base = 20 + 15 * np.sin(2 * np.pi * (h % 24) / 24)
        rows += [(161, ts, int(3 * base)), (132, ts, int(base))]
    demand = spark.createDataFrame(rows, "zone_id INT, hour_ts TIMESTAMP, pickups INT")
    spine = aggregate.hour_spine(spark, date(2024, 1, 1), date(2024, 1, 22))
    weather = spine.selectExpr(
        "hour_ts", "40.0 AS temp_f", "0.0 AS precip_in", "50.0 AS humidity_pct",
        "10.0 AS visibility_mi", "5.0 AS wind_mph", "0 AS is_raining",
    )
    holidays = features.holidays_table(spark, [2024])
    return features.build_features(demand, zones, weather, holidays).dropna(
        subset=features.HISTORY_COLUMNS
    )


def test_naive_is_last_week(gold):
    r = naive.predict(gold).first()
    assert r.prediction == r.lag_168h


def test_glm_learns_level_and_daily_cycle(gold):
    train = gold.filter("hour_ts < '2024-01-18'")
    test = gold.filter("hour_ts >= '2024-01-18'")
    scored = glm.predict(glm.fit(train), test).select("pickups", "prediction").toPandas()
    wape = (scored.prediction - scored.pickups).abs().sum() / scored.pickups.sum()
    assert (scored.prediction > 0).all()  # log link keeps counts positive
    assert wape < 0.05


def test_rolling_contexts_only_see_the_past():
    y = np.arange(10, dtype=np.float32)
    ctx = chronos.rolling_contexts(y, first_target=4, context=3)
    # Targets t = 4..9; each window is y[t-3:t].
    assert ctx.shape == (6, 3)
    assert ctx[0].tolist() == [1, 2, 3]
    assert ctx[-1].tolist() == [6, 7, 8]
    with pytest.raises(ValueError):
        chronos.rolling_contexts(y, first_target=2, context=3)


class _LastValuePipeline:
    """Stand-in for Chronos: predicts the last context value, +/- 1."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        last = inputs[:, -1:]
        q = torch.stack([last - 1, last, last + 1], dim=-1)  # [batch, 1, 3]
        return q, last


def test_forecast_zone_aligns_targets(monkeypatch):
    monkeypatch.setattr(chronos, "_pipeline", _LastValuePipeline())
    monkeypatch.setattr(chronos, "CONTEXT_HOURS", 4)
    hours = pd.date_range("2024-01-01", periods=10, freq="h")
    pdf = pd.DataFrame({"zone_id": 161, "hour_ts": hours, "pickups": np.arange(10) * 10})

    out = chronos.forecast_zone(
        pdf.sample(frac=1, random_state=0), test_start=hours[6], model_path="unused"  # shuffled on purpose
    )
    assert out["hour_ts"].tolist() == list(hours[6:])
    # Forecast for hour t is pickups at t-1 under the stand-in.
    assert out["prediction"].tolist() == [50.0, 60.0, 70.0, 80.0]
    assert out["p10"].tolist() == [49.0, 59.0, 69.0, 79.0]
