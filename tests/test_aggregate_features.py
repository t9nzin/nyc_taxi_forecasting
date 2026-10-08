from datetime import date, datetime, timedelta

import pytest

from taxi import aggregate, features
from taxi.evaluation import overall_metrics

START, END = date(2024, 1, 1), date(2024, 1, 15)  # 14 days = 336 hours
N_HOURS = 14 * 24


@pytest.fixture(scope="module")
def zones(spark):
    return spark.createDataFrame(
        [(161, "Manhattan", "Midtown Center"), (1, "EWR", "Newark Airport")],
        "zone_id INT, borough STRING, zone_name STRING",
    )


@pytest.fixture(scope="module")
def demand(spark, zones):
    # Zone 161: pickups at hour h = h % 7 trips (deterministic pattern).
    # Zone 1: a single trip in the whole window.
    trips = []
    for h in range(N_HOURS):
        ts = datetime(2024, 1, 1) + timedelta(hours=h, minutes=10)
        trips += [("yellow", ts, 161)] * (h % 7)
    trips.append(("green", datetime(2024, 1, 3, 4, 30), 1))
    trips_df = spark.createDataFrame(trips, "taxi_type STRING, pickup_ts TIMESTAMP, pu_zone INT")
    return aggregate.zone_hour_demand(spark, trips_df, zones, START, END).cache()


def test_panel_is_gap_free_with_zeros(demand):
    assert demand.count() == 2 * N_HOURS
    ewr = {r.hour_ts: r.pickups for r in demand.filter("zone_id = 1").collect()}
    assert len(ewr) == N_HOURS
    assert ewr[datetime(2024, 1, 3, 4)] == 1
    assert sum(ewr.values()) == 1
    green = demand.filter("zone_id = 1").agg({"pickups_green": "sum"}).first()[0]
    assert green == 1


def test_modeled_zones_drops_sparse_zone(demand):
    kept = [r.zone_id for r in aggregate.modeled_zones(demand, START, END).collect()]
    assert kept == [161]


def test_lags_and_rolling_use_only_the_past(spark, demand):
    df = features.add_history(demand.filter("zone_id = 161"))
    rows = {r.hour_ts: r for r in df.collect()}
    t = datetime(2024, 1, 1) + timedelta(hours=200)

    def y(hours_back):
        return (200 - hours_back) % 7

    r = rows[t]
    assert r.pickups == y(0)
    assert r.lag_1h == y(1)
    assert r.lag_24h == y(24)
    assert r.lag_168h == y(168)
    assert r.roll_mean_24h == pytest.approx(sum(y(k) for k in range(1, 25)) / 24)
    # First hour of the series has no history.
    first = rows[datetime(2024, 1, 1)]
    assert first.lag_1h is None and first.lag_168h is None


def test_calendar_and_holidays(spark, demand):
    holidays = features.holidays_table(spark, [2024])
    df = features.add_calendar(demand.filter("zone_id = 161"), holidays)
    rows = {r.hour_ts: r for r in df.collect()}

    new_year = rows[datetime(2024, 1, 1, 9)]  # Monday, holiday
    assert (new_year.day_of_week, new_year.is_weekend, new_year.is_holiday, new_year.hour) == (0, 0, 1, 9)
    saturday = rows[datetime(2024, 1, 6, 9)]
    assert (saturday.day_of_week, saturday.is_weekend, saturday.is_holiday) == (5, 1, 0)


def test_weather_gaps_forward_filled(spark):
    spine = aggregate.hour_spine(spark, date(2024, 1, 1), date(2024, 1, 2))
    weather = spark.createDataFrame(
        [(datetime(2024, 1, 1, 0), 30.0, 0.0, 50.0, 10.0, 3.0),
         (datetime(2024, 1, 1, 5), 35.0, 0.1, 60.0, 5.0, 4.0)],
        "hour_ts TIMESTAMP, temp_f DOUBLE, precip_in DOUBLE, humidity_pct DOUBLE, "
        "visibility_mi DOUBLE, wind_mph DOUBLE",
    )
    rows = {r.hour_ts.hour: r for r in features.fill_weather_gaps(weather, spine).collect()}
    assert len(rows) == 24
    assert rows[3].temp_f == 30.0 and rows[3].is_raining == 0
    assert rows[23].temp_f == 35.0 and rows[23].is_raining == 1


def test_metrics(spark):
    df = spark.createDataFrame([(10, 12.0), (0, 1.0), (5, 2.0)], "pickups INT, prediction DOUBLE")
    m = overall_metrics(df)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx((14 / 3) ** 0.5)
    assert m["wape"] == pytest.approx(6 / 15)
