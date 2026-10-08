from datetime import datetime, timedelta

import pytest

from taxi.cleaning import clean_trips, clean_weather

T0 = datetime(2024, 1, 10, 12, 0)


def trip(pickup=T0, minutes=15, zone=161, miles=2.0, fare=12.0, total=15.0):
    return ("yellow", pickup, pickup + timedelta(minutes=minutes), zone, miles, fare, total, 2024, 1)


TRIP_SCHEMA = (
    "taxi_type STRING, pickup_ts TIMESTAMP, dropoff_ts TIMESTAMP, pu_zone INT, "
    "trip_distance DOUBLE, fare_amount DOUBLE, total_amount DOUBLE, source_year INT, source_month INT"
)


def test_each_rule_drops_its_trip(spark):
    rows = [
        trip(),  # kept
        trip(pickup=datetime(2023, 12, 31, 23, 0)),  # pickup_outside_file_month
        trip(zone=264),  # unknown_pickup_zone
        trip(minutes=0),  # impossible_duration
        trip(minutes=60 * 5),  # impossible_duration
        trip(miles=0.0),  # bad_distance
        trip(fare=-5.0),  # bad_fare
    ]
    clean, counts = clean_trips(spark.createDataFrame(rows, TRIP_SCHEMA))

    assert {r.reason: r["count"] for r in counts.collect()} == {
        "kept": 1,
        "pickup_outside_file_month": 1,
        "unknown_pickup_zone": 1,
        "impossible_duration": 2,
        "bad_distance": 1,
        "bad_fare": 1,
    }
    kept = clean.collect()
    assert len(kept) == 1
    assert kept[0].duration_min == pytest.approx(15.0)


WEATHER_SCHEMA = (
    "DATE STRING, REPORT_TYPE STRING, HourlyDryBulbTemperature STRING, "
    "HourlyPrecipitation STRING, HourlyRelativeHumidity STRING, "
    "HourlyVisibility STRING, HourlyWindSpeed STRING"
)


def test_weather_hour_assignment_and_parsing(spark):
    raw = spark.createDataFrame(
        [
            # Winter: EST == LST. 00:51 report describes the hour starting 01:00.
            ("2024-01-09T00:51:00", "FM-15", "38", "T", "80", "10.00", "5"),
            # Special report: ignored.
            ("2024-01-09T00:58:00", "FM-16", "99", "1.00", "80", "1.00", "5"),
            # Summer: EDT = LST + 1h, so 00:51 LST -> 01:51 EDT -> hour 02:00.
            ("2024-07-09T00:51:00", "FM-15 ", "81s", "0.05s", "70", "M", "7"),
        ],
        WEATHER_SCHEMA,
    )
    out = {r.hour_ts: r for r in clean_weather(raw).collect()}

    assert set(out) == {datetime(2024, 1, 9, 1, 0), datetime(2024, 7, 9, 2, 0)}
    winter, summer = out[datetime(2024, 1, 9, 1, 0)], out[datetime(2024, 7, 9, 2, 0)]
    assert winter.temp_f == 38.0
    assert winter.precip_in == pytest.approx(0.001)  # trace
    assert summer.temp_f == 81.0  # 's' suspect flag stripped
    assert summer.precip_in == pytest.approx(0.05)
    assert summer.visibility_mi is None  # 'M' = missing
