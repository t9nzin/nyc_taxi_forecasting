from datetime import datetime

from taxi.ingest import CANONICAL_TRIP_COLUMNS, normalize_trips

PICKUP = datetime(2024, 1, 5, 8, 15)
DROPOFF = datetime(2024, 1, 5, 8, 40)


def test_yellow_2024_schema_drift(spark):
    # 2024 yellow files: int32 VendorID, int64 passenger_count, capitalized Airport_fee.
    raw = spark.createDataFrame(
        [(2, PICKUP, DROPOFF, 1, 3.2, 1.0, 161, 236, 1, 18.4, 4.0, 0.0, 2.5, 1.75, 28.0)],
        "VendorID INT, tpep_pickup_datetime TIMESTAMP, tpep_dropoff_datetime TIMESTAMP, "
        "passenger_count BIGINT, trip_distance DOUBLE, RatecodeID DOUBLE, PULocationID BIGINT, "
        "DOLocationID BIGINT, payment_type BIGINT, fare_amount DOUBLE, tip_amount DOUBLE, "
        "tolls_amount DOUBLE, congestion_surcharge DOUBLE, Airport_fee DOUBLE, total_amount DOUBLE",
    )
    out = normalize_trips(raw, "yellow", 2024, 1)
    row = out.first()

    assert out.columns == ["taxi_type", *CANONICAL_TRIP_COLUMNS, "source_year", "source_month"]
    assert row.taxi_type == "yellow"
    assert row.pickup_ts == PICKUP
    assert row.pu_zone == 161
    assert row.airport_fee == 1.75
    assert row.ratecode_id == 1


def test_green_has_no_airport_fee_and_double_payment_type(spark):
    raw = spark.createDataFrame(
        [(2, PICKUP, DROPOFF, 1.0, 1.1, 1.0, 74, 75, 2.0, 7.0, 0.0, 0.0, 0.0, 9.5)],
        "VendorID BIGINT, lpep_pickup_datetime TIMESTAMP, lpep_dropoff_datetime TIMESTAMP, "
        "passenger_count DOUBLE, trip_distance DOUBLE, RatecodeID DOUBLE, PULocationID BIGINT, "
        "DOLocationID BIGINT, payment_type DOUBLE, fare_amount DOUBLE, tip_amount DOUBLE, "
        "tolls_amount DOUBLE, congestion_surcharge DOUBLE, total_amount DOUBLE",
    )
    out = normalize_trips(raw, "green", 2024, 1)
    row = out.first()
    types = dict(out.dtypes)

    assert row.pickup_ts == PICKUP
    assert row.airport_fee is None
    assert row.payment_type == 2
    assert types["payment_type"] == "int"
    assert types["passenger_count"] == "int"


def test_truncated_weather_download_is_rejected(tmp_path):
    from datetime import date

    from taxi.ingest import _reaches_last_day

    header = '"STATION","DATE","REPORT_TYPE"\n'
    complete = tmp_path / "complete.csv"
    complete.write_text(header + '"725053","2024-01-31T23:51:00","FM-15"\n')
    truncated = tmp_path / "truncated.csv"
    truncated.write_text(header + '"725053","2024-01-12T06:51:00","FM-15"\n')

    check = _reaches_last_day(date(2024, 1, 31))
    assert check(str(complete))
    assert not check(str(truncated))
