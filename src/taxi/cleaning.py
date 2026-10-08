"""Silver-layer cleaning for trips and weather."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from taxi import config

MAX_VALID_ZONE = 263


def _drop_reason() -> Column:
    """First failing rule for a trip, or null if the trip is valid.

    Rules are checked in order, so each dropped trip is counted once.
    """
    duration_min = (F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")) / 60.0
    month_start = F.make_date("source_year", "source_month", F.lit(1)).cast("timestamp")
    month_end = F.add_months(month_start, 1).cast("timestamp")
    return (
        F.when(F.col("pickup_ts").isNull() | F.col("dropoff_ts").isNull(), "missing_timestamp")
        .when(
            (F.col("pickup_ts") < month_start) | (F.col("pickup_ts") >= month_end),
            "pickup_outside_file_month",
        )
        .when(
            F.col("pu_zone").isNull()
            | F.col("pu_zone").isin(*config.UNKNOWN_ZONES)
            | (F.col("pu_zone") < 1)
            | (F.col("pu_zone") > MAX_VALID_ZONE),
            "unknown_pickup_zone",
        )
        .when(
            (duration_min < config.MIN_DURATION_MIN) | (duration_min > config.MAX_DURATION_MIN),
            "impossible_duration",
        )
        .when(
            F.col("trip_distance").isNull()
            | (F.col("trip_distance") <= 0)
            | (F.col("trip_distance") > config.MAX_TRIP_MILES),
            "bad_distance",
        )
        .when(
            F.col("fare_amount").isNull()
            | (F.col("fare_amount") < 0)
            | (F.col("fare_amount") > config.MAX_FARE)
            | (F.col("total_amount") < 0),
            "bad_fare",
        )
    )


def clean_trips(bronze: DataFrame):
    """Return (clean_trips, drop_counts). ``drop_counts`` has one row per reason
    plus a ``kept`` row, which makes a good README table."""
    flagged = bronze.withColumn("drop_reason", _drop_reason())
    counts = flagged.groupBy(F.coalesce("drop_reason", F.lit("kept")).alias("reason")).count()
    clean = (
        flagged.filter(F.col("drop_reason").isNull())
        .drop("drop_reason")
        .withColumn(
            "duration_min",
            (F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")) / 60.0,
        )
    )
    return clean, counts


# --- Weather ---------------------------------------------------------------

def _lcd_number(col: str, trace_value: float = None) -> Column:
    """Parse an LCD field: strips the 's' (suspect) suffix, maps 'T' (trace)
    to ``trace_value``, and turns 'M', '*' and blanks into null."""
    raw = F.trim(F.col(col))
    number = F.regexp_extract(raw, r"^(-?\d+(?:\.\d+)?)", 1)
    parsed = F.when(number != "", number.cast("double"))
    if trace_value is not None:
        parsed = F.when(raw == "T", F.lit(trace_value)).otherwise(parsed)
    return parsed


def clean_weather(raw: DataFrame) -> DataFrame:
    """Hourly weather keyed by NYC wall-clock ``hour_ts``.

    Uses only routine hourly reports (FM-15), which arrive at about :51. The
    report at hh:51 is assigned to hour hh+1, so the weather for an hour is
    the last observation *before* that hour starts and leaks nothing about it.
    LCD times are Local Standard Time all year; they are shifted to UTC and
    then to America/New_York wall-clock time to line up with TLC timestamps.
    """
    # The LCD CSV repeats REPORT_TYPE (identical values), so Spark's CSV
    # reader renames the copies to e.g. REPORT_TYPE2 / REPORT_TYPE95.
    report_type = next(c for c in raw.columns if c.startswith("REPORT_TYPE"))
    obs = (
        raw.filter(F.trim(F.col(report_type)) == "FM-15")
        .select(
            F.to_timestamp("DATE").alias("obs_lst"),
            _lcd_number("HourlyDryBulbTemperature").alias("temp_f"),
            _lcd_number("HourlyPrecipitation", trace_value=0.001).alias("precip_in"),
            _lcd_number("HourlyRelativeHumidity").alias("humidity_pct"),
            _lcd_number("HourlyVisibility").alias("visibility_mi"),
            _lcd_number("HourlyWindSpeed").alias("wind_mph"),
        )
        .dropDuplicates(["obs_lst"])
    )
    hour_lst = F.date_trunc("hour", F.col("obs_lst") + F.expr("INTERVAL 9 MINUTES"))
    utc = hour_lst + F.expr("INTERVAL 5 HOURS")  # LST is UTC-5
    wall = F.from_utc_timestamp(utc, config.LOCAL_TZ)
    return (
        obs.withColumn("hour_ts", wall)
        # The repeated hour when DST ends maps two reports to one wall hour.
        .groupBy("hour_ts")
        .agg(
            F.avg("temp_f").alias("temp_f"),
            F.max("precip_in").alias("precip_in"),
            F.avg("humidity_pct").alias("humidity_pct"),
            F.avg("visibility_mi").alias("visibility_mi"),
            F.avg("wind_mph").alias("wind_mph"),
        )
    )
