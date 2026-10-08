"""Gold-layer modeling features: calendar, holidays, weather, lags and rolling windows.

Each row is (zone_id, hour_ts) with target ``pickups`` for that hour. Every
history-based feature only uses hours strictly before ``hour_ts``, so a row is
exactly what a forecaster would know one hour ahead.
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

LAGS = {"lag_1h": 1, "lag_2h": 2, "lag_24h": 24, "lag_168h": 168}
ROLLING_WINDOWS = {"roll_mean_24h": 24, "roll_mean_168h": 168}

WEATHER_COLUMNS = ["temp_f", "precip_in", "humidity_pct", "visibility_mi", "wind_mph"]
CALENDAR_COLUMNS = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]
HISTORY_COLUMNS = list(LAGS) + list(ROLLING_WINDOWS)


def holidays_table(spark: SparkSession, years) -> DataFrame:
    """US federal + New York State holidays, one row per date."""
    import holidays as hol

    days = hol.country_holidays("US", subdiv="NY", years=list(years))
    rows = [(d, name) for d, name in sorted(days.items())]
    return spark.createDataFrame(rows, "date DATE, holiday_name STRING")


def fill_weather_gaps(weather: DataFrame, spine: DataFrame) -> DataFrame:
    """Put weather on every hour of ``spine`` and forward-fill missing reports.

    Weather has ~26k rows per three years, so a single unpartitioned window
    is cheap here.
    """
    w = Window.orderBy("hour_ts").rowsBetween(Window.unboundedPreceding, 0)
    filled = spine.join(weather, "hour_ts", "left")
    for c in WEATHER_COLUMNS:
        filled = filled.withColumn(c, F.last(c, ignorenulls=True).over(w))
    return filled.withColumn("is_raining", (F.col("precip_in") > 0).cast("int"))


def add_calendar(df: DataFrame, holidays: DataFrame) -> DataFrame:
    return (
        df.withColumn("hour", F.hour("hour_ts"))
        # Spark's dayofweek is 1=Sunday..7=Saturday; shift to 0=Monday..6=Sunday.
        .withColumn("day_of_week", (F.dayofweek("hour_ts") + 5) % 7)
        .withColumn("month", F.month("hour_ts"))
        .withColumn("is_weekend", (F.col("day_of_week") >= 5).cast("int"))
        .withColumn("date", F.to_date("hour_ts"))
        .join(F.broadcast(holidays.select("date", F.lit(1).alias("is_holiday"))), "date", "left")
        .fillna(0, subset=["is_holiday"])
        .drop("date")
    )


def add_history(df: DataFrame) -> DataFrame:
    """Lag and trailing-window features per zone.

    The demand panel is gap-free (one row per zone per hour), so a row offset
    of k equals k hours. Rolling windows end at t-1 and exclude the target hour.
    """
    by_zone = Window.partitionBy("zone_id").orderBy("hour_ts")
    for name, k in LAGS.items():
        df = df.withColumn(name, F.lag("pickups", k).over(by_zone))
    for name, k in ROLLING_WINDOWS.items():
        df = df.withColumn(name, F.avg("pickups").over(by_zone.rowsBetween(-k, -1)))
    return df


def build_features(
    demand: DataFrame, zones: DataFrame, weather_hourly: DataFrame, holidays: DataFrame
) -> DataFrame:
    """Join everything onto the zone x hour panel.

    ``weather_hourly`` should already be gap-filled (see ``fill_weather_gaps``).
    Rows in the first week of each series have null long lags; drop them
    before training with ``dropna(subset=HISTORY_COLUMNS)``.
    """
    df = add_history(demand)
    df = add_calendar(df, holidays)
    df = df.join(F.broadcast(weather_hourly), "hour_ts", "left")
    return df.join(F.broadcast(zones.select("zone_id", "borough", "zone_name")), "zone_id", "left")
