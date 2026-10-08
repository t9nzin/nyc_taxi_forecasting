"""Aggregate cleaned trips to a gap-free zone x hour demand panel."""

from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi import config


def hour_spine(spark: SparkSession, start: date, end: date) -> DataFrame:
    """One row per wall-clock hour in [start, end)."""
    return spark.sql(
        f"""
        SELECT explode(sequence(
            timestamp'{start.isoformat()} 00:00:00',
            timestamp'{end.isoformat()} 00:00:00' - INTERVAL 1 HOUR,
            INTERVAL 1 HOUR
        )) AS hour_ts
        """
    )


def zone_hour_demand(
    spark: SparkSession, trips: DataFrame, zones: DataFrame, start: date, end: date
) -> DataFrame:
    """Pickups per (zone, hour), with explicit zeros for hours without trips.

    ``zones`` needs ``zone_id``. Every zone gets every hour so all series have
    the same length, which lag features and the forecasting models rely on.
    """
    counts = (
        trips.groupBy(
            F.col("pu_zone").alias("zone_id"),
            F.date_trunc("hour", "pickup_ts").alias("hour_ts"),
        )
        .agg(
            F.count("*").alias("pickups"),
            F.sum(F.when(F.col("taxi_type") == "yellow", 1).otherwise(0)).alias("pickups_yellow"),
            F.sum(F.when(F.col("taxi_type") == "green", 1).otherwise(0)).alias("pickups_green"),
        )
    )
    grid = zones.select("zone_id").distinct().crossJoin(hour_spine(spark, start, end))
    return (
        grid.join(counts, ["zone_id", "hour_ts"], "left")
        .fillna(0, subset=["pickups", "pickups_yellow", "pickups_green"])
        .withColumn("pickups", F.col("pickups").cast("int"))
        .withColumn("pickups_yellow", F.col("pickups_yellow").cast("int"))
        .withColumn("pickups_green", F.col("pickups_green").cast("int"))
    )


def modeled_zones(demand: DataFrame, train_start: date, train_end: date) -> DataFrame:
    """Zones with enough training-period volume to model. Chosen on the train
    window only, so the zone set doesn't leak information from the test set."""
    days = (train_end - train_start).days
    return (
        demand.filter(
            (F.col("hour_ts") >= F.lit(train_start).cast("timestamp"))
            & (F.col("hour_ts") < F.lit(train_end).cast("timestamp"))
        )
        .groupBy("zone_id")
        .agg((F.sum("pickups") / F.lit(days)).alias("avg_daily_pickups"))
        .filter(F.col("avg_daily_pickups") >= config.MIN_AVG_DAILY_PICKUPS)
    )
