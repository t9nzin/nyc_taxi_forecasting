"""Run ingest -> clean -> aggregate -> features on one real month, locally.

Uses plain Parquet in ./data instead of Unity Catalog. Catches real-data
surprises (schema drift, weather quirks) before spending Databricks quota.

    JAVA_HOME=... .venv/bin/python scripts/local_smoke.py
"""

import os
import sys
import time
from datetime import date

os.environ["TZ"] = "UTC"
time.tzset()
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from taxi import aggregate, cleaning, config, features, ingest  # noqa: E402

START, END = date(2024, 1, 1), date(2024, 2, 1)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("downloading...")
ingest.download_trips(ROOT, START, END)
lookup, _ = ingest.download_zones(ROOT)
weather_paths = ingest.download_weather(ROOT, START, END)

bronze = None
for taxi_type in config.TAXI_TYPES:
    for y, m in config.months(START, END):
        df = ingest.normalize_trips(
            spark.read.parquet(ingest.trip_file_path(ROOT, taxi_type, y, m)), taxi_type, y, m
        )
        bronze = df if bronze is None else bronze.unionByName(df)

clean, drops = cleaning.clean_trips(bronze)
print("\ndrop reasons:")
drops.orderBy(F.desc("count")).show(truncate=False)

zones = ingest.load_zones(spark, lookup)
spine = aggregate.hour_spine(spark, START, END)
weather = features.fill_weather_gaps(
    cleaning.clean_weather(spark.read.option("header", True).csv(weather_paths)), spine
).cache()
print("weather hours:", weather.count(), "null temp hours:", weather.filter("temp_f IS NULL").count())

demand = aggregate.zone_hour_demand(spark, clean, zones, START, END).cache()
n_zones, n_hours = zones.count(), spine.count()
assert demand.count() == n_zones * n_hours, "panel not gap-free"
keep = aggregate.modeled_zones(demand, START, END)
print(f"panel {n_zones} zones x {n_hours} hours; modeling {keep.count()} zones")

holidays = features.holidays_table(spark, [2024])
gold = features.build_features(demand.join(keep.select("zone_id"), "zone_id"), zones, weather, holidays).dropna(
    subset=features.HISTORY_COLUMNS
)
print("gold rows:", gold.count())
print("\nnull counts in gold:")
gold.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in gold.columns]).show(vertical=True)

print("Midtown Center (161), MLK Day evening:")
(
    gold.filter((F.col("zone_id") == 161) & (F.to_date("hour_ts") == "2024-01-15"))
    .select("hour_ts", "pickups", "lag_1h", "lag_168h", "roll_mean_24h", "is_holiday", "temp_f", "precip_in")
    .orderBy("hour_ts")
    .show(24)
)

# Seasonal-naive sanity number for the README.
naive = gold.withColumn("prediction", F.col("lag_168h").cast("double"))
from taxi.evaluation import overall_metrics  # noqa: E402

print("seasonal-naive on this month:", overall_metrics(naive))
