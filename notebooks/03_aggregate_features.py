# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Aggregate + features → gold
# MAGIC Pickups per taxi zone per hour (gap-free), then calendar, holiday, weather, lag and rolling features.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from taxi import aggregate, features

start, end = config.date_range()
zones = spark.table(config.TABLES["zones"])

demand = aggregate.zone_hour_demand(spark, spark.table(config.TABLES["trips"]), zones, start, end)
demand.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(config.TABLES["zone_hour_demand"])
demand = spark.table(config.TABLES["zone_hour_demand"])

n_zones = demand.select("zone_id").distinct().count()
n_hours = demand.select("hour_ts").distinct().count()
assert demand.count() == n_zones * n_hours, "panel is not gap-free"
print(f"{n_zones} zones x {n_hours} hours")

# COMMAND ----------

# MAGIC %md ## Which zones get modeled (chosen on the training window only)

# COMMAND ----------

train_start, train_end = config.splits()["train"]
keep = aggregate.modeled_zones(demand, train_start, train_end)
print(f"modeling {keep.count()} of {n_zones} zones")

# COMMAND ----------

gold = (
    features.build_features(
        demand.join(keep.select("zone_id"), "zone_id"),
        zones,
        spark.table(config.TABLES["weather"]),
        spark.table(config.TABLES["holidays"]),
    )
    .dropna(subset=features.HISTORY_COLUMNS)  # first week of each series has no lag_168h
)
gold.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(config.TABLES["features"])

g = spark.table(config.TABLES["features"])
display(g.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in g.columns]))  # null audit
display(g.filter(F.col("zone_id") == 161).orderBy("hour_ts").limit(200))  # Midtown Center
