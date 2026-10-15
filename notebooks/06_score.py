# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Batch scoring with `@champion`
# MAGIC Loads whichever model version holds the `@champion` alias in Unity Catalog and forecasts the hour after the data ends for every zone, with p10–p90 bands, into `gold.next_hour_forecasts`. Run hourly in a Job, this is the production forecast.
# MAGIC
# MAGIC The model needs each zone's previous week, not a single row, so scoring runs one zone per task with `applyInPandas` (see `taxi.scoring`). Before writing, it re-forecasts the first test hour and checks the result against what notebook 05 wrote for the same model: the registered artifact has to reproduce the backtest.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt -r ../requirements-ml.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from datetime import datetime, timezone

import mlflow
import pandas as pd
from pyspark.sql import functions as F

from taxi import scoring

mlflow.set_registry_uri("databricks-uc")

model_path, version = scoring.download_champion()
champion_run = mlflow.get_run(version.run_id).data
kind = champion_run.tags["model"]
input_size = int(champion_run.params["input_size"])
print(f"@champion = {config.MODEL_NAME} v{version.version} ({kind}), "
      f"test WAPE {champion_run.metrics['test_wape']:.4f} on {champion_run.params['test_window']}")

features = spark.table(config.TABLES["features"])
holidays = spark.table(config.TABLES["holidays"])

# COMMAND ----------

# MAGIC %md ## Check: the registered model reproduces its backtest

# COMMAND ----------

test_start = pd.Timestamp(config.splits()["test"][0])
check = scoring.score(scoring.scoring_window(features, holidays, test_start, input_size), model_path)
backtest = spark.table(config.TABLES["forecasts"]).filter(
    (F.col("model") == kind) & (F.col("hour_ts") == F.lit(test_start.to_pydatetime()))
)
diff = (
    check.join(backtest.select("zone_id", F.col("prediction").alias("backtest")), "zone_id")
    .agg(F.count("*").alias("zones"), F.max(F.abs(F.col("prediction") - F.col("backtest"))).alias("max_abs_diff"))
    .first()
)
print(f"{diff.zones} zones, max |scored - backtest| = {diff.max_abs_diff:.6f} pickups")
assert diff.zones > 0 and diff.max_abs_diff < 0.01, "registered model doesn't match its backtest"

# COMMAND ----------

# MAGIC %md ## Forecast the next hour

# COMMAND ----------

target = features.agg(F.max("hour_ts")).first()[0] + pd.Timedelta(hours=1)
forecasts = (
    scoring.score(scoring.scoring_window(features, holidays, target, input_size), model_path)
    .withColumn("model", F.lit(kind))
    .withColumn("model_version", F.lit(int(version.version)))
    .withColumn("scored_at", F.lit(datetime.now(timezone.utc).replace(tzinfo=None)))
)

table = config.TABLES["next_hour"]
if spark.catalog.tableExists(table):  # rerunning an hour replaces it
    forecasts.write.mode("overwrite").option("replaceWhere", f"hour_ts = '{target}'").saveAsTable(table)
else:
    forecasts.write.saveAsTable(table)

written = spark.table(table).filter(F.col("hour_ts") == F.lit(target))
n_zones = features.select("zone_id").distinct().count()
assert written.count() == n_zones, f"expected one forecast per zone ({n_zones})"
print(f"wrote {n_zones} zone forecasts for {target} to {table}")

# COMMAND ----------

# MAGIC %md ## Busiest zones next hour

# COMMAND ----------

zones = spark.table(config.TABLES["zones"]).select("zone_id", "zone_name", "borough")
display(
    written.join(zones, "zone_id")
    .select("zone_name", "borough", "hour_ts", F.round("p10", 1).alias("p10"),
            F.round("prediction", 1).alias("forecast"), F.round("p90", 1).alias("p90"))
    .orderBy(F.desc("forecast"))
    .limit(15)
)
