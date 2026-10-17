# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Export forecasts for the map app
# MAGIC Writes every model's test-window forecasts from `gold.forecasts` as one compact Parquet file in the volume. The Streamlit map in `app/` reads it from `app/data/forecasts.parquet`.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import os

from pyspark.sql import functions as F

out_dir = f"{config.RAW_VOLUME_PATH}/app"
os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/forecasts.parquet"

forecasts = spark.table(config.TABLES["forecasts"]).select(
    "model",
    F.col("zone_id").cast("smallint").alias("zone_id"),
    "hour_ts",
    F.col("pickups").cast("smallint").alias("pickups"),
    F.round("prediction", 1).cast("float").alias("prediction"),
    # p10 rounded up and p90 down to 0.1. Pickups are whole numbers, so whether
    # one falls inside the band doesn't change; plain rounding would turn p10 = 0.04
    # into 0.0 and count zero-pickup hours as inside.
    (F.ceil(F.col("p10") * 10) / 10).cast("float").alias("p10"),
    (F.floor(F.col("p90") * 10) / 10).cast("float").alias("p90"),
)
pdf = forecasts.toPandas()
pdf.attrs.clear()  # Spark Connect puts query metrics here, which Parquet metadata can't hold
pdf["model"] = pdf["model"].astype("category")
pdf.sort_values(["model", "hour_ts", "zone_id"]).to_parquet(path, index=False, compression="zstd")

print(f"{len(pdf):,} rows, {pdf.model.nunique()} models, {pdf.hour_ts.min()} → {pdf.hour_ts.max()}")
print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB")
