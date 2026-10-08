# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest → bronze
# MAGIC Downloads TLC monthly trip files, the zone lookup/shapefile and NOAA hourly weather into the UC volume,
# MAGIC then writes schema-normalized bronze Delta tables.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from taxi import ingest

start, end = config.date_range()
root = config.RAW_VOLUME_PATH

trip_paths = ingest.download_trips(root, start, end)
zone_lookup, zone_shapes = ingest.download_zones(root)
weather_paths = ingest.download_weather(root, start, end)
print(f"{len(trip_paths)} trip files, {len(weather_paths)} weather files")

# COMMAND ----------

# MAGIC %md ## Trips (one file at a time — column types drift between months)

# COMMAND ----------

ingest.build_bronze_trips(spark, root, start, end, config.TABLES["bronze_trips"])
bronze = spark.table(config.TABLES["bronze_trips"])
display(bronze.groupBy("taxi_type", "source_year", "source_month").count().orderBy("taxi_type", "source_year", "source_month"))

# COMMAND ----------

# MAGIC %md ## Zones and weather

# COMMAND ----------

ingest.load_zones(spark, zone_lookup).write.mode("overwrite").saveAsTable(config.TABLES["zones"])

(
    spark.read.option("header", True)
    .csv(weather_paths)  # all columns as strings; parsed in 02_clean
    .write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(config.TABLES["bronze_weather"])
)
print("zones:", spark.table(config.TABLES["zones"]).count(),
      "weather rows:", spark.table(config.TABLES["bronze_weather"]).count())
