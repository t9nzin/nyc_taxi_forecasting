# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Clean → silver
# MAGIC Removes impossible trips (zero distance, negative fares, impossible durations, unknown zones, timestamps outside the file's month)
# MAGIC and turns raw NOAA reports into clean hourly weather in NYC wall-clock time.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from taxi import cleaning, features

clean, drop_counts = cleaning.clean_trips(spark.table(config.TABLES["bronze_trips"]))
clean.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(config.TABLES["trips"])

total = drop_counts.agg(F.sum("count")).first()[0]
display(drop_counts.withColumn("pct", F.round(100 * F.col("count") / total, 3)).orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md ## Hourly weather (gap-filled onto every hour) and holidays

# COMMAND ----------

from taxi.aggregate import hour_spine

start, end = config.date_range()
weather = cleaning.clean_weather(spark.table(config.TABLES["bronze_weather"]))
weather_hourly = features.fill_weather_gaps(weather, hour_spine(spark, start, end))
weather_hourly.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(config.TABLES["weather"])

holidays = features.holidays_table(spark, range(start.year, end.year + 1))
holidays.write.mode("overwrite").saveAsTable(config.TABLES["holidays"])

w = spark.table(config.TABLES["weather"])
print("weather hours:", w.count(), "| hours still null:", w.filter(F.col("temp_f").isNull()).count())
display(w.orderBy("hour_ts").limit(48))
