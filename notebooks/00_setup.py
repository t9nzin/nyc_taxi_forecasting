# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog catalog, the bronze/silver/gold/ml schemas, the raw-file volume and the MLflow experiment.
# MAGIC Run once per workspace.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {config.CATALOG}")
except Exception as e:
    raise RuntimeError(
        f"Could not create catalog '{config.CATALOG}'. If your workspace doesn't allow it, "
        "set CATALOG = 'workspace' in src/taxi/config.py and re-run."
    ) from e

for schema in (config.BRONZE, config.SILVER, config.GOLD, config.ML):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {config.RAW_VOLUME}")

display(spark.sql(f"SHOW SCHEMAS IN {config.CATALOG}"))

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
print("MLflow experiment:", config.MLFLOW_EXPERIMENT)
