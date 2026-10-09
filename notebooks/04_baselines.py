# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Baselines
# MAGIC Three baselines the TFT has to beat, each logged as an MLflow run and scored on the same out-of-time test window:
# MAGIC 1. **Seasonal-naive**: same hour last week
# MAGIC 2. **Poisson GLM** (Spark MLlib): zone and hour-of-week effects, weather, holidays, log lags
# MAGIC 3. **Chronos-Bolt** (zero-shot): a pretrained time-series model that only sees each zone's history
# MAGIC
# MAGIC Forecasts go to `gold.forecasts` (one partition per model), which the map app reads.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt -r ../requirements-ml.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow

from taxi.evaluation import log_test_results, metrics_for_mlflow, split, write_forecasts
from taxi.models import chronos, glm, naive

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

features = spark.table(config.TABLES["features"])
validation, test = split(features, "validation"), split(features, "test")
test_start, test_end = config.splits()["test"]

COMMON_PARAMS = {
    "dev_mode": config.DEV_MODE,
    "train_window": "→".join(map(str, config.splits()["train"])),
    "test_window": f"{test_start}→{test_end}",
    "n_zones": features.select("zone_id").distinct().count(),
}

# COMMAND ----------

# MAGIC %md ## 1 · Seasonal-naive

# COMMAND ----------

with mlflow.start_run(run_name=naive.NAME):
    mlflow.set_tags({"stage": "baseline", "model": naive.NAME})
    mlflow.log_params({**COMMON_PARAMS, "rule": "pickups(t - 168h)"})
    mlflow.log_metrics(metrics_for_mlflow(naive.predict(validation), "validation"))
    write_forecasts(naive.predict(test), naive.NAME)
    log_test_results(spark, naive.NAME)

# COMMAND ----------

# MAGIC %md ## 2 · Poisson GLM (Spark MLlib)

# COMMAND ----------

with mlflow.start_run(run_name=glm.NAME):
    mlflow.set_tags({"stage": "baseline", "model": glm.NAME})
    mlflow.log_params({**COMMON_PARAMS, **glm.PARAMS,
                       "categorical": ",".join(glm.CATEGORICAL),
                       "numeric": ",".join(glm.NUMERIC + glm.LOG_HISTORY)})
    model = glm.fit(split(features, "train"))
    mlflow.log_metrics(metrics_for_mlflow(glm.predict(model, validation), "validation"))
    write_forecasts(glm.predict(model, test), glm.NAME)
    log_test_results(spark, glm.NAME)

# COMMAND ----------

# MAGIC %md ## 3 · Chronos-Bolt (zero-shot)
# MAGIC Runs per zone in parallel with `applyInPandas`. Each test hour is forecast from the 512 hours before it.

# COMMAND ----------

# Workers can't download from Hugging Face (read-only disk), so fetch the
# weights once here into the volume and let every worker load from there.
model_path = chronos.download_model(f"{config.MODELS_PATH}/chronos-bolt-small")

modeled = features.select("zone_id").distinct()
demand = spark.table(config.TABLES["zone_hour_demand"]).join(modeled, "zone_id")

with mlflow.start_run(run_name=chronos.NAME):
    mlflow.set_tags({"stage": "baseline", "model": chronos.NAME})
    mlflow.log_params({**COMMON_PARAMS, "model_id": chronos.MODEL_ID,
                       "context_hours": chronos.CONTEXT_HOURS, "quantiles": chronos.QUANTILES})
    preds = chronos.predict(demand, test_start, test_end, model_path)
    # Inner join keeps exactly the rows the other models are scored on.
    write_forecasts(preds.join(test.select("zone_id", "hour_ts", "pickups"), ["zone_id", "hour_ts"]), chronos.NAME)
    log_test_results(spark, chronos.NAME)

# COMMAND ----------

# MAGIC %md ## Comparison

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=[config.MLFLOW_EXPERIMENT],
    filter_string="tags.stage = 'baseline'",
    order_by=["start_time DESC"],
)
cols = ["tags.model", "metrics.test_mae", "metrics.test_rmse", "metrics.test_wape", "metrics.validation_wape"]
display(runs[cols].drop_duplicates("tags.model").sort_values("metrics.test_wape"))
