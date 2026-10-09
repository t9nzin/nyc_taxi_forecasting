# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Neural models: N-HiTS and TFT
# MAGIC Two global models trained across all zones with `neuralforecast`, each predicting the 10th/50th/90th percentile of next-hour pickups:
# MAGIC - **N-HiTS** (main): pickup history only; trains in minutes on CPU
# MAGIC - **TFT**: adds calendar, holiday, weather and borough inputs, and reports which ones it relies on
# MAGIC
# MAGIC Each is trained on the train split, early-stopped on validation, then forecasts every test hour one step ahead (the same rows as the baselines). The better one is registered in Unity Catalog and becomes `@champion` if it beats the current champion.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt -r ../requirements-ml.txt

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import logging
import os
import tempfile
import time

import mlflow
import neuralforecast
import pandas as pd
from mlflow.models import infer_signature

import taxi
from taxi.evaluation import log_test_results, split, write_forecasts
from taxi.models import neural

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

splits = config.splits()
val_start = splits["validation"][0]
test_start, test_end = splits["test"]
test_window = f"{test_start}→{test_end}"

features = spark.table(config.TABLES["features"])
test = split(features, "test")
pdf = features.select(*neural.INPUT_COLUMNS).toPandas()
print(f"{len(pdf):,} rows, {pdf.zone_id.nunique()} zones, {pdf.hour_ts.min()} → {pdf.hour_ts.max()}")


def example_input(n_zones: int = 2) -> pd.DataFrame:
    """A pyfunc input: a few zones' last week before the test window, plus the
    first test hour with pickups left null (the hour to forecast)."""
    start = pd.Timestamp(test_start)
    lookback = pd.Timedelta(hours=neural.SPECS["nhits"]["params"]["input_size"])
    window = pdf[pdf.zone_id.isin(pdf.zone_id.drop_duplicates().head(n_zones))
                 & (pdf.hour_ts >= start - lookback) & (pdf.hour_ts <= start)].copy()
    window["pickups"] = window["pickups"].astype(float).where(window.hour_ts < start)
    return window


def train_and_log(kind: str) -> dict:
    """Backtest one model, log it to MLflow, and return its run and model URI."""
    spec = neural.SPECS[kind]
    with mlflow.start_run(run_name=kind) as run:
        mlflow.set_tags({"stage": "main", "model": kind})
        mlflow.log_params({
            "dev_mode": config.DEV_MODE,
            "train_window": "→".join(map(str, splits["train"])),
            "test_window": test_window,
            "n_zones": pdf.zone_id.nunique(),
            "futr_exog": ",".join(spec["futr_exog"]) or "none",
            "stat_exog": ",".join(spec["stat_exog"]) or "none",
            "quantiles": "0.1,0.5,0.9",
            **spec["params"],
        })

        t0 = time.time()
        forecaster, forecasts = neural.backtest(pdf, val_start, test_start, test_end, kind)
        mlflow.log_metrics({"train_minutes": (time.time() - t0) / 60,
                            "train_steps": forecaster.model.train_trajectories[-1][0]})
        for step, loss in forecaster.model.valid_trajectories:  # early-stopping curve
            mlflow.log_metric("validation_loss", float(loss), step=int(step))
        print(f"{kind}: trained + backtested in {(time.time() - t0) / 60:.1f} min")

        # Inner join keeps exactly the rows the baselines are scored on.
        scored = spark.createDataFrame(forecasts).join(
            test.select("zone_id", "hour_ts", "pickups"), ["zone_id", "hour_ts"]
        )
        # A time-zone slip would silently drop rows here; fail loudly instead.
        assert scored.count() == len(forecasts), "forecast hours didn't line up with gold.features"
        write_forecasts(scored, kind)
        log_test_results(spark, kind)

        if kind == "tft":
            importances = neural.feature_importances(forecaster)
            mlflow.log_figure(neural.importance_figure(importances), "variable_importance.png")
            for group, weights in importances.items():
                mlflow.log_dict(weights.round(4).to_dict(), f"variable_importance_{group}.json")

        window = example_input()
        with tempfile.TemporaryDirectory() as tmp:
            forecaster.save(tmp)  # neuralforecast weights + the train-split scales
            log_kwargs = {"name": "model"} if int(mlflow.__version__.split(".")[0]) >= 3 else {"artifact_path": "model"}
            info = mlflow.pyfunc.log_model(
                **log_kwargs,
                python_model=neural.NeuralForecaster(),
                artifacts={"forecaster": tmp},
                code_paths=[os.path.dirname(taxi.__file__)],  # ships src/taxi with the model
                pip_requirements=[f"neuralforecast=={neuralforecast.__version__}", "torch", "pandas", "numpy"],
                signature=infer_signature(window, forecaster.predict_next_hour(window)),
                input_example=window,
            )
    test_wape = mlflow.get_run(run.info.run_id).data.metrics["test_wape"]
    return {"kind": kind, "run_id": run.info.run_id, "model_uri": info.model_uri, "test_wape": test_wape,
            "forecaster": forecaster}

# COMMAND ----------

# MAGIC %md ## N-HiTS (main)

# COMMAND ----------

results = {"nhits": train_and_log("nhits")}

# COMMAND ----------

# MAGIC %md ## TFT
# MAGIC Much slower on CPU (roughly 25–60 min). Set `TRAIN_TFT = False` to skip it.

# COMMAND ----------

TRAIN_TFT = True
if TRAIN_TFT:
    results["tft"] = train_and_log("tft")
    display(pd.concat(neural.feature_importances(results["tft"]["forecaster"])).rename("weight")
            .reset_index().rename(columns={"level_0": "input_type", "level_1": "input"}))

# COMMAND ----------

# MAGIC %md ## Register in Unity Catalog
# MAGIC The better neural model becomes a new version of `nyc_taxi.ml.demand_forecaster`. It takes each zone's recent history plus the hour to forecast (pickups null) and returns p10/p50/p90; notebook 06 scores with it. `@champion` moves to it if it has a lower test WAPE than the current champion on the same test window.

# COMMAND ----------

best = min(results.values(), key=lambda r: r["test_wape"])
version = mlflow.register_model(best["model_uri"], config.MODEL_NAME).version
print(f"registered {best['kind']} as {config.MODEL_NAME} v{version} (test WAPE {best['test_wape']:.4f})")

client = mlflow.MlflowClient()
try:
    champion = client.get_model_version_by_alias(config.MODEL_NAME, "champion")
    champ_run = client.get_run(champion.run_id).data
    same_window = champ_run.params.get("test_window") == test_window
    champ_wape = champ_run.metrics["test_wape"] if same_window else float("inf")
except Exception:  # no champion yet
    champion, champ_wape = None, float("inf")

if best["test_wape"] < champ_wape:
    client.set_registered_model_alias(config.MODEL_NAME, "champion", version)
    print(f"v{version} is the new @champion")
else:
    print(f"kept @champion v{champion.version} (test WAPE {champ_wape:.4f})")

# COMMAND ----------

# MAGIC %md ## All models on the same test window

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=[config.MLFLOW_EXPERIMENT],
    filter_string=f"params.test_window = '{test_window}'",
    order_by=["start_time DESC"],
)
cols = ["tags.model", "tags.stage", "metrics.test_mae", "metrics.test_rmse", "metrics.test_wape",
        "metrics.test_coverage_p10_p90", "metrics.train_minutes"]
display(runs[cols].drop_duplicates("tags.model").sort_values("metrics.test_wape"))
