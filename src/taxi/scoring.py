"""Batch scoring with the registered ``@champion`` model.

The champion is a panel model: each zone's forecast needs that zone's
previous week, not one feature row, so ``mlflow.pyfunc.spark_udf`` (which
scores rows independently) doesn't fit. Instead the model folder is
downloaded once on the driver into a UC Volume, and ``applyInPandas`` hands
each worker one zone's window; the worker loads the pyfunc from the Volume
once and forecasts that zone.
"""

import os

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from taxi import config, features
from taxi.models import neural

OUTPUT_SCHEMA = "zone_id INT, hour_ts TIMESTAMP, prediction DOUBLE, p10 DOUBLE, p90 DOUBLE"

_model = None


def download_champion(dest_root: str = f"{config.MODELS_PATH}/demand_forecaster", alias: str = "champion"):
    """Copy the model version behind ``alias`` into a folder workers can read.

    Returns ``(path, version)``. A version is never changed once registered,
    so a folder that's already there is reused.
    """
    import mlflow

    version = mlflow.MlflowClient().get_model_version_by_alias(config.MODEL_NAME, alias)
    dest = f"{dest_root}/v{version.version}"
    if not os.path.exists(os.path.join(dest, "MLmodel")):
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{config.MODEL_NAME}/{version.version}", dst_path=dest
        )
    return dest, version


def _load(model_path: str):
    """Load once per Python worker; applyInPandas calls us once per zone."""
    global _model
    if _model is None:
        import mlflow.pyfunc

        _model = mlflow.pyfunc.load_model(model_path)
    return _model


def scoring_window(feats: DataFrame, holidays: DataFrame, target_hour, input_size: int) -> DataFrame:
    """Model input for forecasting ``target_hour``: each zone's ``input_size``
    hours before it, plus one row for ``target_hour`` with ``pickups`` null.

    If ``target_hour`` is already in ``feats`` (a past hour, e.g. to check
    against a backtest), its calendar and observed weather are used. Otherwise
    it's the hour after the data ends: calendar is computed from the
    timestamp and weather is carried forward from the last observed hour.
    """
    target = pd.Timestamp(target_hour).to_pydatetime()
    start = target - pd.Timedelta(hours=input_size).to_pytimedelta()
    rows = feats.select(*neural.INPUT_COLUMNS)
    history = rows.filter((F.col("hour_ts") >= F.lit(start)) & (F.col("hour_ts") < F.lit(target)))

    known = rows.filter(F.col("hour_ts") == F.lit(target))
    if known.limit(1).count():
        future = known
    else:
        last = history.filter(F.col("hour_ts") == F.lit(target - pd.Timedelta(hours=1).to_pytimedelta()))
        future = features.add_calendar(
            last.drop(*features.CALENDAR_COLUMNS).withColumn("hour_ts", F.lit(target)), holidays
        )
    future = future.withColumn("pickups", F.lit(None))
    # Same column order and types as history, so the two union cleanly.
    future = future.select(*[F.col(f.name).cast(f.dataType) for f in history.schema])
    return history.unionByName(future).withColumn("pickups", F.col("pickups").cast("double"))


def score(window: DataFrame, model_path: str) -> DataFrame:
    """Forecast the null-pickups hour of every zone in ``window`` (see
    ``scoring_window``), one zone per task."""
    import sys

    from pyspark import cloudpickle

    # Workers can't import `taxi` from src/; ship this module by value. The
    # model brings its own copy of `taxi` (logged with code_paths).
    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    def forecast_zone(pdf: pd.DataFrame) -> pd.DataFrame:
        return _load(model_path).predict(pdf.sort_values("hour_ts", ignore_index=True))

    return window.groupBy("zone_id").applyInPandas(forecast_zone, schema=OUTPUT_SCHEMA)
