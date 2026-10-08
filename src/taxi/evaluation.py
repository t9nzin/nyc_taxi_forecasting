"""Out-of-time splits and forecast metrics shared by every model."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from taxi import config


def split(df: DataFrame, name: str, ts_col: str = "hour_ts") -> DataFrame:
    """Rows of ``df`` that fall in the named out-of-time split."""
    start, end = config.splits()[name]
    return df.filter(
        (F.col(ts_col) >= F.lit(start).cast("timestamp"))
        & (F.col(ts_col) < F.lit(end).cast("timestamp"))
    )


def metrics(df: DataFrame, actual: str = "pickups", pred: str = "prediction", by=None):
    """MAE, RMSE and WAPE, overall or grouped by ``by`` (e.g. "borough").

    WAPE = sum|error| / sum(actual). Unlike MAPE it stays defined when many
    zone-hours have zero pickups, which is common here.
    """
    err = F.col(pred) - F.col(actual)
    aggs = [
        F.avg(F.abs(err)).alias("mae"),
        F.sqrt(F.avg(err * err)).alias("rmse"),
        (F.sum(F.abs(err)) / F.sum(F.col(actual))).alias("wape"),
        F.count("*").alias("n"),
    ]
    return df.groupBy(by).agg(*aggs) if by else df.agg(*aggs)


def overall_metrics(df: DataFrame, actual: str = "pickups", pred: str = "prediction") -> dict:
    """Overall metrics as a plain dict, ready for ``mlflow.log_metrics``."""
    row = metrics(df, actual, pred).first().asDict()
    return {k: float(v) for k, v in row.items() if k != "n"}


def metrics_for_mlflow(scored: DataFrame, split_name: str) -> dict:
    """Overall + per-borough metrics, keyed like ``test_wape`` and
    ``test_wape_Manhattan`` (MLflow metric names can't contain spaces)."""
    out = {f"{split_name}_{k}": v for k, v in overall_metrics(scored).items()}
    for row in metrics(scored, by="borough").collect():
        borough = row["borough"].replace(" ", "_")
        for k in ("mae", "rmse", "wape"):
            out[f"{split_name}_{k}_{borough}"] = float(row[k])
    return out


FORECAST_COLUMNS = ["model", "zone_id", "hour_ts", "pickups", "prediction", "p10", "p90"]


def write_forecasts(scored: DataFrame, model_name: str, table: str = config.TABLES["forecasts"]):
    """Replace one model's rows in the shared forecasts table, which the map
    app reads. Models without prediction intervals get null p10/p90."""
    for c in ("p10", "p90"):
        if c not in scored.columns:
            scored = scored.withColumn(c, F.lit(None).cast("double"))
    out = scored.withColumn("model", F.lit(model_name)).select(*FORECAST_COLUMNS)
    spark = scored.sparkSession
    if spark.catalog.tableExists(table):
        out.write.mode("overwrite").option("replaceWhere", f"model = '{model_name}'").saveAsTable(table)
    else:
        out.write.partitionBy("model").saveAsTable(table)


def plot_zone(scored: DataFrame, zone_id: int, title: str, days: int = 7):
    """Actual vs forecast for one zone over the first ``days`` of ``scored``,
    with the p10-p90 band when present. Returns a matplotlib figure."""
    import matplotlib.pyplot as plt
    import pandas as pd

    cols = ["hour_ts", "pickups", "prediction"] + [c for c in ("p10", "p90") if c in scored.columns]
    pdf = scored.filter(F.col("zone_id") == zone_id).select(*cols).orderBy("hour_ts").toPandas()
    pdf = pdf[pdf["hour_ts"] < pdf["hour_ts"].min() + pd.Timedelta(days=days)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pdf["hour_ts"], pdf["pickups"], label="actual", color="#222222", linewidth=1.5)
    ax.plot(pdf["hour_ts"], pdf["prediction"], label="forecast", color="#2a6fdb", linewidth=1.2)
    if "p10" in pdf and pdf["p10"].notna().any():
        ax.fill_between(pdf["hour_ts"], pdf["p10"], pdf["p90"], color="#2a6fdb", alpha=0.2, label="p10–p90")
    ax.set_title(title)
    ax.set_ylabel("pickups / hour")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

