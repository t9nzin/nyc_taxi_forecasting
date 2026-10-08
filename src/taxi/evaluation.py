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
