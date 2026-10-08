"""Seasonal-naive baseline: predict the same hour last week."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

NAME = "seasonal_naive"


def predict(features: DataFrame) -> DataFrame:
    return features.withColumn("prediction", F.col("lag_168h").cast("double"))
