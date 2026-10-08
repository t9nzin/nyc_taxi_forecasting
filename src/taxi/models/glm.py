"""Poisson regression baseline in Spark MLlib.

Pickups are counts, so a Poisson GLM with a log link fits them better than
least squares. Lag features enter as log1p so they act multiplicatively on the
log scale (e.g. "this hour ~ k x last week's").
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from taxi.features import HISTORY_COLUMNS

NAME = "poisson_glm"

CATEGORICAL = ["zone_id", "hour_of_week"]
NUMERIC = ["is_holiday", "temp_f", "precip_in", "is_raining", "wind_mph"]
LOG_HISTORY = [f"log1p_{c}" for c in HISTORY_COLUMNS]

PARAMS = {"family": "poisson", "link": "log", "maxIter": 25, "regParam": 1e-4}


def prepare(df: DataFrame) -> DataFrame:
    """Derived columns the pipeline expects."""
    df = df.withColumn("hour_of_week", F.col("day_of_week") * 24 + F.col("hour"))
    for c in HISTORY_COLUMNS:
        df = df.withColumn(f"log1p_{c}", F.log1p(F.col(c).cast("double")))
    return df


def build_pipeline() -> Pipeline:
    # One-hot encoding hour_of_week (0..167) lets weekday and weekend hours
    # have different profiles; zone one-hots give each zone its own level.
    encoder = OneHotEncoder(
        inputCols=CATEGORICAL,
        outputCols=[f"{c}_ohe" for c in CATEGORICAL],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in CATEGORICAL] + NUMERIC + LOG_HISTORY,
        outputCol="features_vec",
    )
    glm = GeneralizedLinearRegression(
        featuresCol="features_vec", labelCol="pickups", predictionCol="prediction", **PARAMS
    )
    return Pipeline(stages=[encoder, assembler, glm])


def fit(train: DataFrame):
    return build_pipeline().fit(prepare(train))


def predict(model, df: DataFrame) -> DataFrame:
    return model.transform(prepare(df))
