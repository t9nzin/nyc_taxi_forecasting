import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from taxi import scoring
from taxi.models import neural
from tests.test_neural import _features

INPUT_SIZE = 48


@pytest.fixture(scope="module")
def champion(spark, tmp_path_factory):
    """A tiny N-HiTS backtest, saved as an MLflow pyfunc the way notebook 05 logs it."""
    pytest.importorskip("neuralforecast")
    mlflow = pytest.importorskip("mlflow")
    from mlflow.models import infer_signature

    import taxi

    # Round-trip through Spark so dtypes match gold.features (the signature enforces them).
    pdf = spark.createDataFrame(_features(hours=24 * 12)).toPandas()
    params = {**neural.SPECS["nhits"]["params"], "input_size": INPUT_SIZE, "max_steps": 5,
              "val_check_steps": 5, "batch_size": 2, "windows_batch_size": 64, "accelerator": "cpu"}
    forecaster, fc = neural.backtest(pdf, "2024-01-10", "2024-01-11", "2024-01-12", "nhits", params)

    root = tmp_path_factory.mktemp("champion")
    forecaster.save(str(root / "forecaster"))
    example = pdf.tail(INPUT_SIZE + 1).assign(pickups=lambda d: d.pickups.astype(float))
    mlflow.pyfunc.save_model(
        str(root / "model"),
        python_model=neural.NeuralForecaster(),
        artifacts={"forecaster": str(root / "forecaster")},
        code_paths=[os.path.dirname(taxi.__file__)],
        signature=infer_signature(example, forecaster.predict_next_hour(example)),
    )
    return pdf, fc, str(root / "model")


def scoring_window_pd(feats, holidays, target):
    return scoring.scoring_window(feats, holidays, target, INPUT_SIZE).toPandas()


def test_scoring_window_known_and_next_hour(spark):
    pdf = _features(hours=72)
    feats = spark.createDataFrame(pdf)
    holidays = spark.createDataFrame([(date(2024, 1, 4), "x")], "date DATE, holiday_name STRING")

    # A past hour: observed calendar + weather, pickups hidden.
    past = scoring_window_pd(feats, holidays, "2024-01-03 00:00")
    assert len(past) == 2 * (INPUT_SIZE + 1)
    target = past[past.hour_ts == "2024-01-03 00:00"]
    assert target.pickups.isna().all() and past.pickups.notna().sum() == 2 * INPUT_SIZE

    # The hour after the data ends: calendar from the timestamp, weather carried forward.
    nxt = scoring_window_pd(feats, holidays, "2024-01-04 00:00")
    target = nxt[nxt.hour_ts == "2024-01-04 00:00"]
    assert len(target) == 2 and target.pickups.isna().all()
    assert (target.hour == 0).all() and (target.day_of_week == 3).all()  # Thursday
    assert (target.is_holiday == 1).all() and (target.temp_f == 40.0).all()


def test_score_with_registered_model_matches_backtest(spark, champion):
    """Spark scoring of the logged pyfunc reproduces the backtest's forecast."""
    pdf, fc, model_path = champion
    target = pd.Timestamp("2024-01-11 00:00")
    holidays = spark.createDataFrame([], "date DATE, holiday_name STRING")
    window = scoring.scoring_window(spark.createDataFrame(pdf), holidays, target, INPUT_SIZE)

    out = scoring.score(window, model_path).toPandas().sort_values("zone_id", ignore_index=True)
    expected = fc[fc.hour_ts == target].sort_values("zone_id", ignore_index=True)
    assert out.hour_ts.eq(target).all() and list(out.zone_id) == list(expected.zone_id)
    np.testing.assert_allclose(out.prediction, expected.prediction, rtol=1e-4)
