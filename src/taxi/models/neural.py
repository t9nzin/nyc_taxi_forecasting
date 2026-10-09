"""Trained neural forecasters (neuralforecast): N-HiTS (main) and TFT.

Each is one global model across all zones that forecasts the next hour's
pickups from the zone's previous week, as the 10th/50th/90th percentiles.

- **N-HiTS** sees pickup history only. It is fast on CPU and, on the DEV split,
  more accurate than versions given weather or calendar inputs: with weeks
  rather than years of training data, those inputs let it memorise dates.
- **TFT** also gets the target hour's calendar and weather (known future
  inputs) and the zone's borough (static), and reports how much it relied on
  each through its variable-selection weights. Weather for the target hour is
  treated as known: one hour ahead, a forecast is close to what is observed.

Scaling is done here, not by neuralforecast. Its built-in scalers normalise
every input inside each training window, which turns a mostly-zero column
(``is_holiday`` in a week without a holiday, ``precip_in`` in a dry week) into
values around 1e6 whenever it fires. Instead, pickups are divided by each
zone's training-period mean and continuous weather is standardised with
training-period statistics; both are fixed, saved with the model, and computed
on the train split only. N-HiTS, which has no exogenous inputs, additionally
uses neuralforecast's per-window robust scaling of pickups, which helps it
follow level shifts.

Everything here works on pandas on the driver; the training panel (zones x
hours) fits in memory even for the full three years.
"""

import json
import os

import numpy as np
import pandas as pd

LEVEL = 80  # central interval -> p10 / p90
ALIAS = "nn"  # neuralforecast prefixes output columns with this ("model" is renamed)

BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island"]
STAT_EXOG = [f"borough_{b.lower().replace(' ', '_')}" for b in BOROUGHS]
# Cyclical encodings so hour 23 sits next to hour 0 (and Sunday next to Monday).
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]
WEATHER = ["temp_f", "precip_in", "is_raining", "wind_mph", "humidity_pct", "visibility_mi"]
FUTR_EXOG = CALENDAR + WEATHER
# Standardised with train-split mean/std; the rest are already 0/1 or in [-1, 1].
CONTINUOUS = ["temp_f", "precip_in", "wind_mph", "humidity_pct", "visibility_mi"]

# Columns the models need from gold.features, and the pyfunc's input schema.
INPUT_COLUMNS = ["zone_id", "hour_ts", "pickups", "borough", "hour", "day_of_week",
                 "is_weekend", "is_holiday"] + WEATHER
OUTPUT_COLUMNS = ["zone_id", "hour_ts", "prediction", "p10", "p90"]

_TRAINING = {
    "h": 1,
    "input_size": 168,  # one week of hourly history
    "learning_rate": 1e-3,
    "max_steps": 1000,
    "val_check_steps": 100,
    "early_stop_patience_steps": 3,  # validation checks without improvement
    "batch_size": 64,  # zones per step
    "windows_batch_size": 256,
    "random_seed": 42,
}

# Sized for CPU. A TFT step costs ~hidden_size^2 x windows, and every
# validation check runs the model over every zone's full history.
SPECS = {
    "nhits": {
        "futr_exog": [],
        "stat_exog": [],
        "params": {**_TRAINING, "scaler_type": "robust"},
    },
    "tft": {
        "futr_exog": FUTR_EXOG,
        "stat_exog": STAT_EXOG,
        "params": {**_TRAINING, "hidden_size": 32, "n_head": 4, "dropout": 0.1, "scaler_type": "identity"},
    },
}


def _weather(pdf: pd.DataFrame) -> pd.DataFrame:
    """Raw continuous weather, with precipitation log-compressed (it's skewed)."""
    w = pdf[CONTINUOUS].astype(float).copy()
    w["precip_in"] = np.log1p(w["precip_in"].clip(lower=0) * 100)
    return w


def fit_scales(pdf: pd.DataFrame, train_end) -> dict:
    """Per-zone pickup scale and weather mean/std, from hours before ``train_end``."""
    train = pdf[pd.to_datetime(pdf["hour_ts"]) < pd.Timestamp(train_end)]
    zone = train.groupby("zone_id")["pickups"].mean().clip(lower=0.1)
    w = _weather(train)
    return {
        "zone": {int(k): float(v) for k, v in zone.items()},
        "mean": w.mean().to_dict(),
        "std": w.std().replace(0, 1).to_dict(),
    }


def _zone_scale(zone_ids, scales: dict) -> np.ndarray:
    scale = pd.Series(np.asarray(zone_ids)).map(scales["zone"])
    if scale.isna().any():
        raise ValueError(f"zones not seen in training: {sorted(set(np.asarray(zone_ids)[scale.isna()]))}")
    return scale.to_numpy()


def to_long(pdf: pd.DataFrame, scales: dict):
    """gold.features rows -> neuralforecast's (df, static_df), scaled.

    ``df`` has ``unique_id`` (zone), ``ds`` (hour), ``y`` (pickups / zone
    scale) and the future exogenous columns; ``static_df`` has one borough
    one-hot row per zone.
    """
    hour = pdf["hour"].to_numpy()
    dow = pdf["day_of_week"].to_numpy()
    w = _weather(pdf)
    df = pd.DataFrame(
        {
            "unique_id": pdf["zone_id"].astype(int).to_numpy(),
            "ds": pd.to_datetime(pdf["hour_ts"]).to_numpy(),
            "y": pdf["pickups"].astype(float).to_numpy() / _zone_scale(pdf["zone_id"], scales),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": pdf["is_weekend"].astype(float).to_numpy(),
            "is_holiday": pdf["is_holiday"].astype(float).to_numpy(),
            "is_raining": pdf["is_raining"].astype(float).to_numpy(),
            **{c: ((w[c] - scales["mean"][c]) / scales["std"][c]).to_numpy() for c in CONTINUOUS},
        }
    ).sort_values(["unique_id", "ds"], ignore_index=True)

    zones = pdf[["zone_id", "borough"]].drop_duplicates("zone_id")
    static_df = pd.DataFrame({"unique_id": zones["zone_id"].astype(int).to_numpy()})
    for borough, col in zip(BOROUGHS, STAT_EXOG):
        static_df[col] = (zones["borough"].to_numpy() == borough).astype(float)
    return df, static_df


def build(kind: str, params: dict = None):
    """An unfitted NeuralForecast around one model from ``SPECS``."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, TFT

    spec = SPECS[kind]
    model = {"nhits": NHITS, "tft": TFT}[kind](
        **(params or spec["params"]),
        futr_exog_list=spec["futr_exog"] or None,
        stat_exog_list=spec["stat_exog"] or None,
        loss=MQLoss(level=[LEVEL]),
        alias=ALIAS,
        # No checkpoints or log folders written next to the notebook.
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return NeuralForecast(models=[model], freq="h")


def _to_output(fc: pd.DataFrame, scales: dict) -> pd.DataFrame:
    """neuralforecast columns -> ``OUTPUT_COLUMNS`` in pickups; demand can't be negative."""
    # The band is named "nn-lo-80" or "nn-lo-80.0" depending on how the
    # level was given; match on the prefix.
    lo, hi = (next(c for c in fc.columns if c.startswith(f"{ALIAS}-{side}-")) for side in ("lo", "hi"))
    scale = _zone_scale(fc["unique_id"], scales)[:, None]
    # Quantile heads are trained separately and can cross; sorting each row
    # keeps p10 <= median <= p90.
    q = np.sort(fc[[lo, f"{ALIAS}-median", hi]].to_numpy() * scale, axis=1).clip(min=0)
    return pd.DataFrame(
        {
            "zone_id": fc["unique_id"].astype("int32").to_numpy(),
            "hour_ts": pd.to_datetime(fc["ds"]).to_numpy(),
            "prediction": q[:, 1],
            "p10": q[:, 0],
            "p90": q[:, 2],
        }
    )


class Forecaster:
    """A fitted NeuralForecast plus the scales it was trained with."""

    def __init__(self, nf, scales: dict):
        self.nf, self.scales = nf, scales

    @property
    def model(self):
        return self.nf.models[0]

    def save(self, path: str):
        self.nf.save(os.path.join(path, "neuralforecast"), save_dataset=False, overwrite=True)
        with open(os.path.join(path, "scales.json"), "w") as f:
            json.dump(self.scales, f)

    @classmethod
    def load(cls, path: str) -> "Forecaster":
        from neuralforecast import NeuralForecast

        with open(os.path.join(path, "scales.json")) as f:
            scales = json.load(f)
        scales["zone"] = {int(k): v for k, v in scales["zone"].items()}  # JSON keys are strings
        return cls(NeuralForecast.load(os.path.join(path, "neuralforecast")), scales)

    def predict_next_hour(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """Forecast the next hour per zone.

        ``pdf`` holds, for each zone, at least ``input_size`` hours of history
        with ``pickups`` filled, plus exactly one later hour with ``pickups``
        null (its calendar and weather columns filled). The model scores that hour.
        """
        is_future = pdf["pickups"].isna()
        history, static_df = to_long(pdf[~is_future], self.scales)
        futr_df = None
        if self.model.futr_exog_list:
            futr_df = to_long(pdf[is_future].assign(pickups=0.0), self.scales)[0].drop(columns="y")
        fc = self.nf.predict(df=history, static_df=static_df, futr_df=futr_df)
        return _to_output(fc, self.scales)


def backtest(pdf: pd.DataFrame, val_start, test_start, test_end, kind: str, params: dict = None):
    """Train once and forecast every test hour one step ahead.

    ``pdf`` is gold.features as pandas. Training uses hours before
    ``val_start``; [val_start, test_start) is the early-stopping validation
    set; each hour in [test_start, test_end) is then forecast from the hours
    before it, with weights frozen. Returns ``(Forecaster, forecasts)``.
    """
    val_start, test_start, test_end = (pd.Timestamp(t) for t in (val_start, test_start, test_end))
    pdf = pdf[pd.to_datetime(pdf["hour_ts"]) < test_end]
    scales = fit_scales(pdf, val_start)
    df, static_df = to_long(pdf, scales)
    hours = df["ds"].drop_duplicates().sort_values()
    val_size = int(((hours >= val_start) & (hours < test_start)).sum())
    test_size = int((hours >= test_start).sum())

    nf = build(kind, params)
    cv = nf.cross_validation(
        df=df, static_df=static_df, n_windows=test_size, step_size=1, val_size=val_size, refit=False
    )
    return Forecaster(nf, scales), _to_output(cv, scales)


def feature_importances(forecaster: Forecaster) -> dict:
    """TFT only: mean variable-selection weight per input, by input type."""
    raw = forecaster.model.feature_importances()
    return {
        "past": raw["Past variable importance over time"].mean().sort_values(ascending=False),
        "future": raw["Future variable importance over time"].mean().sort_values(ascending=False),
        "static": raw["Static covariates"]["importance"].sort_values(ascending=False),
    }


def importance_figure(importances: dict):
    """Horizontal bar chart per input type."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"past": "Past inputs (history window)", "future": "Known future inputs", "static": "Static (borough)"}
    for ax, (kind, s) in zip(axes, importances.items()):
        s = s.sort_values()
        ax.barh(s.index, s.to_numpy(), color="#2a6fdb")
        ax.set_title(titles[kind])
        ax.set_xlabel("mean selection weight")
    fig.suptitle("TFT variable importance")
    fig.tight_layout()
    return fig


try:
    import mlflow.pyfunc

    class NeuralForecaster(mlflow.pyfunc.PythonModel):
        """MLflow wrapper so a model can be registered in Unity Catalog and
        loaded anywhere with ``mlflow.pyfunc.load_model`` / ``spark_udf``.
        Input: rows shaped like ``INPUT_COLUMNS`` (see ``Forecaster.predict_next_hour``).
        Output: ``OUTPUT_COLUMNS``."""

        def load_context(self, context):
            self.forecaster = Forecaster.load(context.artifacts["forecaster"])

        def predict(self, context, model_input, params=None):
            return self.forecaster.predict_next_hour(model_input)

except ImportError:  # mlflow isn't needed for training or tests
    pass
