"""Zero-shot Chronos-Bolt baseline, run per zone with Spark ``applyInPandas``.

Chronos-Bolt is a pretrained time-series model: it sees only each zone's
recent pickup history (no weather, no calendar) and is never trained on this
data. For every target hour t it forecasts t from the ``CONTEXT_HOURS`` hours
before t, which matches the one-hour-ahead setup of the other models.
"""

import numpy as np
import pandas as pd

NAME = "chronos_bolt"
MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT_HOURS = 512
BATCH_SIZE = 256
QUANTILES = [0.1, 0.5, 0.9]

OUTPUT_SCHEMA = "zone_id INT, hour_ts TIMESTAMP, prediction DOUBLE, p10 DOUBLE, p90 DOUBLE"

_pipeline = None


def _load():
    """Load once per Python worker; applyInPandas calls us once per zone."""
    global _pipeline
    if _pipeline is None:
        import os
        import tempfile

        # Serverless workers have a read-only home directory, where Hugging
        # Face caches downloads by default. Point every HF cache (including
        # the separate xet download cache) at the writable temp dir. These are
        # read when huggingface_hub is first imported, so set them before.
        hf_home = os.path.join(tempfile.gettempdir(), "hf_home")
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ["HF_XET_CACHE"] = os.path.join(hf_home, "xet")
        os.environ["XDG_CACHE_HOME"] = os.path.join(hf_home, "xdg")

        import torch
        from chronos import BaseChronosPipeline

        _pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID, device_map="cpu", torch_dtype=torch.float32,
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
    return _pipeline


def rolling_contexts(pickups: np.ndarray, first_target: int, context: int):
    """Context windows for targets ``first_target..len(pickups)-1``.

    Row i is ``pickups[t - context : t]`` for ``t = first_target + i``, i.e.
    only hours strictly before the target.
    """
    if first_target < context:
        raise ValueError(f"need {context} hours of history before the first target")
    windows = np.lib.stride_tricks.sliding_window_view(pickups[:-1], context)
    return windows[first_target - context:]


def forecast_zone(pdf: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """One zone's hourly panel (history + test window) -> one-step forecasts
    for each hour >= ``test_start``."""
    import torch

    pdf = pdf.sort_values("hour_ts").reset_index(drop=True)
    first_target = int((pdf["hour_ts"] < test_start).sum())
    contexts = rolling_contexts(pdf["pickups"].to_numpy(dtype=np.float32), first_target, CONTEXT_HOURS)

    pipeline = _load()
    quantiles = []
    for i in range(0, len(contexts), BATCH_SIZE):
        batch = torch.from_numpy(np.ascontiguousarray(contexts[i:i + BATCH_SIZE]))
        q, _ = pipeline.predict_quantiles(batch, prediction_length=1, quantile_levels=QUANTILES)
        quantiles.append(q[:, 0, :].numpy())
    q = np.clip(np.concatenate(quantiles), 0, None)  # demand can't be negative

    targets = pdf.iloc[first_target:]
    return pd.DataFrame(
        {
            "zone_id": targets["zone_id"].to_numpy(),
            "hour_ts": targets["hour_ts"].to_numpy(),
            "prediction": q[:, 1],
            "p10": q[:, 0],
            "p90": q[:, 2],
        }
    )


def predict(demand, test_start, test_end):
    """Spark entry point. ``demand`` is the zone x hour panel for modeled zones."""
    import sys

    from pyspark import cloudpickle
    from pyspark.sql import functions as F

    # Workers can't import `taxi` (src/ is only on the driver's sys.path), so
    # ship this module's code inside the pickled UDF instead of by reference.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    history_start = pd.Timestamp(test_start) - pd.Timedelta(hours=CONTEXT_HOURS)
    window = demand.filter(
        (F.col("hour_ts") >= F.lit(history_start.to_pydatetime()))
        & (F.col("hour_ts") < F.lit(test_end).cast("timestamp"))
    ).select("zone_id", "hour_ts", "pickups")
    ts = pd.Timestamp(test_start)
    return window.groupBy("zone_id").applyInPandas(
        lambda pdf: forecast_zone(pdf, ts), schema=OUTPUT_SCHEMA
    )
