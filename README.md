# NYC Taxi Demand Forecasting

Forecasting next-hour taxi pickups for every NYC taxi zone, built on **Databricks + PySpark + MLflow**, comparing a zero-shot foundation model (Chronos-Bolt) with trained neural forecasters (**N-HiTS**, **Temporal Fusion Transformer**), plus an interactive forecast map.

> Status: data pipeline, baselines and neural models done and tested (DEV slice). Batch scoring, the map and the full 2022–2024 run are in progress.

## Architecture

```
TLC trip Parquet + zone lookup + NOAA hourly weather + holidays
        │  download into a Unity Catalog volume
        ▼
bronze.trips              schema-normalized raw trips (Delta)
        ▼  PySpark cleaning
silver.trips              outliers removed
        ▼  aggregate + gap-free zone × hour grid
silver.zone_hour_demand
        ▼  weather + holiday joins, window-function lags/rolling
gold.features             modeling table
        ▼
seasonal-naive │ Spark MLlib Poisson GLM │ Chronos-Bolt (zero-shot) │ N-HiTS │ TFT
        ▼  MLflow tracking → UC Model Registry @champion → spark_udf batch scoring
gold.forecasts → interactive 3D zone map (Databricks App)
```

## Data
- **Trips:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow + green, 2022–2024.
- **Weather:** NOAA NCEI Local Climatological Data, Central Park station.
- **Holidays:** US federal + New York State, from the `holidays` package.

## Pipeline details worth knowing
- **Schema drift:** TLC column types and names change between months (e.g. `payment_type` int vs double, `airport_fee` → `Airport_fee` in 2024). Each monthly file is read separately and mapped onto one canonical schema (`taxi.ingest.normalize_trips`).
- **Cleaning:** each dropped trip is tagged with the first rule it fails, so the drop table adds up exactly. January 2024 sample:

  | reason | trips |
  |---|---:|
  | kept | 2,902,052 |
  | bad_distance (≤ 0 or > 100 mi) | 37,926 |
  | impossible_duration (< 1 min or > 4 h) | 37,018 |
  | bad_fare (negative or > $1,000) | 32,000 |
  | unknown_pickup_zone | 12,159 |
  | pickup_outside_file_month | 20 |

- **Time zones:** TLC timestamps are NYC wall-clock time, but NOAA reports use Local Standard Time all year. Weather is shifted onto wall-clock time, so DST lines up.
- **No leakage:** an hour's weather is the last report *before* that hour starts. Lags and rolling windows end at t−1. The set of modeled zones is chosen on the training window only.
- **NOAA quirk:** long API responses get cut off mid-stream, so weather is downloaded one month at a time. Each file is checked to reach the end of its month before it is kept.

## Validation
Out-of-time splits: train 2022–2023, validate 2024 H1, test 2024 H2. Metrics are MAE, RMSE and **WAPE**. WAPE stays defined when many zone-hours have zero pickups, which MAPE does not.

## Results
Interim, on the DEV slice (train Jan 8 – Feb 29 2024, test Mar 16–31 2024, 222 zones). Full 2022–2024 results to follow.

| model | inputs | test WAPE |
|---|---|---:|
| Chronos-Bolt (zero-shot, never trained on this data) | pickup history | 15.5% |
| N-HiTS | pickup history | 16.8% |
| Poisson GLM (Spark MLlib) | zone × hour-of-week, weather, holidays, lags | 19.1% |
| seasonal-naive (same hour last week) | — | 21.6% |
| Temporal Fusion Transformer | history + calendar, weather, borough | 23.1% |

## Neural models: what mattered
- **Per-window scaling breaks sparse inputs.** neuralforecast normalises each input inside every training window, so `is_holiday` in a holiday-free week, or rain in a dry week, becomes ~10⁶ when it fires. Pickups and weather are instead scaled with fixed train-split statistics that are saved with the model (`taxi.models.neural`).
- **With seven weeks of training data, extra inputs hurt.** Weather is the same for every zone and different every hour, so the models used it to memorise dates: N-HiTS went from 16.8% (history only) to 19.0% with calendar inputs and 22.4% with weather too. N-HiTS is therefore the main model, and the TFT is kept for its variable-importance readout.

## Running it

### Databricks (Free Edition)
1. Sign up at [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).
2. In the workspace, go to **Workspace → Create → Git folder** and paste this repo's GitHub URL.
3. Run the notebooks in order on serverless compute:
   - `notebooks/00_setup` creates the catalog, schemas, volume and MLflow experiment. If creating a catalog isn't allowed, set `CATALOG = "workspace"` in `src/taxi/config.py`.
   - `01_ingest` → `02_clean` → `03_aggregate_features`.
   - `04_baselines`: seasonal-naive, Poisson GLM and Chronos-Bolt, each an MLflow run.
   - `05_train_neural`: N-HiTS and TFT; the better one is registered in Unity Catalog as `nyc_taxi.ml.demand_forecaster` (`@champion` if it beats the current one).
4. `config.DEV_MODE = True` (the default) runs on Jan–Mar 2024 only. Set it to `False` for the full 2022–2024 run.

### Locally (tests + one-month smoke run)
Needs Java 17.
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/local_smoke.py   # downloads Jan 2024 into ./data and runs the whole pipeline
```

## Repo layout
```
src/taxi/      importable pipeline logic (config, ingest, cleaning, aggregate, features, evaluation)
  models/      naive, glm (Spark MLlib), chronos (applyInPandas), neural (N-HiTS / TFT + MLflow pyfunc)
notebooks/     thin Databricks notebooks that call src/taxi
tests/         pytest suite on a local SparkSession
scripts/       local smoke run
```
