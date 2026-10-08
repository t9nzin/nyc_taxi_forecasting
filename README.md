# NYC Taxi Demand Forecasting

Forecasting next-hour taxi pickups for every NYC taxi zone, built on **Databricks + PySpark + MLflow** with a **Temporal Fusion Transformer** and an interactive forecast map.

> Status: data pipeline (ingest → clean → features) done and tested. Models, scoring and the map are in progress.

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
seasonal-naive │ Spark MLlib Poisson GLM │ Chronos-Bolt (zero-shot) │ TFT
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
| model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| seasonal-naive (same hour last week) | _tbd_ | _tbd_ | _tbd_ |
| Poisson GLM (Spark MLlib) | _tbd_ | _tbd_ | _tbd_ |
| Chronos-Bolt (zero-shot) | _tbd_ | _tbd_ | _tbd_ |
| **Temporal Fusion Transformer** | _tbd_ | _tbd_ | _tbd_ |

## Running it

### Databricks (Free Edition)
1. Sign up at [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).
2. In the workspace, go to **Workspace → Create → Git folder** and paste this repo's GitHub URL.
3. Run the notebooks in order on serverless compute:
   - `notebooks/00_setup` creates the catalog, schemas, volume and MLflow experiment. If creating a catalog isn't allowed, set `CATALOG = "workspace"` in `src/taxi/config.py`.
   - `01_ingest` → `02_clean` → `03_aggregate_features`.
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
notebooks/     thin Databricks notebooks that call src/taxi
tests/         pytest suite on a local SparkSession
scripts/       local smoke run
```
