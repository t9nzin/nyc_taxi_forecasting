"""Download raw sources into the UC volume and normalize TLC trips to one schema.

Timestamp convention for the whole project: every timestamp is NYC wall-clock
time stored as a Spark TIMESTAMP with ``spark.sql.session.timeZone = UTC``, so
Spark never shifts the values. TLC files already hold wall-clock times.
"""

import os
import shutil
import time
import urllib.request
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi import config

# Canonical trip schema: output column -> (type, candidate source columns).
# Source names are matched case-insensitively because TLC renames columns
# between years (e.g. airport_fee -> Airport_fee in 2024).
CANONICAL_TRIP_COLUMNS = {
    "vendor_id": ("int", ["VendorID"]),
    "pickup_ts": ("timestamp", ["tpep_pickup_datetime", "lpep_pickup_datetime"]),
    "dropoff_ts": ("timestamp", ["tpep_dropoff_datetime", "lpep_dropoff_datetime"]),
    "passenger_count": ("int", ["passenger_count"]),
    "trip_distance": ("double", ["trip_distance"]),
    "ratecode_id": ("int", ["RatecodeID"]),
    "pu_zone": ("int", ["PULocationID"]),
    "do_zone": ("int", ["DOLocationID"]),
    "payment_type": ("int", ["payment_type"]),
    "fare_amount": ("double", ["fare_amount"]),
    "tip_amount": ("double", ["tip_amount"]),
    "tolls_amount": ("double", ["tolls_amount"]),
    "congestion_surcharge": ("double", ["congestion_surcharge"]),
    "airport_fee": ("double", ["airport_fee"]),
    "total_amount": ("double", ["total_amount"]),
}


def download(url: str, dest: str, overwrite: bool = False, validate=None, attempts: int = 3) -> str:
    """Download ``url`` to ``dest`` (a local or /Volumes path). Skips existing files.

    ``validate(tmp_path) -> bool`` is checked before the file is moved into
    place, so a bad download is retried instead of cached.
    """
    if os.path.exists(dest) and not overwrite:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if validate is None or validate(tmp):
            os.replace(tmp, dest)
            return dest
        time.sleep(5 * attempt)
    os.remove(tmp)
    raise RuntimeError(f"Download failed validation after {attempts} attempts: {url}")


def trip_file_path(root: str, taxi_type: str, year: int, month: int) -> str:
    return f"{root}/trips/{taxi_type}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def download_trips(root: str, start: date, end: date, taxi_types=config.TAXI_TYPES):
    """Download every monthly TLC file in [start, end). Returns local paths."""
    paths = []
    for taxi_type in taxi_types:
        for year, month in config.months(start, end):
            dest = trip_file_path(root, taxi_type, year, month)
            download(config.trip_url(taxi_type, year, month), dest)
            paths.append(dest)
    return paths


def normalize_trips(df: DataFrame, taxi_type: str, year: int, month: int) -> DataFrame:
    """Map one raw TLC monthly file onto the canonical schema."""
    by_lower = {c.lower(): c for c in df.columns}
    cols = []
    for out_name, (dtype, candidates) in CANONICAL_TRIP_COLUMNS.items():
        src = next((by_lower[c.lower()] for c in candidates if c.lower() in by_lower), None)
        col = F.col(f"`{src}`") if src else F.lit(None)
        cols.append(col.cast(dtype).alias(out_name))
    return df.select(
        F.lit(taxi_type).alias("taxi_type"),
        *cols,
        F.lit(year).alias("source_year"),
        F.lit(month).alias("source_month"),
    )


def build_bronze_trips(spark: SparkSession, root: str, start: date, end: date, table: str):
    """Read each monthly file on its own (types drift between files), normalize, append."""
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    for taxi_type in config.TAXI_TYPES:
        for year, month in config.months(start, end):
            raw = spark.read.parquet(trip_file_path(root, taxi_type, year, month))
            (
                normalize_trips(raw, taxi_type, year, month)
                .write.mode("append")
                .saveAsTable(table)
            )


# --- Weather ---------------------------------------------------------------

def noaa_url(start: date, end_inclusive: date) -> str:
    # dataTypes filtering returns empty bodies on this endpoint, so pull all
    # columns and select later.
    return (
        f"{config.NOAA_LCD_URL}?dataset=local-climatological-data"
        f"&stations={config.NOAA_STATION}"
        f"&startDate={start.isoformat()}&endDate={end_inclusive.isoformat()}"
        "&format=csv&units=standard"
    )


def _reaches_last_day(last_day: date):
    """Validator: the CSV's final row is on ``last_day``. Long NOAA responses
    get cut off mid-stream, which otherwise looks like a successful download."""

    def check(path: str) -> bool:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            last_line = f.read().decode("utf-8", "replace").strip().splitlines()[-1]
        return f'"{last_day.isoformat()}T' in last_line

    return check


def download_weather(root: str, start: date, end: date):
    """One CSV per month: full-year requests are truncated by the API.

    Also fetches the month before ``start``, because the first hour's weather
    comes from the 23:51 report the day before.
    """
    paths = []
    first = date(start.year, start.month, 1) - timedelta(days=1)
    for year, month in config.months(date(first.year, first.month, 1), end):
        m_start = date(year, month, 1)
        m_last = (m_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        dest = f"{root}/weather/lcd_{config.NOAA_STATION}_{year}-{month:02d}.csv"
        download(noaa_url(m_start, m_last), dest, validate=_reaches_last_day(m_last))
        paths.append(dest)
    return paths


def download_zones(root: str):
    lookup = download(config.ZONE_LOOKUP_URL, f"{root}/zones/taxi_zone_lookup.csv")
    shapes = download(config.ZONE_SHAPEFILE_URL, f"{root}/zones/taxi_zones.zip")
    return lookup, shapes


def load_zones(spark: SparkSession, lookup_csv: str) -> DataFrame:
    """TLC zone lookup without the 'Unknown' / 'Outside of NYC' pseudo-zones."""
    return (
        spark.read.option("header", True)
        .csv(lookup_csv)
        .select(
            F.col("LocationID").cast("int").alias("zone_id"),
            F.col("Borough").alias("borough"),
            F.col("Zone").alias("zone_name"),
            F.col("service_zone"),
        )
        .filter(~F.col("zone_id").isin(*config.UNKNOWN_ZONES))
    )
