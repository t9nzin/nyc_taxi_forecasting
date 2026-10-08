"""Central configuration: Unity Catalog names, date ranges, splits and data sources."""

from datetime import date

# --- Unity Catalog ---------------------------------------------------------
CATALOG = "nyc_taxi"
BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"
GOLD = f"{CATALOG}.gold"
ML = f"{CATALOG}.ml"
RAW_VOLUME = f"{BRONZE}.raw"
RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/bronze/raw"
MODELS_PATH = f"{RAW_VOLUME_PATH}/models"  # pretrained weights (Chronos)

TABLES = {
    "bronze_trips": f"{BRONZE}.trips",
    "bronze_weather": f"{BRONZE}.weather_raw",
    "zones": f"{SILVER}.zones",
    "holidays": f"{SILVER}.holidays",
    "weather": f"{SILVER}.weather_hourly",
    "trips": f"{SILVER}.trips",
    "zone_hour_demand": f"{SILVER}.zone_hour_demand",
    "features": f"{GOLD}.features",
    "forecasts": f"{GOLD}.forecasts",
}

MODEL_NAME = f"{ML}.demand_forecaster"
MLFLOW_EXPERIMENT = "/Shared/nyc_taxi_forecasting"

# --- Data scope ------------------------------------------------------------
TAXI_TYPES = ["yellow", "green"]
START_DATE = date(2022, 1, 1)
END_DATE = date(2025, 1, 1)  # exclusive

# Develop on a small slice first to save Free Edition compute quota.
DEV_MODE = True
DEV_START_DATE = date(2024, 1, 1)
DEV_END_DATE = date(2024, 4, 1)  # exclusive


def date_range():
    """(start, end_exclusive) for the active mode."""
    return (DEV_START_DATE, DEV_END_DATE) if DEV_MODE else (START_DATE, END_DATE)


def months(start: date, end: date):
    """Yield (year, month) for every month in [start, end)."""
    y, m = start.year, start.month
    while date(y, m, 1) < end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


# --- Out-of-time splits (inclusive start, exclusive end) -------------------
SPLITS = {
    "train": (date(2022, 1, 1), date(2024, 1, 1)),
    "validation": (date(2024, 1, 1), date(2024, 7, 1)),
    "test": (date(2024, 7, 1), date(2025, 1, 1)),
}
DEV_SPLITS = {
    "train": (date(2024, 1, 1), date(2024, 3, 1)),
    "validation": (date(2024, 3, 1), date(2024, 3, 16)),
    "test": (date(2024, 3, 16), date(2024, 4, 1)),
}


def splits():
    return DEV_SPLITS if DEV_MODE else SPLITS


# --- Sources ---------------------------------------------------------------
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_URL = f"{TLC_BASE_URL}/misc/taxi_zone_lookup.csv"
ZONE_SHAPEFILE_URL = f"{TLC_BASE_URL}/misc/taxi_zones.zip"


def trip_url(taxi_type: str, year: int, month: int) -> str:
    return f"{TLC_BASE_URL}/trip-data/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


# NOAA NCEI Local Climatological Data, Central Park. The access service wants
# the USAF+WBAN id (725053-94728); the GHCN id USW00094728 returns no rows.
NOAA_STATION = "72505394728"
NOAA_LCD_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
# LCD timestamps are Local Standard Time (UTC-5) all year; trips use wall clock.
LOCAL_TZ = "America/New_York"

# --- Cleaning thresholds ---------------------------------------------------
MAX_TRIP_MILES = 100.0
MAX_FARE = 1000.0
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 240.0
UNKNOWN_ZONES = (264, 265)  # "Unknown" / "Outside of NYC" in the TLC lookup

# Zones averaging fewer pickups per day than this are excluded from modeling
# (they stay on the map).
MIN_AVG_DAILY_PICKUPS = 1.0
