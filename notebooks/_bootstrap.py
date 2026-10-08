# Databricks notebook source
# Shared setup, pulled into other notebooks with `%run ./_bootstrap`.
# Makes `src/` importable and pins the session time zone (see taxi.ingest docstring).
import os
import sys

_src = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

spark.conf.set("spark.sql.session.timeZone", "UTC")  # noqa: F821 (Databricks global)

from taxi import config  # noqa: E402

print(f"DEV_MODE={config.DEV_MODE}  date_range={config.date_range()}  catalog={config.CATALOG}")
