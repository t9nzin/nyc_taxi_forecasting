import os
import sys
import time

import pytest
from pyspark.sql import SparkSession

# PySpark converts naive Python datetimes using the *process* time zone, not
# spark.sql.session.timeZone. Pin both to UTC (as on Databricks) so test
# datetimes round-trip as the wall-clock values they represent.
os.environ["TZ"] = "UTC"
time.tzset()

# Workers must use the same interpreter as the driver, and binding to
# localhost avoids hanging on hostname resolution on laptops/VPNs.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("taxi-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
