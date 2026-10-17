"""Interactive 3D map of hourly taxi-demand forecasts across NYC taxi zones.

Column height is the forecast for the selected hour; colour is either the
forecast error (forecast vs what actually happened) or demand. Click a zone
to see its forecast against actual pickups, with the p10–p90 band.

    streamlit run app/app.py
"""

import json
import os
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
FORECASTS = Path(os.environ.get("FORECASTS_PATH", DATA / "forecasts.parquet"))

MODELS = {
    "chronos_bolt": "Chronos-Bolt (zero-shot)",
    "nhits": "N-HiTS (@champion)",
    "poisson_glm": "Poisson GLM",
    "seasonal_naive": "Seasonal-naive",
}
DEFAULT_ZONE = 161  # Midtown Center
DEFAULT_HOUR = pd.Timestamp("2024-10-17 18:00")  # a Thursday evening rush
ELEVATION_SCALE = 12  # metres per forecast pickup
COLUMN_RADIUS = 230  # metres

UNDER, NEUTRAL, OVER = np.array([59, 130, 246]), np.array([226, 232, 240]), np.array([239, 68, 68])
DEMAND_STOPS = np.array([[38, 23, 77], [94, 38, 128], [182, 54, 121], [251, 136, 97], [252, 253, 191]])
ACTUAL_COLOR, FORECAST_COLOR = "#f59e0b", "#3b82f6"

st.set_page_config(page_title="NYC Taxi Demand Forecasts", page_icon="🚕", layout="wide")


@st.cache_data(show_spinner="Loading forecasts…")
def load():
    fc = pd.read_parquet(FORECASTS)
    fc["model"] = fc["model"].astype(str)
    fc = fc[fc["model"].isin(MODELS)]

    err = (fc["prediction"] - fc["pickups"]).abs()
    by_model = fc.assign(err=err, inside=(fc["pickups"] >= fc["p10"]) & (fc["pickups"] <= fc["p90"])).groupby("model")
    board = pd.DataFrame({
        "WAPE": by_model["err"].sum() / by_model["pickups"].sum(),
        "MAE": by_model["err"].mean(),
        "p10–p90 coverage": by_model["inside"].mean().where(by_model["p10"].count() > 0),
    })
    by_zone = fc.assign(err=err).groupby(["zone_id", "model"])
    zone_wape = (by_zone["err"].sum() / by_zone["pickups"].sum().clip(lower=1)).unstack("model")

    polygons, zones = [], []
    for feature in json.loads((DATA / "zones.geojson").read_text())["features"]:
        geom, props = feature["geometry"], feature["properties"]
        rings = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        polygons += [{"zone_id": props["zone_id"], "polygon": r} for r in rings]
        zones.append(props)

    return (fc.set_index(["model", "hour_ts"]).sort_index(), board, zone_wape,
            pd.DataFrame(zones).set_index("zone_id"), pd.DataFrame(polygons))


def _blend(a, b, t):
    return (a + (b - a) * t[:, None]).round().astype(int)


def colors(frame: pd.DataFrame, color_by: str) -> np.ndarray:
    if color_by == "Forecast error":
        # Signed error relative to the zone's actual demand (floor 5 so quiet zones don't flare up).
        rel = ((frame["prediction"] - frame["pickups"]) / frame["pickups"].clip(lower=5)).clip(-1, 1).to_numpy()
        return np.where(rel[:, None] < 0, _blend(NEUTRAL, UNDER, -rel), _blend(NEUTRAL, OVER, rel))
    # Square root so mid-sized zones aren't all the same colour next to the airports.
    level = np.sqrt((frame["prediction"] / max(frame["prediction"].quantile(0.98), 1)).clip(0, 1).to_numpy())
    pos = level * (len(DEMAND_STOPS) - 1)
    lo = np.floor(pos).astype(int).clip(max=len(DEMAND_STOPS) - 2)
    return _blend(DEMAND_STOPS[lo], DEMAND_STOPS[lo + 1], pos - lo)


def hour_data(fc, zones, model, ts, color_by) -> pd.DataFrame:
    """One row per zone: colour, column height and tooltip text for one hour."""
    data = zones.join(fc.loc[(model, ts)].set_index("zone_id"))
    modeled = data["prediction"].notna()

    rgb = np.tile([90, 98, 112], (len(data), 1))
    rgb[modeled.to_numpy()] = colors(data[modeled], color_by)
    # Plain ints: pydeck serialises numpy ints as strings, which deck.gl can't read.
    data["rgb"] = rgb.tolist()
    data["modeled"] = modeled
    data["elevation"] = data["prediction"].fillna(0)

    fmt = lambda s: s.map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")  # noqa: E731
    data["actual_txt"] = fmt(data["pickups"])
    data["forecast_txt"] = fmt(data["prediction"])
    data["band_txt"] = np.where(data["p10"].notna(), "p10–p90: " + fmt(data["p10"]) + "–" + fmt(data["p90"]), "")
    data.loc[~modeled, "forecast_txt"] = "not modeled (< 1 pickup/day)"
    return data.drop(columns=["pickups", "prediction", "p10", "p90"]).reset_index()


def deck(data: pd.DataFrame, polygons: pd.DataFrame) -> pdk.Deck:
    """Flat zones coloured like their column, plus a 3D column per modeled zone.
    Columns rather than extruded zones, so a big zone (JFK) doesn't look busy
    just because it's big."""
    shapes = polygons.merge(data, on="zone_id")
    shapes["fill"] = [c + [140 if m else 35] for c, m in zip(shapes["rgb"], shapes["modeled"])]
    cols = data[data["modeled"]].copy()
    cols["fill"] = [c + [255] for c in cols["rgb"]]
    common = {"pickable": True, "auto_highlight": True, "highlight_color": [255, 255, 255, 110]}
    layers = [
        pdk.Layer("PolygonLayer", shapes.drop(columns="rgb"), id="zones", get_polygon="polygon",
                  get_fill_color="fill", stroked=True, get_line_color=[255, 255, 255, 50],
                  line_width_min_pixels=0.6, **common),
        pdk.Layer("ColumnLayer", cols.drop(columns="rgb"), id="columns", get_position=["lon", "lat"],
                  get_elevation="elevation", elevation_scale=ELEVATION_SCALE, radius=COLUMN_RADIUS,
                  disk_resolution=12, get_fill_color="fill", extruded=True, **common),
    ]
    view = pdk.ViewState(latitude=40.715, longitude=-73.915, zoom=10.5, pitch=50, bearing=-12)
    tooltip = {
        "html": "<b>{zone}</b> · {borough}<br/>Actual: {actual_txt}<br/>Forecast: {forecast_txt}<br/>{band_txt}",
        "style": {"fontSize": "13px"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style=pdk.map_styles.CARTO_DARK)


def zone_chart(fc, model, zone_id, ts, start, end) -> alt.LayerChart:
    lo, hi = max(ts - pd.Timedelta(days=3), start), min(ts + pd.Timedelta(days=4), end)
    series = fc.loc[model].loc[lo:hi].reset_index()
    series = series[series["zone_id"] == zone_id]
    base = alt.Chart(series).encode(x=alt.X("hour_ts:T", title=None))
    layers = []
    if series["p10"].notna().any():
        layers.append(base.mark_area(opacity=0.22, color=FORECAST_COLOR).encode(
            y=alt.Y("p10:Q", title="pickups / hour"), y2="p90:Q"))
    layers += [
        base.mark_line(color=ACTUAL_COLOR, strokeWidth=2).encode(
            y=alt.Y("pickups:Q", title="pickups / hour"), tooltip=["hour_ts:T", "pickups:Q"]),
        base.mark_line(color=FORECAST_COLOR, strokeWidth=1.6).encode(
            y="prediction:Q", tooltip=["hour_ts:T", alt.Tooltip("prediction:Q", format=".1f")]),
        alt.Chart(pd.DataFrame({"hour_ts": [ts]})).mark_rule(color="#94a3b8", strokeDash=[4, 4]).encode(x="hour_ts:T"),
    ]
    return alt.layer(*layers).properties(height=260)


fc, board, zone_wape, zones, polygons = load()
hours = fc.index.get_level_values("hour_ts").unique()
start, end = hours.min(), hours.max()

if "hour" not in st.session_state:
    st.session_state.hour = DEFAULT_HOUR if start <= DEFAULT_HOUR <= end else start + pd.Timedelta(hours=66)
if "zone_id" not in st.session_state:
    st.session_state.zone_id = DEFAULT_ZONE

# --- Controls --------------------------------------------------------------
with st.sidebar:
    st.header("🚕 NYC taxi demand")
    model = st.radio("Model", list(MODELS), format_func=MODELS.get)
    day = st.date_input("Day", st.session_state.hour.date(), min_value=start.date(), max_value=end.date())
    hour = st.slider("Hour", 0, 23, st.session_state.hour.hour, format="%d:00")
    color_by = st.radio("Colour by", ["Demand", "Forecast error"], horizontal=True)
    play = st.button("▶ Play this day", width="stretch")
    st.caption(
        "Height: forecast pickups in the hour. Forecast-error colour: blue = under-forecast, "
        "red = over-forecast. Faded zones average under one pickup a day and aren't modeled."
    )

ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
if ts not in hours:  # e.g. the skipped hour when clocks go forward
    ts = hours[hours.get_indexer([ts], method="nearest")[0]]
st.session_state.hour = ts

# --- Header ----------------------------------------------------------------
st.title("Next-hour taxi demand forecasts, NYC")
st.caption(
    f"One-hour-ahead forecasts for {fc.loc[model].zone_id.nunique()} taxi zones over the test window "
    f"({start:%b %d} – {end:%b %d %Y}), never seen in training. Built on Databricks, PySpark and MLflow."
)
frame = fc.loc[(model, ts)]
m = board.loc[model]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test WAPE (whole window)", f"{m['WAPE']:.1%}")
c2.metric("p10–p90 coverage", "—" if pd.isna(m["p10–p90 coverage"]) else f"{m['p10–p90 coverage']:.1%}")
c3.metric(f"Actual pickups, {ts:%a %b %d %H:00}", f"{frame['pickups'].sum():,}")
c4.metric("Forecast", f"{frame['prediction'].sum():,.0f}",
          f"{frame['prediction'].sum() - frame['pickups'].sum():+,.0f}", delta_color="off")

# --- Map -------------------------------------------------------------------
map_slot = st.empty()
if play:
    for h in range(24):
        t = pd.Timestamp(day) + pd.Timedelta(hours=h)
        map_slot.pydeck_chart(deck(hour_data(fc, zones, model, t, color_by), polygons), height=560)
        time.sleep(0.5)
event = map_slot.pydeck_chart(
    deck(hour_data(fc, zones, model, ts, color_by), polygons),
    height=560, on_select="rerun", selection_mode="single-object", key="map",
)
objects = (event.selection.get("objects") or {}) if event else {}
picked = objects.get("columns") or objects.get("zones")
if picked and picked[0].get("zone_id") in zone_wape.index:
    st.session_state.zone_id = int(picked[0]["zone_id"])

# --- Zone detail -----------------------------------------------------------
zone_id = st.session_state.zone_id
left, right = st.columns([3, 1])
with left:
    st.subheader(f"{zones.loc[zone_id, 'zone']} · {zones.loc[zone_id, 'borough']}")
    st.caption("Click a zone on the map to switch. Amber: actual pickups. Blue: forecast, with its p10–p90 band.")
    st.altair_chart(zone_chart(fc, model, zone_id, ts, start, end), width="stretch")
with right:
    st.subheader("This zone")
    zw = zone_wape.loc[zone_id].rename(index=MODELS).sort_values().rename("WAPE").to_frame()
    st.dataframe(zw.style.format("{:.1%}"), width="stretch")

# --- All models ------------------------------------------------------------
st.subheader("All models, whole test window")
table = board.rename(index=MODELS).sort_values("WAPE")
st.dataframe(
    pd.DataFrame({
        "WAPE": table["WAPE"].map("{:.1%}".format),
        "MAE": table["MAE"].map("{:.2f}".format),
        "p10–p90 coverage": table["p10–p90 coverage"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
    }),
    width="stretch",
)
