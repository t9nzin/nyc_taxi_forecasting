"""Interactive 3D map of hourly taxi-demand forecasts across NYC taxi zones.

Column height is the forecast for the selected hour; colour is either demand
or the forecast error (forecast vs what actually happened). Click a zone to
see its forecast against actual pickups, with the p10–p90 band.

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
REPO = "https://github.com/t9nzin/nyc_taxi_forecasting"

MODELS = {
    "chronos_bolt": ("Chronos-Bolt", "Pretrained foundation model, used zero-shot", "pickup history"),
    "nhits": ("N-HiTS", "Neural network trained on 2022–23 · @champion", "pickup history"),
    "poisson_glm": ("Poisson GLM", "Spark MLlib regression with weather, holidays and lags", "zone × hour-of-week, weather, lags"),
    "seasonal_naive": ("Seasonal-naive", "Same hour last week", "—"),
}
DEFAULT_ZONE = 161  # Midtown Center
DEFAULT_HOUR = pd.Timestamp("2024-10-17 18:00")  # a Thursday evening rush
ELEVATION_SCALE = 12  # metres per forecast pickup
COLUMN_RADIUS = 210  # metres
DEMAND_MAX = 400  # pickups/hour at the top of the colour scale

# Sequential (one hue, light -> dark) for demand; diverging with a grey midpoint for error.
RAMP = np.array([[219, 234, 254], [147, 197, 253], [59, 130, 246], [37, 99, 235], [30, 58, 138]])
UNDER, NEUTRAL, OVER = np.array([42, 120, 214]), np.array([214, 214, 210]), np.array([235, 104, 52])
SELECTED = [15, 23, 42]  # ink, so it never reads as a demand or error colour
INK, BLUE, MUTED, GRID = "#0f172a", "#2a78d6", "#94a3b8", "#eef0f3"

st.set_page_config(page_title="NYC Taxi Demand Forecasts", page_icon="🚕", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp { background: #f4f5f6; }
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container { padding: 1.6rem 2rem 2.5rem; max-width: 1440px; }
div[class*="st-key-card"] {
  background: #fff; border: 1px solid rgba(15, 23, 42, 0.06) !important; border-radius: 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.06); padding: 18px 20px;
}
.stSlider [data-baseweb="slider"] { padding-top: 4px; }
label, [data-testid="stWidgetLabel"] p { font-size: 12px !important; color: #475569 !important; font-weight: 500 !important; }

.tx-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.tx-logo { width: 40px; height: 40px; border-radius: 11px; background: #0f172a; display: flex; align-items: flex-end;
        justify-content: center; gap: 3px; padding: 9px 8px; flex: 0 0 auto; }
.tx-logo span { width: 6px; border-radius: 2px; }
.tx-brand h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin: 0; padding: 0; color: #0f172a; }
.tx-brand p { margin: 2px 0 0; font-size: 13px; color: #64748b; }
.tx-brand a { margin-left: auto; font-size: 13px; color: #475569; text-decoration: none; border: 1px solid rgba(15,23,42,.1);
           padding: 7px 12px; border-radius: 10px; background: #fff; }
.tx-brand a:hover { color: #0f172a; border-color: rgba(15,23,42,.2); }

.tx-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.tx-kpi { background: #fff; border: 1px solid rgba(15,23,42,.06); border-radius: 16px; padding: 16px 18px;
       box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06); }
.tx-kpi .tx-label { font-size: 12px; color: #64748b; }
.tx-kpi .tx-value { font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin-top: 10px; color: #0f172a;
              font-variant-numeric: tabular-nums; line-height: 1.1; }
.tx-kpi .tx-value small { font-size: 15px; color: #94a3b8; font-weight: 500; margin-left: 4px; }
.tx-kpi .tx-sub { font-size: 12px; color: #94a3b8; margin-top: 4px; }

.tx-card-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 4px; }
.tx-card-title { font-size: 14px; font-weight: 600; color: #0f172a; }
.tx-card-note { font-size: 12px; color: #64748b; }
.tx-legend { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #64748b; }
.tx-legend .tx-ramp { width: 120px; height: 8px; border-radius: 4px; display: inline-block; }

.tx-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #94a3b8; }
.tx-zone-name { font-size: 20px; font-weight: 600; letter-spacing: -.01em; color: #0f172a; margin-top: 2px; }
.tx-chip { font-size: 12px; font-weight: 500; padding: 4px 10px; border-radius: 999px; background: #f1f5f9; color: #475569; }
.tx-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background: #f8fafc; border-radius: 12px;
         padding: 12px 14px; margin: 14px 0 6px; }
.tx-stats .tx-label { font-size: 12px; color: #64748b; }
.tx-stats .tx-num { font-size: 20px; font-weight: 600; color: #0f172a; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tx-stats .tx-num.tx-blue { color: #2a78d6; }
.tx-stats .tx-num.tx-muted { font-size: 15px; color: #475569; font-weight: 500; padding-top: 4px; }
.tx-section { font-size: 12px; font-weight: 600; color: #475569; margin: 10px 0 8px; display: flex; justify-content: space-between; }
.tx-key { font-weight: 500; color: #64748b; display: flex; gap: 10px; align-items: center; }
.tx-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.tx-bars { display: flex; flex-direction: column; gap: 9px; }
.tx-bar { display: grid; grid-template-columns: 108px 1fr 52px; gap: 10px; align-items: center; font-size: 13px; color: #64748b; }
.tx-bar.tx-active { color: #0f172a; font-weight: 600; }
.tx-track { height: 8px; border-radius: 4px; background: #f1f5f9; overflow: hidden; }
.tx-fill { height: 100%; border-radius: 4px; background: #cbd5e1; }
.tx-bar.tx-active .tx-fill { background: #2a78d6; }
.tx-bar .tx-pct { text-align: right; font-variant-numeric: tabular-nums; }

.tx-board { width: 100%; border-collapse: collapse; font-size: 13px; }
.tx-board th { text-align: left; font-weight: 500; color: #64748b; font-size: 12px; padding: 8px 10px; border-bottom: 1px solid #eef0f3; }
.tx-board td { padding: 11px 10px; border-bottom: 1px solid #f4f5f6; color: #334155; }
.tx-board td.tx-num { font-variant-numeric: tabular-nums; }
.tx-board tr.tx-active td { background: #f5f9ff; color: #0f172a; font-weight: 600; }
.tx-board td.tx-num, .tx-board th.tx-num { text-align: right; }
.tx-board .tx-desc { display: block; font-size: 12px; color: #94a3b8; font-weight: 400; }
.tx-foot { font-size: 12px; color: #94a3b8; margin-top: 18px; }
.tx-foot a { color: #64748b; }

@media (max-width: 900px) {
  .tx-kpis { grid-template-columns: repeat(2, 1fr); }
  .block-container { padding: 1rem 1rem 2rem; }
}
</style>
"""


@st.cache_data(show_spinner="Loading forecasts…")
def load(version: float):
    """``version`` is the file's modified time, so a new export replaces the cached copy."""
    fc = pd.read_parquet(FORECASTS)
    fc["model"] = fc["model"].astype(str)
    fc = fc[fc["model"].isin(MODELS)]

    err = (fc["prediction"] - fc["pickups"]).abs()
    by_model = fc.assign(err=err, inside=(fc["pickups"] >= fc["p10"]) & (fc["pickups"] <= fc["p90"])).groupby("model")
    board = pd.DataFrame({
        "wape": by_model["err"].sum() / by_model["pickups"].sum(),
        "mae": by_model["err"].mean(),
        "coverage": by_model["inside"].mean().where(by_model["p10"].count() > 0),
    })
    by_zone = fc.assign(err=err).groupby(["zone_id", "model"])
    zone_wape = (by_zone["err"].sum() / by_zone["pickups"].sum().clip(lower=1)).unstack("model")
    totals = fc.groupby(["model", "hour_ts"])[["pickups", "prediction"]].sum()

    polygons, zones = [], []
    for feature in json.loads((DATA / "zones.geojson").read_text())["features"]:
        geom, props = feature["geometry"], feature["properties"]
        rings = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        polygons += [{"zone_id": props["zone_id"], "polygon": r} for r in rings]
        zones.append(props)

    return (fc.set_index(["model", "hour_ts"]).sort_index(), board, zone_wape, totals,
            pd.DataFrame(zones).set_index("zone_id"), pd.DataFrame(polygons))


def _blend(a, b, t):
    return (a + (b - a) * t[:, None]).round().astype(int)


def colors(frame: pd.DataFrame, color_by: str) -> np.ndarray:
    if color_by == "error":
        # Signed error relative to the zone's actual demand (floor 5 so quiet zones don't flare up).
        rel = ((frame["prediction"] - frame["pickups"]) / frame["pickups"].clip(lower=5)).clip(-1, 1).to_numpy()
        return np.where(rel[:, None] < 0, _blend(NEUTRAL, UNDER, -rel), _blend(NEUTRAL, OVER, rel))
    # Fixed scale, so colours mean the same pickups at every hour; square root so
    # mid-sized zones aren't all the same colour next to Midtown and the airports.
    level = np.sqrt((frame["prediction"] / DEMAND_MAX).clip(0, 1).to_numpy())
    pos = level * (len(RAMP) - 1)
    lo = np.floor(pos).astype(int).clip(max=len(RAMP) - 2)
    return _blend(RAMP[lo], RAMP[lo + 1], pos - lo)


def hour_data(fc, zones, model, ts, color_by, selected) -> pd.DataFrame:
    """One row per zone: colour, column height and tooltip text for one hour."""
    data = zones.join(fc.loc[(model, ts)].set_index("zone_id"))
    modeled = data["prediction"].notna()

    rgb = np.tile([203, 213, 225], (len(data), 1))
    rgb[modeled.to_numpy()] = colors(data[modeled], color_by)
    # Plain ints: pydeck serialises numpy ints as strings, which deck.gl can't read.
    data["rgb"] = rgb.tolist()
    data["modeled"] = modeled
    data["selected"] = data.index == selected
    data["elevation"] = data["prediction"].fillna(0)

    fmt = lambda s: s.map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")  # noqa: E731
    data["actual_txt"] = fmt(data["pickups"])
    data["forecast_txt"] = fmt(data["prediction"])
    data["band_txt"] = np.where(data["p10"].notna(), "80% range " + fmt(data["p10"]) + "–" + fmt(data["p90"]), "")
    data.loc[~modeled, "forecast_txt"] = "not modeled"
    return data.drop(columns=["pickups", "prediction", "p10", "p90"]).reset_index()


def deck(data: pd.DataFrame, polygons: pd.DataFrame) -> pdk.Deck:
    """Flat zones tinted like their column, plus a 3D column per modeled zone.
    Columns rather than extruded zones, so a big zone (JFK) doesn't look busy
    just because it's big."""
    shapes = polygons.merge(data, on="zone_id")
    shapes["fill"] = [c + [95 if m else 25] for c, m in zip(shapes["rgb"], shapes["modeled"])]
    shapes["line"] = [[15, 23, 42, 200] if s else [15, 23, 42, 35] for s in shapes["selected"]]
    shapes["width"] = [80 if s else 1 for s in shapes["selected"]]  # metres; thin lines use the pixel minimum
    cols = data[data["modeled"]].copy()
    cols["fill"] = [SELECTED + [255] if s else c + [255] for c, s in zip(cols["rgb"], cols["selected"])]
    common = {"pickable": True, "auto_highlight": True, "highlight_color": [15, 23, 42, 60]}
    layers = [
        pdk.Layer("PolygonLayer", shapes.drop(columns="rgb"), id="zones", get_polygon="polygon",
                  get_fill_color="fill", stroked=True, get_line_color="line", get_line_width="width",
                  line_width_min_pixels=0.8, **common),
        pdk.Layer("ColumnLayer", cols.drop(columns="rgb"), id="columns", get_position=["lon", "lat"],
                  get_elevation="elevation", elevation_scale=ELEVATION_SCALE, radius=COLUMN_RADIUS,
                  disk_resolution=20, get_fill_color="fill", extruded=True,
                  material={"ambient": 0.55, "diffuse": 0.6, "shininess": 24}, **common),
    ]
    view = pdk.ViewState(latitude=40.708, longitude=-73.905, zoom=10.6, pitch=50, bearing=-18)
    tooltip = {
        "html": "<div style='font-weight:600;font-size:13px'>{zone}</div>"
                "<div style='color:#94a3b8;margin-bottom:4px'>{borough}</div>"
                "Actual <b>{actual_txt}</b><br/>Forecast <b>{forecast_txt}</b><br/>"
                "<span style='color:#64748b'>{band_txt}</span>",
        "style": {"backgroundColor": "#ffffff", "color": INK, "fontFamily": "Inter, sans-serif", "fontSize": "12px",
                  "borderRadius": "10px", "padding": "10px 12px", "boxShadow": "0 8px 24px rgba(15,23,42,.12)"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT)


def zone_chart(fc, model, zone_id, ts, start, end) -> alt.LayerChart:
    lo = max(ts.normalize() - pd.Timedelta(days=3), start)
    hi = min(lo + pd.Timedelta(days=7) - pd.Timedelta(hours=1), end)
    series = fc.loc[model].loc[lo:hi].reset_index()
    series = series[series["zone_id"] == zone_id]
    base = alt.Chart(series).encode(x=alt.X("hour_ts:T", title=None, axis=alt.Axis(format="%a", tickCount=7)))
    y = alt.Y("pickups:Q", title=None, axis=alt.Axis(tickCount=3))
    layers = []
    if series["p10"].notna().any():
        layers.append(base.mark_area(opacity=0.16, color=BLUE).encode(y=alt.Y("p10:Q", title=None), y2="p90:Q"))
    layers += [
        base.mark_line(color=INK, strokeWidth=1.6).encode(y=y),
        base.mark_line(color=BLUE, strokeWidth=1.6).encode(y="prediction:Q"),
        alt.Chart(pd.DataFrame({"hour_ts": [ts]})).mark_rule(color=INK, strokeDash=[3, 3], opacity=0.4).encode(x="hour_ts:T"),
        alt.Chart(series[series["hour_ts"] == ts]).mark_circle(color=BLUE, size=55, opacity=1)
        .encode(x="hour_ts:T", y="prediction:Q"),
        base.mark_rule(opacity=0).encode(tooltip=[
            alt.Tooltip("hour_ts:T", title="Hour", format="%a %b %d, %H:00"),
            alt.Tooltip("pickups:Q", title="Actual"),
            alt.Tooltip("prediction:Q", title="Forecast", format=".0f"),
        ]),
    ]
    return (alt.layer(*layers).properties(height=190, background="#ffffff", padding={"left": 4, "right": 4, "top": 6, "bottom": 2})
            .configure_view(strokeWidth=0, fill="#ffffff")
            .configure_axis(labelColor=MUTED, labelFont="Inter", labelFontSize=11, gridColor=GRID,
                            domain=False, ticks=False, labelPadding=6))


def _rgb(c) -> str:
    return "rgb({},{},{})".format(*np.asarray(c).tolist())


def html(markup: str):
    st.html(markup)


def kpis(model, frame):
    m, naive = board.loc[model], board.loc["seasonal_naive", "wape"]
    act, fc_sum = frame["pickups"].sum(), frame["prediction"].sum()
    diff = (fc_sum - act) / act
    vs_naive = "the baseline" if model == "seasonal_naive" else f"{1 - m['wape'] / naive:.0%} below seasonal-naive"
    cov = "—" if pd.isna(m["coverage"]) else f"{m['coverage']:.1%}"
    cov_sub = "no intervals" if pd.isna(m["coverage"]) else "of actuals inside the 80% band"
    html(f"""
    <div class="tx-kpis">
      <div class="tx-kpi"><div class="tx-label">Test WAPE · Jul–Dec 2024</div><div class="tx-value">{m['wape']:.1%}</div><div class="tx-sub">{vs_naive}</div></div>
      <div class="tx-kpi"><div class="tx-label">Interval coverage</div><div class="tx-value">{cov}</div><div class="tx-sub">{cov_sub}</div></div>
      <div class="tx-kpi"><div class="tx-label">Pickups this hour</div><div class="tx-value">{act:,.0f}</div><div class="tx-sub">actual, across {len(frame)} zones</div></div>
      <div class="tx-kpi"><div class="tx-label">Forecast this hour</div><div class="tx-value">{fc_sum:,.0f}</div><div class="tx-sub">{diff:+.1%} vs actual</div></div>
    </div>""")


fc, board, zone_wape, totals, zones, polygons = load(FORECASTS.stat().st_mtime)
hours = fc.index.get_level_values("hour_ts").unique()
start, end = hours.min(), hours.max()

if "hour" not in st.session_state:
    st.session_state.hour = DEFAULT_HOUR if start <= DEFAULT_HOUR <= end else start + pd.Timedelta(hours=66)
if "zone_id" not in st.session_state:
    st.session_state.zone_id = DEFAULT_ZONE

html(CSS)
html(f"""
<div class="tx-brand">
  <div class="tx-logo"><span style="height:40%;background:#93c5fd"></span><span style="height:70%;background:#3b82f6"></span><span style="height:100%;background:#fff"></span></div>
  <div><h1>NYC Taxi Demand Forecasts</h1>
  <p>Next-hour pickups for {fc.loc['nhits'].zone_id.nunique()} taxi zones over Jul–Dec 2024, a test window no model saw in training.</p></div>
  <a href="{REPO}" target="_blank">View on GitHub ↗</a>
</div>""")

# --- Controls --------------------------------------------------------------
with st.container(key="card_controls"):
    top_l, top_r = st.columns([3, 1.25], vertical_alignment="bottom")
    with top_l:
        model = st.segmented_control("Model", list(MODELS), format_func=lambda m: MODELS[m][0],
                                     default="chronos_bolt", key="model") or "chronos_bolt"
    with top_r:
        color_by = st.segmented_control("Colour by", ["demand", "error"], key="color_by", default="demand",
                                        format_func={"demand": "Demand", "error": "Forecast error"}.get) or "demand"
    c_day, c_hour, c_play = st.columns([1, 3, 0.7], vertical_alignment="bottom")
    day = c_day.date_input("Day", st.session_state.hour.date(), min_value=start.date(), max_value=end.date(),
                           format="MM/DD/YYYY")
    hour = c_hour.slider("Hour", 0, 23, st.session_state.hour.hour, format="%d:00")
    play = c_play.button("▶  Play day", width="stretch", type="primary")

ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
if ts not in hours:  # e.g. the skipped hour when clocks go forward
    ts = hours[hours.get_indexer([ts], method="nearest")[0]]
st.session_state.hour = ts
frame = fc.loc[(model, ts)]

kpi_slot = st.empty()
with kpi_slot.container():
    kpis(model, frame)

# --- Map and zone panel ----------------------------------------------------
left, right = st.columns([2.15, 1], gap="medium")
zone_id = st.session_state.zone_id

with left, st.container(key="card_map"):
    legend = (
        f"<span>0</span><span class='tx-ramp' style='background:linear-gradient(90deg,{','.join(_rgb(c) for c in RAMP)})'></span><span>{DEMAND_MAX}+ / hr</span>"
        if color_by == "demand" else
        f"<span>Under</span><span class='tx-ramp' style='background:linear-gradient(90deg,{_rgb(UNDER)},{_rgb(NEUTRAL)},{_rgb(OVER)})'></span><span>Over</span>"
    )
    title_slot = st.empty()
    title_slot.html(f"""<div class="tx-card-head"><div><div class="tx-card-title">{ts:%A, %b %d · %H:00}</div>
        <div class="tx-card-note">{MODELS[model][1]}. Column height = forecast pickups; click a zone for detail.</div></div>
        <div class="tx-legend">{legend}</div></div>""")
    map_slot = st.empty()
    if play:
        for h in range(24):
            t = pd.Timestamp(day) + pd.Timedelta(hours=h)
            title_slot.html(f"""<div class="tx-card-head"><div><div class="tx-card-title">{t:%A, %b %d · %H:00}</div>
                <div class="tx-card-note">Playing {t:%b %d}…</div></div><div class="tx-legend">{legend}</div></div>""")
            with kpi_slot.container():
                kpis(model, fc.loc[(model, t)])
            map_slot.pydeck_chart(deck(hour_data(fc, zones, model, t, color_by, zone_id), polygons), height=600)
            time.sleep(0.45)
    event = map_slot.pydeck_chart(
        deck(hour_data(fc, zones, model, ts, color_by, zone_id), polygons),
        height=600, on_select="rerun", selection_mode="single-object", key="map",
    )

objects = (event.selection.get("objects") or {}) if event else {}
picked = objects.get("columns") or objects.get("zones")
if picked and picked[0].get("zone_id") in zone_wape.index and int(picked[0]["zone_id"]) != zone_id:
    st.session_state.zone_id = int(picked[0]["zone_id"])
    st.rerun()

with right, st.container(key="card_zone"):
    zf = frame.set_index("zone_id").loc[zone_id]
    band = "—" if pd.isna(zf["p10"]) else f"{zf['p10']:,.0f}–{zf['p90']:,.0f}"
    html(f"""
    <div class="tx-card-head" style="align-items:flex-start">
      <div><div class="tx-eyebrow">Selected zone</div><div class="tx-zone-name">{zones.loc[zone_id, 'zone']}</div></div>
      <span class="tx-chip">{zones.loc[zone_id, 'borough']}</span>
    </div>
    <div class="tx-stats">
      <div><div class="tx-label">Actual</div><div class="tx-num">{zf['pickups']:,.0f}</div></div>
      <div><div class="tx-label">Forecast</div><div class="tx-num tx-blue">{zf['prediction']:,.0f}</div></div>
      <div><div class="tx-label">80% range</div><div class="tx-num tx-muted">{band}</div></div>
    </div>
    <div class="tx-section"><span>Week around this hour</span>
      <span class="tx-key"><span><i class="tx-dot" style="background:{INK}"></i>Actual</span><span><i class="tx-dot" style="background:{BLUE}"></i>Forecast</span></span></div>""")
    st.altair_chart(zone_chart(fc, model, zone_id, ts, start, end), width="stretch", theme=None)
    zw = zone_wape.loc[zone_id]
    top = zw.max()
    rows = "".join(
        f"<div class='tx-bar {'tx-active' if m == model else ''}'><span>{MODELS[m][0]}</span>"
        f"<span class='tx-track'><span class='tx-fill' style='display:block;width:{zw[m] / top:.0%}'></span></span>"
        f"<span class='tx-pct'>{zw[m]:.1%}</span></div>"
        for m in MODELS
    )
    html(f"<div class='tx-section'><span>Accuracy in this zone (WAPE)</span></div><div class='tx-bars'>{rows}</div>")

# --- All models ------------------------------------------------------------
naive = board.loc["seasonal_naive", "wape"]
body = "".join(
    f"<tr class='{'tx-active' if m == model else ''}'><td>{MODELS[m][0]}<span class='tx-desc'>{MODELS[m][1]}</span></td>"
    f"<td>{MODELS[m][2]}</td><td class='tx-num'>{r.wape:.1%}</td><td class='tx-num'>{r.mae:.2f}</td>"
    f"<td class='tx-num'>{'—' if pd.isna(r.coverage) else f'{r.coverage:.1%}'}</td>"
    f"<td class='tx-num'>{'—' if m == 'seasonal_naive' else f'{r.wape / naive - 1:+.0%}'}</td></tr>"
    for m, r in board.loc[list(MODELS)].sort_values("wape").iterrows()
)
with st.container(key="card_board"):
    html(f"""
    <div class="tx-card-head"><div><div class="tx-card-title">All models · Jul 1 – Dec 31 2024</div>
      <div class="tx-card-note">962,688 one-hour-ahead forecasts per model, on identical zone-hours</div></div></div>
    <table class="tx-board"><thead><tr><th>Model</th><th>Inputs</th><th class="tx-num">WAPE</th><th class="tx-num">MAE</th>
      <th class="tx-num">80% coverage</th><th class="tx-num">vs naive</th></tr></thead><tbody>{body}</tbody></table>""")

html(f"""<div class="tx-foot">Data: NYC TLC trip records (yellow + green taxis) and NOAA hourly weather.
Pipeline on Databricks (PySpark, Delta Lake, MLflow). <a href="{REPO}" target="_blank">Source and write-up</a>.</div>""")
