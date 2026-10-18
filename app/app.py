"""Dashboard of next-hour taxi-demand forecasts across NYC taxi zones.

A map of the busiest zones for the selected hour, the model's accuracy, demand
by borough, the selected zone's week, and a table of the busiest zones.

    streamlit run app/app.py
"""

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
FORECASTS = Path(os.environ.get("FORECASTS_PATH", DATA / "forecasts.parquet"))
REPO = "https://github.com/t9nzin/nyc_taxi_forecasting"

MODELS = {
    "chronos_bolt": "Chronos-Bolt",
    "nhits": "N-HiTS",
    "poisson_glm": "Poisson GLM",
    "seasonal_naive": "Seasonal-naive",
}
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
DEFAULT_ZONE = 161  # Midtown Center
DEFAULT_HOUR = pd.Timestamp("2024-10-17 18:00")  # a Thursday evening rush
N_PINS = 14  # busiest zones get a pin; the rest are dots

CREAM = "#e9e2cf"
UNDER, NEUTRAL, OVER = np.array([74, 144, 226]), np.array([120, 120, 115]), np.array([235, 104, 52])

st.set_page_config(page_title="NYC Taxi Demand", page_icon="🚕", layout="wide")


# --- Icons -----------------------------------------------------------------

def img(svg: str, style: str = "") -> str:
    """st.html drops inline <svg>, so SVGs go in as image data URIs."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<img src="{uri}" style="display:block;{style}" alt="">'


def _svg(body: str, size: int = 18, color: str = "#d7d7d2") -> str:
    return img(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
               f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICONS = {
    "arrow": _svg('<path d="M7 17 17 7M8 7h9v9"/>', 20, "#8d8d88"),
    "github": img('<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 16 16" fill="#ffffff"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'),
    "taxi": _svg('<path d="M5 16V12l1.8-4.2A1.5 1.5 0 0 1 8.2 7h7.6a1.5 1.5 0 0 1 1.4.8L19 12v4"/><path d="M3.5 16h17v2.5h-17z"/><path d="M5 12h14"/><circle cx="7.5" cy="18.5" r="1.4"/><circle cx="16.5" cy="18.5" r="1.4"/><path d="M10.5 7V5.5h3V7"/>', 22),
    "target": _svg('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r=".6" fill="#d7d7d2"/>'),
    "ruler": _svg('<path d="M4 15 15 4l5 5L9 20z"/><path d="m8 11 1.5 1.5M11 8l1.5 1.5M14 5l1.5 1.5"/>'),
    "trend": _svg('<path d="M4 17 9.5 11.5l3.5 3.5L20 8"/><path d="M15 8h5v5"/>'),
    "clock": _svg('<circle cx="12" cy="12" r="8"/><path d="M12 8v4l2.5 2"/>'),
    "Manhattan": _svg('<path d="M6 20V8l4-2v14M10 20V4h5v16M15 20V10h3v10M4 20h16"/>', 22),
    "Brooklyn": _svg('<path d="M3 17h18M6 17V9M18 17V9M6 9q6 6 12 0M3 9h3M18 9h3M9 17v-4M12 17v-3M15 17v-4"/>', 22),
    "Queens": _svg('<path d="M3 14.5 21 9.5l-1-2-6.5 1.8L8 4 6 4.6l3 6.3-4 1.1-2-1.6-1.3.4 1.6 3.9z"/><path d="M4 20h16"/>', 22),
    "Bronx": _svg('<path d="M12 4 6 12h3l-4 5h14l-4-5h3z"/><path d="M12 17v3"/>', 22),
    "Staten Island": _svg('<path d="M4 15h16l-2 4H6z"/><path d="M8 15V9h7l2 3v3M11 9V6"/><path d="M3 21q2-1.2 4.5 0t4.5 0 4.5 0 4.5 0"/>', 22),
}


def _pin_url() -> str:
    """Black map pin with a white taxi, as an SVG data URL for deck.gl's IconLayer."""
    svg = """<svg xmlns='http://www.w3.org/2000/svg' width='72' height='84' viewBox='0 0 72 84'>
      <circle cx='36' cy='32' r='28' fill='#141414' stroke='#ffffff' stroke-width='3'/>
      <g fill='none' stroke='#ffffff' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'>
        <path d='M22 39v-7l3.2-7.4c.4-.9 1.3-1.6 2.3-1.6h17c1 0 1.9.7 2.3 1.6L50 32v7'/>
        <path d='M19 39h34v5H19z'/><path d='M22 32h28'/><path d='M32 23v-3h8v3'/>
      </g>
      <circle cx='36' cy='77' r='4.5' fill='#141414' stroke='#ffffff' stroke-width='2'/>
    </svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


PIN = {"url": _pin_url(), "width": 72, "height": 84, "anchorY": 84}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp { background: #0b0b0b; color: #f5f5f4; }
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
/* Space Grotesk for headings, nav and big numbers; Inter for small text */
.tx-brand, .tx-title, .tx-center b, .tx-id, .tx-profile b, .tx-pill, .tx-row > span:nth-child(2), .tx-two b,
.st-key-nav_model button p, .st-key-filter_pick button p {
  font-family: "Space Grotesk", Inter, system-ui, sans-serif !important; letter-spacing: -0.01em;
}
.block-container { padding: 1rem 1.25rem 2rem; max-width: 1520px; }

div[class*="st-key-card"] {
  background: #161616; border: 1px solid rgba(255,255,255,0.06); border-radius: 22px; padding: 22px 24px;
}
.st-key-card_map { padding: 0 !important; overflow: hidden; position: relative; background: #e9e5dc; gap: 0; }
.st-key-map_overlay { position: absolute !important; top: 18px; left: 18px; z-index: 5; width: auto !important; gap: 10px; }

/* white pills over the map */
.st-key-map_overlay [data-testid="stSelectbox"] div, .st-key-map_overlay [data-testid="stDateInput"] div,
.st-key-map_overlay input, .st-key-map_overlay button { background: #fff !important; border-color: transparent !important; }
.st-key-map_overlay [role="group"], .st-key-map_overlay [data-baseweb="input"] {
  border-radius: 999px !important; min-height: 44px; box-shadow: 0 2px 10px rgba(0,0,0,0.14); padding-left: 10px; overflow: hidden;
}
.st-key-map_overlay * { color: #141414 !important; -webkit-text-fill-color: #141414 !important; }
.st-key-map_overlay input { font-size: 14px !important; font-weight: 500; }
.st-key-map_overlay svg { fill: #141414 !important; color: #141414 !important; }
.st-key-day_pick [data-testid="stDateInput"] > div:last-child { border-radius: 999px !important; overflow: hidden;
  box-shadow: 0 2px 10px rgba(0,0,0,0.14); min-height: 44px; padding: 0 8px; display: flex; align-items: center; }
.st-key-card_map canvas.mapboxgl-canvas, .st-key-card_map canvas.maplibregl-canvas { filter: saturate(0.5) sepia(0.16) brightness(1.03); }
.st-key-zone_pick { flex: 0 0 300px !important; width: 300px !important; } .st-key-day_pick { flex: 0 0 150px !important; width: 150px !important; }
.st-key-hour_pick { flex: 0 0 112px !important; width: 112px !important; } .st-key-color_pick { flex: 0 0 165px !important; width: 165px !important; }
.st-key-map_overlay input { background: transparent !important; }

/* nav */
.tx-brand { display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 600; letter-spacing: -0.03em; color: #fff; white-space: nowrap; }
.st-key-nav_model button { background: transparent !important; border: 0 !important; color: #8d8d88 !important;
  padding: 6px 14px !important; box-shadow: none !important; }
.st-key-nav_model button p { font-size: 15px !important; color: #8d8d88 !important; }
.st-key-nav_model button:hover p { color: #fff !important; }
.st-key-nav_model button[aria-checked="true"] p { color: #fff !important; font-weight: 600; }
.st-key-play_btn button { width: 46px; height: 46px; min-height: 46px; border-radius: 50% !important; background: #161616 !important;
  border: 1px solid rgba(255,255,255,0.1) !important; color: #fff !important; padding: 0 !important; }
.st-key-play_btn button:hover { background: #222 !important; }
.tx-circle { width: 46px; height: 46px; border-radius: 50%; background: #161616; border: 1px solid rgba(255,255,255,0.1);
  color: #fff !important; display: grid; place-items: center; }
.tx-circle:hover { background: #222; }
.tx-profile { display: flex; align-items: center; gap: 10px; height: 50px; padding: 4px 18px 4px 5px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.1); background: #161616; width: fit-content; margin-left: auto; }
.tx-avatar { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #e9e2cf, #a59f8f); color: #141414;
  display: grid; place-items: center; font-weight: 700; font-size: 12px; }
.tx-profile b { display: block; font-size: 14px; font-weight: 600; color: #fff; line-height: 1.25; }
.tx-profile span { font-size: 12px; color: #8d8d88; }

/* cards */
.tx-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.tx-title { font-size: 21px; font-weight: 500; letter-spacing: -0.02em; color: #f5f5f4; }
.tx-head svg { color: #8d8d88; }
.tx-muted { color: #8d8d88; }

.tx-gauge-card { background: radial-gradient(120% 90% at 50% 0%, #2c2c29 0%, #161616 65%); border-radius: 18px; margin: -4px -10px 0; padding: 4px 10px 0; }
.tx-pills { position: relative; height: 138px; }
.tx-pill { position: absolute; display: inline-flex; align-items: center; gap: 8px; padding: 9px 15px; border-radius: 999px;
  background: rgba(78,78,74,0.55); border: 1px solid rgba(255,255,255,0.08); font-size: 14px; color: #eee; white-space: nowrap; }
.tx-pill b { font-weight: 500; color: #a9a9a3; }
.tx-pill svg { color: #cfcfca; }
.tx-gauge-labels { display: flex; justify-content: space-between; font-size: 13px; color: #8d8d88; margin-top: -10px; }
.tx-caption { text-align: center; color: #b9b9b3; font-size: 14px; margin-top: 6px; }
.tx-caption b { color: #f5f5f4; font-weight: 500; }

.tx-rows { display: flex; flex-direction: column; gap: 12px; }
.tx-row { display: grid; grid-template-columns: 24px 104px 52px 1fr 40px; align-items: center; gap: 10px; font-size: 15px; color: #f5f5f4; }
.tx-row svg { color: #d7d7d2; }
.tx-count { text-align: right; font-variant-numeric: tabular-nums; }
.tx-share { text-align: right; font-size: 14px; color: #d7d7d2; font-variant-numeric: tabular-nums; }
.tx-ticks { display: flex; gap: 2px; height: 16px; }
.tx-ticks i { flex: 1; border-radius: 1px; background: #3a3a37; }
.tx-ticks i.on { background: #f5f5f4; }

.tx-fuel { display: grid; grid-template-columns: 1fr 1.3fr 1fr; align-items: center; row-gap: 12px; }
.tx-stat b { display: block; font-size: 15px; font-weight: 500; color: #f5f5f4; font-variant-numeric: tabular-nums; }
.tx-stat span { font-size: 13px; color: #8d8d88; }
.tx-stat.right { text-align: right; }
.tx-center { grid-row: span 2; text-align: center; }
.tx-center span { font-size: 14px; color: #a9a9a3; }
.tx-center b { display: block; font-size: 40px; font-weight: 500; letter-spacing: -0.02em; color: #fff; margin-top: 4px; font-variant-numeric: tabular-nums; }
.tx-spark-labels { display: flex; justify-content: space-between; font-size: 12px; color: #8d8d88; margin-top: 6px; }

/* table */
.st-key-filter_pick [data-testid="stButtonGroup"] { border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; padding: 4px; gap: 2px; width: fit-content; margin-left: auto; }
.st-key-filter_pick button { background: transparent !important; border: 0 !important; border-radius: 999px !important;
  padding: 7px 18px !important; box-shadow: none !important; min-height: 38px; }
.st-key-filter_pick button p { font-size: 14px !important; color: #f5f5f4 !important; }
.st-key-filter_pick button[aria-checked="true"] { background: #e9e2cf !important; }
.st-key-filter_pick button[aria-checked="true"] p { color: #141414 !important; }
.tx-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.tx-table th { text-align: left; font-weight: 400; color: #8d8d88; padding: 12px 12px; background: #1c1c1c; }
.tx-table th:first-child { border-radius: 10px 0 0 10px; } .tx-table th:last-child { border-radius: 0 10px 10px 0; text-align: right; }
.tx-table td { padding: 16px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #f5f5f4; vertical-align: middle; }
.tx-table tr:last-child td { border-bottom: 0; }
.tx-table tr.tx-selected td { background: rgba(233,226,207,0.06); }
.tx-id { display: flex; align-items: center; gap: 14px; font-size: 16px; font-weight: 500; }
.tx-id svg { color: #d7d7d2; }
.tx-two b { display: block; font-weight: 400; color: #f5f5f4; }
.tx-two span { color: #8d8d88; }
.tx-route { display: flex; flex-direction: column; gap: 6px; position: relative; padding-left: 16px; }
.tx-route::before { content: ""; position: absolute; left: 3.5px; top: 10px; bottom: 10px; border-left: 1px dashed #55554f; }
.tx-route div { position: relative; font-variant-numeric: tabular-nums; }
.tx-route div::before { content: ""; position: absolute; left: -16px; top: 6px; width: 7px; height: 7px; border-radius: 50%; border: 1.3px solid #d7d7d2; background: #161616; }
.tx-route div.tx-fc::before { background: #d7d7d2; }
.tx-route span { color: #8d8d88; }
.tx-num { font-variant-numeric: tabular-nums; }
.tx-status { text-align: right; color: #d7d7d2; }
.tx-status.tx-over { color: #f0a27f; } .tx-status.tx-under { color: #8fb8ea; }
.tx-foot { color: #6f6f6a; font-size: 12px; margin-top: 4px; }
.tx-foot a { color: #a9a9a3; }
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


def error_colors(frame: pd.DataFrame) -> np.ndarray:
    # Signed error relative to the zone's actual demand (floor 5 so quiet zones don't flare up).
    rel = ((frame["prediction"] - frame["pickups"]) / frame["pickups"].clip(lower=5)).clip(-1, 1).to_numpy()
    return np.where(rel[:, None] < 0, _blend(NEUTRAL, UNDER, -rel), _blend(NEUTRAL, OVER, rel))


def fmt(v, digits=0) -> str:
    return "—" if pd.isna(v) else f"{v:,.{digits}f}"


# --- Map -------------------------------------------------------------------

def deck(frame: pd.DataFrame, selected: int, color_by: str) -> pdk.Deck:
    """Flat street map: black pins on the busiest zones, dots elsewhere, and a
    label on the selected zone. pydeck reads every string as a data expression,
    so string constants are quoted ("'start'")."""
    data = zones.join(frame.set_index("zone_id"), how="inner").reset_index()
    data["actual_txt"] = data["pickups"].map(fmt)
    data["forecast_txt"] = data["prediction"].map(fmt)
    data["band_txt"] = np.where(data["p10"].notna(),
                                "80% range " + data["p10"].map(fmt) + "–" + data["p90"].map(fmt), "")
    rgb = error_colors(data).tolist() if color_by == "error" else [[20, 20, 20]] * len(data)
    data["dot"] = [c + [235] for c in rgb]
    data["halo"] = [c + [120] for c in rgb]

    busiest = data.nlargest(N_PINS, "prediction")["zone_id"]
    pins = data[data["zone_id"].isin(busiest) | (data["zone_id"] == selected)].copy()
    pins["icon"] = [PIN] * len(pins)
    pins["size"] = np.where(pins["zone_id"] == selected, 50, 40)
    dots = data[~data["zone_id"].isin(pins["zone_id"])]

    sel = data[data["zone_id"] == selected].copy()
    sel["label"] = sel["zone"] + "\n" + sel["forecast_txt"] + " forecast, " + sel["actual_txt"] + " actual"
    outline = polygons[polygons["zone_id"] == selected]

    common = {"pickable": True, "auto_highlight": True, "highlight_color": [255, 255, 255, 80]}
    layers = [
        pdk.Layer("PolygonLayer", outline, id="outline", get_polygon="polygon", get_fill_color=[20, 20, 20, 22],
                  stroked=True, get_line_color=[20, 20, 20, 200], line_width_min_pixels=1.5),
        pdk.Layer("ScatterplotLayer", dots, id="dots", get_position=["lon", "lat"], get_fill_color="dot",
                  get_radius=110, radius_min_pixels=3.5 if color_by == "demand" else 5, radius_max_pixels=7,
                  stroked=True, get_line_color=[255, 255, 255, 220], line_width_min_pixels=1, **common),
    ]
    if color_by == "error":
        layers.append(pdk.Layer("ScatterplotLayer", pins, id="halos", get_position=["lon", "lat"],
                                get_fill_color="halo", get_radius=1, radius_min_pixels=25, radius_max_pixels=25,
                                get_pixel_offset=[0, -29]))
    layers += [
        pdk.Layer("IconLayer", pins, id="pins", get_position=["lon", "lat"], get_icon="icon", get_size="size", **common),
        pdk.Layer("TextLayer", sel, id="label", get_position=["lon", "lat"], get_text="label", get_size=13,
                  get_color=[20, 20, 20, 255], get_pixel_offset=[30, -34], get_text_anchor="'start'",
                  get_alignment_baseline="'center'", font_family="'Inter, Helvetica, Arial, sans-serif'",
                  font_weight=600, line_height=1.35, background=True, get_background_color=[255, 255, 255, 250],
                  background_padding=[12, 8, 14, 8]),
    ]
    view = pdk.ViewState(latitude=40.752, longitude=-73.955, zoom=11.35, pitch=0, bearing=0)
    tooltip = {
        "html": "<div style='font-weight:600;font-size:13px'>{zone}</div><div style='color:#8d8d88;margin-bottom:4px'>{borough}</div>"
                "Actual <b>{actual_txt}</b> · Forecast <b>{forecast_txt}</b><br/><span style='color:#8d8d88'>{band_txt}</span>",
        "style": {"backgroundColor": "#ffffff", "color": "#141414", "fontFamily": "Inter, sans-serif", "fontSize": "12px",
                  "borderRadius": "12px", "padding": "10px 12px", "boxShadow": "0 8px 24px rgba(0,0,0,.18)"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style=pdk.map_styles.CARTO_ROAD)


# --- Cards -----------------------------------------------------------------

def card_head(title: str) -> str:
    return f"<div class='tx-head'><span class='tx-title'>{title}</span>{ICONS['arrow']}</div>"


def gauge_card(model: str, frame: pd.DataFrame) -> str:
    m, naive = board.loc[model], board.loc["seasonal_naive", "wape"]
    score = max(0.0, 1 - m["wape"])
    act, fcs = frame["pickups"].sum(), frame["prediction"].sum()
    pills = [
        ("target", "Coverage", "—" if pd.isna(m["coverage"]) else f"{m['coverage']:.1%}", "left:4%;top:14px"),
        ("ruler", "MAE", f"{m['mae']:.2f}", "right:3%;top:4px"),
        ("trend", "vs naive", "baseline" if model == "seasonal_naive" else f"{m['wape'] / naive - 1:+.0%}", "left:2%;top:84px"),
        ("clock", "This hour", f"{(fcs - act) / act:+.1%}", "right:2%;top:70px"),
    ]
    pill_html = "".join(f"<span class='tx-pill' style='{pos}'>{ICONS[i]}{label} <b>{v}</b></span>" for i, label, v, pos in pills)

    # A flat arc (quadratic curve) with the knob at the score.
    p0, p1, p2 = np.array([0, 46]), np.array([200, 8]), np.array([400, 46])
    t = score
    kx, ky = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    gauge = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -34 400 88" width="400" height="88" font-family="Helvetica, Arial, sans-serif">
      <defs><linearGradient id="arc" x1="0" x2="1"><stop offset="0" stop-color="#b8665a"/><stop offset=".55" stop-color="#e9e2cf"/><stop offset="1" stop-color="#9ccc8b"/></linearGradient></defs>
      <path d="M0,46 Q200,8 400,46" stroke="url(#arc)" stroke-width="3" fill="none"/>
      <rect x="{kx - 27:.1f}" y="{ky - 42:.1f}" width="54" height="26" rx="13" fill="#fff"/>
      <text x="{kx:.1f}" y="{ky - 24.5:.1f}" text-anchor="middle" font-size="13" font-weight="600" fill="#141414">{score:.0%}</text>
      <line x1="{kx:.1f}" x2="{kx:.1f}" y1="{ky - 16:.1f}" y2="{ky:.1f}" stroke="#fff" stroke-width="1.5"/>
      <circle cx="{kx:.1f}" cy="{ky:.1f}" r="7" fill="#fff" stroke="#161616" stroke-width="3"/>
    </svg>"""
    gauge = img(gauge, "width:100%;height:auto")
    return f"""{card_head("Forecast accuracy")}
    <div class="tx-gauge-card"><div class="tx-pills">{pill_html}</div>{gauge}
    <div class="tx-gauge-labels"><span>0%</span><span>100%</span></div></div>
    <div class="tx-caption"><b>{m['wape']:.1%}</b> WAPE over Jul–Dec 2024 ({score:.1%} accurate)</div>"""


def borough_card(frame: pd.DataFrame) -> str:
    counts = frame.join(zones["borough"], on="zone_id").groupby("borough")["pickups"].sum().reindex(BOROUGHS).fillna(0)
    total = max(counts.sum(), 1)
    rows = ""
    for b, n in counts.items():
        share = n / total
        on = round(share * 40)
        ticks = "".join(f"<i class='{'on' if k < on else ''}'></i>" for k in range(40))
        rows += (f"<div class='tx-row'>{ICONS[b]}<span>{b}</span><span class='tx-count'>{n:,.0f}</span>"
                 f"<span class='tx-ticks'>{ticks}</span><span class='tx-share'>{share:.0%}</span></div>")
    return f"{card_head('Pickups by borough')}<div class='tx-rows'>{rows}</div>"


def zone_card(model: str, zone_id: int, ts: pd.Timestamp) -> str:
    zf = fc.loc[(model, ts)].set_index("zone_id").loc[zone_id]
    band = "—" if pd.isna(zf["p10"]) else f"{zf['p10']:,.0f}–{zf['p90']:,.0f}"
    diff = (zf["prediction"] - zf["pickups"]) / max(zf["pickups"], 1)

    lo = max(ts.normalize() - pd.Timedelta(days=3), start)
    hi = min(lo + pd.Timedelta(days=7) - pd.Timedelta(hours=1), end)
    week = fc.loc[model].loc[lo:hi]
    week = week[week["zone_id"] == zone_id]
    W, H = 400, 96
    top = max(week["pickups"].max(), week["prediction"].max(), 1) * 1.05
    step = W / len(week)
    bars = "".join(
        f"<rect x='{i * step + step * 0.18:.2f}' y='{H - v / top * H:.2f}' width='{step * 0.64:.2f}' "
        f"height='{v / top * H:.2f}' fill='#f5f5f4' opacity='{1 if h == ts else 0.82}'/>"
        for i, (h, v) in enumerate(zip(week.index, week["pickups"]))
    )
    line = " ".join(f"{'M' if i == 0 else 'L'}{(i + 0.5) * step:.2f},{H - v / top * H:.2f}"
                    for i, v in enumerate(week["prediction"]))
    mx = (list(week.index).index(ts) + 0.5) * step if ts in week.index else 0
    spark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="none" width="{W}" height="{H}">
      {bars}<path d="{line}" fill="none" stroke="{CREAM}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
      <line x1="{mx:.2f}" x2="{mx:.2f}" y1="0" y2="{H}" stroke="{CREAM}" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>
    </svg>"""
    spark = img(spark, "width:100%;height:104px;margin-top:14px")
    return f"""{card_head(zones.loc[zone_id, 'zone'])}
    <div class="tx-fuel">
      <div class="tx-stat"><b>{zf['pickups']:,.0f}</b><span>Actual</span></div>
      <div class="tx-center"><span>Forecast, {ts:%H:00}</span><b>{zf['prediction']:,.0f}</b></div>
      <div class="tx-stat right"><b>{zone_wape.loc[zone_id, model]:.1%}</b><span>Zone WAPE</span></div>
      <div class="tx-stat"><b>{band}</b><span>80% range</span></div>
      <div class="tx-stat right"><b>{diff:+.1%}</b><span>vs actual</span></div>
    </div>{spark}
    <div class="tx-spark-labels"><span>{lo:%b %d}</span><span>bars: actual · <span style="color:{CREAM}">line: forecast</span></span><span>{hi:%b %d}</span></div>"""


def table_html(frame: pd.DataFrame, borough: str, selected: int) -> str:
    t = frame.join(zones[["zone", "borough"]], on="zone_id")
    if borough != "All":
        t = t[t["borough"] == borough]
    rows = ""
    for r in t.nlargest(5, "pickups").itertuples():
        err = (r.prediction - r.pickups) / max(r.pickups, 1)
        if pd.notna(r.p10):
            status, cls = (("Above range", "tx-under") if r.pickups > r.p90 else
                           ("Below range", "tx-over") if r.pickups < r.p10 else ("In range", ""))
        else:
            status, cls = (("Over-forecast", "tx-over") if err > 0.15 else
                           ("Under-forecast", "tx-under") if err < -0.15 else ("On target", ""))
        band = "—" if pd.isna(r.p10) else f"{r.p10:,.0f}–{r.p90:,.0f}"
        rows += f"""<tr class="{'tx-selected' if r.zone_id == selected else ''}">
          <td><div class="tx-id">{ICONS['taxi']}#{r.zone_id:03d}</div></td>
          <td class="tx-two"><b>{r.zone}</b><span>{r.borough}</span></td>
          <td><div class="tx-route"><div>{r.pickups:,.0f} <span>actual</span></div><div class="tx-fc">{r.prediction:,.0f} <span>forecast</span></div></div></td>
          <td class="tx-num">{band}</td><td class="tx-num">{err:+.1%}</td><td class="tx-status {cls}">{status}</td></tr>"""
    return f"""<table class="tx-table"><thead><tr><th>Zone ID</th><th>Zone</th><th>Pickups</th><th>80% range</th>
      <th>Error</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"""


# --- Page ------------------------------------------------------------------

fc, board, zone_wape, zones, polygons = load(FORECASTS.stat().st_mtime)
hours = fc.index.get_level_values("hour_ts").unique()
start, end = hours.min(), hours.max()
modeled = zones.loc[zone_wape.index].sort_values("zone")

if "hour" not in st.session_state:
    st.session_state.hour = DEFAULT_HOUR if start <= DEFAULT_HOUR <= end else start + pd.Timedelta(hours=66)
if "zone_pick" not in st.session_state:
    st.session_state.zone_pick = DEFAULT_ZONE

st.html(CSS)

with st.container(key="nav"):
    brand, tabs, right = st.columns([1.25, 3.2, 2.1], vertical_alignment="center")
    brand.html("<div class='tx-brand'>NYC Taxi</div>")
    with tabs:
        model = st.segmented_control("Model", list(MODELS), format_func=MODELS.get, default="chronos_bolt",
                                     key="nav_model", label_visibility="collapsed") or "chronos_bolt"
    with right:
        c_play, c_gh, c_prof = st.columns([0.55, 0.55, 2.6], vertical_alignment="center", gap="small")
        play = c_play.button("▶", key="play_btn", help="Play this day hour by hour")
        c_gh.html(f"<a class='tx-circle' href='{REPO}' target='_blank' title='Source on GitHub'>{ICONS['github']}</a>")
        c_prof.html("<div class='tx-profile'><div class='tx-avatar'>NYC</div><div><b>Test window</b>"
                    "<span>Jul 1 – Dec 31, 2024 · 218 zones</span></div></div>")

left, right = st.columns([1, 2.08], gap="small")

with right:
    with st.container(key="card_map"):
        with st.container(key="map_overlay", horizontal=True):
            zone_id = st.selectbox("Zone", list(modeled.index), key="zone_pick", label_visibility="collapsed",
                                   format_func=lambda z: f"🔍  {modeled.loc[z, 'zone']}, {modeled.loc[z, 'borough']}")
            day = st.date_input("Day", st.session_state.hour.date(), min_value=start.date(), max_value=end.date(),
                                key="day_pick", label_visibility="collapsed", format="MM/DD/YYYY")
            hour = st.selectbox("Hour", list(range(24)), index=st.session_state.hour.hour, key="hour_pick",
                                label_visibility="collapsed", format_func=lambda h: f"{h:02d}:00")
            color_by = st.selectbox("Show", ["demand", "error"], key="color_pick", label_visibility="collapsed",
                                    format_func={"demand": "Show  Busiest", "error": "Show  Error"}.get)
        ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        if ts not in hours:  # e.g. the skipped hour when clocks go forward
            ts = hours[hours.get_indexer([ts], method="nearest")[0]]
        st.session_state.hour = ts
        map_slot = st.empty()
        if play:
            for h in range(24):
                t = pd.Timestamp(day) + pd.Timedelta(hours=h)
                map_slot.pydeck_chart(deck(fc.loc[(model, t)], zone_id, color_by), height=500)
                time.sleep(0.45)
        event = map_slot.pydeck_chart(deck(fc.loc[(model, ts)], zone_id, color_by),
                                      height=500, on_select="rerun", selection_mode="single-object", key="map")

objects = (event.selection.get("objects") or {}) if event else {}
picked = objects.get("pins") or objects.get("dots")
if picked and int(picked[0]["zone_id"]) != zone_id:
    st.session_state.zone_pick = int(picked[0]["zone_id"])
    st.rerun()

frame = fc.loc[(model, ts)]

with left:
    with st.container(key="card_gauge"):
        st.html(gauge_card(model, frame))
    with st.container(key="card_borough"):
        st.html(borough_card(frame))
    with st.container(key="card_zone"):
        st.html(zone_card(model, zone_id, ts))

with right, st.container(key="card_table"):
    head, filt = st.columns([1, 2.2], vertical_alignment="center")
    head.html(f"<span class='tx-title' style='font-size:24px'>Busiest zones</span>"
              f"<div class='tx-muted' style='font-size:13px;margin-top:2px'>{ts:%A, %b %d · %H:00} · {MODELS[model]}</div>")
    with filt:
        borough = st.segmented_control("Borough", ["All"] + BOROUGHS[:4], default="All", key="filter_pick",
                                       label_visibility="collapsed") or "All"
    st.html(table_html(frame, borough, zone_id))
    st.html(f"<div class='tx-foot'>One-hour-ahead forecasts on a test window no model saw in training. Data: NYC TLC trips, "
            f"NOAA weather. Built on Databricks, PySpark and MLflow · <a href='{REPO}' target='_blank'>source</a></div>")
