"""Dashboard of next-hour taxi-demand forecasts across NYC taxi zones.

A street map with a pin per zone for the selected hour, the model's accuracy,
pickups by borough, the selected zone's week, and the busiest zones.

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
MAP_STYLE = "app/static/map_style.json"  # served by Streamlit (server.enableStaticServing)
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

CREAM = "#e9e2cf"
UNDER, NEUTRAL, OVER = np.array([74, 144, 226]), np.array([150, 150, 145]), np.array([235, 104, 52])

st.set_page_config(page_title="NYC Taxi Demand", page_icon="🚕", layout="wide")


# --- Images ------------------------------------------------------------------
# st.html drops inline <svg>, so every SVG goes in as an image data URI.

def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def img(svg: str, style: str = "") -> str:
    return f'<img src="{data_uri(svg)}" style="display:block;{style}" alt="">'


def icon(body: str, size: int = 18, color: str = "#d7d7d2", stroke: float = 1.6) -> str:
    return img(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
               f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


TAXI = ('<path d="M5 16V12l1.8-4.2A1.5 1.5 0 0 1 8.2 7h7.6a1.5 1.5 0 0 1 1.4.8L19 12v4"/><path d="M3.5 16h17v2.5h-17z"/>'
        '<path d="M5 12h14"/><circle cx="7.5" cy="18.5" r="1.4"/><circle cx="16.5" cy="18.5" r="1.4"/><path d="M10.5 7V5.5h3V7"/>')
ICONS = {
    "arrow": icon('<path d="M7 17 17 7M8 7h9v9"/>', 22, "#9a9a95"),
    "taxi": icon(TAXI, 30, "#e7e7e3", 1.4),
    "chevron": icon('<path d="m6 9 6 6 6-6"/>', 18, "#d7d7d2", 2),
    "target": icon('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/>', 18, "#e7e7e3"),
    "ruler": icon('<path d="M4 15 15 4l5 5L9 20z"/><path d="m8 11 1.5 1.5M11 8l1.5 1.5M14 5l1.5 1.5"/>', 18, "#e7e7e3"),
    "trend": icon('<path d="M4 17 9.5 11.5l3.5 3.5L20 8"/><path d="M15 8h5v5"/>', 18, "#e7e7e3"),
    "clock": icon('<circle cx="12" cy="12" r="8"/><path d="M12 8v4l2.5 2"/>', 18, "#e7e7e3"),
    "Manhattan": icon('<path d="M6 20V8l4-2v14M10 20V4h5v16M15 20V10h3v10M4 20h16"/>', 28, "#e7e7e3", 1.4),
    "Brooklyn": icon('<path d="M3 17h18M6 17V9M18 17V9M6 9q6 6 12 0M3 9h3M18 9h3M9 17v-4M12 17v-3M15 17v-4"/>', 28, "#e7e7e3", 1.4),
    "Queens": icon('<path d="M3 14.5 21 9.5l-1-2-6.5 1.8L8 4 6 4.6l3 6.3-4 1.1-2-1.6-1.3.4 1.6 3.9z"/><path d="M4 20h16"/>', 28, "#e7e7e3", 1.4),
    "Bronx": icon('<path d="M12 4 6 12h3l-4 5h14l-4-5h3z"/><path d="M12 17v3"/>', 28, "#e7e7e3", 1.4),
    "Staten Island": icon('<path d="M4 15h16l-2 4H6z"/><path d="M8 15V9h7l2 3v3M11 9V6"/><path d="M3 21q2-1.2 4.5 0t4.5 0 4.5 0 4.5 0"/>', 28, "#e7e7e3", 1.4),
    "github": img('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 16 16" fill="#ffffff"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'),
}
SEARCH_ICON = data_uri('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" '
                       'stroke="#141414" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>')

# Map pin: black circle with a white taxi and a dot at the location (the anchor).
PIN_GLYPH = ("<g fill='none' stroke='#ffffff' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'>"
             "<path d='M22 39v-7l3.2-7.4c.4-.9 1.3-1.6 2.3-1.6h17c1 0 1.9.7 2.3 1.6L50 32v7'/>"
             "<path d='M19 39h34v5H19z'/><path d='M22 32h28'/><path d='M32 23v-3h8v3'/></g>")
PIN = {"url": data_uri(f"""<svg xmlns='http://www.w3.org/2000/svg' width='72' height='84' viewBox='0 0 72 84'>
  <circle cx='36' cy='32' r='27' fill='#141414' stroke='#ffffff' stroke-width='3.5'/>{PIN_GLYPH}
  <circle cx='36' cy='77' r='4.5' fill='#141414' stroke='#ffffff' stroke-width='2'/></svg>"""),
       "width": 72, "height": 84, "anchorY": 84}


def label_icon(title: str, sub: str) -> dict:
    """The selected zone's pin as a white card with its name and numbers, drawn at 2x."""
    w = 72 + max(len(title) * 8.6, len(sub) * 6.9)
    pad, h = 10, 58
    cx, cy = pad + 29, pad + h / 2
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{(w + 2 * pad) * 2:.0f}' height='{(h + 2 * pad + 18) * 2:.0f}'
      viewBox='0 0 {w + 2 * pad:.1f} {h + 2 * pad + 18}' font-family='Helvetica, Arial, sans-serif'>
      <defs><filter id='s' x='-20%' y='-30%' width='140%' height='170%'><feDropShadow dx='0' dy='3' stdDeviation='4' flood-opacity='.16'/></filter></defs>
      <rect x='{pad}' y='{pad}' width='{w:.1f}' height='{h}' rx='{h / 2}' fill='#ffffff' filter='url(#s)'/>
      <circle cx='{cx}' cy='{cy}' r='21' fill='#141414' stroke='#ffffff' stroke-width='2.5'/>
      <g transform='translate({cx - 36 * 0.62:.2f},{cy - 32 * 0.62:.2f}) scale(0.62)'>{PIN_GLYPH}</g>
      <text x='{pad + 60}' y='{pad + 26}' font-size='15' font-weight='700' fill='#141414'>{title}</text>
      <text x='{pad + 60}' y='{pad + 44}' font-size='12' fill='#8a8a85'>{sub}</text>
      <circle cx='{cx}' cy='{pad + h + 12}' r='3.6' fill='#141414' stroke='#ffffff' stroke-width='1.5'/>
    </svg>"""
    return {"url": data_uri(svg), "width": round((w + 2 * pad) * 2), "height": (h + 2 * pad + 18) * 2,
            "anchorX": cx * 2, "anchorY": (pad + h + 12) * 2}


def skyline() -> str:
    """Dark city backdrop for the accuracy card."""
    rng = np.random.default_rng(7)
    parts, x = [], -10
    while x < 620:
        bw, bh = rng.integers(34, 70), rng.integers(70, 230)
        parts.append(f"<rect x='{x}' y='{260 - bh}' width='{bw}' height='{bh}' fill='url(#b)' stroke='#2a2a27' stroke-width='1'/>")
        for wy in range(260 - bh + 12, 250, 16):
            for wx in range(x + 7, x + bw - 8, 11):
                if rng.random() < 0.18:
                    parts.append(f"<rect x='{wx}' y='{wy}' width='4' height='6' fill='#3a3934'/>")
        x += bw + rng.integers(2, 10)
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='600' height='260' viewBox='0 0 600 260'>"
           f"<defs><linearGradient id='b' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#262624'/>"
           f"<stop offset='1' stop-color='#151515'/></linearGradient></defs>{''.join(parts)}</svg>")
    return data_uri(svg)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.stApp { background: #0c0c0c; color: #f5f5f4; }
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container { padding: 0.9rem 1.1rem 1.5rem; max-width: 1560px; }
[data-testid="stVerticalBlock"] { gap: 12px; }

/* Space Grotesk for headings, nav and figures; Inter for small text */
.tx-brand, .tx-title, .tx-center b, .tx-id, .tx-profile b, .tx-pill, .tx-row, .tx-two b, .tx-route div, .tx-table td,
.tx-stat b, .tx-gauge-labels, .tx-caption, .st-key-nav_model button p, .st-key-filter_pick button p {
  font-family: "Space Grotesk", Inter, system-ui, sans-serif !important;
}

div[class*="st-key-card"] { background: #171717; border: 1px solid rgba(255,255,255,0.05); border-radius: 26px; padding: 26px 28px; }

/* --- nav --- */
.tx-brand { font-size: 28px; font-weight: 600; letter-spacing: -0.03em; color: #fff; white-space: nowrap; padding-left: 6px; }
.st-key-nav_model [data-testid="stButtonGroup"] { gap: 10px; }
.st-key-nav_model button { background: transparent !important; border: 0 !important; padding: 6px 12px !important; box-shadow: none !important; }
.st-key-nav_model button p { font-size: 17px !important; color: #8a8a85 !important; font-weight: 400; }
.st-key-nav_model button:hover p { color: #fff !important; }
.st-key-nav_model button[aria-checked="true"] p { color: #fff !important; font-weight: 500; }
.st-key-play_btn button, .tx-circle { width: 54px; height: 54px; min-height: 54px; border-radius: 50% !important; background: #121212 !important;
  border: 1px solid rgba(255,255,255,0.1) !important; color: #fff !important; padding: 0 !important; display: grid; place-items: center; }
.st-key-play_btn button p { font-size: 16px !important; }
.st-key-play_btn button:hover, .tx-circle:hover { background: #1e1e1e !important; }
.tx-navright { display: flex; justify-content: flex-end; }
.tx-profile { display: flex; align-items: center; gap: 12px; height: 60px; padding: 5px 18px 5px 6px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.1); background: #121212; width: fit-content; margin-left: auto; }
.tx-avatar { width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, #f1ebd9, #9d9787); color: #141414;
  display: grid; place-items: center; font-weight: 700; font-size: 13px; letter-spacing: .02em; }
.tx-profile b { display: block; font-size: 17px; font-weight: 500; color: #fff; line-height: 1.2; }
.tx-profile span { font-size: 14px; color: #8a8a85; }

/* --- map --- */
.st-key-card_map { padding: 0 !important; overflow: hidden; position: relative; background: #ebe7df !important; gap: 0 !important; }
.st-key-map_overlay { position: absolute !important; top: 22px; left: 22px; z-index: 5; width: auto !important; gap: 12px; }
.st-key-map_overlay [data-testid="stSelectbox"] div { background: transparent !important; border-color: transparent !important; }
.st-key-map_overlay [data-testid="stSelectbox"] div[role="group"], .st-key-map_overlay input, .st-key-map_overlay button { background-color: #fff !important; border-color: transparent !important; }
.st-key-map_overlay [role="group"] { border-radius: 999px !important; min-height: 50px; box-shadow: 0 2px 12px rgba(0,0,0,0.10); padding-left: 14px; overflow: hidden; }
.st-key-map_overlay * { color: #141414 !important; -webkit-text-fill-color: #141414 !important; }
.st-key-map_overlay input { font-size: 15.5px !important; font-weight: 500; }
.st-key-map_overlay svg { fill: #141414 !important; }
.st-key-zone_pick { flex: 0 0 330px !important; width: 330px !important; }
.st-key-zone_pick input { background: #fff url("SEARCH_ICON") 0 center / 20px no-repeat !important; padding-left: 30px !important; }
.st-key-color_pick { flex: 0 0 210px !important; width: 210px !important; }
.st-key-card_map [data-testid="stElementToolbar"] { opacity: 1 !important; top: 22px !important; right: 22px !important; }
.st-key-card_map [data-testid="stElementToolbar"] button { width: 54px; height: 54px; border-radius: 50%; background: #fff !important; box-shadow: 0 2px 12px rgba(0,0,0,0.10); }
.st-key-card_map [data-testid="stElementToolbar"] svg { fill: #141414 !important; color: #141414 !important; }
.st-key-card_map div:has(> div > .mapboxgl-ctrl-group) { top: auto !important; bottom: 22px !important; right: 22px !important; width: 54px; height: 122px; }
.st-key-card_map div:has(> .mapboxgl-ctrl-group) { top: auto !important; bottom: 0 !important; right: 0 !important; }
.st-key-card_map [data-testid="stElementToolbar"], .st-key-card_map [data-testid="stElementToolbar"] * { opacity: 1 !important; visibility: visible !important; }
.st-key-card_map .mapboxgl-ctrl-group { margin: 0 !important; }
.st-key-card_map .mapboxgl-ctrl-group { background: transparent !important; box-shadow: none !important; border: 0 !important; display: flex; flex-direction: column; gap: 14px; }
.st-key-card_map .mapboxgl-ctrl-group button { width: 54px !important; height: 54px !important; border-radius: 50% !important; background-color: #fff !important;
  border: 0 !important; box-shadow: 0 2px 12px rgba(0,0,0,0.12); }
.st-key-card_map .mapboxgl-ctrl-icon { filter: none !important; background-size: 30px !important; }
.st-key-card_map .mapboxgl-ctrl-bottom-right { right: 90px; }
.st-key-card_map .mapboxgl-ctrl-attrib { background: transparent !important; font-size: 9px; opacity: .55; }

/* --- accuracy card --- */
.st-key-card_gauge { background: linear-gradient(180deg, rgba(23,23,23,0.15) 0%, rgba(23,23,23,0.35) 45%, #171717 88%),
  url("SKYLINE") center 40px / cover no-repeat, #171717 !important; }
.tx-head { display: flex; justify-content: space-between; align-items: center; }
.tx-title { font-size: 25px; font-weight: 500; letter-spacing: -0.02em; color: #f5f5f4; }
.tx-pills { position: relative; height: 166px; margin-top: 12px; }
.tx-pill { position: absolute; display: inline-flex; align-items: center; gap: 9px; padding: 10px 16px; border-radius: 999px;
  background: rgba(64,64,60,0.72); border: 1px solid rgba(255,255,255,0.07); backdrop-filter: blur(8px);
  font-size: 15.5px; color: #f2f2ef; white-space: nowrap; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
.tx-pill b { font-weight: 400; color: #a6a6a0; }
.tx-arc { margin: 0 -28px; position: relative; }
.tx-gauge-labels { position: absolute; left: 30px; right: 30px; bottom: -14px; display: flex; justify-content: space-between; font-size: 15px; color: #8a8a85; }
.tx-caption { text-align: center; color: #a6a6a0; font-size: 16px; margin-top: 26px; }
.tx-caption b { color: #f5f5f4; font-weight: 500; }

/* --- borough card --- */
.tx-rows { display: flex; flex-direction: column; gap: 14px; margin-top: 18px; }
.tx-row { display: grid; grid-template-columns: 30px 112px 50px 1fr 42px; align-items: center; gap: 10px; font-size: 17px; color: #f5f5f4; }
.tx-count { text-align: right; font-variant-numeric: tabular-nums; }
.tx-share { text-align: right; font-size: 16px; color: #e7e7e3; font-variant-numeric: tabular-nums; }
.tx-ticks { display: flex; gap: 1.5px; height: 18px; }
.tx-ticks i { flex: 1; border-radius: 1px; background: #3d3d3a; }
.tx-ticks i.on { background: #f5f5f4; }

/* --- zone card --- */
.tx-fuel { display: grid; grid-template-columns: 1fr 1.4fr 1fr; align-items: center; row-gap: 16px; margin-top: 22px; }
.tx-stat b { display: block; font-size: 17px; font-weight: 500; color: #f5f5f4; font-variant-numeric: tabular-nums; }
.tx-stat span { font-size: 15px; color: #8a8a85; }
.tx-stat.right { text-align: right; }
.tx-center { grid-row: span 2; text-align: center; }
.tx-center span { font-size: 16px; color: #a6a6a0; }
.tx-center b { display: block; font-size: 52px; font-weight: 500; letter-spacing: -0.03em; color: #fff; margin-top: 2px; line-height: 1.05; font-variant-numeric: tabular-nums; }
.tx-bars { margin: 18px -28px 0; }
.tx-spark-labels { display: flex; justify-content: space-between; font-size: 14px; color: #8a8a85; margin-top: 10px; }

/* --- table card --- */
.st-key-card_table { padding: 26px 0 8px !important; }
.st-key-card_table [data-testid="stHorizontalBlock"] { padding: 0 28px; }
.st-key-filter_pick { margin-left: auto; }
.st-key-filter_pick [data-testid="stButtonGroup"] { border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; padding: 5px; gap: 4px; width: fit-content; margin-left: auto; }
.st-key-filter_pick button { background: transparent !important; border: 0 !important; border-radius: 999px !important; padding: 8px 22px !important; box-shadow: none !important; min-height: 44px; }
.st-key-filter_pick button p { font-size: 16px !important; color: #f5f5f4 !important; }
.st-key-filter_pick button[aria-checked="true"] { background: #e3dcc8 !important; }
.st-key-filter_pick button[aria-checked="true"] p { color: #141414 !important; font-weight: 500; }
div[class*="st-key-pop_"] button { width: 54px; height: 54px; min-height: 54px; border-radius: 50% !important; background: transparent !important;
  border: 1px solid rgba(255,255,255,0.12) !important; padding: 0 !important; }
div[class*="st-key-pop_"] button [data-testid="stIconMaterial"] { color: #e7e7e3; font-size: 22px; }
div[class*="st-key-pop_"] button > div > div:last-child { display: none; }
.tx-table { width: 100%; border-collapse: collapse; margin-top: 18px; }
.tx-table th { text-align: left; font-weight: 400; font-size: 16px; color: #8a8a85; padding: 15px 12px; background: #1e1e1e; }
.tx-table th:first-child, .tx-table td:first-child { padding-left: 32px; }
.tx-table th:last-child, .tx-table td:last-child { padding-right: 32px; text-align: right; }
.tx-table td { padding: 20px 12px; border-bottom: 1px solid rgba(255,255,255,0.07); color: #f5f5f4; vertical-align: middle; font-size: 16.5px; }
.tx-table tr:last-child td { border-bottom: 0; }
.tx-table tr.tx-selected td { background: rgba(233,226,207,0.05); }
.tx-id { display: flex; align-items: center; gap: 18px; font-size: 20px; font-weight: 500; }
.tx-two b { display: block; font-weight: 400; }
.tx-two span { color: #8a8a85; }
.tx-route { display: flex; flex-direction: column; gap: 8px; position: relative; padding-left: 20px; }
.tx-route::before { content: ""; position: absolute; left: 4px; top: 11px; bottom: 11px; border-left: 1px solid #5a5a55; }
.tx-route div { position: relative; font-variant-numeric: tabular-nums; }
.tx-route div::before { content: ""; position: absolute; left: -20px; top: 6px; width: 9px; height: 9px; border-radius: 50%; border: 1.4px solid #c9c9c4; background: #171717; box-sizing: border-box; }
.tx-route div.tx-fc::before { background: #c9c9c4; }
.tx-route span { color: #8a8a85; font-family: Inter, sans-serif !important; font-size: 15px; }
.tx-num { font-variant-numeric: tabular-nums; }
.tx-status.tx-over { color: #f0a27f; } .tx-status.tx-under { color: #8fb8ea; }
.tx-foot { color: #66665f; font-size: 12px; padding: 4px 28px 10px; }
.tx-foot a { color: #a6a6a0; }
</style>
""".replace("SEARCH_ICON", SEARCH_ICON).replace("SKYLINE", skyline())


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
    zones = pd.DataFrame([f["properties"] for f in json.loads((DATA / "zones.geojson").read_text())["features"]])
    return fc.set_index(["model", "hour_ts"]).sort_index(), board, zone_wape, zones.set_index("zone_id")


def _blend(a, b, t):
    return (a + (b - a) * t[:, None]).round().astype(int)


def error_colors(frame: pd.DataFrame) -> np.ndarray:
    # Signed error relative to the zone's actual demand (floor 5 so quiet zones don't flare up).
    rel = ((frame["prediction"] - frame["pickups"]) / frame["pickups"].clip(lower=5)).clip(-1, 1).to_numpy()
    return np.where(rel[:, None] < 0, _blend(NEUTRAL, UNDER, -rel), _blend(NEUTRAL, OVER, rel))


def fmt(v) -> str:
    return "—" if pd.isna(v) else f"{v:,.0f}"


# --- Map -----------------------------------------------------------------------

def deck(frame: pd.DataFrame, selected: int, color_by: str) -> pdk.Deck:
    """A pin per modeled zone; the selected zone's pin becomes a labelled card.
    pydeck reads plain strings as data expressions, so constants are quoted."""
    data = zones.join(frame.set_index("zone_id"), how="inner").reset_index()
    data["actual_txt"] = data["pickups"].map(fmt)
    data["forecast_txt"] = data["prediction"].map(fmt)
    data["band_txt"] = np.where(data["p10"].notna(), "80% range " + data["p10"].map(fmt) + "–" + data["p90"].map(fmt), "")
    pins = data[data["zone_id"] != selected].copy()
    pins["icon"] = [PIN] * len(pins)
    sel = data[data["zone_id"] == selected].copy()
    sel["icon"] = [label_icon(r.zone, f"{r.forecast_txt} forecast · {r.actual_txt} actual") for r in sel.itertuples()]

    common = {"pickable": True, "auto_highlight": False}
    layers = []
    if color_by == "error":
        halos = data.assign(halo=[c + [150] for c in error_colors(data).tolist()])
        layers.append(pdk.Layer("ScatterplotLayer", halos, id="halos", get_position=["lon", "lat"], get_fill_color="halo",
                                get_radius=1, radius_min_pixels=19, radius_max_pixels=19, get_pixel_offset=[0, -21]))
    layers += [
        pdk.Layer("IconLayer", pins, id="pins", get_position=["lon", "lat"], get_icon="icon", get_size=36, **common),
        pdk.Layer("IconLayer", sel, id="label", get_position=["lon", "lat"], get_icon="icon", get_size=96, **common),
    ]
    z = zones.loc[selected]
    view = pdk.ViewState(latitude=float(z["lat"]) - 0.004, longitude=float(z["lon"]) + 0.006, zoom=13.2, pitch=0, bearing=0)
    tooltip = {
        "html": "<div style='font-weight:600;font-size:13px'>{zone}</div><div style='color:#8a8a85;margin-bottom:4px'>{borough}</div>"
                "Actual <b>{actual_txt}</b> · Forecast <b>{forecast_txt}</b><br/><span style='color:#8a8a85'>{band_txt}</span>",
        "style": {"backgroundColor": "#ffffff", "color": "#141414", "fontFamily": "Inter, sans-serif", "fontSize": "12px",
                  "borderRadius": "14px", "padding": "10px 14px", "boxShadow": "0 8px 24px rgba(0,0,0,.16)"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip, map_provider="carto", map_style=MAP_STYLE)


# --- Cards ---------------------------------------------------------------------

def card_head(title: str) -> str:
    return f"<div class='tx-head'><span class='tx-title'>{title}</span>{ICONS['arrow']}</div>"


def gauge_card(model: str, frame: pd.DataFrame) -> str:
    m, naive = board.loc[model], board.loc["seasonal_naive", "wape"]
    score = max(0.0, 1 - m["wape"])
    act, fcs = frame["pickups"].sum(), frame["prediction"].sum()
    pills = [
        ("target", "Coverage", "—" if pd.isna(m["coverage"]) else f"{m['coverage']:.1%}", "left:0;top:6px"),
        ("ruler", "MAE", f"{m['mae']:.2f}", "right:0;top:30px"),
        ("clock", "This hour", f"{(fcs - act) / act:+.1%}", "left:6%;top:86px"),
        ("trend", "vs naive", "baseline" if model == "seasonal_naive" else f"{m['wape'] / naive - 1:+.0%}", "right:2%;top:112px"),
    ]
    pill_html = "".join(f"<span class='tx-pill' style='{pos}'>{ICONS[i]}{label} <b>{v}</b></span>" for i, label, v, pos in pills)

    # A shallow arc (quadratic curve) across the full card width, with the knob at the score.
    W = 600
    p0, p1, p2 = np.array([0, 58]), np.array([W / 2, 20]), np.array([W, 58])
    kx, ky = (1 - score) ** 2 * p0 + 2 * (1 - score) * score * p1 + score ** 2 * p2
    arc = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -34 {W} 106" width="{W}" height="106" font-family="Helvetica, Arial, sans-serif">
      <defs><linearGradient id="a" x1="0" x2="1"><stop offset="0" stop-color="#b9675b"/><stop offset=".5" stop-color="#e9e2cf"/><stop offset="1" stop-color="#a3d38f"/></linearGradient></defs>
      <path d="M0,58 Q{W / 2},20 {W},58" stroke="url(#a)" stroke-width="3.5" fill="none"/>
      <rect x="{kx - 40:.1f}" y="{ky - 60:.1f}" width="80" height="38" rx="19" fill="#fff"/>
      <text x="{kx:.1f}" y="{ky - 34.5:.1f}" text-anchor="middle" font-size="20" font-weight="700" fill="#141414">{score:.0%}</text>
      <line x1="{kx:.1f}" x2="{kx:.1f}" y1="{ky - 22:.1f}" y2="{ky:.1f}" stroke="#fff" stroke-width="2"/>
      <circle cx="{kx:.1f}" cy="{ky:.1f}" r="11" fill="#fff" stroke="#9a9a95" stroke-width="3"/>
    </svg>"""
    return f"""{card_head("Forecast accuracy")}
    <div class="tx-pills">{pill_html}</div>
    <div class="tx-arc"><div class="tx-gauge-labels"><span>0%</span><span>100%</span></div>{img(arc, "width:100%;height:auto")}</div>
    <div class="tx-caption"><b>{m['wape']:.1%}</b> WAPE <span>/</span> Jul–Dec 2024 test window ({score:.0%})</div>"""


def borough_card(frame: pd.DataFrame) -> str:
    counts = frame.join(zones["borough"], on="zone_id").groupby("borough")["pickups"].sum().reindex(BOROUGHS).fillna(0)
    total = max(counts.sum(), 1)
    rows = ""
    for b, n in counts.items():
        share = n / total
        on = round(share * 28)
        ticks = "".join(f"<i class='{'on' if k < on else ''}'></i>" for k in range(28))
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
    W, H = 600, 120
    top = max(week["pickups"].max(), week["prediction"].max(), 1) * 1.08
    step = W / len(week)
    same_day = week.index.normalize() == ts.normalize()
    bars = "".join(
        f"<rect x='{i * step + step * 0.22:.2f}' y='{H - v / top * H:.2f}' width='{step * 0.56:.2f}' height='{v / top * H:.2f}' "
        f"fill='{'#f5f5f4' if today else '#4a4a46'}'/>"
        for i, (v, today) in enumerate(zip(week["pickups"], same_day))
    )
    line = " ".join(f"{'M' if i == 0 else 'L'}{(i + 0.5) * step:.2f},{H - v / top * H:.2f}" for i, v in enumerate(week["prediction"]))
    chart = img(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="none" width="{W}" height="{H}">
      {bars}<path d="{line}" fill="none" stroke="{CREAM}" stroke-width="2" stroke-linejoin="round"/></svg>""",
                "width:100%;height:120px")
    return f"""{card_head(zones.loc[zone_id, 'zone'])}
    <div class="tx-fuel">
      <div class="tx-stat"><b>{zf['pickups']:,.0f}</b><span>Actual</span></div>
      <div class="tx-center"><span>Forecast, {ts:%a %H:00}</span><b>{zf['prediction']:,.0f}</b></div>
      <div class="tx-stat right"><b>{zone_wape.loc[zone_id, model]:.1%}</b><span>Zone WAPE</span></div>
      <div class="tx-stat"><b>{band}</b><span>80% range</span></div>
      <div class="tx-stat right"><b>{diff:+.1%}</b><span>vs actual</span></div>
    </div>
    <div class="tx-bars">{chart}</div>
    <div class="tx-spark-labels"><span>{lo:%b %d}</span><span>{hi:%b %d}</span></div>"""


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


# --- Page ----------------------------------------------------------------------

fc, board, zone_wape, zones = load(FORECASTS.stat().st_mtime)
hours = fc.index.get_level_values("hour_ts").unique()
start, end = hours.min(), hours.max()
modeled = zones.loc[zone_wape.index].sort_values("zone")

# Widgets further down the page (date, hour) drive the map above them, so
# their values are read from session state before they're drawn.
ss = st.session_state
default = DEFAULT_HOUR if start <= DEFAULT_HOUR <= end else start + pd.Timedelta(hours=66)
ss.setdefault("zone_pick", DEFAULT_ZONE)
ss.setdefault("day_pick", default.date())
ss.setdefault("hour_pick", default.hour)
ss.setdefault("color_pick", "demand")
ts = pd.Timestamp(ss.day_pick) + pd.Timedelta(hours=ss.hour_pick)
if ts not in hours:  # e.g. the skipped hour when clocks go forward
    ts = hours[hours.get_indexer([ts], method="nearest")[0]]

st.html(CSS)

with st.container(key="nav"):
    brand, tabs, right = st.columns([1.1, 3.3, 2.4], vertical_alignment="center")
    brand.html("<div class='tx-brand'>NYC Taxi</div>")
    with tabs:
        model = st.segmented_control("Model", list(MODELS), format_func=MODELS.get, default="chronos_bolt",
                                     key="nav_model", label_visibility="collapsed") or "chronos_bolt"
    with right:
        c_play, c_gh, c_prof = st.columns([0.5, 0.5, 2.7], vertical_alignment="center", gap="small")
        play = c_play.button("▶", key="play_btn", help="Play this day hour by hour")
        c_gh.html(f"<a class='tx-circle' href='{REPO}' target='_blank' title='Source on GitHub'>{ICONS['github']}</a>")
        c_prof.html(f"<div class='tx-profile'><div class='tx-avatar'>NYC</div><div><b>{ts:%a, %b %d · %H:00}</b>"
                    f"<span>{MODELS[model]} · test window</span></div>{ICONS['chevron']}</div>")

left, right = st.columns([1, 2.12], gap="small")

with right:
    with st.container(key="card_map"):
        with st.container(key="map_overlay", horizontal=True):
            zone_id = st.selectbox("Zone", list(modeled.index), key="zone_pick", label_visibility="collapsed",
                                   format_func=lambda z: f"{modeled.loc[z, 'zone']}, {modeled.loc[z, 'borough']}")
            color_by = st.selectbox("Show", ["demand", "error"], key="color_pick", label_visibility="collapsed",
                                    format_func={"demand": "Show by  Demand", "error": "Show by  Error"}.get)
        map_slot = st.empty()
        if play:
            for h in range(24):
                t = ts.normalize() + pd.Timedelta(hours=h)
                if t in hours:
                    map_slot.pydeck_chart(deck(fc.loc[(model, t)], zone_id, color_by), height=520)
                    time.sleep(0.45)
        event = map_slot.pydeck_chart(deck(fc.loc[(model, ts)], zone_id, color_by),
                                      height=520, on_select="rerun", selection_mode="single-object", key="map")

objects = (event.selection.get("objects") or {}) if event else {}
picked = objects.get("pins")
if picked and int(picked[0]["zone_id"]) != zone_id:
    ss.zone_pick = int(picked[0]["zone_id"])
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
    head, filt, cal, clock = st.columns([1.15, 3, 0.32, 0.32], vertical_alignment="center", gap="small")
    head.html(f"<span class='tx-title' style='font-size:27px'>Busiest zones</span>")
    with filt:
        borough = st.segmented_control("Borough", ["All"] + BOROUGHS[:4], default="All", key="filter_pick",
                                       label_visibility="collapsed") or "All"
    with cal, st.container(key="pop_day"), st.popover(":material/calendar_today:", help="Day"):
        st.date_input("Day", min_value=start.date(), max_value=end.date(), key="day_pick", format="MM/DD/YYYY")
    with clock, st.container(key="pop_hour"), st.popover(":material/schedule:", help="Hour"):
        st.selectbox("Hour", list(range(24)), key="hour_pick", format_func=lambda h: f"{h:02d}:00")
    st.html(table_html(frame, borough, zone_id))
    st.html(f"<div class='tx-foot'>One-hour-ahead forecasts on a test window no model saw in training. Data: NYC TLC trips, "
            f"NOAA weather. Built on Databricks, PySpark and MLflow · <a href='{REPO}' target='_blank'>source</a></div>")
