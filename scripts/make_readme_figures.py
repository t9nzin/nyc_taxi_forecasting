"""README figures from the exported test-window forecasts (app/data/forecasts.parquet).

    .venv/bin/python scripts/make_readme_figures.py
"""

from pathlib import Path

import holidays
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

MODELS = {  # fixed colour per model across every figure; the baseline is neutral grey
    "chronos_bolt": ("Chronos-Bolt", "#2a78d6"),
    "nhits": ("N-HiTS", "#eb6834"),
    "poisson_glm": ("Poisson GLM", "#1baf7a"),
    "seasonal_naive": ("Seasonal-naive", "#8a8984"),
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def wape(g: pd.DataFrame) -> float:
    return (g["prediction"] - g["pickups"]).abs().sum() / g["pickups"].sum()


def by_hour(fc: pd.DataFrame):
    table = fc.groupby(["model", fc["hour_ts"].dt.hour]).apply(wape).unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model, (label, color) in MODELS.items():
        ax.plot(table.index, table[model] * 100, color=color, linewidth=2)
    # Direct labels at the line ends, spread apart where the lines finish close together.
    ends = (table.iloc[-1] * 100).sort_values()
    placed = []
    for model, y in ends.items():
        y = max(y, placed[-1] + 2.6) if placed else y
        placed.append(y)
        ax.annotate(MODELS[model][0], (23, y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10, annotation_clip=False)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, None)
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Error by hour of day")
    fig.subplots_adjust(right=0.82)
    fig.savefig(OUT / "wape_by_hour.png", dpi=160, bbox_inches="tight")


def holiday_effect(fc: pd.DataFrame):
    days = set(holidays.country_holidays("US", subdiv="NY", years=[2024]))
    day = fc["hour_ts"].dt.date
    week_after = (fc["hour_ts"] - pd.Timedelta(days=7)).dt.date
    fc = fc.assign(kind=day.isin(days).map({True: "Holiday", False: "Other days"}))
    fc.loc[week_after.isin(days) & ~day.isin(days), "kind"] = "Week after a holiday"
    order = ["Other days", "Holiday", "Week after a holiday"]
    table = fc.groupby(["kind", "model"]).apply(wape).unstack("model").loc[order]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.2
    for i, (model, (label, color)) in enumerate(MODELS.items()):
        xs = [x + (i - 1.5) * width for x in range(len(order))]
        bars = ax.bar(xs, table[model] * 100, width=width - 0.03, color=color, label=label)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=9, color=MUTED)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("WAPE (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=10, labelcolor=INK)
    ax.set_ylim(0, table.max().max() * 100 * 1.18)
    ax.set_title("Holidays break \"same hour last week\"")
    fig.savefig(OUT / "holiday_effect.png", dpi=160, bbox_inches="tight")


def thanksgiving_week(fc: pd.DataFrame, zone_id: int = 161):
    week = fc[(fc["zone_id"] == zone_id) & fc["hour_ts"].between("2024-11-24", "2024-12-01 23:00")]
    series = {m: w.set_index("hour_ts").sort_index() for m, w in week.groupby("model")}
    chronos, naive = series["chronos_bolt"], series["seasonal_naive"]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.fill_between(chronos.index, chronos["p10"], chronos["p90"], color=MODELS["chronos_bolt"][1], alpha=0.18,
                    linewidth=0, label="Chronos-Bolt p10–p90")
    ax.plot(chronos.index, chronos["pickups"], color=INK, linewidth=2, label="Actual")
    ax.plot(chronos.index, chronos["prediction"], color=MODELS["chronos_bolt"][1], linewidth=1.6, label="Chronos-Bolt")
    ax.plot(naive.index, naive["prediction"], color=MODELS["seasonal_naive"][1], linewidth=1.4, linestyle="--",
            label="Seasonal-naive")
    ax.axvspan(pd.Timestamp("2024-11-28"), pd.Timestamp("2024-11-29"), color=GRID, alpha=0.6, linewidth=0)
    ax.annotate("Thanksgiving", (pd.Timestamp("2024-11-28 12:00"), ax.get_ylim()[1] * 0.95), ha="center",
                color=MUTED, fontsize=10)
    ax.set_ylabel("pickups / hour")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=10, labelcolor=INK)
    ax.set_title("Midtown Center, Thanksgiving week 2024")
    fig.savefig(OUT / "thanksgiving_midtown.png", dpi=160, bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc = pd.read_parquet(ROOT / "app" / "data" / "forecasts.parquet")
    fc["model"] = fc["model"].astype(str)
    by_hour(fc)
    holiday_effect(fc)
    thanksgiving_week(fc)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
