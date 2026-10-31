"""
Benchmark figure: forecast error against lead time.

One panel per (area, variable) and one line per forecast. Baselines are grey,
dashed and dotted, so the models stand out; every line is also labelled at its
end, so telling them apart never depends on colour alone.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

# Categorical colours in a fixed order, checked for colour-vision deficiency
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
BASELINE_STYLES = {
    "persistence": {"color": "#52514e", "linestyle": "--"},
    "anomaly_persistence": {"color": "#8a8984", "linestyle": ":"},
}
LABELS = {
    "this_model": "this model",
    "persistence": "persistence",
    "anomaly_persistence": "anomaly persistence",
    "graphcast": "GraphCast",
    "hres": "ECMWF HRES",
    "pangu": "Pangu-Weather",
    "fuxi": "FuXi",
    "neuralgcm": "NeuralGCM",
    "fgn": "FGN (WeatherNext 2)",
}
UNITS = {"t850": "K", "z500": "m²/s²", "u10": "m/s", "v10": "m/s"}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e5e4e0"
SURFACE_COLOR = "#fcfcfb"


def plot_error_by_lead(
    results: Dict[str, dict],
    areas: Sequence[str],
    variables: Sequence[str],
    forecasts: Sequence[str],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    RMSE in physical units (log scale) against lead time.

    Args:
        results: benchmark() output, {lead hours: {area: {forecast: metrics}}}.
        areas: One row per area, e.g. "global", "india/monsoon_jjas".
        variables: One column per variable.
        forecasts: Forecasts to draw, in legend order; baselines are grey.
        save_path: Where to save the figure (PNG).
        title: Figure title.

    Returns:
        The matplotlib Figure.
    """
    leads = sorted(int(h) for h in results)
    colors = iter(MODEL_COLORS)
    styles = {
        name: BASELINE_STYLES.get(name) or {"color": next(colors), "linestyle": "-"}
        for name in forecasts
    }

    fig, axes = plt.subplots(
        len(areas), len(variables), squeeze=False,
        figsize=(4.6 * len(variables), 3.6 * len(areas)), facecolor=SURFACE_COLOR,
    )
    for row, area in enumerate(areas):
        for col, var in enumerate(variables):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE_COLOR)
            key = f"rmse_{var}_physical"
            ends = []
            for name in forecasts:
                points = [
                    (h, results[str(h)][area][name][key]) for h in leads
                    if key in results[str(h)].get(area, {}).get(name, {})
                ]
                if not points:
                    continue
                x, y = zip(*points)
                ax.plot(x, y, marker="o", markersize=4, linewidth=2,
                        label=LABELS.get(name, name), **styles[name])
                ends.append((LABELS.get(name, name), x[-1], y[-1]))

            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            _label_line_ends(ax, ends)
            ax.set_xticks(leads)
            ax.set_xlim(leads[0] - 3, leads[-1] * 1.35)  # room for the end labels
            ax.set_xlabel("lead time (hours)", color=TEXT_COLOR)
            ax.set_ylabel(f"{var} RMSE ({UNITS.get(var, '')}), log scale", color=TEXT_COLOR)
            ax.set_title(f"{area.replace('/', ', ').replace('_', ' ')}: {var}", fontsize=10)
            ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE_COLOR)
    return fig


def _label_line_ends(ax, ends: List[Tuple[str, float, float]]):
    """Write each line's name at its right end, nudged apart so labels never overlap."""
    low, high = np.log10(ax.get_ylim())
    min_gap = 0.07 * (high - low)  # in log10 units, about one line of text
    ends = sorted(ends, key=lambda end: end[2])
    positions = list(np.log10([y for _, _, y in ends]))
    for i in range(1, len(positions)):  # push crowded labels up...
        positions[i] = max(positions[i], positions[i - 1] + min_gap)
    if positions:  # ...and back down if the top one left the axes
        positions[-1] = min(positions[-1], high - min_gap / 2)
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - min_gap)
    for (name, x, _), position in zip(ends, positions):
        ax.annotate(name, (x, 10 ** position), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=TEXT_COLOR)
