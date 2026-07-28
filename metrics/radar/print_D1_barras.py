"""Create a grouped-bar replacement for the D1 radar charts.

The figure contains every combination used by the radar:
3 metrics x 3 models x 3 temperatures x 2 conditions = 54 bars.
Values are percentages of the corresponding human baseline (100%).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
READABILITY_FILE = SCRIPT_DIR.parent / "readability" / "readability_limited_news.json"
MTLD_FILE = SCRIPT_DIR.parent / "mtld" / "mtld_limited_news.json"
OUTPUT_DIR = SCRIPT_DIR / "D1"

MODELS = (
    ("1", "Ministral-3B"),
    ("2", "Qwen-2.5-0.5B"),
    ("3", "Llama-3.2-1B"),
)
TEMPERATURES = (
    ("3", "t = 0.50"),
    ("2", "t = 0.75"),
    ("1", "t = 1.00"),
)
CONDITIONS = ("RAG", "NO_RAG")
CONDITION_LABELS = {"RAG": "RAG", "NO_RAG": "NO RAG"}
CONDITION_COLORS = {"RAG": "#D38E45", "NO_RAG": "#5D9BCF"}
CONDITION_HATCHES = {"RAG": "", "NO_RAG": ""}
METRICS = (
    ("MTLD", "mean_mtld", "mtld"),
    ("Tree Depth", "mean_depth", "readability"),
    ("FH Index", "mean_fh", "readability"),
)
METRIC_Y_LIMITS = {
    "MTLD": (50, 200),
    "Tree Depth": (50, 150),
    "FH Index": (25, 125),
}

HUMAN_COLOR = "#008000"
GRID_COLOR = "#E0E0E0"
AXIS_FONT_SIZE = 14.3
AXIS_TITLE_FONT_SIZE = 15.4
MODEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
FIGURE_TITLE_FONT_SIZE = 18


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty JSON object in {path}")
    return data


def process_d1_data() -> dict:
    """Reproduce the radar aggregation and return all normalized values."""

    sources = {
        "readability": load_json(READABILITY_FILE),
        "mtld": load_json(MTLD_FILE),
    }
    results = {}

    for metric_name, mean_key, source_name in METRICS:
        source = sources[source_name]
        newspapers = list(source)
        human_mean = fmean(
            source[newspaper]["real_reference_metrics"][mean_key]
            for newspaper in newspapers
        )

        for model_id, model_name in MODELS:
            for temperature_id, temperature_label in TEMPERATURES:
                for condition in CONDITIONS:
                    model_mean = fmean(
                        source[newspaper]["llm_models"][model_id][temperature_id][condition][mean_key]
                        for newspaper in newspapers
                    )
                    results[(metric_name, model_id, temperature_id, condition)] = {
                        "model": model_name,
                        "temperature": temperature_label,
                        "relative_percent": 100.0 * model_mean / human_mean,
                    }

    expected = len(METRICS) * len(MODELS) * len(TEMPERATURES) * len(CONDITIONS)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} values, found {len(results)}")
    return results


def percent_label(value: float, _position: int) -> str:
    return f"{value:.0f}%"


def generate_bar_chart() -> tuple[Path, Path]:
    data = process_d1_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(METRICS),
        len(MODELS),
        figsize=(15, 12),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x_positions = list(range(len(TEMPERATURES)))
    bar_width = 0.36
    offsets = (-bar_width / 2.0, bar_width / 2.0)

    for metric_index, (metric_name, _mean_key, _source_name) in enumerate(METRICS):
        for model_index, (model_id, model_name) in enumerate(MODELS):
            ax = axes[metric_index][model_index]

            for condition_index, condition in enumerate(CONDITIONS):
                values = [
                    data[(metric_name, model_id, temperature_id, condition)]["relative_percent"]
                    for temperature_id, _label in TEMPERATURES
                ]
                ax.bar(
                    [position + offsets[condition_index] for position in x_positions],
                    values,
                    width=bar_width,
                    color=CONDITION_COLORS[condition],
                    edgecolor="#333333",
                    linewidth=0.8,
                    hatch=CONDITION_HATCHES[condition],
                    zorder=3,
                )

            ax.axhline(
                100.0,
                color=HUMAN_COLOR,
                linestyle="--",
                linewidth=1.8,
                zorder=4,
            )
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9)
            ax.xaxis.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            y_min, y_max = METRIC_Y_LIMITS[metric_name]
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(list(range(y_min, y_max + 1, 25)))
            ax.yaxis.set_major_formatter(FuncFormatter(percent_label))
            ax.set_xticks(x_positions, [label for _temp_id, label in TEMPERATURES])
            show_temperature_labels = metric_name == "FH Index"
            if show_temperature_labels:
                ax.tick_params(
                    axis="x",
                    labelbottom=True,
                    labelsize=AXIS_FONT_SIZE,
                    pad=6,
                    labelrotation=0,
                )
                for tick_label in ax.get_xticklabels():
                    tick_label.set_horizontalalignment("center")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="y", labelsize=AXIS_FONT_SIZE)

            if metric_index == 0:
                ax.set_title(model_name, fontsize=MODEL_FONT_SIZE, fontweight="bold", color="#2c3e50")
            if model_index == 0:
                ax.set_ylabel(metric_name, fontsize=AXIS_FONT_SIZE, fontweight="bold", labelpad=10)

    legend = [
        Patch(
            facecolor=CONDITION_COLORS[condition],
            edgecolor="#333333",
            hatch=CONDITION_HATCHES[condition],
            label=CONDITION_LABELS[condition],
        )
        for condition in CONDITIONS
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            color=HUMAN_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Human baseline",
        )
    )

    fig.suptitle(
        "Syntactic-Lexical Complexity (D1)",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.supxlabel("Temperature", fontsize=AXIS_TITLE_FONT_SIZE, fontweight="bold", y=0.025)
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.87,
        bottom=0.08,
        wspace=0.13,
        hspace=0.28,
    )

    png_path = OUTPUT_DIR / "d1_barras.png"
    pdf_path = OUTPUT_DIR / "d1_barras.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png_output, pdf_output = generate_bar_chart()
    print(f"Generated: {png_output}")
    print(f"Generated: {pdf_output}")
