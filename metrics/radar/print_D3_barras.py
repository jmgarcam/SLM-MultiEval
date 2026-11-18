"""Create a grouped-bar replacement for the D3 radar charts.

The figure contains every D3 combination:
4 metrics x 3 models x 3 temperatures x 2 conditions = 72 bars.
Scores are normalized against the empirical human baseline, which is 100%.
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
NLI_FILE = (
    SCRIPT_DIR.parent
    / 'nli_llm'
    / 'new_journal'
    / 'nli_results'
    / 'nli_llm_results.json'
)
FACT_RAG_FILE = SCRIPT_DIR.parent / "factual_llm" / "factual_llm_RAG_results.json"
FACT_NO_RAG_FILE = (
    SCRIPT_DIR.parent / "factual_llm" / "factual_llm_NO_RAG_results.json"
)
FACT_REAL_FILE = SCRIPT_DIR.parent / "factual_llm" / "factual_llm_REAL_results.json"
OUTPUT_DIR = SCRIPT_DIR / "D3"

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
METRICS = ("NLI_E", "1 - NLI_C", "Correct", "1 - Incorrect")
METRIC_LABELS = {
    "NLI_E": r"NLI$_{\mathrm{E}}$",
    "1 - NLI_C": r"1 - NLI$_{\mathrm{C}}$",
    "Correct": "Correct",
    "1 - Incorrect": "1 - Incorrect",
}
METRIC_Y_LIMITS = {
    "NLI_E": (0, 125),
    "1 - NLI_C": (25, 125),
    "Correct": (0, 125),
    "1 - Incorrect": (25, 125),
}

HUMAN_COLOR = "#008000"
GRID_COLOR = "#E0E0E0"
AXIS_FONT_SIZE = 14.3
AXIS_TITLE_FONT_SIZE = 15.4
NLI_LABEL_FONT_SIZE = AXIS_FONT_SIZE * 1.2
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


def process_d3_data() -> dict:
    """Aggregate D3 scores and normalize them to the empirical human means."""

    nli = load_json(NLI_FILE)
    factual_sources = {
        "RAG": load_json(FACT_RAG_FILE),
        "NO_RAG": load_json(FACT_NO_RAG_FILE),
    }
    factual_real = load_json(FACT_REAL_FILE)
    results = {}

    if "REAL" not in nli:
        raise KeyError(f"Missing condition REAL in {NLI_FILE}")

    nli_real_blocks = [
        models["REAL"]["1"]
        for models in nli["REAL"].values()
        if "REAL" in models and "1" in models["REAL"]
    ]
    if not nli_real_blocks:
        raise ValueError(f"No empirical NLI baseline found in {NLI_FILE}")

    factual_real_ratios = []
    for models in factual_real.values():
        if "REAL" not in models or "1" not in models["REAL"]:
            continue
        block = models["REAL"]["1"]
        total = block["correct"] + block["incorrect"] + block["partially_correct"]
        if total <= 0:
            continue
        factual_real_ratios.append(
            (block["correct"] / total, 1.0 - block["incorrect"] / total)
        )
    if not factual_real_ratios:
        raise ValueError(f"No empirical factual baseline found in {FACT_REAL_FILE}")

    human_baselines = {
        "NLI_E": fmean(
            block["mean_entailment_softmax"] for block in nli_real_blocks
        ),
        "1 - NLI_C": 1.0
        - fmean(block["mean_contradiction_softmax"] for block in nli_real_blocks),
        "Correct": fmean(value[0] for value in factual_real_ratios),
        "1 - Incorrect": fmean(value[1] for value in factual_real_ratios),
    }
    for metric_name, baseline in human_baselines.items():
        if baseline <= 0:
            raise ValueError(
                f"The empirical human baseline for {metric_name} must be positive"
            )

    for condition in CONDITIONS:
        if condition not in nli:
            raise KeyError(f"Missing condition {condition} in {NLI_FILE}")
        factual = factual_sources[condition]

        for model_id, model_name in MODELS:
            for temperature_id, temperature_label in TEMPERATURES:
                nli_blocks = [
                    models[model_id][temperature_id]
                    for models in nli[condition].values()
                    if model_id in models and temperature_id in models[model_id]
                ]
                factual_blocks = [
                    models[model_id][temperature_id]
                    for models in factual.values()
                    if model_id in models and temperature_id in models[model_id]
                ]
                if not nli_blocks or not factual_blocks:
                    raise ValueError(
                        "Missing D3 observations for "
                        f"{condition}/model {model_id}/temperature {temperature_id}"
                    )

                entailment = fmean(
                    block["mean_entailment_softmax"] for block in nli_blocks
                )
                contradiction = fmean(
                    block["mean_contradiction_softmax"] for block in nli_blocks
                )

                correct_ratios = []
                incorrect_ratios = []
                for block in factual_blocks:
                    total = (
                        block["correct"]
                        + block["incorrect"]
                        + block["partially_correct"]
                    )
                    if total <= 0:
                        continue
                    correct_ratios.append(block["correct"] / total)
                    incorrect_ratios.append(block["incorrect"] / total)
                if not correct_ratios:
                    raise ValueError(
                        "No valid factual observations for "
                        f"{condition}/model {model_id}/temperature {temperature_id}"
                    )

                metric_values = {
                    "NLI_E": entailment,
                    "1 - NLI_C": 1.0 - contradiction,
                    "Correct": fmean(correct_ratios),
                    "1 - Incorrect": 1.0 - fmean(incorrect_ratios),
                }
                for metric_name, value in metric_values.items():
                    results[(metric_name, model_id, temperature_id, condition)] = {
                        "model": model_name,
                        "temperature": temperature_label,
                        "relative_percent": (
                            100.0 * value / human_baselines[metric_name]
                        ),
                    }

    expected = len(METRICS) * len(MODELS) * len(TEMPERATURES) * len(CONDITIONS)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} values, found {len(results)}")
    return results


def percent_label(value: float, _position: int) -> str:
    return f"{value:.0f}%"


def generate_bar_chart() -> tuple[Path, Path]:
    data = process_d3_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(METRICS),
        len(MODELS),
        figsize=(15, 15),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x_positions = list(range(len(TEMPERATURES)))
    bar_width = 0.36
    offsets = (-bar_width / 2.0, bar_width / 2.0)

    for metric_index, metric_name in enumerate(METRICS):
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
            ax.set_xticks(
                x_positions,
                [label for _temperature_id, label in TEMPERATURES],
            )

            show_temperature_labels = metric_index == len(METRICS) - 1
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
                ax.set_title(
                    model_name,
                    fontsize=MODEL_FONT_SIZE,
                    fontweight="bold",
                    color="#2c3e50",
                )
            if model_index == 0:
                metric_font_size = (
                    NLI_LABEL_FONT_SIZE
                    if metric_name in ("NLI_E", "1 - NLI_C")
                    else AXIS_FONT_SIZE
                )
                ax.set_ylabel(
                    METRIC_LABELS[metric_name],
                    fontsize=metric_font_size,
                    fontweight="bold",
                    labelpad=10,
                )

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
        "Consistency (D3)",
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
    fig.supxlabel(
        "Temperature",
        fontsize=AXIS_TITLE_FONT_SIZE,
        fontweight="bold",
        y=0.025,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.87,
        bottom=0.075,
        wspace=0.13,
        hspace=0.28,
    )

    png_path = OUTPUT_DIR / 'd3_barras_v2.png'
    pdf_path = OUTPUT_DIR / 'd3_barras_v2.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png_output, pdf_output = generate_bar_chart()
    print(f"Generated: {png_output}")
    print(f"Generated: {pdf_output}")
