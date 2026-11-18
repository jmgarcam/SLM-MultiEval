import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "nli_results" / "nli_llm_results.json"

OUTPUT_DIR = BASE_DIR / "results_nli_llm_one_graph"
OUTPUT_FILENAME = "nli_llm_V5"

COLOR_PALETTE = {
    "NO RAG": "#e74c3c",
    "RAG": "#2ecc71",
    "REAL": "#3498db"
}

MARKER_MAP = {
    "Ministral-3B": "o",
    "Qwen-2.5-0.5B": "s",
    "Llama-3.2-1B": "^",
    "REAL": "D"
}

TEMP_CONFIG = {
    "1": {"label": "t=1", "size": 150},
    "2": {"label": "t=0.75", "size": 350},
    "3": {"label": "t=0.5", "size": 600}
}

MODEL_ID_MAP = {
    "1": "Ministral-3B",
    "2": "Qwen-2.5-0.5B",
    "3": "Llama-3.2-1B"
}


def load_softmax_means(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rows = []

    for condition_key, newspapers in raw_data.items():
        condition_label = condition_key.replace("_", " ")

        for newspaper, llm_dict in newspapers.items():
            for llm_id, feature_dict in llm_dict.items():

                if condition_label == "REAL":
                    model_name = "REAL"
                else:
                    model_name = MODEL_ID_MAP.get(str(llm_id), f"Model {llm_id}")

                for feature_id, metrics in feature_dict.items():

                    temp_config = TEMP_CONFIG.get(
                        str(feature_id),
                        {"label": f"feat_{feature_id}", "size": 150}
                    )

                    rows.append({
                        "Model": model_name,
                        "Condition": condition_label,
                        "Newspaper": newspaper,
                        "Temp_Label": temp_config["label"],
                        "Size_Value": temp_config["size"],
                        "Entailment": metrics.get("mean_entailment_softmax", 0),
                        "Contradiction": metrics.get("mean_contradiction_softmax", 0),
                    })

    return pd.DataFrame(rows)


def build_legend(ax, df_plot):
    legend_elements = []

    # Condition (excluding REAL)
    for label, color in COLOR_PALETTE.items():
        if label == "REAL":
            continue
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=14,
                linestyle="None"
            )
        )

    # REAL (single entry)
    legend_elements.append(
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            label="REAL",
            markerfacecolor=COLOR_PALETTE["REAL"],
            markeredgecolor="black",
            markersize=14,
            linestyle="None"
        )
    )

    # Model (excluding REAL)
    for label, marker in MARKER_MAP.items():
        if label == "REAL":
            continue
        legend_elements.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="black",
                label=label,
                linestyle="None",
                markersize=14
            )
        )

    # Temperature
    for temp_id in ["1", "2", "3"]:
        conf = TEMP_CONFIG[temp_id]
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="gray",
                label=conf["label"],
                linestyle="None",
                markersize=(conf["size"] ** 0.5) / 2
            )
        )

    available_labels = (
        set(df_plot["Condition"])
        | set(df_plot["Model"])
        | set(df_plot["Temp_Label"])
    )
    legend_elements = [
        element
        for element in legend_elements
        if element.get_label() in available_labels
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=20,
        frameon=True
    )


def main():
    df = load_softmax_means(JSON_FILE)

    if df.empty:
        print("No data available")
        return

    # REAL average across newspapers
    df_real = df[df["Condition"] == "REAL"]

    df_real_avg = (
        df_real.groupby(["Condition"])
        .mean(numeric_only=True)
        .reset_index()
    )

    df_real_avg["Model"] = "REAL"
    df_real_avg["Temp_Label"] = "REAL"
    df_real_avg["Size_Value"] = 800

    # RAG / NO RAG
    df_other = df[df["Condition"] != "REAL"]

    df_other_avg = (
        df_other.groupby(["Model", "Condition", "Temp_Label", "Size_Value"])
        .mean(numeric_only=True)
        .reset_index()
    )

    df_plot = pd.concat([df_other_avg, df_real_avg], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, row in df_plot.iterrows():
        ax.scatter(
            row["Contradiction"],
            row["Entailment"],
            c=COLOR_PALETTE.get(row["Condition"], "#7f8c8d"),
            marker=MARKER_MAP.get(row["Model"], "o"),
            s=row["Size_Value"],
            alpha=0.95,
            edgecolors="black",
            linewidths=1.5 if row["Condition"] == "REAL" else 0.8,
            zorder=4 if row["Condition"] == "REAL" else 3
        )

    ax.set_xlabel(r"$NLI_C$", fontsize=28, labelpad=12)
    ax.set_ylabel(r"$NLI_E$", fontsize=28, labelpad=12)

    ax.tick_params(axis="both", which="major", labelsize=28)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    yticks = ax.get_yticks()
    yticks = [y for y in yticks if y != 0]
    ax.set_yticks(yticks)

    ax.axhline(0, color="black", linewidth=1, alpha=0.5, zorder=2)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5, zorder=2)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

    build_legend(ax, df_plot)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME + ".pdf"), bbox_inches="tight")

    print("Plot generated successfully")


if __name__ == "__main__":
    main()
