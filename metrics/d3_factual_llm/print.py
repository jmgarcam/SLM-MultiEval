import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================
# CONFIG
# =====================================================
RAG_FILE = "factual_llm_RAG_results.json"
NO_RAG_FILE = "factual_llm_NO_RAG_results.json"
REAL_FILE = "factual_llm_REAL_results.json"

OUTPUT_DIR = "./results_factual_one_graph"
OUTPUT_FILENAME = "factual_consistency_V4"

MARKER_MAP = {
    "Ministral-3B": "o",
    "Qwen-2.5-0.5B": "s",
    "Llama-3.2-1B": "^"
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

FORCED_DENOMINATORS = {}
IMPUTED_LABEL_COUNTS = {}

CMAP = plt.cm.plasma

# =====================================================
# LOAD DATA
# =====================================================
def load_factual(filepath, condition_label):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rows = []

    for newspaper, models in raw_data.items():
        for llm_id, feature_dict in models.items():
            model_name = MODEL_ID_MAP.get(str(llm_id), f"Model {llm_id}")

            for feature_id, metrics in feature_dict.items():

                temp_conf = TEMP_CONFIG.get(
                    str(feature_id),
                    {"label": f"feat_{feature_id}", "size": 150}
                )

                report_key = (
                    condition_label,
                    str(newspaper),
                    str(llm_id),
                    str(feature_id),
                )
                imputed_counts = IMPUTED_LABEL_COUNTS.get(report_key, {})
                correct_count = metrics["correct"] + imputed_counts.get("correct", 0)
                partial_count = metrics["partially_correct"] + imputed_counts.get(
                    "partially_correct", 0
                )
                incorrect_count = metrics["incorrect"] + imputed_counts.get(
                    "incorrect", 0
                )
                total = FORCED_DENOMINATORS.get(
                    report_key,
                    correct_count + partial_count + incorrect_count,
                )

                if total == 0:
                    continue
                if correct_count + partial_count + incorrect_count != total:
                    raise ValueError(
                        "Reported factual label counts must sum to the "
                        f"denominator: {report_key}"
                    )

                rows.append({
                    "Model": model_name,
                    "Condition": condition_label,
                    "Temp_Label": temp_conf["label"],
                    "Size_Value": temp_conf["size"],
                    "Correct": correct_count / total,
                    "Incorrect": incorrect_count / total,
                    "Partial": partial_count / total
                })

    return pd.DataFrame(rows)


# =====================================================
# LOAD REAL
# =====================================================
def load_real(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rows = []

    for newspaper, models in raw_data.items():
        metrics = models["REAL"]["1"]

        total = (
            metrics["correct"]
            + metrics["partially_correct"]
            + metrics["incorrect"]
        )

        if total == 0:
            continue

        rows.append({
            "Correct": metrics["correct"] / total,
            "Incorrect": metrics["incorrect"] / total,
            "Partial": metrics["partially_correct"] / total
        })

    df = pd.DataFrame(rows)

    return pd.DataFrame([{
        "Model": "REAL",
        "Condition": "REAL",
        "Temp_Label": "REAL",
        "Size_Value": 400,
        "Correct": df["Correct"].mean(),
        "Incorrect": df["Incorrect"].mean(),
        "Partial": df["Partial"].mean()
    }])


# =====================================================
# LEGEND
# =====================================================
def build_legend(ax):
    legend_elements = []

    legend_elements.append(
        Line2D([0], [0],
               marker="o",
               color="black",
               label="RAG",
               markerfacecolor="white",
               linestyle="None",
               markersize=12,
               markeredgewidth=0.5)
    )

    legend_elements.append(
        Line2D([0], [0],
               marker="o",
               color="black",
               label="NO RAG",
               markerfacecolor="white",
               linestyle="None",
               markersize=12,
               markeredgewidth=3.0)
    )

    legend_elements.append(
        Line2D([0], [0],
               marker="D",
               color="black",
               label="REAL",
               markerfacecolor="none",
               linestyle="None",
               markersize=12,
               markeredgewidth=1.5)
    )

    for label, marker in MARKER_MAP.items():
        legend_elements.append(
            Line2D([0], [0],
                   marker=marker,
                   color="black",
                   label=label,
                   linestyle="None",
                   markersize=12)
        )

    for temp_id in ["1", "2", "3"]:
        conf = TEMP_CONFIG[temp_id]
        legend_elements.append(
            Line2D([0], [0],
                   marker="o",
                   color="gray",
                   label=conf["label"],
                   linestyle="None",
                   markersize=(conf["size"] ** 0.5) / 2)
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=16, frameon=True)


# =====================================================
# MAIN
# =====================================================
def main():
    df_rag = load_factual(RAG_FILE, "RAG")
    df_no_rag = load_factual(NO_RAG_FILE, "NO RAG")
    df_real = load_real(REAL_FILE)

    df = pd.concat([df_rag, df_no_rag, df_real], ignore_index=True)

    if df.empty:
        print("No data available")
        return

    df_avg = (
        df.groupby(["Model", "Condition", "Temp_Label", "Size_Value"])
        .mean(numeric_only=True)
        .reset_index()
    )

    vmin = df_avg["Partial"].min()
    vmax = df_avg["Partial"].max()

    fig, ax = plt.subplots(figsize=(12, 8))

    for _, row in df_avg.iterrows():

        is_rag = row["Condition"] == "RAG"
        is_real = row["Condition"] == "REAL"

        ax.scatter(
            row["Incorrect"],
            row["Correct"],
            c=[row["Partial"]],
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            marker="D" if is_real else MARKER_MAP.get(row["Model"], "o"),
            s=800 if is_real else row["Size_Value"],
            edgecolors="black",
            linewidths=1.5 if is_real else (0.5 if is_rag else 3.0),
            zorder=5 if is_real else 3
        )

    # =========================
    # COLORBAR
    # =========================
    cbar_ax = fig.add_axes([0.155, 0.87, 0.57, 0.02])

    cbar = plt.colorbar(
        ax.collections[0],
        cax=cbar_ax,
        orientation="horizontal"
    )

    cbar.set_label("Partially Correct", fontsize=20, labelpad=15)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom') 
    cbar.ax.tick_params(labelsize=22)

    # =========================
    # AXES
    # =========================
    ax.set_xlabel("Incorrect", fontsize=28, labelpad=12)
    ax.set_ylabel("Correct", fontsize=28, labelpad=12)

    ax.tick_params(axis="both", labelsize=28)

    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0.0, 1.2)

    yticks = ax.get_yticks()
    yticks = [y for y in yticks if y > 0.001]
    ax.set_yticks(yticks)

    ax.axvline(0, alpha=0)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)

    build_legend(ax)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME + ".pdf"), bbox_inches="tight")

    print("Plot generated successfully")


if __name__ == "__main__":
    main()
