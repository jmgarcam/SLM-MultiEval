import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE = 'sbert_results.json'
OUTPUT_DIR = './graphs'

MODEL_NAMES = {
    "1": "Ministral-3B", 
    "2": "Qwen-2.5-0.5B", 
    "3": "Llama-3.2-1B"
}
TEMP_MAP = {"1": "t = 1", "2": "t = 0.75", "3": "t = 0.5"}

COLOR_NO_RAG = '#5D9BCF'
COLOR_RAG    = '#D38E45'
COLOR_HUMAN  = '#008000'
COLOR_GRID   = '#E0E0E0'

PALETTE = {"NO RAG": COLOR_NO_RAG, "RAG": COLOR_RAG}
HUE_ORDER = ["NO RAG", "RAG"]

# =====================================================
# 2. DATA LOADING
# =====================================================
def load_data(filepath):
    """Loads JSON data from the specified path."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# =====================================================
# 3. PREPARE DATAFRAME (MEAN)
# =====================================================
def prepare_dataframe(json_data):
    """Parses JSON data to calculate averages across sources and human baselines."""
    rows = []
    baseline_values = []
    baseline_ci_lowers = []
    baseline_ci_uppers = []

    sources = sorted([k for k, v in json_data.items() if "llm_models" in v])

    for source in sources:
        content = json_data[source]
        baseline_block = content.get("human_baseline", {})
        base_val = baseline_block.get("mean_sbert", 0)
        if base_val > 0:
            baseline_values.append(base_val)
        baseline_ci = baseline_block.get("ci95_sbert", {})
        if baseline_ci.get("lower") is not None:
            baseline_ci_lowers.append(baseline_ci["lower"])
        if baseline_ci.get("upper") is not None:
            baseline_ci_uppers.append(baseline_ci["upper"])

        llm_models = content.get("llm_models", {})
        for model_id, features in llm_models.items():
            model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")

            for temp_id, features_data in features.items():
                try:
                    rag_block = features_data["RAG"]
                    no_rag_block = features_data["NO_RAG"]
                    rag_val = rag_block["mean_sbert"]
                    no_rag_val = no_rag_block["mean_sbert"]

                    rows.append({
                        "ModelName": model_name,
                        "TempID": temp_id,
                        "Group": f"{model_id}_{temp_id}",
                        "Configuration": "NO RAG",
                        "Value": no_rag_val,
                        "CILower": no_rag_block.get("ci95_sbert", {}).get("lower"),
                        "CIUpper": no_rag_block.get("ci95_sbert", {}).get("upper")
                    })

                    rows.append({
                        "ModelName": model_name,
                        "TempID": temp_id,
                        "Group": f"{model_id}_{temp_id}",
                        "Configuration": "RAG",
                        "Value": rag_val,
                        "CILower": rag_block.get("ci95_sbert", {}).get("lower"),
                        "CIUpper": rag_block.get("ci95_sbert", {}).get("upper")
                    })
                except KeyError:
                    continue

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, {"mean": 0, "lower": None, "upper": None}

    df_mean = raw_df.groupby(
        ["ModelName", "TempID", "Group", "Configuration"], as_index=False
    ).agg({
        "Value": "mean",
        "CILower": "mean",
        "CIUpper": "mean"
    })

    df_mean['ModelName'] = pd.Categorical(
        df_mean['ModelName'], 
        categories=["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"], 
        ordered=True
    )
    df_mean = df_mean.sort_values(['ModelName', 'TempID'])

    baseline_stats = {
        "mean": sum(baseline_values) / len(baseline_values) if baseline_values else 0,
        "lower": (
            sum(baseline_ci_lowers) / len(baseline_ci_lowers)
            if baseline_ci_lowers else None
        ),
        "upper": (
            sum(baseline_ci_uppers) / len(baseline_ci_uppers)
            if baseline_ci_uppers else None
        )
    }
    return df_mean, baseline_stats

# =====================================================
# 4. STYLE & PLOTTING FUNCTION
# =====================================================
def apply_custom_style(ax, df_data, baseline_stats):
    """Applies specific styling, labels, and legends to the provided axis."""
    ax.set_axisbelow(True)
    # Horizontal grid only
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1)
    ax.xaxis.grid(False)

    sns.barplot(
        data=df_data,
        x="Group",
        y="Value",
        hue="Configuration",
        palette=PALETTE,
        ax=ax,
        edgecolor='black',
        linewidth=1,
        hue_order=HUE_ORDER,
        errorbar=None,
        legend=False  # Se desactiva la leyenda automática de Seaborn
    )

    x_order = df_data["Group"].drop_duplicates().tolist()
    for container, configuration in zip(ax.containers, HUE_ORDER):
        config_rows = (
            df_data[df_data["Configuration"] == configuration]
            .set_index("Group")
            .reindex(x_order)
        )

        for patch, (_, row) in zip(container.patches, config_rows.iterrows()):
            lower = row.get("CILower")
            upper = row.get("CIUpper")
            value = row.get("Value")

            if pd.isna(lower) or pd.isna(upper) or pd.isna(value):
                continue

            x_center = patch.get_x() + patch.get_width() / 2
            yerr = [
                [max(0, value - lower)],
                [max(0, upper - value)]
            ]
            ax.errorbar(
                x_center, value, yerr=yerr, fmt='none',
                ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5,
                zorder=5
            )

    unique_models = df_data["ModelName"].drop_duplicates().tolist()
    temps_per_model = df_data["TempID"].drop_duplicates().tolist()
    n_temps = len(temps_per_model)

    tick_labels = [TEMP_MAP.get(t, t) for t in temps_per_model] * len(unique_models)

    # --- X-Axis (Temperatures) ---
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(
        tick_labels, 
        fontsize=26, 
        rotation=30, 
        ha='right', 
        rotation_mode='anchor'
    )
    ax.tick_params(axis='x', pad=6)
    ax.set_xlabel("")

    # --- X-Axis (Secondary: Model Names) ---
    for idx, model_name in enumerate(unique_models):
        center = (idx * n_temps + (idx + 1) * n_temps - 1) / 2
        ax.text(center, -0.18,
                model_name,
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=26,
                fontweight='bold',
                color='#2c3e50')

        if idx < len(unique_models) - 1:
            ax.axvline(idx * n_temps + n_temps - 0.5,
                        color='gray',
                        linestyle=':',
                        alpha=0.5)

    # --- Human Baseline ---
    baseline_val = baseline_stats.get("mean", 0)
    baseline_lower = baseline_stats.get("lower")
    baseline_upper = baseline_stats.get("upper")
    if baseline_val > 0:
        if baseline_lower is not None and baseline_upper is not None:
            ax.axhspan(ymin=baseline_lower, ymax=baseline_upper,
                       color=COLOR_HUMAN,
                       alpha=0.12,
                       zorder=2)
        ax.axhline(y=baseline_val,
                    color=COLOR_HUMAN,
                    linestyle='--',
                    linewidth=2,
                    zorder=6)

    # --- Y-Axis ---
    ax.set_ylabel("SBERTScore", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(0.2, 0.8)

# =====================================================
# 5. GENERATE REPORT
# =====================================================
def generate_report():
    """Generates and saves the SBERTScore mean plot."""
    df_mean, baseline_stats = prepare_dataframe(loaded_data)

    if df_mean.empty:
        print("No data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    apply_custom_style(ax, df_mean, baseline_stats)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.savefig(os.path.join(OUTPUT_DIR, "sbert.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "sbert.pdf"), format='pdf', bbox_inches='tight')

    print(f"Successfully generated SBERT plot in {OUTPUT_DIR}/")
    plt.close()

# =====================================================
# EXECUTION
# =====================================================
if __name__ == "__main__":
    loaded_data = load_data(INPUT_FILE)

    if loaded_data:
        sns.set_theme(style="white")
        generate_report()
