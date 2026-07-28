import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE_RAG = '../coherence_RAG_results.json'
INPUT_FILE_NO_RAG = '../coherence_NO_RAG_results.json'
OUTPUT_DIR = './results_geval'

MODEL_NAMES = {
    "1": "Ministral-3B", 
    "2": "Qwen-2.5-0.5B", 
    "3": "Llama-3.2-1B"
}

TEMP_MAP = {
    "1": "t = 1",
    "2": "t = 0.75",
    "3": "t = 0.5"
}

COLOR_NO_RAG = '#5D9BCF'
COLOR_RAG    = '#D38E45'
COLOR_GRID   = '#E0E0E0'

PALETTE = {
    "NO RAG": COLOR_NO_RAG,
    "RAG": COLOR_RAG
}
HUE_ORDER = ["NO RAG", "RAG"]

# =====================================================
# 2. DATA LOADING
# =====================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_ci95_from_distribution(distribution, sample_size, confidence=0.95, decimals=3):
    """Rebuilds a 95% CI from score frequencies when the JSON has no ci95 field."""
    if not distribution or sample_size is None or sample_size < 2:
        return {"lower": None, "upper": None}

    score_counts = {
        int(score): int(count)
        for score, count in distribution.items()
    }
    effective_n = sum(score_counts.values())

    if effective_n < 2:
        return {"lower": None, "upper": None}

    sum_values = sum(score * count for score, count in score_counts.items())
    sum_sq_values = sum((score ** 2) * count for score, count in score_counts.items())
    mean_val = sum_values / effective_n
    variance_numerator = sum_sq_values - effective_n * (mean_val ** 2)
    sample_variance = max(variance_numerator / (effective_n - 1), 0.0)
    sem = (sample_variance / effective_n) ** 0.5
    alpha = 1 - confidence
    margin = t.ppf(1 - alpha / 2, df=effective_n - 1) * sem

    return {
        "lower": round(mean_val - margin, decimals),
        "upper": round(mean_val + margin, decimals)
    }

# =====================================================
# 3. PREPARE DATAFRAME (MEAN)
# =====================================================
def prepare_dataframe(rag_data, no_rag_data):
    rows = []

    def process(data, label):
        for source, models in data.items():
            for model_id, temps in models.items():
                model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")

                for temp_id, values in temps.items():
                    ci_data = values.get("ci95_coherence")
                    if not ci_data or ci_data.get("lower") is None or ci_data.get("upper") is None:
                        ci_data = compute_ci95_from_distribution(
                            values.get("dist", {}),
                            values.get("samples", 0)
                        )
                    rows.append({
                        "ModelName": model_name,
                        "TempID": temp_id,
                        "Group": f"{model_id}_{temp_id}",
                        "Configuration": label,
                        "Value": values["avg_coherence"],
                        "CILower": ci_data.get("lower"),
                        "CIUpper": ci_data.get("upper")
                    })

    process(rag_data, "RAG")
    process(no_rag_data, "NO RAG")

    raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        return raw_df

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

    return df_mean

# =====================================================
# 4. STYLE & PLOTTING FUNCTION
# =====================================================
def apply_custom_style(ax, df_data):
    ax.set_axisbelow(True)

    # Grid
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1)
    ax.xaxis.grid(False)

    # Barplot
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
        legend=False
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
            yerr = [[max(0, value - lower)], [max(0, upper - value)]]
            ax.errorbar(
                x_center, value, yerr=yerr, fmt='none',
                ecolor='black', elinewidth=1.5, capsize=4, capthick=1.5,
                zorder=5
            )

    unique_models = df_data["ModelName"].drop_duplicates().tolist()
    temps_per_model = ["1", "2", "3"]
    n_temps = len(temps_per_model)

    tick_labels = [TEMP_MAP[t] for t in temps_per_model] * len(unique_models)

    # X ticks
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

    # Model labels
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

    # Y axis
    ax.set_ylabel("Coherence", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)

    # IMPORTANTE: zoom en rango útil
    ax.set_ylim(1.0, 5.1)

# =====================================================
# 5. GENERATE REPORT
# =====================================================
def generate_report():
    rag_data = load_data(INPUT_FILE_RAG)
    no_rag_data = load_data(INPUT_FILE_NO_RAG)

    if not rag_data or not no_rag_data:
        print("Missing data.")
        return

    df_mean = prepare_dataframe(rag_data, no_rag_data)

    if df_mean.empty:
        print("No data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    apply_custom_style(ax, df_mean)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.savefig(os.path.join(OUTPUT_DIR, "geval_coherence_V2.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "geval_coherence_V2.pdf"), format='pdf', bbox_inches='tight')

    print(f"Successfully generated G-Eval plot in {OUTPUT_DIR}/")
    plt.close()

# =====================================================
# EXECUTION
# =====================================================
if __name__ == "__main__":
    sns.set_theme(style="white")
    generate_report()
