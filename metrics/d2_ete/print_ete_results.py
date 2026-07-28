import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE = 'ete_results.json'
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
# 2. DATA LOADING & PROCESSING
# =====================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataframe(json_data):
    rows = []
    baseline_values = []
    baseline_ci_lowers = []
    baseline_ci_uppers = []
    sources = [k for k, v in json_data.items() if "llm_models" in v]
    
    for source in sources:
        content = json_data[source]
        base_val = content.get("real_reference_metrics", {}).get("mean_entity_entropy", 0)
        if base_val > 0:
            baseline_values.append(base_val)
        baseline_ci = content.get("real_reference_metrics", {}).get("ci95_entity_entropy", {})
        if baseline_ci.get("lower") is not None:
            baseline_ci_lowers.append(baseline_ci["lower"])
        if baseline_ci.get("upper") is not None:
            baseline_ci_uppers.append(baseline_ci["upper"])
            
        llm_models = content.get("llm_models", {})
        for model_id, features in llm_models.items():
            model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")
            for temp_id, strategies in features.items():
                try:
                    no_rag_block = strategies["NO_RAG"]
                    rag_block = strategies["RAG"]
                    rows.append({
                        "ModelName": model_name, "TempID": temp_id, "Group": f"{model_id}_{temp_id}", 
                        "Configuration": "NO RAG", "Value": float(no_rag_block["mean_entity_entropy"]),
                        "CILower": no_rag_block.get("ci95_entity_entropy", {}).get("lower"),
                        "CIUpper": no_rag_block.get("ci95_entity_entropy", {}).get("upper")
                    })
                    rows.append({
                        "ModelName": model_name, "TempID": temp_id, "Group": f"{model_id}_{temp_id}", 
                        "Configuration": "RAG", "Value": float(rag_block["mean_entity_entropy"]),
                        "CILower": rag_block.get("ci95_entity_entropy", {}).get("lower"),
                        "CIUpper": rag_block.get("ci95_entity_entropy", {}).get("upper")
                    })
                except (KeyError, TypeError): continue
                    
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
    df_mean['ModelName'] = pd.Categorical(df_mean['ModelName'], categories=MODEL_NAMES.values(), ordered=True)
    df_mean = df_mean.sort_values(['ModelName', 'TempID'])
    baseline_stats = {
        "mean": (sum(baseline_values) / len(baseline_values) if baseline_values else 0),
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
# 3. STYLE & PLOTTING
# =====================================================
def apply_custom_style(ax, df_data, baseline_stats):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=1)
    ax.xaxis.grid(False)

    sns.barplot(
        data=df_data, x="Group", y="Value", hue="Configuration",
        palette=PALETTE, ax=ax, edgecolor='black', linewidth=1, zorder=3,
        hue_order=HUE_ORDER, errorbar=None
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

    unique_models = df_data["ModelName"].unique()
    temps_per_model = df_data["TempID"].unique()
    n_temps = len(temps_per_model)
    
    # Primary X-axis (Temperatures)
    tick_labels = [TEMP_MAP.get(t, t) for t in temps_per_model] * len(unique_models)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=26, rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', pad=6)
    ax.set_xlabel("")

    # Y-axis
    ax.set_ylabel("ETE (b)", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(0.2, 1.2)

    # Model Names and Separators
    for idx, model_name in enumerate(unique_models):
        center = (idx * n_temps + (idx + 1) * n_temps - 1) / 2
        ax.text(center, -0.18, model_name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')
        if idx < len(unique_models) - 1:
            ax.axvline(idx * n_temps + n_temps - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Human Baseline
    baseline_val = baseline_stats.get("mean", 0)
    baseline_lower = baseline_stats.get("lower")
    baseline_upper = baseline_stats.get("upper")
    if baseline_val > 0:
        if baseline_lower is not None and baseline_upper is not None:
            ax.axhspan(baseline_lower, baseline_upper, color=COLOR_HUMAN, alpha=0.12, zorder=6)
        ax.axhline(y=baseline_val, color=COLOR_HUMAN, linestyle='--', linewidth=2, zorder=7)

    # LEYENDA DENTRO ARRIBA A LA DERECHA
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove automatic "Configuration" label if present
    filtered_handles, filtered_labels = [], []
    for h, l in zip(handles, labels):
        if l != "Configuration":
            filtered_handles.append(h)
            filtered_labels.append(l)

    # Add Human baseline reference
    human_line = mlines.Line2D([], [], color=COLOR_HUMAN, linestyle='--', linewidth=2, label='Human baseline')
    filtered_handles.append(human_line)
    filtered_labels.append('Human baseline')

    ax.legend(
        handles=filtered_handles,
        labels=filtered_labels,
        loc='upper right', # Posición interna
        bbox_to_anchor=(0.99, 0.98), # Ajuste preciso en la esquina
        fontsize=26, 
        frameon=True, 
        facecolor='white',
        framealpha=1.0, # Fondo opaco para legibilidad
        edgecolor='#cccccc',
        title=""
    )

# =====================================================
# 4. MAIN EXECUTION
# =====================================================
def generate_report():
    data = load_data(INPUT_FILE)
    if not data: return
    df, baseline_stats = prepare_dataframe(data)
    if df.empty: return

    fig, ax = plt.subplots(figsize=(10, 9))
    apply_custom_style(ax, df, baseline_stats)

    plt.tight_layout()
    # Bottom adjustment for model names
    plt.subplots_adjust(bottom=0.20) 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ete.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "ete.pdf"), bbox_inches='tight')
    print(f"ETE chart generated successfully in {OUTPUT_DIR}")
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="white")
    generate_report()
