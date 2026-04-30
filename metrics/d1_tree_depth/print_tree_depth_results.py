import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE = 'tree_depth_results.json'
OUTPUT_DIR = './graphs'

MODEL_NAMES = {"1": "Ministral-3B", "2": "Qwen-2.5-0.5B", "3": "Llama-3.2-1B"}
TEMP_MAP = {"1": "t = 1", "2": "t = 0.75", "3": "t = 0.5"}

COLOR_NO_RAG = '#5D9BCF'
COLOR_RAG    = '#D38E45'
COLOR_HUMAN  = '#008000'
COLOR_GRID   = '#E0E0E0'

PALETTE = {"NO RAG": COLOR_NO_RAG, "RAG": COLOR_RAG}

# =====================================================
# 2. DATA LOADING & PROCESSING
# =====================================================
def load_data(filepath):
    """Loads JSON data from the specified filepath."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataframe(json_data, metric_key):
    """Parses raw JSON data, extracts values across all sources, and returns a DataFrame."""
    rows = []
    baseline_values = []
    sources = [k for k, v in json_data.items() if "llm_models" in v]
    
    for source in sources:
        content = json_data[source]
        baseline_val = content.get("real_reference_metrics", {}).get(metric_key)
        if baseline_val is not None:
            baseline_values.append(baseline_val)
        
        llm_models = content.get("llm_models", {})
        for model_id, features in llm_models.items():
            model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")
            for temp_id, features_data in features.items():
                try:
                    rag_val = features_data["RAG"][metric_key]
                    no_rag_val = features_data["NO_RAG"][metric_key]
                    
                    rows.append({"ModelName": model_name, "TempID": temp_id, 
                                 "Group": f"{model_id}_{temp_id}", "Configuration": "NO RAG", "Value": no_rag_val})
                    rows.append({"ModelName": model_name, "TempID": temp_id, 
                                 "Group": f"{model_id}_{temp_id}", "Configuration": "RAG", "Value": rag_val})
                except KeyError:
                    continue
                    
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, 0
        
    averaged_df = raw_df.groupby(["ModelName", "TempID", "Group", "Configuration"])["Value"].mean().reset_index()
    averaged_df = averaged_df.sort_values(by=["Group", "Configuration"])
    mean_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
    
    return averaged_df, mean_baseline

# =====================================================
# 3. STYLE & PLOTTING
# =====================================================
def apply_custom_style(ax, df_source, baseline_val, ylabel, y_min=None, y_max=None):
    """Applies custom styling to the axis."""
    ax.grid(False) 
    ax.set_axisbelow(True)
    
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=1)
    ax.xaxis.grid(False)

    # Plot the bars
    sns.barplot(
        data=df_source, x="Group", y="Value", hue="Configuration",
        palette=PALETTE, ax=ax, edgecolor='black', linewidth=1, zorder=3,
        legend=False
    )

    unique_models = df_source["ModelName"].unique()
    temps_per_model = df_source["TempID"].unique()
    n_temps = len(temps_per_model)
    
    tick_labels = [TEMP_MAP.get(t, t) for t in temps_per_model] * len(unique_models)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=26, rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', pad=6)
    ax.set_xlabel("")

    for idx, model_name in enumerate(unique_models):
        center = (idx * n_temps + (idx + 1) * n_temps - 1) / 2
        ax.text(center, -0.18, model_name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')
        if idx < len(unique_models) - 1:
            ax.axvline(idx * n_temps + n_temps - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

    if baseline_val > 0:
        ax.axhline(y=baseline_val, color=COLOR_HUMAN, linestyle='--', linewidth=2, zorder=4)

    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylabel(ylabel, fontsize=26, fontweight='bold', labelpad=15)
    
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    else:
        ax.margins(y=0.15) 

    if "Tree Depth" in ylabel:
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

# =====================================================
# 4. EXECUTION
# =====================================================
def generate_report(metric_key, ylabel, filename, y_min=None, y_max=None):
    df, mean_baseline = prepare_dataframe(loaded_data, metric_key)
    if df.empty: return

    fig, ax = plt.subplots(figsize=(10, 9))
    apply_custom_style(ax, df, mean_baseline, ylabel, y_min, y_max)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), bbox_inches='tight')
    print(f"Successfully generated: {filename}")
    plt.close()

if __name__ == "__main__":
    loaded_data = load_data(INPUT_FILE)
    if loaded_data:
        sns.set_theme(style="white")
        generate_report("mean_depth", "Tree Depth (levels)", "DT", y_min=5, y_max=8)