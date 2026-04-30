import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE = 'mtld_results.json'
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
    """Loads raw JSON data from the specified path."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataframe(json_data, metric_key):
    """Processes JSON data to calculate means across sources and extract human baselines."""
    rows = []
    baseline_values = []
    sources = [k for k, v in json_data.items() if "llm_models" in v]
    
    for source in sources:
        content = json_data[source]
        
        # Collect human reference values for later averaging
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
        
    # Group results by Model, Temperature, and Configuration to calculate the MEAN
    averaged_df = raw_df.groupby(["ModelName", "TempID", "Group", "Configuration"])["Value"].mean().reset_index()
    
    # Sort data by groups to ensure consistent X-axis ordering
    averaged_df = averaged_df.sort_values(by=["Group", "Configuration"])
    
    # Calculate the global average for the human baseline
    mean_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
    
    return averaged_df, mean_baseline

# =====================================================
# 3. STYLE & PLOTTING
# =====================================================
def apply_custom_style(ax, df_source, baseline_val, ylabel, y_min=None, y_max=None):
    """Applies visual styling, axes formatting, and unified legends to the plot."""
    ax.grid(False) 
    ax.set_axisbelow(True)
    
    # Enable horizontal grid lines
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=1)
    ax.xaxis.grid(False)

    # Render bar plots
    sns.barplot(
        data=df_source, x="Group", y="Value", hue="Configuration",
        palette=PALETTE, ax=ax, edgecolor='black', linewidth=1, zorder=3
    )

    unique_models = df_source["ModelName"].unique()
    temps_per_model = df_source["TempID"].unique()
    n_temps = len(temps_per_model)
    
    # Configure primary X-axis labels (Temperatures)
    tick_labels = [TEMP_MAP.get(t, t) for t in temps_per_model] * len(unique_models)
    ax.set_xticks(range(len(tick_labels)))
    
    # Slanted X-axis labels with padding
    ax.set_xticklabels(tick_labels, fontsize=26, rotation=30, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', pad=6)
    ax.set_xlabel("")

    # Add secondary X-axis labels for Model Names and vertical delimiters
    for idx, model_name in enumerate(unique_models):
        center = (idx * n_temps + (idx + 1) * n_temps - 1) / 2
        ax.text(center, -0.18, model_name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')
        if idx < len(unique_models) - 1:
            # Vertical lines to differentiate between models
            ax.axvline(idx * n_temps + n_temps - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Add horizontal reference for human baseline
    if baseline_val > 0:
        ax.axhline(y=baseline_val, color=COLOR_HUMAN, linestyle='--', linewidth=2, zorder=4)

    # Y-axis tick labels font size
    ax.tick_params(axis='y', labelsize=26)
    
    # Y-axis label styling with padding
    ax.set_ylabel(ylabel, fontsize=26, fontweight='bold', labelpad=15)
    
    # Set fixed Y-axis limits if provided
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    else:
        ax.margins(y=0.15) 

    # --- MAIN LEGEND ---
    handles, labels = ax.get_legend_handles_labels()
    human_line = mlines.Line2D([], [], color=COLOR_HUMAN, linestyle='--', linewidth=2, label='Human baseline')
    handles.append(human_line)
    
    # CAMBIO REALIZADO: loc='upper left' a loc='upper right'
    main_legend = ax.legend(handles=handles, loc='upper right', fontsize=26, frameon=True, edgecolor='#cccccc')
    main_legend.set_zorder(10)
    
    # Ensure main legend stays on top
    ax.add_artist(main_legend)

def generate_report(metric_key, ylabel, filename, y_min=None, y_max=None):
    """Orchestrates data preparation and figure saving for a specific metric."""
    df, mean_baseline = prepare_dataframe(loaded_data, metric_key)
    if df.empty: 
        print(f"No data found for metric: {metric_key}")
        return

    # Match the (10, 9) size from the previous scripts
    fig, ax = plt.subplots(figsize=(10, 9))
    
    apply_custom_style(ax, df, mean_baseline, ylabel, y_min, y_max)

    # Match layout and margins from the previous scripts
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), bbox_inches='tight')
    print(f"Successfully generated: {filename}")
    plt.close()

# =====================================================
# 4. EXECUTION
# =====================================================
if __name__ == "__main__":
    loaded_data = load_data(INPUT_FILE)
    if loaded_data:
        sns.set_theme(style="white")
        # Generate the MTLD plot with specific Y-axis range
        generate_report("mean_mtld", "MTLD (words)", "mtld", y_min=50, y_max=110)