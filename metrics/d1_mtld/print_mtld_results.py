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
HUE_ORDER = ["NO RAG", "RAG"]

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
    baseline_ci_lowers = []
    baseline_ci_uppers = []
    sources = [k for k, v in json_data.items() if "llm_models" in v]
    ci_key = metric_key.replace("mean_", "ci95_")
    
    for source in sources:
        content = json_data[source]
        
        # Collect human reference values for later averaging
        baseline_val = content.get("real_reference_metrics", {}).get(metric_key)
        baseline_ci = content.get("real_reference_metrics", {}).get(ci_key, {})
        if baseline_val is not None:
            baseline_values.append(baseline_val)
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
                    rag_val = rag_block[metric_key]
                    no_rag_val = no_rag_block[metric_key]
                    rag_ci = rag_block.get(ci_key, {})
                    no_rag_ci = no_rag_block.get(ci_key, {})
                    
                    rows.append({"ModelName": model_name, "TempID": temp_id, 
                                 "Group": f"{model_id}_{temp_id}", "Configuration": "NO RAG", "Value": no_rag_val,
                                 "CILower": no_rag_ci.get("lower"), "CIUpper": no_rag_ci.get("upper")})
                    rows.append({"ModelName": model_name, "TempID": temp_id, 
                                 "Group": f"{model_id}_{temp_id}", "Configuration": "RAG", "Value": rag_val,
                                 "CILower": rag_ci.get("lower"), "CIUpper": rag_ci.get("upper")})
                except KeyError:
                    continue
                    
    raw_df = pd.DataFrame(rows)
    
    if raw_df.empty:
        return raw_df, {"mean": 0, "lower": None, "upper": None}
        
    # Group results by Model, Temperature, and Configuration to calculate the MEAN
    averaged_df = raw_df.groupby(
        ["ModelName", "TempID", "Group", "Configuration"], as_index=False
    ).agg({
        "Value": "mean",
        "CILower": "mean",
        "CIUpper": "mean"
    })
    
    # Sort data by groups to ensure consistent X-axis ordering
    averaged_df = averaged_df.sort_values(by=["Group", "Configuration"])
    
    # Calculate the global average for the human baseline
    mean_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
    baseline_stats = {
        "mean": mean_baseline,
        "lower": (
            sum(baseline_ci_lowers) / len(baseline_ci_lowers)
            if baseline_ci_lowers else None
        ),
        "upper": (
            sum(baseline_ci_uppers) / len(baseline_ci_uppers)
            if baseline_ci_uppers else None
        )
    }
    
    return averaged_df, baseline_stats

# =====================================================
# 3. STYLE & PLOTTING
# =====================================================
def apply_custom_style(ax, df_source, baseline_stats, ylabel, y_min=None, y_max=None):
    """Applies visual styling, axes formatting, and unified legends to the plot."""
    ax.grid(False) 
    ax.set_axisbelow(True)
    
    # Enable horizontal grid lines
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=1)
    ax.xaxis.grid(False)

    # Render bar plots
    sns.barplot(
        data=df_source, x="Group", y="Value", hue="Configuration",
        palette=PALETTE, ax=ax, edgecolor='black', linewidth=1, zorder=3,
        hue_order=HUE_ORDER, errorbar=None
    )

    x_order = df_source["Group"].drop_duplicates().tolist()
    for container, configuration in zip(ax.containers, HUE_ORDER):
        config_rows = (
            df_source[df_source["Configuration"] == configuration]
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
    baseline_val = baseline_stats.get("mean", 0)
    baseline_lower = baseline_stats.get("lower")
    baseline_upper = baseline_stats.get("upper")
    if baseline_val > 0:
        if baseline_lower is not None and baseline_upper is not None:
            ax.axhspan(baseline_lower, baseline_upper, color=COLOR_HUMAN, alpha=0.12, zorder=6)
        ax.axhline(y=baseline_val, color=COLOR_HUMAN, linestyle='--', linewidth=2, zorder=7)

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
    df, baseline_stats = prepare_dataframe(loaded_data, metric_key)
    if df.empty: 
        print(f"No data found for metric: {metric_key}")
        return

    # Match the (10, 9) size from the previous scripts
    fig, ax = plt.subplots(figsize=(10, 9))
    
    apply_custom_style(ax, df, baseline_stats, ylabel, y_min, y_max)

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
