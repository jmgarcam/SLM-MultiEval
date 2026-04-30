import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# =====================================================
# 1. CONFIGURATION
# =====================================================
INPUT_FILE = 'uer_results.json'
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

# =====================================================
# 2. DATA LOADING
# =====================================================
def load_data(filepath):
    """Loads JSON data from the specified filepath."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# =====================================================
# 3. PREPARE DATAFRAME (MEAN ACROSS SOURCES)
# =====================================================
def prepare_dataframe(json_data):
    """Parses JSON data and calculates means across sources."""
    rows = []
    baseline_values = []
    
    sources = [k for k, v in json_data.items() if "llm_models" in v]
    
    for source in sources:
        content = json_data[source]
        
        base_val = content.get("real_reference_metrics", {}).get("mean_uer", 0)
        if base_val > 0:
            baseline_values.append(base_val * 100)
            
        llm_models = content.get("llm_models", {})
        for model_id, features in llm_models.items():
            model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")
            for temp_id, features_data in features.items():
                try:
                    rag_val = list(features_data["RAG"].values())[0] * 100
                    no_rag_val = list(features_data["NO_RAG"].values())[0] * 100
                    
                    rows.append({
                        "ModelName": model_name, 
                        "TempID": temp_id, 
                        "Group": f"{model_id}_{temp_id}", 
                        "Configuration": "NO RAG", 
                        "Value": no_rag_val
                    })
                    rows.append({
                        "ModelName": model_name, 
                        "TempID": temp_id, 
                        "Group": f"{model_id}_{temp_id}", 
                        "Configuration": "RAG", 
                        "Value": rag_val
                    })
                except (KeyError, IndexError):
                    continue
                    
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, 0
        
    df_mean = raw_df.groupby(["ModelName", "TempID", "Group", "Configuration"])["Value"].mean().reset_index()
    df_mean['ModelName'] = pd.Categorical(df_mean['ModelName'], categories=MODEL_NAMES.values(), ordered=True)
    df_mean = df_mean.sort_values(['ModelName', 'TempID'])
    
    mean_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
    return df_mean, mean_baseline

# =====================================================
# 4. STYLE & PLOTTING FUNCTION
# =====================================================
def apply_custom_style(ax, df_data, baseline_val):
    """Applies specific styling and labels to the axis. Legend is disabled."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=COLOR_GRID, linestyle='-', linewidth=1, zorder=1)
    ax.xaxis.grid(False)

    sns.barplot(
        data=df_data, x="Group", y="Value", hue="Configuration",
        palette=PALETTE, ax=ax, edgecolor='black', linewidth=1, zorder=3,
        legend=False  # Desactiva la leyenda automática de Seaborn
    )

    unique_models = df_data["ModelName"].unique()
    temps_per_model = df_data["TempID"].unique()
    n_temps = len(temps_per_model)
    
    # --- X-Axis (Temperatures) ---
    tick_labels = [TEMP_MAP.get(t, t) for t in temps_per_model] * len(unique_models)
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
        ax.text(center, -0.18, model_name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')
        
        if idx < len(unique_models) - 1:
            ax.axvline(idx * n_temps + n_temps - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # --- Human Baseline ---
    if baseline_val > 0:
        ax.axhline(y=baseline_val, color=COLOR_HUMAN, linestyle='--', linewidth=2, zorder=4)

    # --- Y-Axis ---
    ax.set_ylabel("UER (%)", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(2, 7)

# =====================================================
# 5. GENERATE REPORT
# =====================================================
def generate_report():
    """Generates and saves the UER mean plot."""
    df, mean_baseline = prepare_dataframe(loaded_data)
    if df.empty: return

    fig, ax = plt.subplots(figsize=(10, 9))
    apply_custom_style(ax, df, mean_baseline)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "uer.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "uer.pdf"), bbox_inches='tight')
    print(f"Successfully generated: uer")
    plt.close()

if __name__ == "__main__":
    loaded_data = load_data(INPUT_FILE)
    if loaded_data:
        sns.set_theme(style="white")
        generate_report()