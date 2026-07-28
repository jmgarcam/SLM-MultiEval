import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D 

# ==========================================
# 1. CONFIGURATION
# ==========================================
JSON_FILE = "bert_results.json"
OUTPUT_DIR = "./graphs" 
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model_map = {
    "1": "Ministral-3B", 
    "2": "Qwen-2.5-0.5B", 
    "3": "Llama-3.2-1B"
}

temp_key_map = {
    "1": "t=1",      
    "2": "t=0.75",   
    "3": "t=0.5"     
}

# ==========================================
# 2. DATA LOADING AND PROCESSING
# ==========================================
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"File not found: {JSON_FILE}")
    exit()

rows = []
baselines = {}

for newspaper_key, content in data.items():
    if "human_baseline" in content and content["human_baseline"]:
        baselines[newspaper_key] = content["human_baseline"]
    
    if "llm_models" in content:
        for mod_id, feats in content["llm_models"].items():
            model_name = model_map.get(mod_id, f"Mod {mod_id}")
            for temp_id, configs in feats.items():
                str_id = str(temp_id)
                if str_id not in temp_key_map:
                    continue
                temp_label = temp_key_map[str_id]
                for config, metrics in configs.items():
                    if metrics.get("mean_f1", 0) > 0:
                        clean_config = config.replace("_", " ")
                        precision_ci = metrics.get("ci95_precision", {})
                        recall_ci = metrics.get("ci95_recall", {})
                        f1_ci = metrics.get("ci95_f1", {})
                        rows.append({
                            "Newspaper": newspaper_key,
                            "Model": model_name,
                            "Configuration": clean_config, 
                            "Temperature": temp_label, 
                            "Precision": metrics.get("mean_precision", 0),
                            "PrecisionLower": precision_ci.get("lower"),
                            "PrecisionUpper": precision_ci.get("upper"),
                            "Recall": metrics.get("mean_recall", 0),
                            "RecallLower": recall_ci.get("lower"),
                            "RecallUpper": recall_ci.get("upper"),
                            "F1": metrics.get("mean_f1", 0),
                            "F1Lower": f1_ci.get("lower"),
                            "F1Upper": f1_ci.get("upper")
                        })

raw_df = pd.DataFrame(rows)
if raw_df.empty:
    print("Error: No data found.")
    exit()

metric_columns = [
    "Precision", "PrecisionLower", "PrecisionUpper",
    "Recall", "RecallLower", "RecallUpper",
    "F1", "F1Lower", "F1Upper"
]
df_mean = (
    raw_df.groupby(["Model", "Configuration", "Temperature"])[metric_columns]
    .mean()
    .reset_index()
)

df_mean['Model'] = pd.Categorical(
    df_mean['Model'], 
    categories=["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"], 
    ordered=True
)
df_mean = df_mean.sort_values('Model')

mean_baselines = {}
if NEWSPAPER_A in baselines and NEWSPAPER_B in baselines:
    for metric in ["mean_precision", "mean_recall", "mean_f1"]:
        v1 = baselines[NEWSPAPER_A].get(metric, 0)
        v2 = baselines[NEWSPAPER_B].get(metric, 0)
        ci_key = metric.replace("mean_", "ci95_")
        ci1 = baselines[NEWSPAPER_A].get(ci_key, {})
        ci2 = baselines[NEWSPAPER_B].get(ci_key, {})
        lower_values = [
            value for value in (ci1.get("lower"), ci2.get("lower"))
            if value is not None
        ]
        upper_values = [
            value for value in (ci1.get("upper"), ci2.get("upper"))
            if value is not None
        ]
        mean_baselines[metric] = {
            "mean": (v1 + v2) / 2,
            "lower": (
                sum(lower_values) / len(lower_values)
                if lower_values else None
            ),
            "upper": (
                sum(upper_values) / len(upper_values)
                if upper_values else None
            )
        }

# ==========================================
# 3. PLOTTING
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

palette_colors = {"NO RAG": "#e74c3c", "RAG": "#2ecc71"} 
markers_map = {"t=1": "o", "t=0.75": "D", "t=0.5": "s"}
temp_order = ["t=1", "t=0.75", "t=0.5"]

def plot_metric_detailed(metric_col, base_filename):
    print(f"Generating plots for: {metric_col}...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    
    # SEPARACIÓN VISUAL: Líneas verticales entre modelos
    for x_sep in [0.5, 1.5]:
        ax.axvline(x=x_sep, color='#dddddd', linestyle='-', linewidth=1.5, zorder=0)

    sns.scatterplot(
        data=df_mean,
        x="Model",
        y=metric_col,
        hue="Configuration",
        style="Temperature",
        markers=markers_map,
        hue_order=["NO RAG", "RAG"], 
        style_order=temp_order,
        s=220,
        alpha=0.9,
        palette=palette_colors,
        linewidth=1.2,            
        edgecolor="#333333",
        ax=ax,
        zorder=3
    )
    
    # --- CONFIGURACIÓN DEL EJE X (SEPARACIÓN DE NOMBRES A -0.18) ---
    unique_models = ["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"]

    lower_col = f"{metric_col}Lower"
    upper_col = f"{metric_col}Upper"
    for _, row in df_mean.iterrows():
        lower = row.get(lower_col)
        upper = row.get(upper_col)
        value = row.get(metric_col)
        model_name = str(row.get("Model"))
        if (
            model_name not in unique_models
            or pd.isna(lower)
            or pd.isna(upper)
            or pd.isna(value)
        ):
            continue

        x_position = unique_models.index(model_name)
        yerr = [
            [max(0, value - lower)],
            [max(0, upper - value)]
        ]
        ax.errorbar(
            x_position, value, yerr=yerr, fmt='none',
            ecolor='#333333', elinewidth=1.3, capsize=4, capthick=1.3,
            alpha=0.75, zorder=2
        )

    ax.set_xticks(range(len(unique_models)))
    ax.set_xticklabels([]) # Eliminamos etiquetas automáticas
    ax.set_xlabel("")
    ax.set_xlim(-0.5, 2.5)

    for idx, model_name in enumerate(unique_models):
        ax.text(idx, -0.05, model_name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.1) 
    
    ax.set_ylabel(f"BERTScore {metric_col}", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)
    
    baseline_key = f"mean_{metric_col.lower()}"
    baseline_stats = mean_baselines.get(
        baseline_key,
        {"mean": 0, "lower": None, "upper": None}
    )
    val = baseline_stats.get("mean", 0)
    baseline_lower = baseline_stats.get("lower")
    baseline_upper = baseline_stats.get("upper")
    
    if val > 0:
        if baseline_lower is not None and baseline_upper is not None:
            ax.axhspan(
                baseline_lower, baseline_upper, color='#008000',
                alpha=0.12, zorder=1
            )
        ax.axhline(y=val, color='#008000', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    
    filtered_handles = []
    filtered_labels = []
    for h, l in zip(handles, labels):
        if l != "Configuration":
            filtered_handles.append(h)
            filtered_labels.append(l)
            
    handles = filtered_handles
    labels = filtered_labels

    baseline_line = Line2D([0], [0], color='green', linestyle='--', linewidth=2.5)
    handles.append(baseline_line)
    labels.append("Human baseline") 
    
    ax.legend(
        handles=handles, 
        labels=labels, 
        loc='lower right', 
        bbox_to_anchor=(1.012, 0.06),  
        frameon=True,                  
        facecolor='white',             
        framealpha=1.0,               
        edgecolor='#CCCCCC',          
        fontsize=20,                  
        title=""
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) # Espacio extra para los nombres a -0.18

    base_path = os.path.join(OUTPUT_DIR, base_filename)
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    print(f"   Saved: {base_filename}")
    plt.close()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__:
    if not df_mean.empty:
        plot_metric_detailed("Precision", "bert_precision")
        plot_metric_detailed("Recall", "bert_recall")
        plot_metric_detailed("F1", "bert_f1")
        print(f"\nProcess finished. Files saved in: {OUTPUT_DIR}/")
    else:
        print("Error: DataFrame is empty.")
