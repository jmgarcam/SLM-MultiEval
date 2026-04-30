import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE_RAG = '../consistency_RAG_results.json'
INPUT_FILE_NO_RAG = '../consistency_NO_RAG_results.json'
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

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataframe(rag_data, no_rag_data):
    rows = []

    def process(data, label):
        for source, models in data.items():
            for model_id, temps in models.items():
                model_name = MODEL_NAMES.get(model_id, f"Model {model_id}")

                for temp_id, values in temps.items():
                    rows.append({
                        "ModelName": model_name,
                        "TempID": temp_id,
                        "Group": f"{model_id}_{temp_id}",
                        "Configuration": label,
                        "Value": values["avg_consistency"]
                    })

    process(rag_data, "RAG")
    process(no_rag_data, "NO RAG")

    raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        return raw_df

    df_mean = raw_df.groupby(
        ["ModelName", "TempID", "Group", "Configuration"]
    )["Value"].mean().reset_index()

    df_mean['ModelName'] = pd.Categorical(
        df_mean['ModelName'], 
        categories=["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"], 
        ordered=True
    )

    df_mean = df_mean.sort_values(['ModelName', 'TempID'])

    return df_mean

def apply_custom_style(ax, df_data):
    ax.set_axisbelow(True)

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
        linewidth=1
    )

    unique_models = df_data["ModelName"].drop_duplicates().tolist()
    temps_per_model = ["1", "2", "3"]
    n_temps = len(temps_per_model)

    tick_labels = [TEMP_MAP[t] for t in temps_per_model] * len(unique_models)

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

    ax.set_ylabel("Consistency", fontsize=26, fontweight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=26)

    ax.set_ylim(1.0, 5.1)

    legend = ax.get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]

    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.98),
        fontsize=26,
        frameon=True,
        facecolor='white',
        framealpha=1.0,
        edgecolor='#cccccc',
        title=""
    )

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

    plt.savefig(os.path.join(OUTPUT_DIR, "geval_consistency.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "geval_consistency.pdf"), format='pdf', bbox_inches='tight')

    print(f"Successfully generated G-Eval plot in {OUTPUT_DIR}/")
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="white")
    generate_report()