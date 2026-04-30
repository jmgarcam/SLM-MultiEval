import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =====================================================
# CONFIG
# =====================================================
FILE_NLI = "../d3_nli_llm/nli_llm_results.json"
FILE_FACT_RAG = "../d3_factual_llm/factual_llm_RAG_results.json"
FILE_FACT_NO_RAG = "../d3_factual_llm/factual_llm_NO_RAG_results.json"

OUTPUT_DIR = "./D3"

MODEL_NAMES = ["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"]
MODEL_IDS = ["1", "2", "3"]
COLORS = ['#e74c3c', '#2ecc71', '#3498db']

TEMP_IDS = ["3", "2", "1"]
TEMP_MARKERS = ['o', 's', '^']
TEMP_LINESTYLES = [':', '--', '-']
TEMP_LABELS = ['t = 0.5', 't = 0.75', 't = 1.0']

METRICS = ['NLI_E', '1 - NLI_C', 'Correct', '1 - Incorrect']

SECTOR_COLORS = [
    ('#f1c40f', 0.35),
    ('#f1c40f', 0.15),
    ('#7f8c8d', 0.35),
    ('#7f8c8d', 0.15)
]

CONDITIONS = ["RAG", "NO_RAG"]

# =====================================================
# LOAD
# =====================================================
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================
# ANGULAR DISTRIBUTION
# =====================================================
def build_trace_offsets(sector_halfwidth=np.pi / 4, margin_ratio=0.22):
    usable_halfwidth = sector_halfwidth * (1 - margin_ratio)
    offsets = np.linspace(-usable_halfwidth, usable_halfwidth, len(MODEL_IDS)*len(TEMP_IDS))

    trace_offsets = {}
    k = 0
    for m in MODEL_IDS:
        for t in TEMP_IDS:
            trace_offsets[(m, t)] = offsets[k]
            k += 1
    return trace_offsets

# =====================================================
# PROCESS DATA
# =====================================================
def process_d3_data():
    nli = load_json(FILE_NLI)
    fact_rag = load_json(FILE_FACT_RAG)
    fact_no_rag = load_json(FILE_FACT_NO_RAG)

    model_data = {cond: {m: {t: [] for t in TEMP_IDS} for m in MODEL_IDS} for cond in CONDITIONS}

    for cond in CONDITIONS:
        fact_src = fact_rag if cond == "RAG" else fact_no_rag

        for m_id in MODEL_IDS:
            for t_id in TEMP_IDS:

                e_vals, c_vals, corr_vals, inc_vals = [], [], [], []

                for nw in nli[cond]:
                    if m_id in nli[cond][nw] and t_id in nli[cond][nw][m_id]:
                        m = nli[cond][nw][m_id][t_id]
                        e_vals.append(m["mean_entailment_softmax"])
                        c_vals.append(m["mean_contradiction_softmax"])

                for nw in fact_src:
                    if m_id in fact_src[nw] and t_id in fact_src[nw][m_id]:
                        m = fact_src[nw][m_id][t_id]
                        total = m["correct"] + m["incorrect"] + m["partially_correct"]
                        if total > 0:
                            corr_vals.append(m["correct"] / total)
                            inc_vals.append(m["incorrect"] / total)

                vals = [
                    np.mean(e_vals) if e_vals else 0,
                    1 - (np.mean(c_vals) if c_vals else 0),
                    np.mean(corr_vals) if corr_vals else 0,
                    1 - (np.mean(inc_vals) if inc_vals else 0)
                ]

                model_data[cond][m_id][t_id] = vals

    return model_data

# =====================================================
# PLOT
# =====================================================
def generate_d3():
    model_data = process_d3_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centers = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    edges = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4, 9*np.pi/4]
    trace_offsets = build_trace_offsets()

    for cond in CONDITIONS:

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1, polar=True)

        ax.set_ylim(0.0, 1.25)

        # sectores
        for i in range(4):
            color, alpha = SECTOR_COLORS[i]
            theta = np.linspace(edges[i], edges[i+1], 200)
            ax.fill_between(theta, 0, 1.25, color=color, alpha=alpha)

        # divisores
        for edge in edges[:-1]:
            ax.plot([edge, edge], [0, 1.25], color='black', alpha=0.3)

        # baseline
        ax.plot(np.linspace(0, 2*np.pi, 200), [1.0]*200, color='black', linewidth=2.5)

        # polígonos
        for m_idx, m_id in enumerate(MODEL_IDS):
            for t_idx, t_id in enumerate(TEMP_IDS):

                vals = model_data[cond][m_id][t_id]
                offset = trace_offsets[(m_id, t_id)]

                angles = [centers[i] + offset for i in range(4)]
                angles += [angles[0]]
                vals_closed = vals + [vals[0]]

                ax.plot(angles, vals_closed,
                        color=COLORS[m_idx],
                        linestyle=TEMP_LINESTYLES[t_idx],
                        linewidth=1.5,
                        alpha=0.6)

                for i in range(4):
                    ax.plot(angles[i], vals[i],
                            marker=TEMP_MARKERS[t_idx],
                            color=COLORS[m_idx],
                            markersize=14,
                            markeredgecolor='white')

        # ejes
        ax.set_theta_offset(0)
        ax.set_theta_direction(1)
        ax.set_xticks(centers)
        ax.set_xticklabels([])

        LABEL_ANGLE_DEG = 260 if cond == "NO_RAG" else 235
        ax.set_rlabel_position(LABEL_ANGLE_DEG)

        grid_ticks = [0.5, 0.75]
        tick_labels = ["50%", "75%"]

        ax.set_yticks(grid_ticks)
        ax.set_yticklabels([])

        # etiquetas manuales (BAJADAS)
        label_angle_rad = np.radians(LABEL_ANGLE_DEG)

        for r, label in zip(grid_ticks, tick_labels):
            ax.text(
                label_angle_rad,
                r - 0.04,
                label,
                ha='center',
                va='center',
                fontsize=18,
                fontweight='bold'
            )

        ax.grid(False)

        # grid con hueco (estilo D1)
        for r in grid_ticks:
            gap = 0.28 / r
            theta = np.linspace(label_angle_rad + gap/2,
                                label_angle_rad + 2*np.pi - gap/2, 200)
            ax.plot(theta, [r]*200, color='gray', linestyle=':', alpha=0.6)

        # leyendas
        legend_metrics = [
            Patch(facecolor=SECTOR_COLORS[i][0],
                  alpha=SECTOR_COLORS[i][1],
                  edgecolor='black',
                  label=METRICS[i])
            for i in range(4)
        ]
        legend_metrics.append(Line2D([0],[0],color='black',lw=2.5,label='Human baseline'))

        legend_models = [Line2D([0],[0],color=COLORS[i],lw=4,label=MODEL_NAMES[i]) for i in range(3)]

        legend_temps = [Line2D([0],[0],color='gray',
                               linestyle=TEMP_LINESTYLES[i],
                               marker=TEMP_MARKERS[i],
                               markersize=14,
                               label=TEMP_LABELS[i]) for i in range(3)]

        leg_met = ax.legend(handles=legend_metrics, loc='upper left',
                            bbox_to_anchor=(1.05,0.9),
                            title="Metrics",
                            frameon=False,
                            fontsize=20,
                            title_fontsize=22)
        leg_met.get_title().set_fontweight('bold')

        leg_mod = ax.legend(handles=legend_models, loc='center left',
                            bbox_to_anchor=(1.05,0.35),
                            title="Models",
                            frameon=False,
                            fontsize=20,
                            title_fontsize=22)
        leg_mod.get_title().set_fontweight('bold')

        leg_tmp = ax.legend(handles=legend_temps, loc='lower left',
                            bbox_to_anchor=(1.05,-0.10),
                            title="Temperatures",
                            frameon=False,
                            fontsize=20,
                            title_fontsize=22)
        leg_tmp.get_title().set_fontweight('bold')

        ax.add_artist(leg_met)
        ax.add_artist(leg_mod)

        all_legends = [leg_met, leg_mod, leg_tmp]

        if cond == "NO_RAG":
            for leg in all_legends:
                leg.get_title().set_alpha(0)
                for t in leg.get_texts():
                    t.set_alpha(0)
                for h in leg.legend_handles:
                    h.set_alpha(0)

        plt.savefig(os.path.join(OUTPUT_DIR, f"d3_{cond.lower()}.png"),
                    dpi=300, bbox_extra_artists=all_legends, bbox_inches='tight')

        plt.savefig(os.path.join(OUTPUT_DIR, f"d3_{cond.lower()}.pdf"),
                    bbox_extra_artists=all_legends, bbox_inches='tight')

        plt.close()

if __name__ == "__main__":
    generate_d3()