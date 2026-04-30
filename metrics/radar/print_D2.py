import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =====================================================
# 1. CONFIGURATION
# =====================================================
FILE_BERT = '../d2_bert/bert_results.json'
FILE_SBERT = '../d2_sbert/sbert_results.json'
FILE_ETE = '../d2_ete/ete_results.json'
FILE_UER = '../d2_uer/uer_results.json'

OUTPUT_DIR = './D2' 

MODEL_NAMES = ["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"]
MODEL_IDS = ["1", "2", "3"]
COLORS = ['#e74c3c', '#2ecc71', '#3498db'] 

TEMP_IDS = ["3", "2", "1"]
TEMP_MARKERS = ['o', 's', '^']
TEMP_LINESTYLES = [':', '--', '-']
TEMP_LABELS = ['t = 0.5', 't = 0.75', 't = 1.0']

METRICS = ['BERT', 'SBERT', 'ETE', 'UER']
SECTOR_COLORS = ['#fde0dd', '#e0f3db', '#fee8c8', '#ece7f2']

CONDITIONS = ["RAG", "NO_RAG"]

# =====================================================
# 2. DATA PROCESSING
# =====================================================
def load_json(filepath):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_key(metric_name):
    keys = {'BERT': 'mean_f1', 'SBERT': 'mean_sbert', 'ETE': 'mean_entity_entropy', 'UER': 'mean_uer'}
    return keys.get(metric_name, f'mean_{metric_name.lower()}')

def process_radial_data():
    data_sources = {m: load_json(f) for m, f in zip(METRICS, [FILE_BERT, FILE_SBERT, FILE_ETE, FILE_UER])}
    
    reference_values = {}
    for metric in METRICS:
        k = get_key(metric)
        src = data_sources[metric]
        
        # CORRECCIÓN: Obtener referencia de human_baseline para BERT/SBERT
        if metric in ['BERT', 'SBERT']:
            try:
                ref_vals = [src[nw]["human_baseline"][k] for nw in src if "human_baseline" in src[nw] and k in src[nw]["human_baseline"]]
                reference_values[metric] = np.mean(ref_vals) if ref_vals else 1.0
            except:
                reference_values[metric] = 1.0
        else:
            # Para ETE y UER se mantiene la búsqueda en real_reference_metrics
            try:
                ref_vals = [src[nw]["real_reference_metrics"][k] for nw in src if "real_reference_metrics" in src[nw]]
                reference_values[metric] = np.mean(ref_vals) if ref_vals else 1.0
            except: 
                reference_values[metric] = 1.0
            
    model_data = {cond: {m: {t: [] for t in TEMP_IDS} for m in MODEL_IDS} for cond in CONDITIONS}
    
    for cond in CONDITIONS:
        for m_id in MODEL_IDS:
            for t_id in TEMP_IDS:
                vals = []
                for metric in METRICS:
                    try:
                        k = get_key(metric)
                        src = data_sources[metric]
                        m_vals = [src[nw]["llm_models"][m_id][t_id][cond][k] for nw in src if "llm_models" in src[nw]]
                        # Normalización: (Media LLM / Media Humano)
                        vals.append(np.mean(m_vals) / reference_values[metric] if m_vals else 0.0)
                    except: vals.append(0.0)
                model_data[cond][m_id][t_id] = vals
    return model_data

# =====================================================
# 3. CHART GENERATION
# =====================================================
def generate_separate_radial_figures():
    model_data = process_radial_data()
    if not model_data: return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cond in CONDITIONS:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1, polar=True)
        
        upper_limit = 2.75 
        ax.set_ylim(0.0, upper_limit)

        centers = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        edges = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4, 9*np.pi/4] 
        offsets = [-np.pi/9, 0, np.pi/9]
        
        for i in range(4):
            theta_sector = np.linspace(edges[i], edges[i+1], 50)
            ax.fill_between(theta_sector, 0, upper_limit, color=SECTOR_COLORS[i], alpha=0.7, zorder=0)

        for edge in edges[:-1]: 
            ax.plot([edge, edge], [0.0, upper_limit], color='black', linewidth=1.5, alpha=0.3, zorder=1)

        ax.plot(np.linspace(0, 2*np.pi, 200), [1.0]*200, color='black', linewidth=2.5, zorder=2)

        for m_idx, m_id in enumerate(MODEL_IDS):
            for t_idx, t_id in enumerate(TEMP_IDS):
                poly_angles = [centers[i] + offsets[m_idx] for i in range(4)]
                poly_vals = [model_data[cond][m_id][t_id][i] for i in range(4)]
                poly_angles.append(poly_angles[0]); poly_vals.append(poly_vals[0])
                ax.plot(poly_angles, poly_vals, color=COLORS[m_idx], linewidth=1.2, 
                        linestyle=TEMP_LINESTYLES[t_idx], alpha=0.5, zorder=3)

        for i in range(4):
            for m_idx, m_id in enumerate(MODEL_IDS):
                angle = centers[i] + offsets[m_idx]
                vals = [model_data[cond][m_id][t_id][i] for t_id in TEMP_IDS]
                ax.plot([angle, angle], [min(vals), max(vals)], color=COLORS[m_idx], linewidth=3.0, alpha=0.3, zorder=4)
                for t_idx, t_id in enumerate(TEMP_IDS):
                    ax.plot(angle, vals[t_idx], marker=TEMP_MARKERS[t_idx], color=COLORS[m_idx], 
                            markersize=16, markeredgecolor='white', markeredgewidth=1.0, zorder=5)

        ax.set_theta_offset(0)
        ax.set_theta_direction(1)
        ax.set_xticks(centers)
        ax.set_xticklabels([]) 
        
        LABEL_ANGLE_DEG = 235
        ax.set_rlabel_position(LABEL_ANGLE_DEG) 
        grid_ticks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
        tick_labels = ["25%", "", "75%", "", "125%", "", "175%", "", "225%", ""]
        ax.set_yticks(grid_ticks)
        ax.set_yticklabels(tick_labels, color="#333333", size=18, fontweight='bold', va='center', ha='center')
        ax.grid(False)
        
        label_angle_rad = np.radians(LABEL_ANGLE_DEG)
        for r, label in zip(grid_ticks, tick_labels):
            if label != "":
                gap = 0.45 / r 
                theta = np.linspace(label_angle_rad + gap/2, label_angle_rad + 2*np.pi - gap/2, 200)
                ax.plot(theta, [r]*200, color='gray', linestyle=':', alpha=0.6, zorder=1)
            elif r != 1.0:
                ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200, color='gray', linestyle=':', alpha=0.6, zorder=1)

        # LEYENDAS
        legend_elements_models = [Line2D([0], [0], color=COLORS[i], lw=4, label=MODEL_NAMES[i]) for i in range(3)]
        legend_elements_temps = [Line2D([0], [0], color='gray', linestyle=TEMP_LINESTYLES[i], lw=1.5,
                                        marker=TEMP_MARKERS[i], markerfacecolor='gray', markeredgecolor='white',
                                        markersize=14, label=TEMP_LABELS[i]) for i in range(3)]
        legend_elements_sectors = [Patch(facecolor=SECTOR_COLORS[i], edgecolor='black', alpha=0.5, 
                                         label=METRICS[i]) for i in range(4)]
        legend_elements_sectors.append(Line2D([0], [0], color='black', lw=2.5, label='Human baseline'))
        
        leg_met = ax.legend(handles=legend_elements_sectors, loc='upper left', bbox_to_anchor=(1.05, 0.9), 
                            title="Metrics", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_met.get_title().set_fontweight('bold')

        leg_mod = ax.legend(handles=legend_elements_models, loc='center left', bbox_to_anchor=(1.05, 0.35), 
                            title="Models", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_mod.get_title().set_fontweight('bold')
        
        leg_tmp = ax.legend(handles=legend_elements_temps, loc='lower left', bbox_to_anchor=(1.05, -0.10), 
                            title="Temperatures", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_tmp.get_title().set_fontweight('bold')
        
        ax.add_artist(leg_met)
        ax.add_artist(leg_mod)
        all_legends = [leg_met, leg_mod, leg_tmp]

        if cond == "NO_RAG":
            for leg in all_legends:
                leg.get_title().set_alpha(0)
                for t in leg.get_texts(): t.set_alpha(0)
                for h in leg.legend_handles: h.set_alpha(0)

        base_filename = f"d2_{cond.lower()}"
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_filename}.png"), dpi=300, 
                    bbox_extra_artists=all_legends, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_filename}.pdf"), 
                    bbox_extra_artists=all_legends, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_separate_radial_figures()