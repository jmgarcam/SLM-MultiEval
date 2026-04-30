import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =====================================================
# 1. CONFIGURATION
# =====================================================
FILE_FH = '../d1_fh/fh_results.json'
FILE_TREE_DEPTH = '../d1_tree_depth/tree_depth_results.json'
FILE_MTLD = '../d1_mtld/mtld_results.json'
OUTPUT_DIR = './D1'

MODEL_NAMES = ["Ministral-3B", "Qwen-2.5-0.5B", "Llama-3.2-1B"]
MODEL_IDS = ["1", "2", "3"]
COLORS = ['#e74c3c', '#2ecc71', '#3498db'] 

TEMP_IDS = ["3", "2", "1"]
TEMP_MARKERS = ['o', 's', '^']
TEMP_LINESTYLES = [':', '--', '-']
TEMP_LABELS = ['t = 0.5', 't = 0.75', 't = 1.0']

METRICS = ['MTLD', 'Tree Depth', 'FH Index']
SECTOR_COLORS = ['#acc67d', '#e6e6e6', '#e3c1a2'] 

CONDITIONS = ["RAG", "NO_RAG"]

# =====================================================
# 2. DATA PROCESSING
# =====================================================
def load_json(filepath):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_radial_data():
    fh_data = load_json(FILE_FH)
    tree_data = load_json(FILE_TREE_DEPTH)
    mtld_data = load_json(FILE_MTLD)
    if not fh_data or not tree_data or not mtld_data: return None, 0

    human_fh = np.mean([fh_data[nw]["real_reference_metrics"]["mean_fh"] for nw in fh_data])
    human_depth = np.mean([tree_data[nw]["real_reference_metrics"]["mean_depth"] for nw in tree_data])
    human_mtld = np.mean([mtld_data[nw]["real_reference_metrics"]["mean_mtld"] for nw in mtld_data])
    human_baselines = {"mtld": human_mtld, "depth": human_depth, "fh": human_fh}
    
    model_data = {cond: {m: {t: [] for t in TEMP_IDS} for m in MODEL_IDS} for cond in CONDITIONS}
    global_max = 1.0 
    
    for cond in CONDITIONS:
        for m_id in MODEL_IDS:
            for t_id in TEMP_IDS:
                fh_vals, depth_vals, mtld_vals = [], [], []
                for nw in fh_data:
                    try:
                        fh_vals.append(fh_data[nw]["llm_models"][m_id][t_id][cond]["mean_fh"])
                    except KeyError: pass
                for nw in tree_data:
                    try:
                        depth_vals.append(tree_data[nw]["llm_models"][m_id][t_id][cond]["mean_depth"])
                    except KeyError: pass
                for nw in mtld_data:
                    try:
                        mtld_vals.append(mtld_data[nw]["llm_models"][m_id][t_id][cond]["mean_mtld"])
                    except KeyError: pass
                
                n_mtld = np.mean(mtld_vals)/human_baselines["mtld"] if mtld_vals else 0
                n_depth = np.mean(depth_vals)/human_baselines["depth"] if depth_vals else 0
                n_fh = np.mean(fh_vals)/human_baselines["fh"] if fh_vals else 0
                
                vals = [n_mtld, n_depth, n_fh]
                model_data[cond][m_id][t_id] = vals
                global_max = max(global_max, max(vals))
    return model_data, global_max

# =====================================================
# 3. CHART GENERATION
# =====================================================
def generate_separate_radial_figures():
    model_data, global_max = process_radial_data()
    if not model_data: return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cond in CONDITIONS:
        fig = plt.figure(figsize=(9, 9)) 
        ax = fig.add_subplot(1, 1, 1, polar=True)
        ax.set_ylim(0.0, 2.0)

        centers = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
        edges = [np.pi/6, 5*np.pi/6, 9*np.pi/6, 13*np.pi/6] 
        offsets = [-np.pi/9, 0, np.pi/9]
        
        for i in range(3):
            theta_sector = np.linspace(edges[i], edges[i+1], 50)
            ax.fill_between(theta_sector, 0, 2.0, color=SECTOR_COLORS[i], alpha=0.4, zorder=0)

        for edge in edges[:-1]: 
            ax.plot([edge, edge], [0.0, 2.0], color='black', linewidth=1.5, zorder=1)

        ax.plot(np.linspace(0, 2*np.pi, 200), [1.0]*200, color='black', linewidth=2.5, zorder=2)

        for m_idx, m_id in enumerate(MODEL_IDS):
            for t_idx, t_id in enumerate(TEMP_IDS):
                poly_angles = [centers[i] + offsets[m_idx] for i in range(3)]
                poly_vals = [model_data[cond][m_id][t_id][i] for i in range(3)]
                poly_angles.append(poly_angles[0]); poly_vals.append(poly_vals[0])
                ax.plot(poly_angles, poly_vals, color=COLORS[m_idx], linewidth=1.2, 
                        linestyle=TEMP_LINESTYLES[t_idx], alpha=0.4, zorder=3)

        for i in range(3):
            for m_idx, m_id in enumerate(MODEL_IDS):
                angle = centers[i] + offsets[m_idx]
                vals = [model_data[cond][m_id][t_id][i] for t_id in TEMP_IDS]
                ax.plot([angle, angle], [min(vals), max(vals)], color=COLORS[m_idx], linewidth=3.0, alpha=0.3, zorder=4)
                for t_idx, t_id in enumerate(TEMP_IDS):
                    ax.plot(angle, vals[t_idx], marker=TEMP_MARKERS[t_idx], color=COLORS[m_idx], 
                            markersize=16, markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        ax.set_theta_offset(0)
        ax.set_theta_direction(1)
        ax.set_xticks(centers)
        ax.set_xticklabels([]) 
        
        ax.set_rlabel_position(50) 
        grid_ticks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        tick_labels = ["25%", "", "75%", "", "125%", "", "175%", ""]
        ax.set_yticks(grid_ticks)
        ax.set_yticklabels(tick_labels, color="#333333", size=18, fontweight='bold', va='center', ha='center')
        ax.grid(False)
        
        for r, label in zip(grid_ticks, tick_labels):
            if label != "":
                gap = 0.28 / r 
                theta = np.linspace(np.radians(50) + gap/2, np.radians(50) + 2*np.pi - gap/2, 200)
                ax.plot(theta, [r]*200, color='gray', linestyle=':', alpha=0.5, zorder=1)
            elif r != 1.0:
                ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200, color='gray', linestyle=':', alpha=0.5, zorder=1)

        # =====================================================
        # 4. LEYENDAS (TÉCNICA DE OCULTACIÓN PARA NO_RAG)
        # =====================================================
        legend_elements_models = [Line2D([0], [0], color=COLORS[i], lw=4, label=MODEL_NAMES[i]) for i in range(3)]
        legend_elements_temps = [Line2D([0], [0], color='gray', linestyle=TEMP_LINESTYLES[i], lw=1.5,
                                        marker=TEMP_MARKERS[i], markerfacecolor='gray', markeredgecolor='white',
                                        markersize=14, label=TEMP_LABELS[i]) for i in range(3)]
        legend_elements_sectors = [Patch(facecolor=SECTOR_COLORS[i], edgecolor='black', alpha=0.5, label=METRICS[i]) for i in range(3)]
        legend_elements_sectors.append(Line2D([0], [0], color='black', lw=2.5, label='Human baseline'))
        
        leg_met = ax.legend(handles=legend_elements_sectors, loc='upper left', bbox_to_anchor=(1.05, 1.0), 
                         title="Metrics", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_met.get_title().set_fontweight('bold')

        leg_mod = ax.legend(handles=legend_elements_models, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                         title="Models", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_mod.get_title().set_fontweight('bold')
        
        leg_tmp = ax.legend(handles=legend_elements_temps, loc='lower left', bbox_to_anchor=(1.05, 0.05), 
                         title="Temperatures", frameon=False, ncol=1, fontsize=20, title_fontsize=22)
        leg_tmp.get_title().set_fontweight('bold')
        
        ax.add_artist(leg_met)
        ax.add_artist(leg_mod)
        all_legends = [leg_met, leg_mod, leg_tmp]

        # OCULTAR EN NO_RAG (Mantiene el espacio para que el gráfico no cambie de tamaño)
        if cond == "NO_RAG":
            for leg in all_legends:
                leg.get_title().set_alpha(0)
                for text in leg.get_texts(): text.set_alpha(0)
                for handle in leg.legend_handles: handle.set_alpha(0)

        base_filename = f"d1_{cond.lower()}"
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_filename}.png"), dpi=300, 
                    bbox_extra_artists=all_legends, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_filename}.pdf"), 
                    bbox_extra_artists=all_legends, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_separate_radial_figures()