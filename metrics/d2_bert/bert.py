import json
import re
import sys
import os
import torch
from bert_score import score
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from news_queries.news_data import  get_synthetic_news_from_json

JSON_RESULTS = "bert_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"

# ==========================================
# 1. AUXILIARY FUNCTIONS
# ==========================================

def remove_html(text):
    """
    Removes HTML tags, non-breaking spaces, and specific keywords from text.
    """
    if not text: return ""
    text = text.replace('&nbsp;', ' ')
    clean_tags = re.compile('<.*?>')
    text = re.sub(clean_tags, ' ', text)
    text = text.replace('Leer', '') 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 2. BERT PROCESSING
# ==========================================

def process_batch(news_list):
    """
    Calculates BERTScore (P, R, F1) for synthetic descriptions against headlines.
    """
    cands = []
    refs = []
    valid_indices = []

    for i, news in enumerate(news_list):
        cand_text = remove_html(news.get("synthetic_description", ""))
        ref_text = remove_html(news.get("headline", "")) 
        
        if cand_text and ref_text:
            cands.append(cand_text)
            refs.append(ref_text)
            valid_indices.append(i)
        else:
            news_list[i]["bert_precision"] = 0.0
            news_list[i]["bert_recall"] = 0.0
            news_list[i]["bert_f1"] = 0.0

    if cands and refs:
        print(f"Comparing {len(cands)} pairs (Headline vs Synthetic)...")
        try:
            P, R, F1 = score(cands, refs, lang="es", verbose=False)
            
            for idx, p_val, r_val, f1_val in zip(valid_indices, P, R, F1):
                news_list[idx]["bert_precision"] = round(p_val.item(), 4)
                news_list[idx]["bert_recall"] = round(r_val.item(), 4)
                news_list[idx]["bert_f1"] = round(f1_val.item(), 4)
                
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")

def calculate_real_baseline(news_list):
    """
    Calculates the mean BERTScore for human-written descriptions against headlines.
    """
    cands = [] 
    refs = []  
    
    for news in news_list:
        cand_text = remove_html(news.get("real_description", ""))
        ref_text = remove_html(news.get("headline", ""))
        
        if cand_text and ref_text:
            cands.append(cand_text)
            refs.append(ref_text)
            
    if not cands:
        return None

    print(f"Calculating HUMAN BASELINE for {len(cands)} news...")
    try:
        P, R, F1 = score(cands, refs, lang="es", verbose=False)
        return {
            "mean_precision": round(P.mean().item(), 4),
            "mean_recall": round(R.mean().item(), 4),
            "mean_f1": round(F1.mean().item(), 4),
            "sample_size": len(cands)
        }
    except Exception as e:
        print(f"Error in baseline: {e}")
        return None

# ==========================================
# 3. STATISTICAL ANALYSIS & DATA STRUCTURE
# ==========================================

def extract_metrics_by_feature(news):
    """
    Groups BERTScore results by feature ID into lists for statistical testing.
    """
    data = {1: {'p': [], 'r': [], 'f1': []}, 2: {'p': [], 'r': [], 'f1': []}, 3: {'p': [], 'r': [], 'f1': []}}
    for item in news:
        feat_id = int(item.get('id_feature', -1))
        if feat_id in data and 'bert_f1' in item:
            data[feat_id]['p'].append(item.get('bert_precision', 0.0))
            data[feat_id]['r'].append(item.get('bert_recall', 0.0))
            data[feat_id]['f1'].append(item.get('bert_f1', 0.0))
    return data

def get_p_value(group_a, group_b):
    """
    Computes the p-value using an independent T-test (Welch's T-test).
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 1.0
    _, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    return p_val

def save_structured_json(path, no_rag_raw, rag_raw, data_real):
    """
    Calculates final means and p-values, then saves the results to a JSON file.
    """
    final_structure = {}
    
    for paper in no_rag_raw.keys():
        final_structure[paper] = {
            "human_baseline": data_real.get(paper, {}),
            "llm_models": {}
        }
        
        for llm_id in no_rag_raw[paper].keys():
            final_structure[paper]["llm_models"][llm_id] = {}
            
            for feat_id in [1, 2, 3]:
                nr = no_rag_raw[paper][llm_id].get(feat_id, {'p': [], 'r': [], 'f1': []})
                r = rag_raw[paper][llm_id].get(feat_id, {'p': [], 'r': [], 'f1': []})
                
                final_structure[paper]["llm_models"][llm_id][feat_id] = {
                    "NO_RAG": {
                        "mean_precision": round(sum(nr['p'])/len(nr['p']), 4) if nr['p'] else 0,
                        "mean_recall": round(sum(nr['r'])/len(nr['r']), 4) if nr['r'] else 0,
                        "mean_f1": round(sum(nr['f1'])/len(nr['f1']), 4) if nr['f1'] else 0
                    },
                    "RAG": {
                        "mean_precision": round(sum(r['p'])/len(r['p']), 4) if r['p'] else 0,
                        "mean_recall": round(sum(r['r'])/len(r['r']), 4) if r['r'] else 0,
                        "mean_f1": round(sum(r['f1'])/len(r['f1']), 4) if r['f1'] else 0
                    },
                    "T_Student_p_values": {
                        "p_value_precision": get_p_value(nr['p'], r['p']),
                        "p_value_recall": get_p_value(nr['r'], r['r']),
                        "p_value_f1": get_p_value(nr['f1'], r['f1'])
                    }
                }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=4, ensure_ascii=False)
    print(f"File saved: {path}")

# ==========================================
# 4. EXECUTION
# ==========================================

ids_to_process = [1, 2, 3] 
newspapers = [NEWSPAPER_A, NEWSPAPER_B]

no_rag_raw = {np: {} for np in newspapers}
rag_raw = {np: {} for np in newspapers}
real_data = {np: None for np in newspapers}

for paper in newspapers:
    baseline_done = False
    for llm_id in ids_to_process:
        print(f"Processing {paper} | LLM {llm_id}")
        
        data_nr = get_synthetic_news_from_json(INPUT_NORAG, paper, llm_id)
        data_r = get_synthetic_news_from_json(INPUT_RAG, paper, llm_id)

        if not baseline_done and data_nr:
            real_data[paper] = calculate_real_baseline(data_nr)
            baseline_done = True

        process_batch(data_r)
        process_batch(data_nr)

        rag_raw[paper][llm_id] = extract_metrics_by_feature(data_r)
        no_rag_raw[paper][llm_id] = extract_metrics_by_feature(data_nr)

save_structured_json(JSON_RESULTS, no_rag_raw, rag_raw, real_data)