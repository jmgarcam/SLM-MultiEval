import json
import re
import sys
import os
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from news_queries.news_data import get_synthetic_news, get_synthetic_news_from_json


JSON_RESULTS = "sbert_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"
OUTPUT_DIR = "."

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory verified: {OUTPUT_DIR}")

# --- GLOBAL MODEL LOADING ---
print("Loading SBERT model...")
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---
def remove_html(text):
    """Cleans HTML tags and noise from the text."""
    if not text: return ""
    text = text.replace('&nbsp;', ' ')
    clean_tags = re.compile('<.*?>')
    text = re.sub(clean_tags, ' ', text)
    text = text.replace('Leer', '') 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_paired_scores(text_list_a, text_list_b):
    """
    Calculates pair-wise cosine similarity (A[0] vs B[0], A[1] vs B[1]).
    Returns a list of floats.
    """
    if not text_list_a or not text_list_b:
        return []

    # Generate Embeddings
    emb_a = model.encode(text_list_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(text_list_b, convert_to_tensor=True, show_progress_bar=False)

    # Paired Cosine Calculation (1 to 1)
    # Using PyTorch cosine_similarity to avoid giant NxN matrices
    scores = torch.nn.functional.cosine_similarity(emb_a, emb_b)
    
    # Convert to rounded Python list
    return [round(s.item(), 4) for s in scores]

# --- SYNTHETIC VS HEADLINE PROCESSING ---
def process_synthetic_batch(news_list):
    candidates = []
    references = []
    valid_indices = []

    for i, news in enumerate(news_list):
        cand_text = remove_html(news.get("synthetic_description", "")) 
        ref_text = remove_html(news.get("headline", "")) 
        
        if cand_text and ref_text:
            candidates.append(cand_text)
            references.append(ref_text)
            valid_indices.append(i)
        else:
            news_list[i]["sbert_headline_raw"] = 0.0 

    if not candidates:
        return

    scores = calculate_paired_scores(candidates, references)

    for k, idx in enumerate(valid_indices):
        news_list[idx]["sbert_headline_raw"] = scores[k]

# --- HUMAN BASELINE CALCULATION (Real Desc vs Headline) ---
def calculate_human_baseline_stats(news_list):
    """
    Calculates the mean SBERT score between real_description and headline
    using a sample of news.
    """
    real_descs = []
    headlines = []
    
    for news in news_list:
        d = remove_html(news.get("real_description", ""))
        h = remove_html(news.get("headline", ""))
        if d and h:
            real_descs.append(d)
            headlines.append(h)
    
    if not real_descs:
        return {"mean_sbert": 0.0, "sample_size": 0}

    scores = calculate_paired_scores(real_descs, headlines)
    
    avg_score = sum(scores) / len(scores)
    
    return {
        "mean_sbert": round(avg_score, 4),
        "sample_size": len(scores)
    }

# --- MEAN AND STATS AGGREGATION ---
def calculate_means_synthetic(news, results):
    metrics = {}
    for feature in [1, 2, 3]:
        metrics[feature] = {'count': 0, 'sum': 0.0, 'raw_scores': []}      

    for item in news:
        feature_id = int(item.get('id_feature', -1))
        if feature_id in metrics and 'sbert_headline_raw' in item:
            val = item['sbert_headline_raw']
            metrics[feature_id]['count'] += 1
            metrics[feature_id]['sum'] += val            
            metrics[feature_id]['raw_scores'].append(val)

    for feature in [1, 2, 3]:
        count = metrics[feature]['count']
        avg_val = metrics[feature]['sum'] / count if count > 0 else 0.0
        results[str(feature)] = {
            'mean_sbert': round(avg_val, 4), 
            'sample_size': count,
            'raw_scores': metrics[feature]['raw_scores']
        }

def initialize_structure(newspaper_list):
    data = {}
    for newspaper in newspaper_list:
        data[newspaper] = {}
        for llm in ["1", "2", "3"]:
            data[newspaper][llm] = {}
            for feature in ["1", "2", "3"]:
                data[newspaper][llm][feature] = {}
    return data

def save_custom_json(path, data_no_rag, data_rag, baselines, t_tests):
    final_json = {}
    newspapers = data_no_rag.keys()

    for newspaper in newspapers:
        final_json[newspaper] = {
            "human_baseline": baselines.get(newspaper, {"mean_sbert": 0.0, "sample_size": 0}),
            "llm_models": {}
        }

        for llm_id in ["1", "2", "3"]:
            if llm_id not in final_json[newspaper]["llm_models"]:
                final_json[newspaper]["llm_models"][llm_id] = {}

            for feature_id in ["1", "2", "3"]:
                val_no_rag = data_no_rag[newspaper][llm_id].get(feature_id, {})
                val_rag = data_rag[newspaper][llm_id].get(feature_id, {})
                t_test_stats = t_tests[newspaper][llm_id].get(feature_id, {})

                final_json[newspaper]["llm_models"][llm_id][feature_id] = {
                    "NO_RAG": val_no_rag,
                    "RAG": val_rag,
                    "t_test_ind": t_test_stats
                }

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        print(f"JSON saved to: {path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# --- CONFIGURATION ---
API_IP = "192.168.1.82"
FLASK_API_PORT = 5010
LLM_NEWS_PORT = 6002

start_date = "26/10/2025"
end_date = "11/11/2025"
ids_to_process = [1, 2, 3]

# --- INITIALIZATION ---
results_no_rag = initialize_structure([NEWSPAPER_A, NEWSPAPER_B])
results_rag = initialize_structure([NEWSPAPER_A, NEWSPAPER_B])
t_test_results = initialize_structure([NEWSPAPER_A, NEWSPAPER_B])
human_baselines = {} 

# --- EXECUTION ---
for newspaper in [NEWSPAPER_A, NEWSPAPER_B]:
    baseline_calculated = False 

    for llm_id_int in ids_to_process:
        llm_id_str = str(llm_id_int)
        print(f"\n>>>> {newspaper} | LLM ID: {llm_id_str} | Processing... <<<<")

        synthetic_rag = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id_int)
        synthetic_no_rag = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id_int)
        # 2. Calculate Human Baseline
        if not baseline_calculated and synthetic_no_rag:
            print(f"Calculating Human Baseline for {newspaper} (Real Desc vs Headline)...")
            human_baselines[newspaper] = calculate_human_baseline_stats(synthetic_no_rag)
            print(f" -> Baseline Result: {human_baselines[newspaper]}")
            baseline_calculated = True

        # 3. Process Synthetic Data
        process_synthetic_batch(synthetic_rag)
        process_synthetic_batch(synthetic_no_rag)

        # 4. Calculate Synthetic Means and Collect Raw Scores
        calculate_means_synthetic(synthetic_no_rag, results_no_rag[newspaper][llm_id_str])
        calculate_means_synthetic(synthetic_rag, results_rag[newspaper][llm_id_str])

        # 5. Compute Student's t-test and Clean Up Raw Scores
        for feature_id in ["1", "2", "3"]:
            scores_nr = results_no_rag[newspaper][llm_id_str][feature_id].pop("raw_scores", [])
            scores_r = results_rag[newspaper][llm_id_str][feature_id].pop("raw_scores", [])

            if len(scores_nr) > 1 and len(scores_r) > 1:
                # Welch's t-test (assumes unequal variance)
                t_stat, p_val = stats.ttest_ind(scores_nr, scores_r, equal_var=False)
                t_test_results[newspaper][llm_id_str][feature_id] = {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_val, 4)
                }
            else:
                t_test_results[newspaper][llm_id_str][feature_id] = {
                    "t_statistic": None,
                    "p_value": None
                }

# --- FINAL SAVE ---
full_path = os.path.join(OUTPUT_DIR, JSON_RESULTS)
save_custom_json(full_path, results_no_rag, results_rag, human_baselines, t_test_results)