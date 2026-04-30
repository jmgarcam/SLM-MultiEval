import json
import re
import sys
import os
import numpy as np
from scipy.stats import ttest_ind
from lexicalrichness import LexicalRichness

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from news_queries.news_data import (
    get_real_news_from_json,
    get_synthetic_news_from_json
)

JSON_RESULTS = "mtld_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
INPUT_REAL = "../reales.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"

def remove_html(text):
    """
    Removes HTML tags, non-breaking spaces, and specific stop words from a string.
    """
    if not text: 
        return ""
    
    text = text.replace('&nbsp;', ' ')
    clean_tags = re.compile('<.*?>')
    text = re.sub(clean_tags, ' ', text)
    text = text.replace('Leer', '')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_batch(news_list, text_key):
    """
    Iterates through a list of news dictionaries, cleans the text, calculates 
    the MTLD score, and appends it to each dictionary.
    """
    for news in news_list:
        raw_text = news.get(text_key, "")
        clean_text = remove_html(raw_text)
        
        mtld_val = 0.0
        
        if clean_text and len(clean_text.split()) > 10:
            try:
                lex = LexicalRichness(clean_text)
                mtld_val = lex.mtld(threshold=0.72)
            except ValueError:
                mtld_val = 0.0
        
        news["mtld"] = round(mtld_val, 2)

def calculate_means(news, results, is_real=False):
    """
    Calculates the mean MTLD scores for a list of news items and stores 
    the results in the provided dictionary, grouping by features if synthetic.
    """
    mtld_data = {}
    
    if not is_real:
        for feature in [1, 2, 3]:
            mtld_data[feature] = {'count': 0, 'sum': 0}      

        for item in news:
            feature_id = int(item.get('id_feature', -1))
            mtld_data[feature_id]['count'] += 1
            mtld_data[feature_id]['sum'] += item['mtld']            

        for feature in [1, 2, 3]:
            avg_mtld = mtld_data[feature]['sum'] / mtld_data[feature]['count'] if mtld_data[feature]['count'] > 0 else 0
            
            results[feature] = {
                'mean_mtld': avg_mtld,
            }

    else:
        mtld_data = {'count': 0, 'sum': 0}

        for item in news:
            mtld_data['count'] += 1
            mtld_data['sum'] += item['mtld']
         
        results['mean_mtld'] = mtld_data['sum'] / mtld_data['count'] if mtld_data['count'] > 0 else 0

def calculate_t_student(news_no_rag, news_rag, p_values_dict):
    """
    Calculates independent T-Student p-values comparing MTLD scores 
    with and without RAG for each feature.
    """
    for feature in [1, 2, 3]:
        mtld_no_rag = [n['mtld'] for n in news_no_rag if int(n.get('id_feature', -1)) == feature and n['mtld'] > 0]
        mtld_rag = [n['mtld'] for n in news_rag if int(n.get('id_feature', -1)) == feature and n['mtld'] > 0]

        if len(mtld_no_rag) > 1 and len(mtld_rag) > 1:
            _, p_mtld = ttest_ind(mtld_no_rag, mtld_rag, nan_policy='omit')
        else:
            p_mtld = None
            
        p_values_dict[feature] = {
            'p_value_mtld': float(p_mtld) if p_mtld is not None else None
        }

def initialize_empty_structure(newspaper_list, include_p_values=False):
    """
    Creates and returns an empty nested dictionary structure to store the metrics.
    """
    data = {}

    for newspaper in newspaper_list:
        data[newspaper] = {}
        for llm in [1, 2, 3]:
            data[newspaper][llm] = {}
            for feature in [1, 2, 3]:
                if include_p_values:
                    data[newspaper][llm][feature] = {
                        'p_value_mtld': None
                    }
                else:
                    data[newspaper][llm][feature] = {
                        'mtld': 0.0,
                    }

    return data

def save_structured_json(path, means_no_rag, means_rag, means_real, p_values):
    """
    Constructs the final nested dictionary with all calculated means and 
    p-values, and saves it to a JSON file.
    """
    final_structure = {}

    for newspaper in means_no_rag.keys():
        final_structure[newspaper] = {
            "real_reference_metrics": means_real.get(newspaper, {}),
            "llm_models": {}
        }

        for llm_id, features_dict in means_no_rag[newspaper].items():
            final_structure[newspaper]["llm_models"][llm_id] = {}

            for feature_id in features_dict.keys():
                final_structure[newspaper]["llm_models"][llm_id][feature_id] = {
                    "RAG": means_rag[newspaper][llm_id].get(feature_id, {}),
                    "NO_RAG": means_no_rag[newspaper][llm_id].get(feature_id, {}),
                    "T_Student_p_values": p_values[newspaper][llm_id].get(feature_id, {})
                }

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_structure, f, indent=4, ensure_ascii=False)
        print(f"File successfully saved at: {path}")
    except Exception as e:
        print(f"Error saving file: {e}")
       

ids_to_process = [1, 2, 3]

results_no_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
results_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
p_values_results = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B], include_p_values=True)

results_real = {
    NEWSPAPER_A: {'mean_mtld': 0},
    NEWSPAPER_B: {'mean_mtld': 0}
}

for newspaper in [NEWSPAPER_A, NEWSPAPER_B]:

    real_content = {}
    real_content[newspaper] = get_real_news_from_json(INPUT_REAL, newspaper)

    for llm_id in ids_to_process:
        print(f"Processing: {newspaper}...")

        synthetic_rag = {}
        synthetic_no_rag = {}

        print(f"Downloading: {newspaper} (ID {llm_id})...")

        synthetic_rag[newspaper] = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag[newspaper] = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(real_content[newspaper], "description")
        process_batch(synthetic_rag[newspaper], "synthetic_description")
        process_batch(synthetic_no_rag[newspaper], "synthetic_description")

        calculate_means(
            synthetic_no_rag.get(newspaper, []),
            results_no_rag[newspaper][llm_id],
            is_real=False
        )
        calculate_means(
            synthetic_rag.get(newspaper, []),
            results_rag[newspaper][llm_id],
            is_real=False
        )
        calculate_means(
            real_content.get(newspaper, []),
            results_real[newspaper],
            is_real=True
        )
        
        calculate_t_student(
            synthetic_no_rag.get(newspaper, []), 
            synthetic_rag.get(newspaper, []), 
            p_values_results[newspaper][llm_id]
        )
       
save_structured_json(
    JSON_RESULTS, 
    results_no_rag, 
    results_rag, 
    results_real, 
    p_values_results
)