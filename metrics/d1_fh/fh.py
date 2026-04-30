import pandas as pd
import numpy as np
import spacy
import textstat
import re
import html
import sys
import os
import json
from scipy.stats import ttest_ind 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from news_queries.news_data import (
    get_synthetic_news_from_json,
    get_real_news_from_json
)

JSON_RESULTS = "fh_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
INPUT_REAL = "../reales.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"

def remove_html(text):
    """Remove HTML tags, URLs, and noisy tokens from text."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    trash_patterns = r'\b(nbsp|leer|más|ver más|el mundo)\b'
    text = re.sub(trash_patterns, "", text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================
# LIBRARY CONFIGURATION
# ==========================================

textstat.set_lang('es')

print("Loading SpaCy model...")
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Downloading es_core_news_sm model...")
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

nlp.max_length = 2000000


# ==========================================
# READABILITY FUNCTIONS
# ==========================================

def calculate_fernandez_huerta(text):
    """Compute Fernandez-Huerta readability score."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    try:
        clean_text = html.unescape(text)
        return textstat.fernandez_huerta(clean_text)
    except Exception:
        return np.nan


def process_batch(news_list, text_key, newspaper_name):
    """Clean text and compute readability for each news item."""
    for news in news_list:
        news[text_key] = remove_html(news.get(text_key))
        text = news.get(text_key)

        fh = calculate_fernandez_huerta(text)
        news["fernandez_huerta"] = fh
    return None


def initialize_empty_structure(newspaper_list):
    """Create nested structure for storing FH metrics."""
    data = {}
    for newspaper in newspaper_list:
        data[newspaper] = {}
        for llm in [1, 2, 3]:
            data[newspaper][llm] = {}
            for feature in [1, 2, 3]:
                data[newspaper][llm][feature] = {
                    'fh': 0.0
                }
    return data


def calculate_means(news, results, is_real=False):
    """Compute mean FH scores per feature or globally."""
    fh_data = {}

    if not is_real:
        for feature in [1, 2, 3]:
            fh_data[feature] = {'count': 0, 'sum': 0}

        for item in news:
            feature_id = int(item.get('id_feature', -1))
            fh_data[feature_id]['count'] += 1
            fh_data[feature_id]['sum'] += item['fernandez_huerta']

        for feature in [1, 2, 3]:
            avg_fh = fh_data[feature]['sum'] / fh_data[feature]['count'] if fh_data[feature]['count'] > 0 else 0
            results[feature] = {
                'mean_fh': avg_fh
            }
    else:
        fh_data = {'count': 0, 'sum': 0}

        for item in news:
            fh_data['count'] += 1
            fh_data['sum'] += item['fernandez_huerta']

        results['mean_fh'] = fh_data['sum'] / fh_data['count']


def calculate_t_student(news_no_rag, news_rag, p_values_dict):
    """Compute T-test p-values for FH between RAG and NO-RAG."""
    for feature in [1, 2, 3]:
        fh_no_rag = [n['fernandez_huerta'] for n in news_no_rag if int(n.get('id_feature', -1)) == feature and not np.isnan(n['fernandez_huerta'])]
        fh_rag = [n['fernandez_huerta'] for n in news_rag if int(n.get('id_feature', -1)) == feature and not np.isnan(n['fernandez_huerta'])]
        
        if len(fh_no_rag) > 1 and len(fh_rag) > 1:
            _, p_fh = ttest_ind(fh_no_rag, fh_rag, nan_policy='omit')
        else:
            p_fh = None

        p_values_dict[feature] = {
            'p_value_fh': float(p_fh) if p_fh is not None else None
        }


def save_structured_json(path, means_no_rag, means_rag, means_real, p_values):
    """Save aggregated results and p-values to JSON."""
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


# ==========================================
# MAIN
# ==========================================

ids_to_process = [1, 2, 3]

results_no_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
results_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
p_values_results = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])

results_real = {
    NEWSPAPER_A: {'mean_fh': 0},
    NEWSPAPER_B: {'mean_fh': 0}
}

for newspaper in [NEWSPAPER_A, NEWSPAPER_B]:

    real_content = {}
    real_content[newspaper] = get_real_news_from_json("../reales.json", newspaper)

    for llm_id in ids_to_process:
        print(f"\n>>>> STARTING PROCESS FOR ID_LLM = {llm_id} <<<<")

        synthetic_rag = {}
        synthetic_no_rag = {}

        print(f"Downloading: {newspaper} (ID {llm_id})...")

        synthetic_rag[newspaper] = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag[newspaper] = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(real_content.get(newspaper, []), "description", newspaper)
        process_batch(synthetic_rag.get(newspaper, []), "synthetic_description", newspaper)
        process_batch(synthetic_no_rag.get(newspaper, []), "synthetic_description", newspaper)

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