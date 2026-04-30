import pandas as pd
import numpy as np
import spacy
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

JSON_RESULTS = "tree_depth_results.json"
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
# METRIC FUNCTION
# ==========================================

def calculate_syntactic_depth(text):
    """Compute average dependency tree depth per sentence."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    try:
        clean_text = html.unescape(text)
        doc = nlp(clean_text)
        sentence_depths = []

        def get_subtree_depth(node):
            if not list(node.children):
                return 1
            return 1 + max(get_subtree_depth(child) for child in node.children)

        for sent in doc.sents:
            root = sent.root
            if root:
                depth = get_subtree_depth(root)
                sentence_depths.append(depth)

        if not sentence_depths:
            return 0.0

        return np.mean(sentence_depths)

    except Exception as e:
        print(f"Error calculating depth: {e}")
        return np.nan


def process_batch(news_list, text_key, newspaper_name):
    """Clean text and compute syntactic depth for each item."""
    for news in news_list:
        news[text_key] = remove_html(news.get(text_key))
        text = news.get(text_key)

        depth = float(calculate_syntactic_depth(text))
        news["tree_depth"] = depth
    return None


def initialize_empty_structure(newspaper_list):
    """Create nested dictionary structure for storing results."""
    data = {}
    for newspaper in newspaper_list:
        data[newspaper] = {}
        for llm in [1, 2, 3]:
            data[newspaper][llm] = {}
            for feature in [1, 2, 3]:
                data[newspaper][llm][feature] = {
                    'tree': 0.0
                }
    return data


def calculate_means(news, results, is_real=False):
    """Compute mean syntactic depth per feature or globally."""
    tree_depth_data = {}

    if not is_real:
        for feature in [1, 2, 3]:
            tree_depth_data[feature] = {'count': 0, 'sum': 0}

        for item in news:
            feature_id = int(item.get('id_feature', -1))
            tree_depth_data[feature_id]['count'] += 1
            tree_depth_data[feature_id]['sum'] += item['tree_depth']

        for feature in [1, 2, 3]:
            avg_depth = (
                tree_depth_data[feature]['sum'] / tree_depth_data[feature]['count']
                if tree_depth_data[feature]['count'] > 0 else 0
            )
            results[feature] = {
                'mean_depth': avg_depth
            }
    else:
        tree_depth_data = {'count': 0, 'sum': 0}

        for item in news:
            tree_depth_data['count'] += 1
            tree_depth_data['sum'] += item['tree_depth']

        results['mean_depth'] = tree_depth_data['sum'] / tree_depth_data['count']


def calculate_t_student(news_no_rag, news_rag, p_values_dict):
    """Compute T-test p-values comparing RAG vs NO-RAG depth."""
    for feature in [1, 2, 3]:
        tree_no_rag = [n['tree_depth'] for n in news_no_rag if int(n.get('id_feature', -1)) == feature and not np.isnan(n['tree_depth'])]
        tree_rag = [n['tree_depth'] for n in news_rag if int(n.get('id_feature', -1)) == feature and not np.isnan(n['tree_depth'])]

        if len(tree_no_rag) > 1 and len(tree_rag) > 1:
            _, p_tree = ttest_ind(tree_no_rag, tree_rag, nan_policy='omit')
        else:
            p_tree = None

        p_values_dict[feature] = {
            'p_value_tree': float(p_tree) if p_tree is not None else None
        }


def save_structured_json(path, means_no_rag, means_rag, means_real, p_values):
    """Save final results structure to JSON file."""
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
    NEWSPAPER_A: {'mean_depth': 0},
    NEWSPAPER_B: {'mean_depth': 0}
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