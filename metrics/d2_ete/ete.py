import json
import re
import sys
import os
import spacy
import math
from collections import Counter
from scipy.stats import ttest_ind

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from news_queries.news_data import (
    get_synthetic_news_from_json,
    get_real_news_from_json
)

JSON_RESULTS = "ete_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
INPUT_REAL = "../reales.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"


# --- LOAD SPACY MODEL ---
print("Loading spaCy Spanish model...")
try:
    nlp = spacy.load("es_core_news_sm")
    print("Model loaded successfully.")
except OSError:
    print("Error: spaCy model not found. Run: python -m spacy download es_core_news_sm")
    sys.exit(1)


# --- HELPER FUNCTIONS ---
def remove_html(text):
    """Removes HTML tags and formatting noise from text."""
    if not text:
        return ""

    text = text.replace('&nbsp;', ' ')
    clean_tags = re.compile('<.*?>')
    text = re.sub(clean_tags, ' ', text)
    text = text.replace('Leer', '')
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def calculate_shannon_entropy(labels_list):
    """Computes Shannon entropy over entity type labels."""
    total_count = len(labels_list)
    if total_count == 0:
        return 0.0

    counts = Counter(labels_list)
    entropy = 0.0

    for count in counts.values():
        p = count / total_count
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def process_batch(news_list, text_key):
    """Processes a batch and computes entity type entropy."""
    for news in news_list:
        raw_text = news.get(text_key, "")
        clean_text = remove_html(raw_text)

        entropy_val = 0.0

        if clean_text:
            doc = nlp(clean_text)
            entity_types = [ent.label_ for ent in doc.ents]
            entropy_val = calculate_shannon_entropy(entity_types)

        news["entity_type_entropy"] = round(entropy_val, 4)


def calculate_means(news, results, is_real=False):
    """Calculates mean entity type entropy."""
    metrics = {}

    if not is_real:
        for feature in [1, 2, 3]:
            metrics[feature] = {'count': 0, 'sum': 0.0}

        for item in news:
            feature_id = int(item.get('id_feature', -1))
            if feature_id in metrics:
                metrics[feature_id]['count'] += 1
                metrics[feature_id]['sum'] += item['entity_type_entropy']

        for feature in [1, 2, 3]:
            avg_val = (
                metrics[feature]['sum'] / metrics[feature]['count']
                if metrics[feature]['count'] > 0 else 0.0
            )

            results[feature] = {
                'mean_entity_entropy': round(avg_val, 4),
            }

    else:
        metrics = {'count': 0, 'sum': 0.0}

        for item in news:
            metrics['count'] += 1
            metrics['sum'] += item['entity_type_entropy']

        results['mean_entity_entropy'] = (
            metrics['sum'] / metrics['count']
            if metrics['count'] > 0 else 0.0
        )


def calculate_t_student(news_no_rag, news_rag, p_values_dict):
    """Calculates independent Student's t-test comparing entropy with and without RAG."""
    for feature in [1, 2, 3]:
        entropy_no_rag = [
            n['entity_type_entropy'] for n in news_no_rag
            if int(n.get('id_feature', -1)) == feature and n['entity_type_entropy'] > 0
        ]

        entropy_rag = [
            n['entity_type_entropy'] for n in news_rag
            if int(n.get('id_feature', -1)) == feature and n['entity_type_entropy'] > 0
        ]

        if len(entropy_no_rag) > 1 and len(entropy_rag) > 1:
            _, p_entropy = ttest_ind(entropy_no_rag, entropy_rag, nan_policy='omit')
        else:
            p_entropy = None

        p_values_dict[feature] = {
            'p_value_entropy': float(p_entropy) if p_entropy is not None else None
        }


def initialize_empty_structure(newspaper_list, include_p_values=False):
    """Creates empty nested dictionary structure."""
    data = {}

    for newspaper in newspaper_list:
        data[newspaper] = {}
        for llm in [1, 2, 3]:
            data[newspaper][llm] = {}
            for feature in [1, 2, 3]:
                if include_p_values:
                    data[newspaper][llm][feature] = {
                        'p_value_entropy': None
                    }
                else:
                    data[newspaper][llm][feature] = {
                        'entity_type_entropy': 0.0,
                    }

    return data


def save_structured_json(path, means_no_rag, means_rag, means_real, p_values):
    """Saves structured entropy results to JSON."""
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

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=4, ensure_ascii=False)

    print(f"File successfully saved at: {path}")


ids_to_process = [1, 2, 3]

results_no_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
results_rag = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B])
p_values_results = initialize_empty_structure([NEWSPAPER_A, NEWSPAPER_B], include_p_values=True)

results_real = {
    NEWSPAPER_A: {'mean_entity_entropy': 0},
    NEWSPAPER_B: {'mean_entity_entropy': 0}
}

# --- EXECUTION ---
for newspaper in [NEWSPAPER_A, NEWSPAPER_B]:

    real_news = get_real_news_from_json(INPUT_REAL, newspaper)

    for llm_id in ids_to_process:
        print(f"Processing: {newspaper} | LLM ID: {llm_id}")

        synthetic_rag = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(real_news, "description")
        process_batch(synthetic_rag, "synthetic_description")
        process_batch(synthetic_no_rag, "synthetic_description")

        calculate_means(
            synthetic_no_rag,
            results_no_rag[newspaper][llm_id],
            is_real=False
        )

        calculate_means(
            synthetic_rag,
            results_rag[newspaper][llm_id],
            is_real=False
        )

        calculate_means(
            real_news,
            results_real[newspaper],
            is_real=True
        )

        calculate_t_student(
            synthetic_no_rag,
            synthetic_rag,
            p_values_results[newspaper][llm_id]
        )

# --- FINAL SAVE ---
save_structured_json(
    JSON_RESULTS,
    results_no_rag,
    results_rag,
    results_real,
    p_values_results
)