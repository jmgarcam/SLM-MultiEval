import json
import re
import sys
import os
import spacy
from scipy.stats import ttest_ind

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from news_queries.news_data import (
    get_synthetic_news_from_json,
    get_real_news_from_json
)


JSON_RESULTS = "uer_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
INPUT_REAL = "../reales.json"
NEWSPAPER_A = "NEWSPAPER_A"
NEWSPAPER_B = "NEWSPAPER_B"
OUTPUT_DIR = "."

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory verified: {OUTPUT_DIR}")

# --- GLOBAL SPACY MODEL LOADING ---
print("Loading spaCy Spanish model...")
try:
    nlp = spacy.load("es_core_news_sm")
    print("Model loaded successfully.")
except OSError:
    print("Error: spaCy model not found.")
    print("Run: python -m spacy download es_core_news_sm")
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


def calculate_uer(text):
    """Computes Unique Entity Ratio (UER) for a given text."""
    if not text:
        return 0.0

    doc = nlp(text)
    total_tokens = len(doc)

    if total_tokens == 0:
        return 0.0

    entities = [ent.text.lower() for ent in doc.ents]
    unique_entities_count = len(set(entities))

    return round(unique_entities_count / total_tokens, 4)


def process_batch(news_list, text_key):
    """Processes a batch of news and adds UER to each item."""
    for news in news_list:
        raw_text = news.get(text_key, "")
        clean_text = remove_html(raw_text)
        news["uer"] = calculate_uer(clean_text)


def calculate_means(news, results, is_real=False):
    """Calculates mean UER grouped by feature if synthetic."""
    uer_data = {}

    if not is_real:
        for feature in [1, 2, 3]:
            uer_data[feature] = {'count': 0, 'sum': 0.0}

        for item in news:
            feature_id = int(item.get('id_feature', -1))
            if feature_id in uer_data:
                uer_data[feature_id]['count'] += 1
                uer_data[feature_id]['sum'] += item['uer']

        for feature in [1, 2, 3]:
            avg_uer = (
                uer_data[feature]['sum'] / uer_data[feature]['count']
                if uer_data[feature]['count'] > 0 else 0.0
            )

            results[feature] = {
                'mean_uer': round(avg_uer, 4),
            }

    else:
        uer_data = {'count': 0, 'sum': 0.0}

        for item in news:
            uer_data['count'] += 1
            uer_data['sum'] += item['uer']

        results['mean_uer'] = (
            uer_data['sum'] / uer_data['count']
            if uer_data['count'] > 0 else 0.0
        )


def calculate_t_student(news_no_rag, news_rag, p_values_dict):
    """Calculates independent Student's t-test comparing UER with and without RAG."""
    for feature in [1, 2, 3]:
        uer_no_rag = [
            n['uer'] for n in news_no_rag
            if int(n.get('id_feature', -1)) == feature and n['uer'] > 0
        ]

        uer_rag = [
            n['uer'] for n in news_rag
            if int(n.get('id_feature', -1)) == feature and n['uer'] > 0
        ]

        if len(uer_no_rag) > 1 and len(uer_rag) > 1:
            _, p_uer = ttest_ind(uer_no_rag, uer_rag, nan_policy='omit')
        else:
            p_uer = None

        p_values_dict[feature] = {
            'p_value_uer': float(p_uer) if p_uer is not None else None
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
                        'p_value_uer': None
                    }
                else:
                    data[newspaper][llm][feature] = {
                        'uer': 0.0,
                    }

    return data


def save_structured_json(path, means_no_rag, means_rag, means_real, p_values):
    """Saves final structured UER results to JSON."""
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
    NEWSPAPER_A: {'mean_uer': 0},
    NEWSPAPER_B: {'mean_uer': 0}
}

# --- EXECUTION ---
for newspaper in [NEWSPAPER_A, NEWSPAPER_B]:

    real_news = get_real_news_from_json(INPUT_REAL, newspaper)

    for llm_id in ids_to_process:
        print(f"Processing: {newspaper}...")

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
full_path = os.path.join(OUTPUT_DIR, JSON_RESULTS)

save_structured_json(
    full_path,
    results_no_rag,
    results_rag,
    results_real,
    p_values_results
)