import json
import re
import sys
import os
import spacy
import math
import numpy as np
from collections import Counter
from statistics import stdev
from time import perf_counter_ns
from scipy.stats import t, ttest_ind

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

NANOSECONDS_PER_SECOND = 1_000_000_000


def summarize_timing_samples(samples_ns):
    """Calculate total, mean, and two-sided Student 95% CI."""
    sample_size = len(samples_ns)
    total_ns = sum(samples_ns)
    if sample_size == 0:
        return {
            "n_news": 0,
            "total_seconds": 0.0,
            "mean_seconds_per_news": None,
            "ci95_seconds_per_news": {"lower": None, "upper": None},
        }

    mean_ns = total_ns / sample_size
    if sample_size == 1:
        interval = {"lower": None, "upper": None}
    else:
        standard_error_ns = stdev(samples_ns) / math.sqrt(sample_size)
        margin_ns = float(t.ppf(0.975, df=sample_size - 1)) * standard_error_ns
        interval = {
            "lower": (mean_ns - margin_ns) / NANOSECONDS_PER_SECOND,
            "upper": (mean_ns + margin_ns) / NANOSECONDS_PER_SECOND,
        }

    return {
        "n_news": sample_size,
        "total_seconds": total_ns / NANOSECONDS_PER_SECOND,
        "mean_seconds_per_news": mean_ns / NANOSECONDS_PER_SECOND,
        "ci95_seconds_per_news": interval,
    }


class MetricTimingCollector:
    """Time one complete metric call for every input entry."""

    def __init__(self, clock_ns=perf_counter_ns):
        self.clock_ns = clock_ns
        self.samples_ns = []

    def measure(self, operation, *args):
        started_ns = self.clock_ns()
        result = operation(*args)
        self.samples_ns.append(self.clock_ns() - started_ns)
        return result

    def as_dict(self):
        return {
            "unit": "seconds",
            **summarize_timing_samples(self.samples_ns),
        }


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


def calculate_entity_type_entropy(text):
    """Calculate entity type entropy for one already-cleaned text."""
    if not text:
        return 0.0

    doc = nlp(text)
    entity_types = [ent.label_ for ent in doc.ents]
    return calculate_shannon_entropy(entity_types)


def process_batch(news_list, text_key, timing_collector=None):
    """Processes a batch and computes entity type entropy."""
    for news in news_list:
        raw_text = news.get(text_key, "")
        clean_text = remove_html(raw_text)

        if timing_collector is None:
            entropy_val = calculate_entity_type_entropy(clean_text)
        else:
            entropy_val = timing_collector.measure(
                calculate_entity_type_entropy,
                clean_text,
            )

        news["entity_type_entropy"] = round(entropy_val, 4)


def calculate_mean_and_ci95(values, confidence=0.95, decimals=4):
    """Calculate mean and two-sided Student 95% CI for valid values."""
    clean_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(numeric_value):
            clean_values.append(numeric_value)

    if not clean_values:
        return {
            "mean": 0.0,
            "ci95": {"lower": None, "upper": None},
            "n": 0
        }

    mean_value = float(np.mean(clean_values))
    sample_size = len(clean_values)
    if sample_size < 2:
        return {
            "mean": round(mean_value, decimals),
            "ci95": {"lower": None, "upper": None},
            "n": sample_size
        }

    standard_error = np.std(clean_values, ddof=1) / np.sqrt(sample_size)
    alpha = 1 - confidence
    margin = t.ppf(1 - alpha / 2, df=sample_size - 1) * standard_error
    return {
        "mean": round(mean_value, decimals),
        "ci95": {
            "lower": round(mean_value - margin, decimals),
            "upper": round(mean_value + margin, decimals)
        },
        "n": sample_size
    }


def calculate_means(news, results, is_real=False):
    """Calculates mean entity type entropy."""
    if not is_real:
        for feature in [1, 2, 3]:
            feature_news = [
                item for item in news
                if int(item.get('id_feature', -1)) == feature
            ]
            entropy_stats = calculate_mean_and_ci95(
                [item.get('entity_type_entropy', np.nan) for item in feature_news]
            )

            results[feature] = {
                'mean_entity_entropy': entropy_stats['mean'],
                'ci95_entity_entropy': entropy_stats['ci95'],
                'n_entity_entropy': entropy_stats['n']
            }
    else:
        entropy_stats = calculate_mean_and_ci95(
            [item.get('entity_type_entropy', np.nan) for item in news]
        )
        results['mean_entity_entropy'] = entropy_stats['mean']
        results['ci95_entity_entropy'] = entropy_stats['ci95']
        results['n_entity_entropy'] = entropy_stats['n']


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


def save_structured_json(
    path,
    means_no_rag,
    means_rag,
    means_real,
    p_values,
    calculation_times
):
    """Saves structured entropy results to JSON."""
    final_structure = {}

    for newspaper in means_no_rag.keys():
        final_structure[newspaper] = {
            "real_reference_metrics": means_real.get(newspaper, {}),
            "llm_models": {},
            "calculation_times": calculation_times.get(newspaper, {})
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
newspapers = [NEWSPAPER_A, NEWSPAPER_B]

results_no_rag = initialize_empty_structure(newspapers)
results_rag = initialize_empty_structure(newspapers)
p_values_results = initialize_empty_structure(newspapers, include_p_values=True)

results_real = {
    NEWSPAPER_A: {'mean_entity_entropy': 0},
    NEWSPAPER_B: {'mean_entity_entropy': 0}
}

ete_timers = {
    newspaper: MetricTimingCollector()
    for newspaper in newspapers
}

# --- EXECUTION ---
for newspaper in newspapers:

    real_news = get_real_news_from_json(INPUT_REAL, newspaper)

    process_batch(real_news, "description", ete_timers[newspaper])
    calculate_means(
        real_news,
        results_real[newspaper],
        is_real=True
    )

    for llm_id in ids_to_process:
        print(f"Processing: {newspaper} | LLM ID: {llm_id}")

        synthetic_rag = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(synthetic_rag, "synthetic_description", ete_timers[newspaper])
        process_batch(synthetic_no_rag, "synthetic_description", ete_timers[newspaper])

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
    p_values_results,
    {
        newspaper: {
            "entity_type_entropy": ete_timers[newspaper].as_dict()
        }
        for newspaper in newspapers
    }
)
