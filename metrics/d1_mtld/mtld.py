import json
import re
import sys
import os
import numpy as np
from math import sqrt
from statistics import stdev
from time import perf_counter_ns
from scipy.stats import t, ttest_ind
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
        standard_error_ns = stdev(samples_ns) / sqrt(sample_size)
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


def calculate_mtld(text):
    """Calculate MTLD for one already-cleaned text."""
    if not text or len(text.split()) <= 10:
        return 0.0

    try:
        lex = LexicalRichness(text)
        return lex.mtld(threshold=0.72)
    except ValueError:
        return 0.0


def process_batch(news_list, text_key, timing_collector=None):
    """
    Iterates through a list of news dictionaries, cleans the text, calculates 
    the MTLD score, and appends it to each dictionary.
    """
    for news in news_list:
        raw_text = news.get(text_key, "")
        clean_text = remove_html(raw_text)
        
        if timing_collector is None:
            mtld_val = calculate_mtld(clean_text)
        else:
            mtld_val = timing_collector.measure(calculate_mtld, clean_text)
        
        news["mtld"] = round(mtld_val, 2)


def calculate_mean_and_ci95(values, confidence=0.95):
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
            "mean": mean_value,
            "ci95": {"lower": None, "upper": None},
            "n": sample_size
        }

    standard_error = np.std(clean_values, ddof=1) / np.sqrt(sample_size)
    alpha = 1 - confidence
    margin = t.ppf(1 - alpha / 2, df=sample_size - 1) * standard_error
    return {
        "mean": mean_value,
        "ci95": {
            "lower": float(mean_value - margin),
            "upper": float(mean_value + margin)
        },
        "n": sample_size
    }


def calculate_means(news, results, is_real=False):
    """
    Calculates the mean MTLD scores for a list of news items and stores 
    the results in the provided dictionary, grouping by features if synthetic.
    """
    if not is_real:
        for feature in [1, 2, 3]:
            feature_values = [
                item.get("mtld", np.nan)
                for item in news
                if int(item.get("id_feature", -1)) == feature
            ]
            stats = calculate_mean_and_ci95(feature_values)
            results[feature] = {
                "mean_mtld": stats["mean"],
                "ci95_mtld": stats["ci95"],
                "n_mtld": stats["n"]
            }
    else:
        stats = calculate_mean_and_ci95([
            item.get("mtld", np.nan)
            for item in news
        ])
        results["mean_mtld"] = stats["mean"]
        results["ci95_mtld"] = stats["ci95"]
        results["n_mtld"] = stats["n"]

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

def save_structured_json(
    path,
    means_no_rag,
    means_rag,
    means_real,
    p_values,
    calculation_times
):
    """
    Constructs the final nested dictionary with all calculated means and 
    p-values, and saves it to a JSON file.
    """
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

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_structure, f, indent=4, ensure_ascii=False)
        print(f"File successfully saved at: {path}")
    except Exception as e:
        print(f"Error saving file: {e}")
       

ids_to_process = [1, 2, 3]
newspapers = [NEWSPAPER_A, NEWSPAPER_B]

results_no_rag = initialize_empty_structure(newspapers)
results_rag = initialize_empty_structure(newspapers)
p_values_results = initialize_empty_structure(newspapers, include_p_values=True)

results_real = {
    NEWSPAPER_A: {'mean_mtld': 0},
    NEWSPAPER_B: {'mean_mtld': 0}
}

mtld_timers = {
    newspaper: MetricTimingCollector()
    for newspaper in newspapers
}

for newspaper in newspapers:

    real_content = {}
    real_content[newspaper] = get_real_news_from_json(INPUT_REAL, newspaper)

    process_batch(
        real_content[newspaper],
        "description",
        mtld_timers[newspaper]
    )
    calculate_means(
        real_content.get(newspaper, []),
        results_real[newspaper],
        is_real=True
    )

    for llm_id in ids_to_process:
        print(f"Processing: {newspaper}...")

        synthetic_rag = {}
        synthetic_no_rag = {}

        print(f"Downloading: {newspaper} (ID {llm_id})...")

        synthetic_rag[newspaper] = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag[newspaper] = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(
            synthetic_rag[newspaper],
            "synthetic_description",
            mtld_timers[newspaper]
        )
        process_batch(
            synthetic_no_rag[newspaper],
            "synthetic_description",
            mtld_timers[newspaper]
        )

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
    p_values_results,
    {
        newspaper: {
            "mtld": mtld_timers[newspaper].as_dict()
        }
        for newspaper in newspapers
    }
)
