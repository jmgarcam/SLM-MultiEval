import pandas as pd
import numpy as np
import spacy
import textstat
import re
import html
import sys
import os
import json
from math import sqrt
from statistics import stdev
from time import perf_counter_ns
from scipy.stats import t, ttest_ind

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

NANOSECONDS_PER_SECOND = 1_000_000_000
BOILERPLATE_RE = re.compile(r"\b(?:leer|ver)\s+más\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"http\S+|www\S+", flags=re.IGNORECASE)


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

def clean_metric_text(value):
    """Remove markup, URLs, and explicit read-more boilerplate only."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\bnbsp\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def remove_html(text):
    """Backward-compatible wrapper around the metric cleaner."""
    return clean_metric_text(text)


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


def process_batch(news_list, text_key, newspaper_name, fernandez_huerta_timer=None):
    """Clean text and compute readability for each news item."""
    for news in news_list:
        news[text_key] = remove_html(news.get(text_key))
        text = news.get(text_key)

        if fernandez_huerta_timer is None:
            fh = calculate_fernandez_huerta(text)
        else:
            fh = fernandez_huerta_timer.measure(
                calculate_fernandez_huerta,
                text
            )
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
    """Compute mean FH scores, sample sizes, and 95% confidence intervals."""
    if not is_real:
        for feature in [1, 2, 3]:
            feature_values = [
                item.get("fernandez_huerta", np.nan)
                for item in news
                if int(item.get("id_feature", -1)) == feature
            ]
            stats = calculate_mean_and_ci95(feature_values)
            results[feature] = {
                "mean_fh": stats["mean"],
                "ci95_fh": stats["ci95"],
                "n_fh": stats["n"]
            }
    else:
        stats = calculate_mean_and_ci95([
            item.get("fernandez_huerta", np.nan)
            for item in news
        ])
        results["mean_fh"] = stats["mean"]
        results["ci95_fh"] = stats["ci95"]
        results["n_fh"] = stats["n"]


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


def save_structured_json(
    path,
    means_no_rag,
    means_rag,
    means_real,
    p_values,
    calculation_times
):
    """Save aggregated results and p-values to JSON."""
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


# ==========================================
# MAIN
# ==========================================

ids_to_process = [1, 2, 3]
newspapers = [NEWSPAPER_A, NEWSPAPER_B]

results_no_rag = initialize_empty_structure(newspapers)
results_rag = initialize_empty_structure(newspapers)
p_values_results = initialize_empty_structure(newspapers)

results_real = {
    NEWSPAPER_A: {'mean_fh': 0},
    NEWSPAPER_B: {'mean_fh': 0}
}

fernandez_huerta_timers = {
    newspaper: MetricTimingCollector()
    for newspaper in newspapers
}

for newspaper in newspapers:

    real_content = {}
    real_content[newspaper] = get_real_news_from_json("../reales.json", newspaper)

    process_batch(
        real_content.get(newspaper, []),
        "description",
        newspaper,
        fernandez_huerta_timer=fernandez_huerta_timers[newspaper]
    )
    calculate_means(
        real_content.get(newspaper, []),
        results_real[newspaper],
        is_real=True
    )

    for llm_id in ids_to_process:
        print(f"\n>>>> STARTING PROCESS FOR ID_LLM = {llm_id} <<<<")

        synthetic_rag = {}
        synthetic_no_rag = {}

        print(f"Downloading: {newspaper} (ID {llm_id})...")

        synthetic_rag[newspaper] = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id)
        synthetic_no_rag[newspaper] = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id)

        process_batch(
            synthetic_rag.get(newspaper, []),
            "synthetic_description",
            newspaper,
            fernandez_huerta_timer=fernandez_huerta_timers[newspaper]
        )
        process_batch(
            synthetic_no_rag.get(newspaper, []),
            "synthetic_description",
            newspaper,
            fernandez_huerta_timer=fernandez_huerta_timers[newspaper]
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
            "fernandez_huerta": fernandez_huerta_timers[newspaper].as_dict()
        }
        for newspaper in newspapers
    }
)
