import json
import re
import sys
import os
import torch
import numpy as np
from math import sqrt
from statistics import stdev
from time import perf_counter_ns
from bert_score import BERTScorer
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from news_queries.news_data import  get_synthetic_news_from_json

JSON_RESULTS = "bert_results.json"
INPUT_NORAG = "../sinteticas_NORAG.json"
INPUT_RAG = "../sinteticas_RAG.json"
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
        margin_ns = float(stats.t.ppf(0.975, df=sample_size - 1)) * standard_error_ns
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
    """Time one complete BERTScore call for every input entry."""

    def __init__(self, synchronize=None, clock_ns=perf_counter_ns):
        self.synchronize = synchronize
        self.clock_ns = clock_ns
        self.samples_ns = []

    def _synchronize(self):
        if self.synchronize is not None:
            self.synchronize()

    def measure(self, operation, *args):
        self._synchronize()
        started_ns = self.clock_ns()
        result = operation(*args)
        self._synchronize()
        self.samples_ns.append(self.clock_ns() - started_ns)
        return result

    def as_dict(self):
        return {
            "unit": "seconds",
            **summarize_timing_samples(self.samples_ns),
        }


def synchronize_torch_device():
    """Wait for pending accelerated operations before reading the CPU clock."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


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

def calculate_single_bertscore(candidate, reference, scorer):
    """Calculate and materialize BERTScore for exactly one text pair."""
    precision, recall, f1 = scorer.score(
        [candidate],
        [reference],
        verbose=False,
        batch_size=1,
    )
    return precision.item(), recall.item(), f1.item()


def process_batch(news_list, scorer, timing_collector):
    """
    Calculates BERTScore (P, R, F1) for synthetic descriptions against headlines.
    """
    for i, news in enumerate(news_list):
        cand_text = remove_html(news.get("synthetic_description", ""))
        ref_text = remove_html(news.get("headline", "")) 
        
        if cand_text and ref_text:
            try:
                p_val, r_val, f1_val = timing_collector.measure(
                    calculate_single_bertscore,
                    cand_text,
                    ref_text,
                    scorer,
                )
                news["bert_precision"] = round(p_val, 4)
                news["bert_recall"] = round(r_val, 4)
                news["bert_f1"] = round(f1_val, 4)
            except Exception as e:
                print(f"Error calculating BERTScore for entry {i}: {e}")
                news["bert_precision"] = 0.0
                news["bert_recall"] = 0.0
                news["bert_f1"] = 0.0
        else:
            news["bert_precision"] = 0.0
            news["bert_recall"] = 0.0
            news["bert_f1"] = 0.0

        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{len(news_list)} pairs processed", end="\r")


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
    margin = stats.t.ppf(1 - alpha / 2, df=sample_size - 1) * standard_error
    return {
        "mean": round(mean_value, decimals),
        "ci95": {
            "lower": round(mean_value - margin, decimals),
            "upper": round(mean_value + margin, decimals)
        },
        "n": sample_size
    }


def calculate_real_baseline(news_list, scorer, timing_collector):
    """
    Calculates the mean BERTScore for human-written descriptions against headlines.
    """
    precision_values = []
    recall_values = []
    f1_values = []
    
    for i, news in enumerate(news_list):
        cand_text = remove_html(news.get("real_description", ""))
        ref_text = remove_html(news.get("headline", ""))
        
        if cand_text and ref_text:
            try:
                p_val, r_val, f1_val = timing_collector.measure(
                    calculate_single_bertscore,
                    cand_text,
                    ref_text,
                    scorer,
                )
                precision_values.append(p_val)
                recall_values.append(r_val)
                f1_values.append(f1_val)
            except Exception as e:
                print(f"Error calculating baseline entry {i}: {e}")

        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{len(news_list)} baseline pairs processed", end="\r")
            
    if not precision_values:
        return None

    precision_stats = calculate_mean_and_ci95(precision_values)
    recall_stats = calculate_mean_and_ci95(recall_values)
    f1_stats = calculate_mean_and_ci95(f1_values)
    return {
        "mean_precision": precision_stats["mean"],
        "ci95_precision": precision_stats["ci95"],
        "n_precision": precision_stats["n"],
        "mean_recall": recall_stats["mean"],
        "ci95_recall": recall_stats["ci95"],
        "n_recall": recall_stats["n"],
        "mean_f1": f1_stats["mean"],
        "ci95_f1": f1_stats["ci95"],
        "n_f1": f1_stats["n"],
        "sample_size": len(precision_values)
    }

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

def save_structured_json(path, no_rag_raw, rag_raw, data_real, calculation_times):
    """
    Calculates final means and p-values, then saves the results to a JSON file.
    """
    final_structure = {}
    
    for paper in no_rag_raw.keys():
        final_structure[paper] = {
            "human_baseline": data_real.get(paper, {}),
            "llm_models": {},
            "calculation_times": calculation_times.get(paper, {})
        }
        
        for llm_id in no_rag_raw[paper].keys():
            final_structure[paper]["llm_models"][llm_id] = {}
            
            for feat_id in [1, 2, 3]:
                nr = no_rag_raw[paper][llm_id].get(feat_id, {'p': [], 'r': [], 'f1': []})
                r = rag_raw[paper][llm_id].get(feat_id, {'p': [], 'r': [], 'f1': []})
                nr_precision_stats = calculate_mean_and_ci95(nr['p'])
                nr_recall_stats = calculate_mean_and_ci95(nr['r'])
                nr_f1_stats = calculate_mean_and_ci95(nr['f1'])
                r_precision_stats = calculate_mean_and_ci95(r['p'])
                r_recall_stats = calculate_mean_and_ci95(r['r'])
                r_f1_stats = calculate_mean_and_ci95(r['f1'])
                
                final_structure[paper]["llm_models"][llm_id][feat_id] = {
                    "NO_RAG": {
                        "mean_precision": nr_precision_stats["mean"],
                        "ci95_precision": nr_precision_stats["ci95"],
                        "n_precision": nr_precision_stats["n"],
                        "mean_recall": nr_recall_stats["mean"],
                        "ci95_recall": nr_recall_stats["ci95"],
                        "n_recall": nr_recall_stats["n"],
                        "mean_f1": nr_f1_stats["mean"],
                        "ci95_f1": nr_f1_stats["ci95"],
                        "n_f1": nr_f1_stats["n"]
                    },
                    "RAG": {
                        "mean_precision": r_precision_stats["mean"],
                        "ci95_precision": r_precision_stats["ci95"],
                        "n_precision": r_precision_stats["n"],
                        "mean_recall": r_recall_stats["mean"],
                        "ci95_recall": r_recall_stats["ci95"],
                        "n_recall": r_recall_stats["n"],
                        "mean_f1": r_f1_stats["mean"],
                        "ci95_f1": r_f1_stats["ci95"],
                        "n_f1": r_f1_stats["n"]
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

device = "cuda" if torch.cuda.is_available() else "cpu"
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Loading BERTScore model on {device}...")
bertscore_scorer = BERTScorer(lang="es", batch_size=1, device=device)
bertscore_timers = {
    paper: MetricTimingCollector(synchronize=synchronize_torch_device)
    for paper in newspapers
}

for paper in newspapers:
    baseline_done = False
    for llm_id in ids_to_process:
        print(f"Processing {paper} | LLM {llm_id}")
        
        data_nr = get_synthetic_news_from_json(INPUT_NORAG, paper, llm_id)
        data_r = get_synthetic_news_from_json(INPUT_RAG, paper, llm_id)

        if not baseline_done and data_nr:
            real_data[paper] = calculate_real_baseline(
                data_nr,
                bertscore_scorer,
                bertscore_timers[paper]
            )
            baseline_done = True

        process_batch(data_r, bertscore_scorer, bertscore_timers[paper])
        process_batch(data_nr, bertscore_scorer, bertscore_timers[paper])

        rag_raw[paper][llm_id] = extract_metrics_by_feature(data_r)
        no_rag_raw[paper][llm_id] = extract_metrics_by_feature(data_nr)

save_structured_json(
    JSON_RESULTS,
    no_rag_raw,
    rag_raw,
    real_data,
    {
        paper: {
            "bertscore": bertscore_timers[paper].as_dict()
        }
        for paper in newspapers
    }
)
