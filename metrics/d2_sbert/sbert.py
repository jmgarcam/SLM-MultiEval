import json
import re
import sys
import os
import torch
import numpy as np
from math import sqrt
from statistics import stdev
from time import perf_counter_ns
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
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
    "snapshots"
)

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
    """Time one complete metric call for every input text pair."""

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

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory verified: {OUTPUT_DIR}")

# --- GLOBAL MODEL LOADING ---
def get_local_model_path():
    """Return the newest complete local Hugging Face snapshot, if present."""
    if not os.path.isdir(LOCAL_MODEL_CACHE_DIR):
        return None

    snapshots = [
        entry.path for entry in os.scandir(LOCAL_MODEL_CACHE_DIR)
        if entry.is_dir()
    ]
    snapshots.sort(key=os.path.getmtime, reverse=True)

    required_files = ["config.json", "modules.json", "sentence_bert_config.json"]
    for snapshot in snapshots:
        if all(os.path.exists(os.path.join(snapshot, name)) for name in required_files):
            return snapshot

    return None


print("Loading SBERT model...")
try:
    model_path = get_local_model_path() or MODEL_ID
    print(f"Using model source: {model_path}")
    model = SentenceTransformer(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def synchronize_model_device():
    """Wait for pending model operations before reading the CPU clock."""
    device_type = getattr(model.device, "type", str(model.device))
    if device_type == "cuda":
        torch.cuda.synchronize(model.device)
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


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


def calculate_single_sbert_score(text_a, text_b):
    """Calculate SBERT similarity for one already-cleaned text pair."""
    if not text_a or not text_b:
        return 0.0

    emb_a = model.encode(
        [text_a],
        batch_size=1,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    emb_b = model.encode(
        [text_b],
        batch_size=1,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    score = torch.nn.functional.cosine_similarity(emb_a, emb_b)
    return score.item()


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

# --- SYNTHETIC VS HEADLINE PROCESSING ---
def process_synthetic_batch(news_list, timing_collector=None):
    for news in news_list:
        cand_text = remove_html(news.get("synthetic_description", "")) 
        ref_text = remove_html(news.get("headline", "")) 

        if timing_collector is None:
            score = calculate_single_sbert_score(cand_text, ref_text)
        else:
            score = timing_collector.measure(
                calculate_single_sbert_score,
                cand_text,
                ref_text,
            )

        news["sbert_headline_raw"] = round(score, 4)

# --- HUMAN BASELINE CALCULATION (Real Desc vs Headline) ---
def calculate_human_baseline_stats(news_list, timing_collector=None):
    """
    Calculates the mean SBERT score between real_description and headline
    using a sample of news.
    """
    scores = []
    
    for news in news_list:
        d = remove_html(news.get("real_description", ""))
        h = remove_html(news.get("headline", ""))

        if timing_collector is None:
            score = calculate_single_sbert_score(d, h)
        else:
            score = timing_collector.measure(
                calculate_single_sbert_score,
                d,
                h,
            )

        if d and h:
            scores.append(round(score, 4))
    
    if not scores:
        return {
            "mean_sbert": 0.0,
            "ci95_sbert": {"lower": None, "upper": None},
            "sample_size": 0
        }

    score_stats = calculate_mean_and_ci95(scores)
    
    return {
        "mean_sbert": score_stats["mean"],
        "ci95_sbert": score_stats["ci95"],
        "sample_size": score_stats["n"]
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
        score_stats = calculate_mean_and_ci95(metrics[feature]['raw_scores'])
        results[str(feature)] = {
            'mean_sbert': score_stats['mean'],
            'ci95_sbert': score_stats['ci95'],
            'sample_size': score_stats['n'],
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

def save_custom_json(
    path,
    data_no_rag,
    data_rag,
    baselines,
    t_tests,
    calculation_times
):
    final_json = {}
    newspapers = data_no_rag.keys()

    for newspaper in newspapers:
        final_json[newspaper] = {
            "human_baseline": baselines.get(newspaper, {"mean_sbert": 0.0, "sample_size": 0}),
            "llm_models": {},
            "calculation_times": calculation_times.get(newspaper, {})
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
newspapers = [NEWSPAPER_A, NEWSPAPER_B]

# --- INITIALIZATION ---
results_no_rag = initialize_structure(newspapers)
results_rag = initialize_structure(newspapers)
t_test_results = initialize_structure(newspapers)
human_baselines = {} 
sbert_timers = {
    newspaper: MetricTimingCollector(synchronize=synchronize_model_device)
    for newspaper in newspapers
}

# --- EXECUTION ---
for newspaper in newspapers:
    baseline_calculated = False 

    for llm_id_int in ids_to_process:
        llm_id_str = str(llm_id_int)
        print(f"\n>>>> {newspaper} | LLM ID: {llm_id_str} | Processing... <<<<")

        synthetic_rag = get_synthetic_news_from_json(INPUT_RAG, newspaper, llm_id_int)
        synthetic_no_rag = get_synthetic_news_from_json(INPUT_NORAG, newspaper, llm_id_int)
        # 2. Calculate Human Baseline
        if not baseline_calculated and synthetic_no_rag:
            print(f"Calculating Human Baseline for {newspaper} (Real Desc vs Headline)...")
            human_baselines[newspaper] = calculate_human_baseline_stats(
                synthetic_no_rag,
                sbert_timers[newspaper]
            )
            print(f" -> Baseline Result: {human_baselines[newspaper]}")
            baseline_calculated = True

        # 3. Process Synthetic Data
        process_synthetic_batch(synthetic_rag, sbert_timers[newspaper])
        process_synthetic_batch(synthetic_no_rag, sbert_timers[newspaper])

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
save_custom_json(
    full_path,
    results_no_rag,
    results_rag,
    human_baselines,
    t_test_results,
    {
        newspaper: {
            "sbert": sbert_timers[newspaper].as_dict()
        }
        for newspaper in newspapers
    }
)
