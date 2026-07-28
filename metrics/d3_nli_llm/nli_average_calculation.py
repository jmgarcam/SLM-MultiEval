import json
import math
import os
import numpy as np
from collections import defaultdict
from scipy.stats import t

# =====================================================
# INPUT FILES
# =====================================================
INPUT_JSON_RAG = "evaluation_rag_grouped.json"
INPUT_JSON_NO_RAG = "evaluation_no_rag_grouped.json"
INPUT_JSON_REAL = "evaluation_real_description_grouped.json"

OUTPUT_JSON = "nli_llm_results.json"


# =====================================================
# SAFE LOAD (robust encoding handling)
# =====================================================
def load_json_safe(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    raise ValueError(f"Could not read file: {path}")


# =====================================================
# SOFTMAX
# =====================================================
def softmax_three(entailment, contradiction, no_neutral):
    values = [float(entailment), float(contradiction), float(no_neutral)]
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [x / total for x in exps]


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


# =====================================================
# RAG / NO RAG PROCESSING
# =====================================================
def compute_feature_means(input_path):
    data = load_json_safe(input_path)

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "entailment_softmax_values": [],
        "contradiction_softmax_values": [],
        "no_neutral_softmax_values": []
    })))

    for newspaper, llm_dict in data.items():

        if not isinstance(llm_dict, dict):
            continue

        for llm_id, feature_dict in llm_dict.items():

            if not isinstance(feature_dict, dict):
                continue

            for feature_id, news_list in feature_dict.items():

                if not isinstance(news_list, list):
                    continue

                for news_item in news_list:

                    e = news_item.get("entailment", 0)
                    c = news_item.get("contradiction", 0)
                    n = news_item.get("no_neutral", 0)

                    # Skip invalid entries
                    if e == 0 and c == 0 and n == 0:
                        continue

                    e_s, c_s, n_s = softmax_three(e, c, n)

                    bucket = accum[newspaper][str(llm_id)][str(feature_id)]
                    bucket["entailment_softmax_values"].append(e_s)
                    bucket["contradiction_softmax_values"].append(c_s)
                    bucket["no_neutral_softmax_values"].append(n_s)

    return finalize(accum)


# =====================================================
# REAL DATA PROCESSING (different structure)
# =====================================================
def compute_real_means(input_path):
    data = load_json_safe(input_path)

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "entailment_softmax_values": [],
        "contradiction_softmax_values": [],
        "no_neutral_softmax_values": []
    })))

    for newspaper, news_list in data.items():

        # Skip invalid placeholder entry
        if newspaper.lower() == "periodico":
            continue

        if not isinstance(news_list, list):
            continue

        for news_item in news_list:

            e = news_item.get("entailment", 0)
            c = news_item.get("contradiction", 0)
            n = news_item.get("no_neutral", 0)

            # Skip invalid rows
            if e == 0 and c == 0 and n == 0:
                continue

            e_s, c_s, n_s = softmax_three(e, c, n)

            # Assign synthetic identifiers for REAL
            llm_id = "REAL"
            feature_id = "1"

            bucket = accum[newspaper][llm_id][feature_id]
            bucket["entailment_softmax_values"].append(e_s)
            bucket["contradiction_softmax_values"].append(c_s)
            bucket["no_neutral_softmax_values"].append(n_s)

    return finalize(accum)


# =====================================================
# FINALIZE RESULTS
# =====================================================
def finalize(accum):
    result = {}

    for newspaper, llm_dict in accum.items():
        result[newspaper] = {}

        for llm_id, feature_dict in llm_dict.items():
            result[newspaper][llm_id] = {}

            for feature_id, stats in feature_dict.items():
                entailment_stats = calculate_mean_and_ci95(
                    stats["entailment_softmax_values"]
                )
                contradiction_stats = calculate_mean_and_ci95(
                    stats["contradiction_softmax_values"]
                )
                no_neutral_stats = calculate_mean_and_ci95(
                    stats["no_neutral_softmax_values"]
                )
                num_news = entailment_stats["n"]

                if num_news == 0:
                    continue

                result[newspaper][llm_id][feature_id] = {
                    "mean_entailment_softmax": entailment_stats["mean"],
                    "ci95_entailment_softmax": entailment_stats["ci95"],
                    "mean_contradiction_softmax": contradiction_stats["mean"],
                    "ci95_contradiction_softmax": contradiction_stats["ci95"],
                    "mean_no_neutral_softmax": no_neutral_stats["mean"],
                    "ci95_no_neutral_softmax": no_neutral_stats["ci95"],
                    "num_news": num_news
                }

    return result


# =====================================================
# MAIN
# =====================================================
def main():
    rag_means = compute_feature_means(INPUT_JSON_RAG)
    no_rag_means = compute_feature_means(INPUT_JSON_NO_RAG)
    real_means = compute_real_means(INPUT_JSON_REAL)

    combined = {
        "RAG": rag_means,
        "NO_RAG": no_rag_means,
        "REAL": real_means
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Generated file: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
