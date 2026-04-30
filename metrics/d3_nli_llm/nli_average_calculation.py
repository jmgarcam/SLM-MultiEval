import json
import math
import os
from collections import defaultdict

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


# =====================================================
# RAG / NO RAG PROCESSING
# =====================================================
def compute_feature_means(input_path):
    data = load_json_safe(input_path)

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "sum_entailment_softmax": 0.0,
        "sum_contradiction_softmax": 0.0,
        "sum_no_neutral_softmax": 0.0,
        "num_news": 0
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
                    bucket["sum_entailment_softmax"] += e_s
                    bucket["sum_contradiction_softmax"] += c_s
                    bucket["sum_no_neutral_softmax"] += n_s
                    bucket["num_news"] += 1

    return finalize(accum)


# =====================================================
# REAL DATA PROCESSING (different structure)
# =====================================================
def compute_real_means(input_path):
    data = load_json_safe(input_path)

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "sum_entailment_softmax": 0.0,
        "sum_contradiction_softmax": 0.0,
        "sum_no_neutral_softmax": 0.0,
        "num_news": 0
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
            bucket["sum_entailment_softmax"] += e_s
            bucket["sum_contradiction_softmax"] += c_s
            bucket["sum_no_neutral_softmax"] += n_s
            bucket["num_news"] += 1

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
                num_news = stats["num_news"]

                if num_news == 0:
                    continue

                result[newspaper][llm_id][feature_id] = {
                    "mean_entailment_softmax": stats["sum_entailment_softmax"] / num_news,
                    "mean_contradiction_softmax": stats["sum_contradiction_softmax"] / num_news,
                    "mean_no_neutral_softmax": stats["sum_no_neutral_softmax"] / num_news,
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