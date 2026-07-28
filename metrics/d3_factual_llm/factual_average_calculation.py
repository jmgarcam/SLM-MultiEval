import json
import os

import numpy as np
import pandas as pd
from scipy.stats import t


# =====================================================
# CONFIGURATION
# =====================================================
RAG_FILE = "factual_llm_RAG_results.json"
NO_RAG_FILE = "factual_llm_NO_RAG_results.json"
REAL_FILE = "factual_llm_REAL_results.json"

OUTPUT_DIR = "./results_factual_one_graph"
OUTPUT_JSON = "factual_means_by_newspaper.json"

MODEL_ID_MAP = {
    "1": "Ministral-3B",
    "2": "Qwen-2.5-0.5B",
    "3": "Llama-3.2-1B"
}


# =====================================================
# LOAD FACTUAL DATA
# =====================================================
def load_factual(filepath, condition):
    """Load RAG/NO-RAG count data while retaining anonymous source keys."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    for newspaper, models in data.items():
        for model_id, temperatures in models.items():
            for temperature_id, metrics in temperatures.items():
                total = (
                    metrics["correct"]
                    + metrics["partially_correct"]
                    + metrics["incorrect"]
                )

                if total == 0:
                    continue

                rows.append({
                    "Newspaper": newspaper,
                    "Model": MODEL_ID_MAP[model_id],
                    "Condition": condition,
                    "Temp": temperature_id,
                    "CorrectCount": metrics["correct"],
                    "IncorrectCount": metrics["incorrect"],
                    "PartialCount": metrics["partially_correct"],
                    "NumNews": total
                })

    return pd.DataFrame(rows)


# =====================================================
# PROPORTION STATISTICS
# =====================================================
def calculate_proportion_ci95(
    success_count,
    total_count,
    confidence=0.95,
    decimals=4
):
    """Calculate a proportion and its two-sided Student 95% CI."""
    if total_count <= 0:
        return {
            "mean": 0.0,
            "ci95": {"lower": None, "upper": None}
        }

    proportion = float(success_count) / float(total_count)

    if total_count < 2:
        return {
            "mean": round(proportion, decimals),
            "ci95": {"lower": None, "upper": None}
        }

    sample_variance = (
        proportion * (1 - proportion) * total_count
    ) / (total_count - 1)
    standard_error = np.sqrt(sample_variance / total_count)
    alpha = 1 - confidence
    margin = (
        t.ppf(1 - alpha / 2, df=total_count - 1)
        * standard_error
    )

    lower = max(0.0, proportion - margin)
    upper = min(1.0, proportion + margin)

    return {
        "mean": round(proportion, decimals),
        "ci95": {
            "lower": round(lower, decimals),
            "upper": round(upper, decimals)
        }
    }


# =====================================================
# LOAD REAL DATA
# =====================================================
def load_real(filepath):
    """Convert real-news count data into proportions and confidence intervals."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    result = {}

    for newspaper in data:
        metrics = data[newspaper]["REAL"]["1"]

        total = (
            metrics["correct"]
            + metrics["partially_correct"]
            + metrics["incorrect"]
        )

        if total == 0:
            continue

        correct_stats = calculate_proportion_ci95(
            metrics["correct"],
            total
        )
        partial_stats = calculate_proportion_ci95(
            metrics["partially_correct"],
            total
        )
        incorrect_stats = calculate_proportion_ci95(
            metrics["incorrect"],
            total
        )

        result[newspaper] = {
            "REAL": {
                "1": {
                    "correct": correct_stats["mean"],
                    "ci95_correct": correct_stats["ci95"],
                    "partially_correct": partial_stats["mean"],
                    "ci95_partially_correct": partial_stats["ci95"],
                    "incorrect": incorrect_stats["mean"],
                    "ci95_incorrect": incorrect_stats["ci95"]
                }
            }
        }

    return result


# =====================================================
# MAIN
# =====================================================
def main():
    rag_data = load_factual(RAG_FILE, "RAG")
    no_rag_data = load_factual(NO_RAG_FILE, "NO_RAG")

    combined_data = pd.concat(
        [rag_data, no_rag_data],
        ignore_index=True
    )

    aggregated_data = (
        combined_data.groupby(
            ["Newspaper", "Condition", "Model", "Temp"]
        )
        .sum(numeric_only=True)
        .reset_index()
    )

    result = {}

    for _, row in aggregated_data.iterrows():
        newspaper = row["Newspaper"]
        condition = row["Condition"]
        model = row["Model"]
        temperature = row["Temp"]
        total = int(row["NumNews"])

        correct_stats = calculate_proportion_ci95(
            int(row["CorrectCount"]),
            total
        )
        partial_stats = calculate_proportion_ci95(
            int(row["PartialCount"]),
            total
        )
        incorrect_stats = calculate_proportion_ci95(
            int(row["IncorrectCount"]),
            total
        )

        result.setdefault(condition, {})
        result[condition].setdefault(newspaper, {})
        result[condition][newspaper].setdefault(model, {})

        result[condition][newspaper][model][temperature] = {
            "correct": correct_stats["mean"],
            "ci95_correct": correct_stats["ci95"],
            "partially_correct": partial_stats["mean"],
            "ci95_partially_correct": partial_stats["ci95"],
            "incorrect": incorrect_stats["mean"],
            "ci95_incorrect": incorrect_stats["ci95"]
        }

    result["REAL"] = load_real(REAL_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print(f"Generated file: {output_path}")


if __name__ == "__main__":
    main()
