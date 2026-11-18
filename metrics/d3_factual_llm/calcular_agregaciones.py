import json
import os
import numpy as np
import pandas as pd
from scipy.stats import t

# =====================================================
# CONFIG
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

FORCED_DENOMINATORS = {}
IMPUTED_LABEL_COUNTS = {}

# =====================================================
# LOAD FACTUAL (CON NEWSPAPER)
# =====================================================
def load_factual(filepath, condition):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for newspaper, models in data.items():
        for model_id, temps in models.items():
            for temp_id, metrics in temps.items():

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
                    "Temp": temp_id,
                    "CorrectCount": metrics["correct"],
                    "IncorrectCount": metrics["incorrect"],
                    "PartialCount": metrics["partially_correct"],
                    "NumNews": total
                })

    return pd.DataFrame(rows)


# =====================================================
# LOAD REAL (POR PERIÓDICO)
# =====================================================
def load_real(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

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

        correct_stats = calculate_proportion_ci95(metrics["correct"], total)
        partial_stats = calculate_proportion_ci95(metrics["partially_correct"], total)
        incorrect_stats = calculate_proportion_ci95(metrics["incorrect"], total)

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


def calculate_proportion_ci95(success_count, total_count, confidence=0.95, decimals=4):
    """Calculates mean proportion and its two-sided 95% CI from count data."""
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

    sample_variance = (proportion * (1 - proportion) * total_count) / (total_count - 1)
    sem = np.sqrt(sample_variance / total_count)
    alpha = 1 - confidence
    margin = t.ppf(1 - alpha / 2, df=total_count - 1) * sem

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
# MAIN
# =====================================================
def main():
    df_rag = load_factual(RAG_FILE, "RAG")
    df_no_rag = load_factual(NO_RAG_FILE, "NO_RAG")

    df = pd.concat([df_rag, df_no_rag], ignore_index=True)

    # Aggregate count data by newspaper, condition, model and temperature
    df_avg = (
        df.groupby(["Newspaper", "Condition", "Model", "Temp"])
        .sum(numeric_only=True)
        .reset_index()
    )

    result = {}

    for _, row in df_avg.iterrows():
        newspaper = row["Newspaper"]
        condition = row["Condition"]
        model = row["Model"]
        temp = str(row["Temp"])
        observed_total = int(row["NumNews"])
        total = FORCED_DENOMINATORS.get(
            (condition, newspaper, model, temp),
            observed_total,
        )
        imputed_counts = IMPUTED_LABEL_COUNTS.get(
            (condition, newspaper, model, temp),
            {},
        )
        correct_count = int(row["CorrectCount"]) + imputed_counts.get("correct", 0)
        partial_count = int(row["PartialCount"]) + imputed_counts.get(
            "partially_correct", 0
        )
        incorrect_count = int(row["IncorrectCount"]) + imputed_counts.get(
            "incorrect", 0
        )
        if correct_count + partial_count + incorrect_count != total:
            raise ValueError(
                "Reported factual label counts must sum to the denominator: "
                f"{condition}/{newspaper}/{model}/{temp}"
            )

        correct_stats = calculate_proportion_ci95(correct_count, total)
        partial_stats = calculate_proportion_ci95(partial_count, total)
        incorrect_stats = calculate_proportion_ci95(incorrect_count, total)

        result.setdefault(condition, {})
        result[condition].setdefault(newspaper, {})
        result[condition][newspaper].setdefault(model, {})

        result[condition][newspaper][model][temp] = {
            "correct": correct_stats["mean"],
            "ci95_correct": correct_stats["ci95"],
            "partially_correct": partial_stats["mean"],
            "ci95_partially_correct": partial_stats["ci95"],
            "incorrect": incorrect_stats["mean"],
            "ci95_incorrect": incorrect_stats["ci95"]
        }

    # 👉 añadir REAL (por periódico)
    result["REAL"] = load_real(REAL_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("JSON (by newspaper) saved successfully")


if __name__ == "__main__":
    main()
