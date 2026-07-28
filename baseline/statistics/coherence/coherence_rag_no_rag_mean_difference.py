"""Paired analysis of G-Eval coherence scores under RAG and NO RAG.

Each generated news item is paired within its newspaper, LLM and feature group
using ``id_news``.  The reported difference is always ``RAG - NO_RAG``.  A
two-sided 95% Student t confidence interval is calculated over the paired
differences.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NO_RAG_INPUT = BASE_DIR / "coherence_NO_RAG_fh_td_by_news.json"
DEFAULT_RAG_INPUT = BASE_DIR / "coherence_RAG_fh_td_by_news.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "coherence_rag_no_rag_mean_difference.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "coherence_rag_no_rag_paired_differences.csv"

SCORE_KEY = "coherence"
CONFIDENCE_LEVEL = 0.95
DECIMALS = 3
PAIRING_FIELDS = ("newspaper", "id_llm", "id_feature", "id_news")
MODEL_LABELS = {
    "1": "Ministral-3B",
    "2": "Qwen-2.5-0.5B",
    "3": "Llama-3.2-1B",
}
TEMPERATURE_LABELS = {
    "1": "1.00",
    "2": "0.75",
    "3": "0.50",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the paired mean difference and 95% confidence interval "
            "for RAG versus NO RAG G-Eval coherence scores."
        )
    )
    parser.add_argument("--no-rag", type=Path, default=DEFAULT_NO_RAG_INPUT)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def is_valid_score(value):
    if value is None or isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


def flatten_records(source_data, source_label):
    records = {}

    for newspaper, llm_dict in source_data.items():
        for llm_id, feature_dict in llm_dict.items():
            for feature_id, news_list in feature_dict.items():
                for news_item in news_list:
                    news_id = news_item.get("id_news")
                    if news_id is None or str(news_id).strip() == "":
                        raise ValueError(
                            f"{source_label}: record without id_news in "
                            f"{newspaper}/LLM {llm_id}/feature {feature_id}"
                        )

                    key = (
                        str(newspaper),
                        str(llm_id),
                        str(feature_id),
                        str(news_id),
                    )
                    if key in records:
                        raise ValueError(
                            f"{source_label}: duplicate pairing key {key}"
                        )

                    records[key] = news_item.get(SCORE_KEY)

    return records


def pair_records(no_rag_records, rag_records):
    paired_rows = []

    for key in sorted(no_rag_records.keys() | rag_records.keys()):
        no_rag_present = key in no_rag_records
        rag_present = key in rag_records
        no_rag_raw = no_rag_records.get(key)
        rag_raw = rag_records.get(key)

        if not no_rag_present:
            status = "missing_no_rag_pair"
        elif not rag_present:
            status = "missing_rag_pair"
        elif not is_valid_score(no_rag_raw):
            status = "invalid_no_rag_score"
        elif not is_valid_score(rag_raw):
            status = "invalid_rag_score"
        else:
            status = "valid"

        no_rag_score = float(no_rag_raw) if is_valid_score(no_rag_raw) else None
        rag_score = float(rag_raw) if is_valid_score(rag_raw) else None
        difference = (
            rag_score - no_rag_score
            if status == "valid"
            else None
        )

        paired_rows.append(
            {
                "newspaper": key[0],
                "id_llm": key[1],
                "id_feature": key[2],
                "id_news": key[3],
                "coherence_no_rag": no_rag_score,
                "coherence_rag": rag_score,
                "difference_rag_minus_no_rag": difference,
                "status": status,
            }
        )

    return paired_rows


def calculate_mean_and_ci95(
    sum_values,
    sum_sq_values,
    sample_size,
    confidence=CONFIDENCE_LEVEL,
    decimals=DECIMALS,
):
    """Replicate the CI calculation from baselines/agregator_coherence.py."""
    if sample_size <= 0:
        return {
            "mean": 0.0,
            "ci95": {"lower": None, "upper": None},
        }

    mean_value = sum_values / sample_size

    if sample_size < 2:
        return {
            "mean": round(mean_value, decimals),
            "ci95": {"lower": None, "upper": None},
        }

    variance_numerator = sum_sq_values - sample_size * (mean_value ** 2)
    sample_variance = max(variance_numerator / (sample_size - 1), 0.0)
    standard_error = (sample_variance / sample_size) ** 0.5
    alpha = 1.0 - confidence
    margin = t.ppf(1.0 - alpha / 2.0, df=sample_size - 1) * standard_error

    return {
        "mean": round(mean_value, decimals),
        "ci95": {
            "lower": round(mean_value - margin, decimals),
            "upper": round(mean_value + margin, decimals),
        },
    }


def calculate_summary(valid_rows):
    no_rag_scores = [row["coherence_no_rag"] for row in valid_rows]
    rag_scores = [row["coherence_rag"] for row in valid_rows]
    differences = [row["difference_rag_minus_no_rag"] for row in valid_rows]
    sample_size = len(differences)

    no_rag_stats = calculate_mean_and_ci95(
        sum(no_rag_scores),
        sum(value ** 2 for value in no_rag_scores),
        sample_size,
    )
    rag_stats = calculate_mean_and_ci95(
        sum(rag_scores),
        sum(value ** 2 for value in rag_scores),
        sample_size,
    )
    difference_stats = calculate_mean_and_ci95(
        sum(differences),
        sum(value ** 2 for value in differences),
        sample_size,
    )

    return {
        "n_pairs": sample_size,
        "mean_coherence_no_rag": no_rag_stats["mean"],
        "mean_coherence_rag": rag_stats["mean"],
        "mean_difference_rag_minus_no_rag": difference_stats["mean"],
        "confidence_interval_95": difference_stats["ci95"],
        "n_rag_higher": sum(value > 0 for value in differences),
        "n_equal": sum(value == 0 for value in differences),
        "n_rag_lower": sum(value < 0 for value in differences),
    }


def summarize_by(valid_rows, fields):
    grouped_rows = defaultdict(list)
    for row in valid_rows:
        group_key = tuple(row[field] for field in fields)
        grouped_rows[group_key].append(row)

    summaries = []
    for group_key in sorted(grouped_rows):
        group_labels = dict(zip(fields, group_key))
        summaries.append(
            {
                **group_labels,
                **calculate_summary(grouped_rows[group_key]),
            }
        )
    return summaries


def summarize_by_model_temperature(valid_rows):
    summaries = summarize_by(valid_rows, ("id_llm", "id_feature"))
    results = []

    for summary in summaries:
        model_id = summary.pop("id_llm")
        temperature_id = summary.pop("id_feature")
        results.append(
            {
                "id_llm": model_id,
                "model": MODEL_LABELS.get(model_id, model_id),
                "id_temperature": temperature_id,
                "temperature": TEMPERATURE_LABELS.get(
                    temperature_id, temperature_id
                ),
                **summary,
            }
        )

    return results


def build_results(no_rag_path, rag_path, paired_rows):
    valid_rows = [row for row in paired_rows if row["status"] == "valid"]
    status_counts = defaultdict(int)
    for row in paired_rows:
        status_counts[row["status"]] += 1

    return {
        "analysis": {
            "metric": SCORE_KEY,
            "difference_definition": "RAG - NO_RAG",
            "pairing_key": list(PAIRING_FIELDS),
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_method": (
                "Same method as baselines/agregator_coherence.py: two-sided "
                "Student t interval from sum and sum of squares, rounded to "
                "3 decimals"
            ),
            "input_no_rag": str(no_rag_path),
            "input_rag": str(rag_path),
        },
        "pairing_diagnostics": {
            "n_keys_in_union": len(paired_rows),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "by_model_temperature": summarize_by_model_temperature(valid_rows),
    }


def write_json(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False, allow_nan=False)


def write_csv(path, paired_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *PAIRING_FIELDS,
        "coherence_no_rag",
        "coherence_rag",
        "difference_rag_minus_no_rag",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)


def main():
    args = parse_args()
    no_rag_records = flatten_records(load_json(args.no_rag), "NO_RAG")
    rag_records = flatten_records(load_json(args.rag), "RAG")
    paired_rows = pair_records(no_rag_records, rag_records)
    results = build_results(args.no_rag, args.rag, paired_rows)

    write_json(args.output_json, results)
    write_csv(args.output_csv, paired_rows)

    print("Model | Temperature | N | Mean difference | 95% CI")
    for row in results["by_model_temperature"]:
        interval = row["confidence_interval_95"]
        print(
            f"{row['model']} | {row['temperature']} | {row['n_pairs']} | "
            f"{row['mean_difference_rag_minus_no_rag']:.3f} | "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        )
    print(f"JSON output: {args.output_json}")
    print(f"CSV output: {args.output_csv}")


if __name__ == "__main__":
    main()
