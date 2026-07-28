"""Paired RAG versus NO RAG analysis for G-Eval metrics.

For every metric, records are paired by id_news within newspaper, model and
temperature. Results aggregate both newspapers into the nine model-temperature
combinations. Confidence intervals reproduce the method used by the project's
baseline aggregators.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
BASELINES_DIR = BASE_DIR.parent
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

METRIC_CONFIGS = {
    "consistency": {
        "no_rag": (
            BASELINES_DIR
            / "files"
            / "consistency"
            / "consistency_NO_RAG_by_news.json"
        ),
        "rag": (
            BASELINES_DIR
            / "files"
            / "consistency"
            / "consistency_RAG_by_news.json"
        ),
        "output_dir": BASE_DIR / "consistency",
    },
    "fluency": {
        "no_rag": BASE_DIR / "fluency" / "fluency_NO_RAG_mtld_fh_by_news.json",
        "rag": BASE_DIR / "fluency" / "fluency_RAG_mtld_fh_by_news.json",
        "output_dir": BASE_DIR / "fluency",
    },
    "relevance": {
        "no_rag": (
            BASE_DIR
            / "relevance"
            / "relevance_NO_RAG_sbert_bertscore_by_news.json"
        ),
        "rag": (
            BASE_DIR
            / "relevance"
            / "relevance_RAG_sbert_bertscore_by_news.json"
        ),
        "output_dir": BASE_DIR / "relevance",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate paired mean RAG - NO RAG differences and 95% "
            "confidence intervals for G-Eval metrics."
        )
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        choices=tuple(METRIC_CONFIGS),
        default=tuple(METRIC_CONFIGS),
    )
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


def flatten_records(source_data, source_label, metric):
    records = {}

    for newspaper, llm_dict in source_data.items():
        for llm_id, feature_dict in llm_dict.items():
            for feature_id, news_list in feature_dict.items():
                for news_item in news_list:
                    news_id = news_item.get("id_news")
                    if news_id is None or str(news_id).strip() == "":
                        raise ValueError(
                            f"{source_label}/{metric}: record without id_news in "
                            f"{newspaper}/LLM {llm_id}/temperature {feature_id}"
                        )

                    key = (
                        str(newspaper),
                        str(llm_id),
                        str(feature_id),
                        str(news_id),
                    )
                    if key in records:
                        raise ValueError(
                            f"{source_label}/{metric}: duplicate pairing key {key}"
                        )

                    records[key] = news_item.get(metric)

    return records


def pair_records(no_rag_records, rag_records, metric):
    paired_rows = []
    no_rag_column = f"{metric}_no_rag"
    rag_column = f"{metric}_rag"

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
                no_rag_column: no_rag_score,
                rag_column: rag_score,
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
    """Replicate calculate_mean_and_ci95 from the baseline aggregators."""
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


def calculate_summary(valid_rows, metric):
    no_rag_column = f"{metric}_no_rag"
    rag_column = f"{metric}_rag"
    no_rag_scores = [row[no_rag_column] for row in valid_rows]
    rag_scores = [row[rag_column] for row in valid_rows]
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
        f"mean_{metric}_no_rag": no_rag_stats["mean"],
        f"mean_{metric}_rag": rag_stats["mean"],
        "mean_difference_rag_minus_no_rag": difference_stats["mean"],
        "confidence_interval_95": difference_stats["ci95"],
        "n_rag_higher": sum(value > 0 for value in differences),
        "n_equal": sum(value == 0 for value in differences),
        "n_rag_lower": sum(value < 0 for value in differences),
    }


def summarize_by_model_temperature(valid_rows, metric):
    grouped_rows = defaultdict(list)
    for row in valid_rows:
        grouped_rows[(row["id_llm"], row["id_feature"])].append(row)

    results = []
    for model_id, temperature_id in sorted(grouped_rows):
        results.append(
            {
                "id_llm": model_id,
                "model": MODEL_LABELS.get(model_id, model_id),
                "id_temperature": temperature_id,
                "temperature": TEMPERATURE_LABELS.get(
                    temperature_id, temperature_id
                ),
                **calculate_summary(
                    grouped_rows[(model_id, temperature_id)],
                    metric,
                ),
            }
        )

    expected_groups = len(MODEL_LABELS) * len(TEMPERATURE_LABELS)
    if len(results) != expected_groups:
        raise ValueError(
            f"{metric}: expected {expected_groups} model-temperature groups, "
            f"found {len(results)}"
        )

    return results


def build_results(metric, config, paired_rows):
    valid_rows = [row for row in paired_rows if row["status"] == "valid"]
    status_counts = defaultdict(int)
    for row in paired_rows:
        status_counts[row["status"]] += 1

    return {
        "analysis": {
            "metric": metric,
            "difference_definition": "RAG - NO_RAG",
            "unit_of_report": "model x temperature, newspapers pooled",
            "pairing_key": list(PAIRING_FIELDS),
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_method": (
                "Same method as the baseline aggregators: two-sided Student t "
                "interval from sum and sum of squares, rounded to 3 decimals"
            ),
            "input_no_rag": str(config["no_rag"]),
            "input_rag": str(config["rag"]),
        },
        "pairing_diagnostics": {
            "n_keys_in_union": len(paired_rows),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "by_model_temperature": summarize_by_model_temperature(
            valid_rows,
            metric,
        ),
    }


def write_json(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False, allow_nan=False)


def write_csv(path, paired_rows, metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *PAIRING_FIELDS,
        f"{metric}_no_rag",
        f"{metric}_rag",
        "difference_rag_minus_no_rag",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)


def run_metric(metric):
    config = METRIC_CONFIGS[metric]
    no_rag_records = flatten_records(
        load_json(config["no_rag"]),
        "NO_RAG",
        metric,
    )
    rag_records = flatten_records(
        load_json(config["rag"]),
        "RAG",
        metric,
    )
    paired_rows = pair_records(no_rag_records, rag_records, metric)
    results = build_results(metric, config, paired_rows)

    output_json = (
        config["output_dir"] / f"{metric}_rag_no_rag_mean_difference.json"
    )
    output_csv = (
        config["output_dir"] / f"{metric}_rag_no_rag_paired_differences.csv"
    )
    write_json(output_json, results)
    write_csv(output_csv, paired_rows, metric)

    print(f"\n{metric.upper()}")
    print("Model | Temperature | N | Mean difference | 95% CI")
    for row in results["by_model_temperature"]:
        interval = row["confidence_interval_95"]
        print(
            f"{row['model']} | {row['temperature']} | {row['n_pairs']} | "
            f"{row['mean_difference_rag_minus_no_rag']:.3f} | "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        )
    print(f"JSON output: {output_json}")
    print(f"CSV output: {output_csv}")

    return results


def main():
    args = parse_args()
    for metric in args.metrics:
        run_metric(metric)


if __name__ == "__main__":
    main()
