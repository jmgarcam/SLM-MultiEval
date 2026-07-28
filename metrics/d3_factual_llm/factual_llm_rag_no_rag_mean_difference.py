"""Paired RAG versus NO RAG analysis for Factual-LLM decisions.

The script reads the tagged CSV files used by the historical factual_llm
aggregators. Each categorical decision is represented by one indicator for
``correct``, ``partially_correct``, and ``incorrect``. RAG and NO_RAG rows
are paired by newspaper, model, temperature, and news identifier, and the
script reports ``RAG - NO_RAG``. A two-sided 95% Student t confidence
interval is calculated over the paired indicator differences for every
model-temperature group.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "files"

DEFAULT_NO_RAG_INPUT = FILES_DIR / "NO_RAG_news_csvs_tag"
DEFAULT_RAG_INPUT = FILES_DIR / "RAG_news_csvs_tag"
DEFAULT_OUTPUT_DIR = BASE_DIR

PAIRING_FIELDS = ("newspaper", "id_llm", "id_feature", "id_news")
METRICS = ("correct", "partially_correct", "incorrect")
FILENAME_PATTERN = re.compile(
    r"^(?P<newspaper>.+)_(?P<llm>\d+)_(?P<feature>\d+)_[ab]\.csv$"
)

CONFIDENCE_LEVEL = 0.95
DECIMALS = 3

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
            "Calculate paired mean RAG - NO_RAG differences and 95% "
            "confidence intervals for Factual-LLM decision proportions."
        )
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        choices=METRICS,
        default=METRICS,
        help="Factual categories to report; all are reported by default.",
    )
    parser.add_argument(
        "--no-rag-dir",
        type=Path,
        default=DEFAULT_NO_RAG_INPUT,
        help="Directory containing tagged NO_RAG CSV files.",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=DEFAULT_RAG_INPUT,
        help="Directory containing tagged RAG CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for result JSON and paired CSV files.",
    )
    return parser.parse_args()


def parse_decision(value, source_label, csv_path, row_number):
    if value is None:
        return None

    decision = str(value).strip()
    if not decision:
        return None
    if decision not in METRICS:
        raise ValueError(
            f"{source_label}: {csv_path.name}, row {row_number}: "
            f"unknown decision {decision!r}"
        )

    return decision


def load_records(input_dir, source_label):
    records = {}
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"{source_label}: no CSV files found in {input_dir}"
        )

    for csv_path in csv_paths:
        match = FILENAME_PATTERN.fullmatch(csv_path.name)
        if match is None:
            raise ValueError(
                f"{source_label}: unrecognized CSV filename {csv_path.name!r}"
            )

        newspaper = match.group("newspaper")
        llm_id = match.group("llm")
        feature_id = match.group("feature")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            required_columns = {"id_news", "decision"}
            missing_columns = required_columns - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"{source_label}: {csv_path.name}: missing columns "
                    f"{sorted(missing_columns)}"
                )

            for row_number, row in enumerate(reader, start=2):
                news_id = str(row.get("id_news", "")).strip()
                key = (
                    newspaper,
                    llm_id,
                    feature_id,
                    news_id,
                )
                if not news_id:
                    raise ValueError(
                        f"{source_label}: {csv_path.name}, row {row_number}: "
                        "missing id_news"
                    )
                if key in records:
                    raise ValueError(
                        f"{source_label}: duplicate pairing key {key}"
                    )

                records[key] = parse_decision(
                    row.get("decision"),
                    source_label,
                    csv_path,
                    row_number,
                )

    return records


def calculate_scores(records):
    scores = {}
    for key, decision in records.items():
        if decision is None:
            scores[key] = None
            continue

        scores[key] = {
            metric: float(decision == metric)
            for metric in METRICS
        }

    return scores


def is_valid_score(value):
    if value is None or isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


def pair_records(no_rag_records, rag_records, no_rag_scores, rag_scores, metric):
    paired_rows = []
    no_rag_column = f"{metric}_no_rag"
    rag_column = f"{metric}_rag"

    for key in sorted(no_rag_records.keys() | rag_records.keys()):
        no_rag_present = key in no_rag_records
        rag_present = key in rag_records
        no_rag_group = no_rag_scores.get(key)
        rag_group = rag_scores.get(key)
        no_rag_raw = (
            no_rag_group.get(metric)
            if no_rag_group is not None
            else None
        )
        rag_raw = (
            rag_group.get(metric)
            if rag_group is not None
            else None
        )

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
    margin = t.ppf(
        1.0 - alpha / 2.0,
        df=sample_size - 1,
    ) * standard_error

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
                    temperature_id,
                    temperature_id,
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


def build_results(metric, args, paired_rows):
    valid_rows = [row for row in paired_rows if row["status"] == "valid"]
    status_counts = defaultdict(int)
    for row in paired_rows:
        status_counts[row["status"]] += 1

    return {
        "analysis": {
            "metric": metric,
            "metric_family": "factual_llm",
            "difference_definition": "RAG - NO_RAG",
            "unit_of_report": "model x temperature, newspapers pooled",
            "pairing_key": list(PAIRING_FIELDS),
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_method": (
                "Two-sided Student t interval over paired categorical "
                "indicator differences from sum and sum of squares, rounded "
                "to 3 decimals"
            ),
            "score_definition": (
                f"Indicator equal to 1 when decision is {metric!r}, otherwise 0"
            ),
            "decision_categories": list(METRICS),
            "input_no_rag": str(args.no_rag_dir.resolve()),
            "input_rag": str(args.rag_dir.resolve()),
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
        json.dump(
            results,
            file_obj,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


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


def print_results(metric, results, output_json, output_csv):
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


def main():
    args = parse_args()
    no_rag_records = load_records(args.no_rag_dir, "NO_RAG")
    rag_records = load_records(args.rag_dir, "RAG")
    no_rag_scores = calculate_scores(no_rag_records)
    rag_scores = calculate_scores(rag_records)

    print(f"NO_RAG records: {len(no_rag_records)}")
    print(f"RAG records: {len(rag_records)}")

    for metric in args.metrics:
        paired_rows = pair_records(
            no_rag_records,
            rag_records,
            no_rag_scores,
            rag_scores,
            metric,
        )
        results = build_results(metric, args, paired_rows)

        output_json = (
            args.output_dir / f"{metric}_rag_no_rag_mean_difference.json"
        )
        output_csv = (
            args.output_dir / f"{metric}_rag_no_rag_paired_differences.csv"
        )
        write_json(output_json, results)
        write_csv(output_csv, paired_rows, metric)
        print_results(metric, results, output_json, output_csv)


if __name__ == "__main__":
    main()
