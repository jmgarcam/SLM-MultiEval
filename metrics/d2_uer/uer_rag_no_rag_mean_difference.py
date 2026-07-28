"""Paired RAG versus NO RAG analysis for Unique Entity Ratio (UER).

The script reproduces the per-news calculation from ``uer_limited_news.py``,
pairs RAG and NO_RAG records by newspaper, model, temperature, and news
identifier, and reports ``RAG - NO_RAG``. A two-sided 95% Student t
confidence interval is calculated over the paired differences for every
model-temperature group.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import spacy
from scipy.stats import t


BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR.parent

DEFAULT_NO_RAG_INPUT = METRICS_DIR / "sinteticas_NORAG.json"
DEFAULT_RAG_INPUT = METRICS_DIR / "sinteticas_RAG.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "uer_rag_no_rag_mean_difference.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "uer_rag_no_rag_paired_differences.csv"

TEXT_FIELD = "synthetic_description"
METRIC = "uer"
DEFAULT_SPACY_MODEL = "es_core_news_sm"
PAIRING_FIELDS = ("newspaper", "id_llm", "id_feature", "id_news")

CONFIDENCE_LEVEL = 0.95
SCORE_DECIMALS = 4
REPORT_DECIMALS = 3

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
            "Calculate the paired mean RAG - NO_RAG difference and 95% "
            "confidence interval for UER."
        )
    )
    parser.add_argument(
        "--no-rag",
        type=Path,
        default=DEFAULT_NO_RAG_INPUT,
        help="Nested NO_RAG synthetic-news JSON.",
    )
    parser.add_argument(
        "--rag",
        type=Path,
        default=DEFAULT_RAG_INPUT,
        help="Nested RAG synthetic-news JSON.",
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help="spaCy pipeline used for Spanish named-entity recognition.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")

    return data


def remove_html(text):
    """Apply exactly the cleaning used by uer_limited_news.py."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("&nbsp;", " ")
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("Leer", "")
    return re.sub(r"\s+", " ", text).strip()


def calculate_uer(text, nlp):
    """Reproduce the rounded per-news UER from uer_limited_news.py."""
    clean_text = remove_html(text)
    if not clean_text:
        return 0.0

    doc = nlp(clean_text)
    total_tokens = len(doc)
    if total_tokens == 0:
        return 0.0

    entities = [entity.text.lower() for entity in doc.ents]
    unique_entities_count = len(set(entities))
    return round(unique_entities_count / total_tokens, SCORE_DECIMALS)


def flatten_records(source_data, source_label, nlp):
    records = {}

    for newspaper, llm_dict in source_data.items():
        if not isinstance(llm_dict, dict):
            raise ValueError(
                f"{source_label}: {newspaper!r} must contain an object"
            )

        for llm_id, feature_dict in llm_dict.items():
            if not isinstance(feature_dict, dict):
                raise ValueError(
                    f"{source_label}: {newspaper}/{llm_id} must contain an object"
                )

            for feature_id, news_list in feature_dict.items():
                if not isinstance(news_list, list):
                    raise ValueError(
                        f"{source_label}: {newspaper}/{llm_id}/{feature_id} "
                        "must contain a list"
                    )

                for news_item in news_list:
                    if not isinstance(news_item, dict):
                        raise ValueError(
                            f"{source_label}: news entries must be objects"
                        )

                    news_id = news_item.get("id_news")
                    if news_id is None or str(news_id).strip() == "":
                        raise ValueError(
                            f"{source_label}: record without id_news in "
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
                            f"{source_label}: duplicate pairing key {key}"
                        )

                    records[key] = calculate_uer(
                        news_item.get(TEXT_FIELD, ""),
                        nlp,
                    )

    return records


def is_valid_score(value):
    if value is None or isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


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
                "uer_no_rag": no_rag_score,
                "uer_rag": rag_score,
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
    decimals=REPORT_DECIMALS,
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


def calculate_summary(valid_rows):
    no_rag_scores = [row["uer_no_rag"] for row in valid_rows]
    rag_scores = [row["uer_rag"] for row in valid_rows]
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
        "mean_uer_no_rag": no_rag_stats["mean"],
        "mean_uer_rag": rag_stats["mean"],
        "mean_difference_rag_minus_no_rag": difference_stats["mean"],
        "confidence_interval_95": difference_stats["ci95"],
        "n_rag_higher": sum(value > 0 for value in differences),
        "n_equal": sum(value == 0 for value in differences),
        "n_rag_lower": sum(value < 0 for value in differences),
    }


def summarize_by_model_temperature(valid_rows):
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
                    grouped_rows[(model_id, temperature_id)]
                ),
            }
        )

    expected_groups = len(MODEL_LABELS) * len(TEMPERATURE_LABELS)
    if len(results) != expected_groups:
        raise ValueError(
            f"UER: expected {expected_groups} model-temperature groups, "
            f"found {len(results)}"
        )

    return results


def get_distribution_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_results(args, paired_rows, nlp):
    valid_rows = [row for row in paired_rows if row["status"] == "valid"]
    status_counts = defaultdict(int)
    for row in paired_rows:
        status_counts[row["status"]] += 1

    return {
        "analysis": {
            "metric": METRIC,
            "difference_definition": "RAG - NO_RAG",
            "unit_of_report": "model x temperature, newspapers pooled",
            "pairing_key": list(PAIRING_FIELDS),
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_method": (
                "Two-sided Student t interval over paired differences from "
                "sum and sum of squares, rounded to 3 decimals"
            ),
            "text_field": TEXT_FIELD,
            "score_definition": (
                "number of distinct lowercased spaCy entity spans divided "
                "by the spaCy token count"
            ),
            "score_rounding_decimals": SCORE_DECIMALS,
            "spacy_model": args.spacy_model,
            "spacy_model_version": nlp.meta.get("version"),
            "spacy_version": spacy.__version__,
            "scipy_version": get_distribution_version("scipy"),
            "input_no_rag": str(args.no_rag.resolve()),
            "input_rag": str(args.rag.resolve()),
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
        json.dump(
            results,
            file_obj,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_csv(path, paired_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *PAIRING_FIELDS,
        "uer_no_rag",
        "uer_rag",
        "difference_rag_minus_no_rag",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)


def main():
    args = parse_args()
    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    print("Calculating NO_RAG UER scores...")
    no_rag_records = flatten_records(
        load_json(args.no_rag),
        "NO_RAG",
        nlp,
    )
    print("Calculating RAG UER scores...")
    rag_records = flatten_records(
        load_json(args.rag),
        "RAG",
        nlp,
    )
    paired_rows = pair_records(no_rag_records, rag_records)
    results = build_results(args, paired_rows, nlp)

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
