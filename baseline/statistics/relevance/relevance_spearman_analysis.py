"""Correlate per-news relevance with SBERT and BERTScore F1.

Run ``relevance_sbert_bertscore.py`` first when the intermediate JSON files
need to be regenerated. Those JSON files are the only analytical inputs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = {
    "NO_RAG": BASE_DIR / "relevance_NO_RAG_sbert_bertscore_by_news.json",
    "RAG": BASE_DIR / "relevance_RAG_sbert_bertscore_by_news.json",
}
TARGET_METRIC = ("relevance", "Relevance")
COMPARISON_METRICS = (
    ("sbert_headline_raw", "SBERT"),
    ("bert_f1", "BERTScore"),
)
ALL_METRICS = (TARGET_METRIC, *COMPARISON_METRICS)
OUTPUT_JSON = BASE_DIR / "relevance_spearman_results.json"
FIGURE_STEM = BASE_DIR / "relevance_spearman_matrix_RAG_NO_RAG"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required intermediate JSON not found: {path}. "
            "Run relevance_sbert_bertscore.py first."
        )
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty JSON object in {path}")
    return data


def finite_float(value: object, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value at {context}: {value!r}")
    return number


def flatten_news_items(source_data: dict, label: str) -> list[dict]:
    flattened: list[dict] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for newspaper, llm_dict in source_data.items():
        for llm_id, feature_dict in llm_dict.items():
            for feature_id, news_list in feature_dict.items():
                if not isinstance(news_list, list):
                    raise ValueError(
                        f"{label}: expected a list at {newspaper}/{llm_id}/{feature_id}"
                    )
                for news_item in news_list:
                    news_id = str(news_item.get("id_news", "")).strip()
                    key = (str(newspaper), str(llm_id), str(feature_id), news_id)
                    if not news_id:
                        raise ValueError(f"{label}: empty id_news at {key[:-1]}")
                    if key in seen_keys:
                        raise ValueError(f"{label}: duplicate observation key {key}")
                    seen_keys.add(key)
                    context = f"{label}:{newspaper}/{llm_id}/{feature_id}/{news_id}"
                    validated = dict(news_item)
                    for metric_key, _ in ALL_METRICS:
                        validated[metric_key] = finite_float(
                            news_item.get(metric_key), f"{context}:{metric_key}"
                        )
                    flattened.append(validated)
    if not flattened:
        raise ValueError(f"{label}: no news items found")
    return flattened


def duplicate_summary(news_items: list[dict]) -> dict:
    counts = Counter(str(item.get("synthetic_description", "")) for item in news_items)
    duplicated = [count for description, count in counts.items() if description and count > 1]
    return {
        "num_distinct_descriptions": len(counts),
        "num_duplicated_descriptions": len(duplicated),
        "num_rows_in_duplicated_descriptions": sum(duplicated),
        "max_description_occurrences": max(duplicated, default=1),
        "policy": "retained",
    }


def build_condition_results(
    condition: str, input_path: Path, news_items: list[dict]
) -> dict:
    target_key, target_label = TARGET_METRIC
    comparison_labels = [name for _, name in COMPARISON_METRICS]
    target_values = np.asarray([item[target_key] for item in news_items], dtype=float)
    rho_values: list[float] = []
    p_values: list[float] = []
    pairwise: dict[str, dict[str, float | int]] = {}
    for comparison_key, _ in COMPARISON_METRICS:
        comparison_values = np.asarray(
            [item[comparison_key] for item in news_items], dtype=float
        )
        result = spearmanr(target_values, comparison_values)
        rho = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(rho) or not math.isfinite(p_value):
            raise ValueError(
                f"{condition}: undefined Spearman result for "
                f"{target_key} vs {comparison_key}"
            )
        rho_values.append(rho)
        p_values.append(p_value)
        pairwise[f"{target_key}_vs_{comparison_key}"] = {
            "rho": rho,
            "p_value": p_value,
            "n": len(news_items),
        }
    return {
        "source_file": input_path.name,
        "unit_of_analysis": "news item",
        "num_news": len(news_items),
        "target_metric": target_label,
        "comparison_metrics": comparison_labels,
        "rho": [rho_values],
        "p_value": [p_values],
        "n": [[len(news_items)] * len(COMPARISON_METRICS)],
        "pairwise_spearman": pairwise,
        "duplicate_descriptions": duplicate_summary(news_items),
    }


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")


def write_matrix_csv(path: Path, analysis: dict, field: str) -> None:
    values = analysis[field][0]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", *analysis["comparison_metrics"]])
        writer.writerow([analysis["target_metric"], *values])


def plot_results(results: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    image = None
    for axis, condition in zip(axes, ("RAG", "NO_RAG"), strict=True):
        analysis = results[condition]
        matrix = np.asarray(analysis["rho"], dtype=float)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for column, rho in enumerate(matrix[0]):
            text_color = "white" if abs(rho) >= 0.55 else "black"
            displayed_rho = 0.0 if abs(rho) < 0.005 else rho
            axis.text(
                column, 0, f"{displayed_rho:.2f}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=text_color,
            )
        axis.set_xticks(range(len(analysis["comparison_metrics"])))
        axis.set_xticklabels(analysis["comparison_metrics"], fontsize=11)
        axis.set_yticks([0])
        axis.set_yticklabels([analysis["target_metric"]], fontsize=11)
        axis.set_title(
            f"{condition.replace('_', ' ')}\n(n = {analysis['num_news']})",
            fontsize=15, fontweight="bold", pad=18,
        )
        axis.tick_params(length=0)
    if image is None:
        raise RuntimeError("No results available to plot")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.04)
    colorbar.set_label(r"Spearman $\rho$", fontsize=12)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(FIGURE_STEM.with_suffix(f".{extension}"), dpi=300)
    plt.close(figure)


def main() -> None:
    results: dict[str, dict] = {}
    for condition, input_path in INPUT_FILES.items():
        news_items = flatten_news_items(load_json(input_path), condition)
        results[condition] = build_condition_results(condition, input_path, news_items)
        print(f"{condition}: {len(news_items)} news items loaded from {input_path.name}")

    write_json(OUTPUT_JSON, results)
    for condition, analysis in results.items():
        write_matrix_csv(
            BASE_DIR / f"relevance_spearman_rho_{condition}.csv", analysis, "rho"
        )
        write_matrix_csv(
            BASE_DIR / f"relevance_spearman_p_value_{condition}.csv",
            analysis, "p_value",
        )
        write_matrix_csv(
            BASE_DIR / f"relevance_spearman_n_{condition}.csv", analysis, "n"
        )
    plot_results(results)
    print(f"Saved Spearman results to {OUTPUT_JSON}")
    print(f"Saved figure to {FIGURE_STEM}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
