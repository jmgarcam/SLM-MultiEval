"""Correlate per-news coherence with FH and tree depth.

Run ``coherence_fh_td.py`` first. The enriched per-news JSON files are the
only analytical inputs. FH versus tree depth belongs to the aggregated D1
analysis and is intentionally not repeated here.
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
    "NO_RAG": BASE_DIR / "coherence_NO_RAG_fh_td_by_news.json",
    "RAG": BASE_DIR / "coherence_RAG_fh_td_by_news.json",
}
TARGET_METRIC = ("coherence", "Coherence")
COMPARISON_METRICS = (("fh", "FH"), ("tree_depth", "Tree Depth"))
ALL_METRICS = (TARGET_METRIC, *COMPARISON_METRICS)
OUTPUT_JSON = BASE_DIR / "coherence_spearman_results.json"
FIGURE_STEM = BASE_DIR / "coherence_spearman_matrix_RAG_NO_RAG"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required intermediate JSON not found: {path}. "
            "Run coherence_fh_td.py first."
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
                        raise ValueError(f"{label}: record without id_news")
                    if key in seen_keys:
                        raise ValueError(f"{label}: duplicate observation key {key}")
                    seen_keys.add(key)
                    context = "/".join(key)
                    flattened.append(
                        {
                            "synthetic_description": news_item.get(
                                "synthetic_description", ""
                            ),
                            **{
                                metric_key: finite_float(
                                    news_item.get(metric_key), f"{context}/{metric_key}"
                                )
                                for metric_key, _ in ALL_METRICS
                            },
                        }
                    )
    if len(flattened) < 2:
        raise ValueError(f"{label}: at least two complete observations are required")
    return flattened


def duplicate_summary(news_items: list[dict]) -> dict:
    counts = Counter(str(item["synthetic_description"]) for item in news_items)
    duplicated_counts = [count for count in counts.values() if count > 1]
    return {
        "num_distinct_descriptions": len(counts),
        "num_duplicated_descriptions": len(duplicated_counts),
        "num_rows_in_duplicated_descriptions": sum(duplicated_counts),
        "max_description_occurrences": max(duplicated_counts, default=1),
        "policy": "retained",
    }


def analyze_file(path: Path, label: str) -> dict:
    news_items = flatten_news_items(load_json(path), label)
    target_key, target_label = TARGET_METRIC
    comparison_labels = [name for _, name in COMPARISON_METRICS]
    rho_values: list[float] = []
    p_values: list[float] = []
    pairwise_results: dict[str, dict] = {}

    for comparison_key, _ in COMPARISON_METRICS:
        result = spearmanr(
            [item[target_key] for item in news_items],
            [item[comparison_key] for item in news_items],
        )
        rho = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(rho) or not math.isfinite(p_value):
            raise ValueError(
                f"{label}: undefined Spearman result for "
                f"{target_key} versus {comparison_key}"
            )
        rho_values.append(rho)
        p_values.append(p_value)
        pairwise_results[f"{target_key}_vs_{comparison_key}"] = {
            "rho": rho, "p_value": p_value, "n": len(news_items)
        }

    duplicates = duplicate_summary(news_items)
    if duplicates["num_rows_in_duplicated_descriptions"]:
        print(
            f"[WARNING] {label}: "
            f"{duplicates['num_rows_in_duplicated_descriptions']} rows belong "
            "to duplicated synthetic descriptions; rows are retained."
        )
    return {
        "source_file": path.name,
        "unit_of_analysis": "news item",
        "num_news": len(news_items),
        "target_metric": target_label,
        "comparison_metrics": comparison_labels,
        "rho": [rho_values],
        "p_value": [p_values],
        "n": [[len(news_items)] * len(COMPARISON_METRICS)],
        "pairwise_spearman": pairwise_results,
        "duplicate_descriptions": duplicates,
    }


def write_matrix_csv(
    path: Path, analysis: dict, matrix_name: str, value_type: str
) -> None:
    row = analysis[matrix_name][0]
    values = (
        [f"{float(value):.10g}" for value in row]
        if value_type == "float"
        else [int(value) for value in row]
    )
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", *analysis["comparison_metrics"]])
        writer.writerow([analysis["target_metric"], *values])


def write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    temporary_path.replace(path)


def write_heatmap(results: dict[str, dict]) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    image = None
    for axis, label in zip(axes, ("RAG", "NO_RAG"), strict=True):
        analysis = results[label]
        matrix = np.asarray(analysis["rho"], dtype=float)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axis.set_title(
            f"{label.replace('_', ' ')}\n(n = {analysis['num_news']})",
            fontsize=15, fontweight="bold", pad=12,
        )
        axis.set_xticks(range(len(analysis["comparison_metrics"])))
        axis.set_xticklabels(analysis["comparison_metrics"], fontsize=11)
        axis.set_yticks([0])
        axis.set_yticklabels([analysis["target_metric"]], fontsize=11)
        axis.tick_params(length=0)
        for column_index, value in enumerate(matrix[0]):
            axis.text(
                column_index, 0, f"{value:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if abs(value) >= 0.65 else "#202020",
            )
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.subplots_adjust(
        left=0.13, right=0.87, bottom=0.20, top=0.78, wspace=0.48
    )
    colorbar_axis = figure.add_axes((0.91, 0.22, 0.018, 0.52))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman ρ", fontsize=11)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    for extension in ("svg", "png", "pdf"):
        figure.savefig(
            FIGURE_STEM.with_suffix(f".{extension}"), dpi=300, facecolor="white"
        )
    plt.close(figure)


def main() -> None:
    results = {label: analyze_file(path, label) for label, path in INPUT_FILES.items()}
    write_json(OUTPUT_JSON, results)
    for label, analysis in results.items():
        for matrix_name, value_type in (
            ("rho", "float"), ("p_value", "float"), ("n", "int")
        ):
            write_matrix_csv(
                BASE_DIR / f"coherence_spearman_{matrix_name}_{label}.csv",
                analysis, matrix_name, value_type,
            )
    write_heatmap(results)
    print(f"Saved Spearman results to {OUTPUT_JSON}")
    for label, analysis in results.items():
        print(f"{label}: n = {analysis['num_news']}")


if __name__ == "__main__":
    main()
