"""Correlate G-Eval Consistency with NLI and factual LLM scores by news.

G-Eval and factual values come from the enriched per-news JSON files. NLI-E
and NLI-C are replaced with the evaluations from the new-journal NLI study
before applying the same three-way softmax used by metrics/nli_llm. Factual
categories are converted to binary indicators.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
NEW_NLI_RESULTS_DIR = (
    PROJECT_ROOT / "metrics/nli_llm/new_journal/nli_results"
)
NEW_NLI_FILES = {
    "NO_RAG": NEW_NLI_RESULTS_DIR / "evaluation_no_rag_agrupado.json",
    "RAG": NEW_NLI_RESULTS_DIR / "evaluacion_rag_agrupado.json",
}
INPUT_FILES = {
    "NO_RAG": BASE_DIR / "consistency_NO_RAG_by_news.json",
    "RAG": BASE_DIR / "consistency_RAG_by_news.json",
}
TARGET_METRIC = ("g_eval_consistency", "G-Eval Consistency")
COMPARISON_METRICS = (
    ("nli_e", "NLI-E"),
    ("one_minus_nli_c", "1−NLI-C"),
    ("factual_correct", "Factual Correct"),
    ("one_minus_incorrect", "1−Incorrect"),
)
OUTPUT_JSON = BASE_DIR / "consistency_spearman_results.json"
FIGURE_STEM = BASE_DIR / "consistency_spearman_matrix_RAG_NO_RAG"
FACTUAL_DECISIONS = {"correct", "partially_correct", "incorrect"}

DERIVED_METRICS = {
    "NLI-E": (
        "softmax(entailment, contradiction, no_neutral)[entailment]"
    ),
    "1−NLI-C": (
        "1 - softmax(entailment, contradiction, no_neutral)[contradiction]"
    ),
    "Factual Correct": "1 if factual_llm.decision == 'correct', else 0",
    "1−Incorrect": "1 if factual_llm.decision != 'incorrect', else 0",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required intermediate JSON not found: {path}. "
            "Run build_llm_results_by_news.py first."
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


def build_new_nli_index(
    source_data: dict, label: str
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """Index the new per-news NLI evaluations by their pairing key."""

    index: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for newspaper, llm_dict in source_data.items():
        if not isinstance(llm_dict, dict):
            raise ValueError(f"{label}: expected an object at {newspaper}")
        for llm_id, feature_dict in llm_dict.items():
            if not isinstance(feature_dict, dict):
                raise ValueError(
                    f"{label}: expected an object at {newspaper}/{llm_id}"
                )
            for feature_id, news_list in feature_dict.items():
                if not isinstance(news_list, list):
                    raise ValueError(
                        f"{label}: expected a list at "
                        f"{newspaper}/{llm_id}/{feature_id}"
                    )
                for news_item in news_list:
                    if not isinstance(news_item, dict):
                        raise ValueError(f"{label}: NLI entry is not an object")
                    news_id = str(news_item.get("id_news", "")).strip()
                    key = (
                        str(newspaper),
                        str(llm_id),
                        str(feature_id),
                        news_id,
                    )
                    if not news_id:
                        raise ValueError(f"{label}: NLI record without id_news")
                    if key in index:
                        raise ValueError(f"{label}: duplicate NLI key {key}")

                    raw_scores = {
                        field: finite_float(
                            news_item.get(field),
                            f"{label}/{'/'.join(key)}/{field}",
                        )
                        for field in ("entailment", "contradiction", "no_neutral")
                    }
                    if any(
                        not 0.0 <= value <= 3.0
                        for value in raw_scores.values()
                    ):
                        raise ValueError(
                            f"{label}: NLI value outside [0, 3] at {key}: "
                            f"{raw_scores}"
                        )
                    index[key] = {
                        "entailment": raw_scores["entailment"],
                        "contradiction": raw_scores["contradiction"],
                        "no_mention": raw_scores["no_neutral"],
                    }
    return index


def inject_new_nli_scores(
    source_data: dict, nli_data: dict, label: str
) -> dict:
    """Replace embedded legacy NLI scores with the new-study evaluations."""

    nli_index = build_new_nli_index(nli_data, label)
    used_keys: set[tuple[str, str, str, str]] = set()

    for newspaper, llm_dict in source_data.items():
        for llm_id, feature_dict in llm_dict.items():
            for feature_id, news_list in feature_dict.items():
                for news_item in news_list:
                    news_id = str(news_item.get("id_news", "")).strip()
                    key = (
                        str(newspaper),
                        str(llm_id),
                        str(feature_id),
                        news_id,
                    )
                    if key not in nli_index:
                        raise ValueError(
                            f"{label}: new NLI evaluation missing for {key}"
                        )
                    news_item["nli_llm"] = nli_index[key]
                    used_keys.add(key)

    extra_keys = set(nli_index) - used_keys
    if (
        len(extra_keys) != 1
        or not all(key[-1].endswith("__dup1") for key in extra_keys)
    ):
        raise ValueError(
            f"{label}: expected only the unmatched completion duplicate in "
            f"new NLI; found {len(extra_keys)} extra keys: "
            f"{sorted(extra_keys)[:3]}"
        )
    return {
        "available_new_nli_rows": len(nli_index),
        "matched_rows": len(used_keys),
        "unmatched_completion_duplicates": [list(key) for key in sorted(extra_keys)],
    }


def softmax_three(values: tuple[float, float, float]) -> tuple[float, float, float]:
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end + 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def pearson_correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered, right_centered, strict=True
        )
    )
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant variable")
    return max(-1.0, min(1.0, numerator / denominator))


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result

    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"Incomplete beta x outside [0, 1]: {x}")
    if x in (0.0, 1.0):
        return x

    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def spearman_with_p_value(
    left: list[float], right: list[float]
) -> tuple[float, float]:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("Spearman requires equally sized samples with n >= 3")
    rho = pearson_correlation(average_ranks(left), average_ranks(right))
    if abs(rho) == 1.0:
        return rho, 0.0
    degrees_of_freedom = len(left) - 2
    statistic = abs(rho) * math.sqrt(
        degrees_of_freedom / ((1.0 - rho) * (1.0 + rho))
    )
    beta_x = degrees_of_freedom / (
        degrees_of_freedom + statistic * statistic
    )
    p_value = regularized_incomplete_beta(
        beta_x, degrees_of_freedom / 2.0, 0.5
    )
    return rho, max(0.0, min(1.0, p_value))


def derive_metrics(news_item: dict, context: str) -> dict[str, float]:
    consistency = finite_float(
        news_item.get("consistency"), f"{context}/consistency"
    )
    if not 1.0 <= consistency <= 5.0:
        raise ValueError(
            f"G-Eval Consistency outside [1, 5] at {context}: {consistency}"
        )

    nli = news_item.get("nli_llm")
    if not isinstance(nli, dict):
        raise ValueError(f"Missing nli_llm object at {context}")
    raw_nli = tuple(
        finite_float(nli.get(field), f"{context}/nli_llm/{field}")
        for field in ("entailment", "contradiction", "no_mention")
    )
    if any(not 0.0 <= value <= 3.0 for value in raw_nli):
        raise ValueError(f"NLI value outside [0, 3] at {context}: {raw_nli}")
    nli_e, nli_c, _ = softmax_three(raw_nli)

    factual = news_item.get("factual_llm")
    if not isinstance(factual, dict):
        raise ValueError(f"Missing factual_llm object at {context}")
    decision = str(factual.get("decision", "")).strip()
    if decision not in FACTUAL_DECISIONS:
        raise ValueError(f"Invalid factual decision at {context}: {decision!r}")

    return {
        "g_eval_consistency": consistency,
        "nli_e": nli_e,
        "one_minus_nli_c": 1.0 - nli_c,
        "factual_correct": float(decision == "correct"),
        "one_minus_incorrect": float(decision != "incorrect"),
    }


def flatten_news_items(source_data: dict, label: str) -> list[dict]:
    flattened: list[dict] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for newspaper, llm_dict in source_data.items():
        if not isinstance(llm_dict, dict):
            raise ValueError(f"{label}: expected an object at {newspaper}")
        for llm_id, feature_dict in llm_dict.items():
            if not isinstance(feature_dict, dict):
                raise ValueError(
                    f"{label}: expected an object at {newspaper}/{llm_id}"
                )
            for feature_id, news_list in feature_dict.items():
                if not isinstance(news_list, list):
                    raise ValueError(
                        f"{label}: expected a list at "
                        f"{newspaper}/{llm_id}/{feature_id}"
                    )
                for news_item in news_list:
                    if not isinstance(news_item, dict):
                        raise ValueError(f"{label}: news entry is not an object")
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
                            **derive_metrics(news_item, context),
                        }
                    )

    if len(flattened) != 899:
        raise ValueError(f"{label}: expected 899 complete observations, got {len(flattened)}")
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


def analyze_file(path: Path, nli_path: Path, label: str) -> dict:
    source_data = load_json(path)
    nli_join = inject_new_nli_scores(
        source_data,
        load_json(nli_path),
        label,
    )
    news_items = flatten_news_items(source_data, label)
    target_key, target_label = TARGET_METRIC
    comparison_labels = [name for _, name in COMPARISON_METRICS]
    rho_values: list[float] = []
    p_values: list[float] = []
    pairwise_results: dict[str, dict] = {}

    target_values = [item[target_key] for item in news_items]
    for comparison_key, comparison_label in COMPARISON_METRICS:
        rho, p_value = spearman_with_p_value(
            target_values,
            [item[comparison_key] for item in news_items],
        )
        if not math.isfinite(rho) or not math.isfinite(p_value):
            raise ValueError(
                f"{label}: undefined Spearman result for "
                f"{target_key} versus {comparison_key}"
            )
        rho_values.append(rho)
        p_values.append(p_value)
        pairwise_results[f"{target_key}_vs_{comparison_key}"] = {
            "target": target_label,
            "comparison": comparison_label,
            "rho": rho,
            "p_value": p_value,
            "n": len(news_items),
        }

    return {
        "source_file": path.name,
        "nli_source_file": str(nli_path),
        "nli_join": nli_join,
        "unit_of_analysis": "news item",
        "num_news": len(news_items),
        "target_metric": target_label,
        "comparison_metrics": comparison_labels,
        "derived_metrics": DERIVED_METRICS,
        "rho": [rho_values],
        "p_value": [p_values],
        "n": [[len(news_items)] * len(COMPARISON_METRICS)],
        "pairwise_spearman": pairwise_results,
        "duplicate_descriptions": duplicate_summary(news_items),
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


def heatmap_color(value: float) -> str:
    neutral = (247, 247, 247)
    endpoint = (33, 102, 172) if value < 0.0 else (178, 24, 43)
    weight = abs(value)
    rgb = tuple(
        round(neutral[channel] * (1.0 - weight) + endpoint[channel] * weight)
        for channel in range(3)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_heatmap(results: dict[str, dict]) -> None:
    width = 1120
    height = 470
    cell_width = 190
    cell_height = 66
    matrix_left = 300
    panel_tops = {"RAG": 38, "NO_RAG": 220}
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#202020">',
    ]

    for label in ("RAG", "NO_RAG"):
        analysis = results[label]
        top = panel_tops[label]
        row_top = top + 70
        parts.append(
            f'<text x="{width / 2}" y="{top}" text-anchor="middle" '
            f'font-size="20" font-weight="700">'
            f'{escape(label.replace("_", " "))} '
            f'(n = {analysis["num_news"]})</text>'
        )
        for column, metric in enumerate(analysis["comparison_metrics"]):
            center_x = matrix_left + column * cell_width + cell_width / 2
            parts.append(
                f'<text x="{center_x}" y="{top + 43}" text-anchor="middle" '
                f'font-size="15">{escape(metric)}</text>'
            )
        parts.append(
            f'<text x="{matrix_left - 20}" y="{row_top + 40}" '
            f'text-anchor="end" font-size="15">'
            f'{escape(analysis["target_metric"])}</text>'
        )
        for column, value in enumerate(analysis["rho"][0]):
            x = matrix_left + column * cell_width
            text_color = "white" if abs(value) >= 0.65 else "#202020"
            parts.extend(
                [
                    (
                        f'<rect x="{x}" y="{row_top}" width="{cell_width}" '
                        f'height="{cell_height}" fill="{heatmap_color(value)}" '
                        'stroke="white" stroke-width="2"/>'
                    ),
                    (
                        f'<text x="{x + cell_width / 2}" '
                        f'y="{row_top + 42}" text-anchor="middle" '
                        f'font-size="19" font-weight="700" '
                        f'fill="{text_color}">{value:.2f}</text>'
                    ),
                ]
            )

    legend_left = 360
    legend_top = 420
    legend_width = 500
    steps = 100
    for step in range(steps):
        value = -1.0 + 2.0 * step / (steps - 1)
        parts.append(
            f'<rect x="{legend_left + step * legend_width / steps:.2f}" '
            f'y="{legend_top}" width="{legend_width / steps + 0.2:.2f}" '
            f'height="14" fill="{heatmap_color(value)}"/>'
        )
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        x = legend_left + (value + 1.0) * legend_width / 2.0
        parts.append(
            f'<text x="{x}" y="{legend_top + 31}" text-anchor="middle" '
            f'font-size="12">{value:g}</text>'
        )
    parts.append(
        f'<text x="{legend_left - 16}" y="{legend_top + 13}" '
        'text-anchor="end" font-size="13">Spearman rho</text>'
    )
    parts.extend(["</g>", "</svg>"])
    FIGURE_STEM.with_suffix(".svg").write_text(
        "\n".join(parts) + "\n", encoding="utf-8"
    )


def main() -> None:
    results = {
        label: analyze_file(path, NEW_NLI_FILES[label], label)
        for label, path in INPUT_FILES.items()
    }
    write_json(OUTPUT_JSON, results)
    for label, analysis in results.items():
        for matrix_name, value_type in (
            ("rho", "float"),
            ("p_value", "float"),
            ("n", "int"),
        ):
            write_matrix_csv(
                BASE_DIR / f"consistency_spearman_{matrix_name}_{label}.csv",
                analysis,
                matrix_name,
                value_type,
            )
    write_heatmap(results)
    print(f"Saved Spearman results to {OUTPUT_JSON}")
    for label, analysis in results.items():
        rho_text = ", ".join(
            f"{metric}={rho:.6f}"
            for metric, rho in zip(
                analysis["comparison_metrics"], analysis["rho"][0], strict=True
            )
        )
        print(f"{label}: n={analysis['num_news']}; {rho_text}")


if __name__ == "__main__":
    main()
