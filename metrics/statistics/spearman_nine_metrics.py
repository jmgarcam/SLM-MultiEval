"""Generate Spearman matrices for the eleven D1/D2/D3 evaluation metrics.

Metrics:
MTLD, Tree Depth, FH, BERTScore F1, SBERT, ETE, UER, NLI entailment,
one minus NLI contradiction, the proportion of factual judgements labelled
Correct, and one minus the proportion labelled Incorrect.

The shared analysis and output routines live in ``spearman_metric_matrix``.
This entry point adds the two D3 layouts, which differ from the seven regular
metric JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import spearman_metric_matrix as core


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spearman_results"

METRICS = (
    core.MetricSpec("MTLD", "metrics/mtld/mtld_limited_news.json", "mean_mtld"),
    core.MetricSpec(
        "Tree Depth",
        "metrics/readability/readability_limited_news.json",
        "mean_depth",
    ),
    core.MetricSpec(
        "FH", "metrics/readability/readability_limited_news.json", "mean_fh"
    ),
    core.MetricSpec(
        "BERTScore", "metrics/bert/bertscore_limited_news.json", "mean_f1"
    ),
    core.MetricSpec("SBERT", "metrics/sbert/sbert_limited_news.json", "mean_sbert"),
    core.MetricSpec(
        "ETE", "metrics/ete/ete_limited_news.json", "mean_entity_entropy"
    ),
    core.MetricSpec("UER", "metrics/uer/uer_limited_news.json", "mean_uer"),
    core.MetricSpec(
        "NLI_E",
        "metrics/nli_llm/nli_llm_results.json",
        "mean_entailment_softmax",
    ),
    core.MetricSpec(
        "1 - NLI_C",
        "metrics/nli_llm/nli_llm_results.json",
        "mean_contradiction_softmax",
    ),
    core.MetricSpec(
        "Factual Correct",
        "metrics/factual_llm/factual_llm_RAG_results.json",
        "correct",
    ),
    core.MetricSpec(
        "1 - Incorrect",
        "metrics/factual_llm/factual_llm_RAG_results.json",
        "incorrect",
    ),
)

NLI_ENTAILMENT_SPEC = METRICS[-4]
NLI_NON_CONTRADICTION_SPEC = METRICS[-3]
FACTUAL_CORRECT_SPEC = METRICS[-2]
FACTUAL_NON_INCORRECT_SPEC = METRICS[-1]
FACTUAL_PATHS = {
    "RAG": core.PROJECT_ROOT / "metrics/factual_llm/factual_llm_RAG_results.json",
    "NO_RAG": core.PROJECT_ROOT
    / "metrics/factual_llm/factual_llm_NO_RAG_results.json",
}
FACTUAL_FORCED_DENOMINATORS = {}
FACTUAL_IMPUTED_LABEL_COUNTS = {}
DIMENSIONS = {
    "D1": {
        "title": "Syntactic-Lexical Complexity",
        "metrics": METRICS[:3],
    },
    "D2": {
        "title": "Semantic Similarity and Entity Diversity",
        "metrics": METRICS[3:7],
    },
    "D3": {
        "title": "Factual Consistency",
        "metrics": METRICS[7:],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def flatten_nli_metric(
    spec: core.MetricSpec,
    *,
    invert: bool = False,
) -> dict[core.ObservationKey, float]:
    data = core.load_json(spec.path)
    values: dict[core.ObservationKey, float] = {}
    for condition in core.CONDITIONS:
        if condition not in data:
            raise KeyError(f"Missing {condition} block in {spec.path}")
        for newspaper, model_data in data[condition].items():
            for model_id, temperature_data in model_data.items():
                for temperature_id, block in temperature_data.items():
                    if spec.value_key not in block:
                        raise KeyError(
                            f"Missing {spec.value_key!r} in {spec.path}: "
                            f"{condition}/{newspaper}/{model_id}/{temperature_id}"
                        )
                    key = (
                        str(newspaper),
                        str(model_id),
                        str(temperature_id),
                        condition,
                    )
                    raw_value = core.as_finite_float(
                        block[spec.value_key],
                        "/".join(key) + f"/{spec.value_key}",
                    )
                    values[key] = 1.0 - raw_value if invert else raw_value
    if not values:
        raise ValueError(f"No observations found for {spec.name}")
    return values


def flatten_factual_metric(
    spec: core.MetricSpec,
    *,
    invert: bool = False,
) -> dict[core.ObservationKey, float]:
    values: dict[core.ObservationKey, float] = {}
    for condition, path in FACTUAL_PATHS.items():
        data = core.load_json(path)
        for newspaper, model_data in data.items():
            for model_id, temperature_data in model_data.items():
                for temperature_id, block in temperature_data.items():
                    location = f"{condition}/{newspaper}/{model_id}/{temperature_id}"
                    counts = {
                        label: core.as_finite_float(
                            block.get(label), f"{location}/{label}"
                        )
                        for label in ("correct", "partially_correct", "incorrect")
                    }
                    observed_total = math.fsum(counts.values())
                    if observed_total <= 0.0:
                        raise ValueError(f"Factual label total must be positive at {location}")
                    denominator_key = (
                        condition,
                        str(newspaper),
                        str(model_id),
                        str(temperature_id),
                    )
                    imputed_counts = FACTUAL_IMPUTED_LABEL_COUNTS.get(
                        denominator_key,
                        {},
                    )
                    counts = {
                        label: value + imputed_counts.get(label, 0.0)
                        for label, value in counts.items()
                    }
                    key = (
                        str(newspaper),
                        str(model_id),
                        str(temperature_id),
                        condition,
                    )
                    total = FACTUAL_FORCED_DENOMINATORS.get(
                        denominator_key,
                        observed_total,
                    )
                    if not math.isclose(math.fsum(counts.values()), total):
                        raise ValueError(
                            "Reported factual label counts must sum to the "
                            f"denominator at {location}"
                        )
                    ratio = counts[spec.value_key] / total
                    values[key] = 1.0 - ratio if invert else ratio
    if not values:
        raise ValueError(f"No observations found for {spec.name}")
    return values


def build_observations() -> list[dict[str, object]]:
    metric_values: dict[str, dict[core.ObservationKey, float]] = {}
    for spec in METRICS[:7]:
        metric_values[spec.name] = core.flatten_metric(core.load_json(spec.path), spec)
    metric_values[NLI_ENTAILMENT_SPEC.name] = flatten_nli_metric(
        NLI_ENTAILMENT_SPEC
    )
    metric_values[NLI_NON_CONTRADICTION_SPEC.name] = flatten_nli_metric(
        NLI_NON_CONTRADICTION_SPEC,
        invert=True,
    )
    metric_values[FACTUAL_CORRECT_SPEC.name] = flatten_factual_metric(
        FACTUAL_CORRECT_SPEC
    )
    metric_values[FACTUAL_NON_INCORRECT_SPEC.name] = flatten_factual_metric(
        FACTUAL_NON_INCORRECT_SPEC,
        invert=True,
    )

    reference_name = METRICS[0].name
    reference_keys = set(metric_values[reference_name])
    for spec in METRICS[1:]:
        current_keys = set(metric_values[spec.name])
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys, key=core.observation_sort_key)
            extra = sorted(current_keys - reference_keys, key=core.observation_sort_key)
            raise ValueError(
                f"Observation mismatch for {spec.name}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    observations: list[dict[str, object]] = []
    for newspaper, model_id, temperature_id, condition in sorted(
        reference_keys, key=core.observation_sort_key
    ):
        key = (newspaper, model_id, temperature_id, condition)
        row: dict[str, object] = {
            "newspaper": newspaper,
            "model_id": model_id,
            "model": core.MODEL_LABELS.get(model_id, model_id),
            "temperature_id": temperature_id,
            "temperature": core.TEMPERATURE_LABELS.get(temperature_id, temperature_id),
            "condition": condition,
        }
        for spec in METRICS:
            row[spec.name] = metric_values[spec.name][key]
        observations.append(row)
    return observations


def correct_summary_metadata(
    output_dir: Path,
    dimension: str = "ALL",
    dimension_title: str = "All evaluation dimensions",
) -> None:
    """Record dimension and D3 metadata in the generated summary."""

    path = output_dir / "spearman_summary.json"
    with path.open("r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    definitions = {
        definition["name"]: definition
        for definition in summary["metric_definitions"]
    }
    if NLI_ENTAILMENT_SPEC.name in definitions:
        definitions[NLI_ENTAILMENT_SPEC.name].update(
            {
                "definition": "mean entailment softmax from the LLM NLI evaluator",
                "aggregation_n_per_cell": 25,
            }
        )
    if NLI_NON_CONTRADICTION_SPEC.name in definitions:
        definitions[NLI_NON_CONTRADICTION_SPEC.name].update(
            {
                "definition": "1 minus mean contradiction softmax from the LLM NLI evaluator",
                "aggregation_n_per_cell": 25,
            }
        )
    if FACTUAL_CORRECT_SPEC.name in definitions:
        definitions[FACTUAL_CORRECT_SPEC.name].update(
            {
                "source": [
                    str(path.relative_to(core.PROJECT_ROOT))
                    for path in FACTUAL_PATHS.values()
                ],
                "definition": "correct / reporting denominator",
                "aggregation_n_per_cell": 50,
            }
        )
    if FACTUAL_NON_INCORRECT_SPEC.name in definitions:
        definitions[FACTUAL_NON_INCORRECT_SPEC.name].update(
            {
                "source": [
                    str(path.relative_to(core.PROJECT_ROOT))
                    for path in FACTUAL_PATHS.values()
                ],
                "definition": "1 - incorrect / reporting denominator",
                "aggregation_n_per_cell": 50,
            }
        )
    summary["dimension"] = dimension
    summary["dimension_title"] = dimension_title
    summary["metric_selection"] = [spec.name for spec in core.METRICS]
    summary["factual_forced_denominators"] = [
        {
            "condition": condition,
            "newspaper": newspaper,
            "model_id": model_id,
            "temperature_id": temperature_id,
            "denominator": denominator,
        }
        for (
            condition,
            newspaper,
            model_id,
            temperature_id,
        ), denominator in FACTUAL_FORCED_DENOMINATORS.items()
    ]
    summary["factual_imputed_label_counts"] = [
        {
            "condition": condition,
            "newspaper": newspaper,
            "model_id": model_id,
            "temperature_id": temperature_id,
            "counts": counts,
        }
        for (
            condition,
            newspaper,
            model_id,
            temperature_id,
        ), counts in FACTUAL_IMPUTED_LABEL_COUNTS.items()
    ]
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> None:
    args = parse_args()
    # The shared routines derive their labels and output columns from METRICS.
    core.METRICS = METRICS
    observations = build_observations()
    analyses = [
        core.analyze_scope(observations, scope) for scope in core.CONDITION_SCOPES
    ]
    core.write_outputs(args.output_dir, observations, analyses)
    correct_summary_metadata(args.output_dir)
    core.print_summary(analyses, args.output_dir)

    dimension_root = args.output_dir / "by_dimension"
    for dimension, definition in DIMENSIONS.items():
        core.METRICS = definition["metrics"]
        dimension_analyses = [
            core.analyze_scope(observations, scope)
            for scope in core.CONDITION_SCOPES
        ]
        for analysis in dimension_analyses:
            analysis["dimension"] = dimension
            analysis["dimension_title"] = definition["title"]
        dimension_output = dimension_root / dimension
        core.write_outputs(
            dimension_output,
            observations,
            dimension_analyses,
        )
        correct_summary_metadata(
            dimension_output,
            dimension=dimension,
            dimension_title=definition["title"],
        )
        core.print_summary(dimension_analyses, dimension_output)

    core.METRICS = METRICS


if __name__ == "__main__":
    main()
