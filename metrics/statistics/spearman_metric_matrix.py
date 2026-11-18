"""Spearman correlation matrices for the nine evaluation metrics.

The unit of analysis is one aggregated experimental cell:
newspaper x model x temperature x condition.  With the current inputs this
produces 36 complete observations (18 RAG and 18 NO_RAG).

The script uses only the Python standard library for the analysis and a local
headless browser for rendering. It writes CSV matrices, a machine-readable
JSON summary, and PNG/PDF heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

CONDITIONS = ("RAG", "NO_RAG")
CONDITION_SCOPES = ("ALL", "RAG", "NO_RAG")
MODEL_LABELS = {
    "1": "Ministral-3B",
    "2": "Qwen-2.5-0.5B",
    "3": "Llama-3.2-1B",
}
TEMPERATURE_LABELS = {"1": "1.00", "2": "0.75", "3": "0.50"}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    relative_path: str
    value_key: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.relative_path


# These are the nine scalar automatic metrics described in the repository.
# BERTScore contributes its three standard components.
METRICS = (
    MetricSpec("FH", "metrics/readability/readability_limited_news.json", "mean_fh"),
    MetricSpec(
        "Tree Depth",
        "metrics/readability/readability_limited_news.json",
        "mean_depth",
    ),
    MetricSpec("MTLD", "metrics/mtld/mtld_limited_news.json", "mean_mtld"),
    MetricSpec(
        "BERT-P", "metrics/bert/bertscore_limited_news.json", "mean_precision"
    ),
    MetricSpec(
        "BERT-R", "metrics/bert/bertscore_limited_news.json", "mean_recall"
    ),
    MetricSpec("BERTScore", "metrics/bert/bertscore_limited_news.json", "mean_f1"),
    MetricSpec("SBERT", "metrics/sbert/sbert_limited_news.json", "mean_sbert"),
    MetricSpec(
        "ETE", "metrics/ete/ete_limited_news.json", "mean_entity_entropy"
    ),
    MetricSpec("UER", "metrics/uer/uer_limited_news.json", "mean_uer"),
)

ObservationKey = tuple[str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty JSON object in {path}")
    return data


def as_finite_float(value: object, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value at {context}: {value!r}")
    return number


def flatten_metric(data: dict, spec: MetricSpec) -> dict[ObservationKey, float]:
    """Extract one metric value for every synthetic experimental cell."""

    values: dict[ObservationKey, float] = {}
    for newspaper, newspaper_data in data.items():
        if not isinstance(newspaper_data, dict) or "llm_models" not in newspaper_data:
            continue
        for model_id, temperature_data in newspaper_data["llm_models"].items():
            for temperature_id, blocks in temperature_data.items():
                for condition in CONDITIONS:
                    if condition not in blocks:
                        raise KeyError(
                            f"Missing {condition} block in {spec.path}: "
                            f"{newspaper}/{model_id}/{temperature_id}"
                        )
                    block = blocks[condition]
                    if spec.value_key not in block:
                        raise KeyError(
                            f"Missing {spec.value_key!r} in {spec.path}: "
                            f"{newspaper}/{model_id}/{temperature_id}/{condition}"
                        )
                    key = (
                        str(newspaper),
                        str(model_id),
                        str(temperature_id),
                        condition,
                    )
                    context = "/".join(key) + f"/{spec.value_key}"
                    values[key] = as_finite_float(block[spec.value_key], context)
    if not values:
        raise ValueError(f"No synthetic observations found for {spec.name} in {spec.path}")
    return values


def observation_sort_key(key: ObservationKey) -> tuple:
    newspaper, model_id, temperature_id, condition = key
    condition_order = {name: index for index, name in enumerate(CONDITIONS)}
    return (
        newspaper,
        int(model_id) if model_id.isdigit() else model_id,
        int(temperature_id) if temperature_id.isdigit() else temperature_id,
        condition_order.get(condition, len(condition_order)),
    )


def build_observations() -> list[dict[str, object]]:
    """Load every metric and enforce the same observation keys across inputs."""

    metric_values: dict[str, dict[ObservationKey, float]] = {}
    for spec in METRICS:
        metric_values[spec.name] = flatten_metric(load_json(spec.path), spec)

    reference_name = METRICS[0].name
    reference_keys = set(metric_values[reference_name])
    for spec in METRICS[1:]:
        current_keys = set(metric_values[spec.name])
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys, key=observation_sort_key)
            extra = sorted(current_keys - reference_keys, key=observation_sort_key)
            raise ValueError(
                f"Observation mismatch for {spec.name}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    observations: list[dict[str, object]] = []
    for newspaper, model_id, temperature_id, condition in sorted(
        reference_keys, key=observation_sort_key
    ):
        key = (newspaper, model_id, temperature_id, condition)
        row: dict[str, object] = {
            "newspaper": newspaper,
            "model_id": model_id,
            "model": MODEL_LABELS.get(model_id, model_id),
            "temperature_id": temperature_id,
            "temperature": TEMPERATURE_LABELS.get(temperature_id, temperature_id),
            "condition": condition,
        }
        for spec in METRICS:
            row[spec.name] = metric_values[spec.name][key]
        observations.append(row)
    return observations


def average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, including correct handling of ties."""

    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    return ranks


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Correlation vectors must have the same length")
    if len(x_values) < 2:
        return None
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    denominator = math.sqrt(
        math.fsum(value * value for value in x_centered)
        * math.fsum(value * value for value in y_centered)
    )
    if denominator == 0.0:
        return None
    rho = math.fsum(
        x_value * y_value for x_value, y_value in zip(x_centered, y_centered)
    ) / denominator
    return max(-1.0, min(1.0, rho))


def calculate_spearman(x_values: Sequence[float], y_values: Sequence[float]) -> dict:
    paired = [
        (float(x_value), float(y_value))
        for x_value, y_value in zip(x_values, y_values)
        if math.isfinite(float(x_value)) and math.isfinite(float(y_value))
    ]
    sample_size = len(paired)
    if sample_size < 2:
        return {"rho": None, "n": sample_size}
    x_clean, y_clean = zip(*paired)
    rho = pearson_correlation(average_ranks(x_clean), average_ranks(y_clean))
    if rho is None:
        return {"rho": None, "n": sample_size}
    return {"rho": rho, "n": sample_size}


def analyze_scope(observations: Sequence[dict[str, object]], scope: str) -> dict:
    rows = (
        list(observations)
        if scope == "ALL"
        else [row for row in observations if row["condition"] == scope]
    )
    metric_names = [spec.name for spec in METRICS]
    size = len(metric_names)
    rho_matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
    n_matrix: list[list[int]] = [[0] * size for _ in range(size)]
    pairs: list[dict[str, object]] = []

    for row_index, metric_x in enumerate(metric_names):
        for column_index in range(row_index, size):
            metric_y = metric_names[column_index]
            result = calculate_spearman(
                [float(row[metric_x]) for row in rows],
                [float(row[metric_y]) for row in rows],
            )
            for x_index, y_index in (
                (row_index, column_index),
                (column_index, row_index),
            ):
                rho_matrix[x_index][y_index] = result["rho"]
                n_matrix[x_index][y_index] = result["n"]
            if row_index != column_index:
                pairs.append(
                    {
                        "metric_x": metric_x,
                        "metric_y": metric_y,
                        **result,
                    }
                )

    pairs.sort(
        key=lambda pair: abs(float(pair["rho"])) if pair["rho"] is not None else -1.0,
        reverse=True,
    )
    return {
        "scope": scope,
        "num_observations": len(rows),
        "metrics": metric_names,
        "rho": rho_matrix,
        "n": n_matrix,
        "pairs": pairs,
    }


def format_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_matrix_csv(path: Path, metric_names: Sequence[str], matrix: Sequence[Sequence]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", *metric_names])
        for metric_name, row in zip(metric_names, matrix):
            writer.writerow([metric_name, *(format_csv_value(value) for value in row)])


def write_observations_csv(path: Path, observations: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "newspaper",
        "model_id",
        "model",
        "temperature_id",
        "temperature",
        "condition",
        *(spec.name for spec in METRICS),
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(observations)


def interpolate_color(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    channels = [round(a + (b - a) * ratio) for a, b in zip(start, end)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def rho_color(rho: float | None) -> str:
    if rho is None:
        return "#d9d9d9"
    blue = (49, 54, 149)
    white = (255, 255, 255)
    red = (165, 0, 38)
    if rho < 0:
        return interpolate_color(blue, white, rho + 1.0)
    return interpolate_color(white, red, rho)


def xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def metric_label_markup(metric_name: object) -> str:
    """Return escaped SVG markup, rendering NLI class suffixes as subscripts."""

    label = xml_escape(metric_name)
    for suffix in ("C", "E"):
        label = label.replace(
            f"NLI_{suffix}",
            f'NLI<tspan baseline-shift="sub" font-size="75%">{suffix}</tspan>',
        )
    return label


def build_heatmap_markup(analysis: dict) -> tuple[str, int, int]:
    metric_names = analysis["metrics"]
    rho_matrix = analysis["rho"]
    show_sample_size = analysis["num_observations"] != 18
    cell_size = 66
    left_margin = 150
    top_margin = 135 if show_sample_size else 115
    right_margin = 100
    # The colorbar label is 91 px below the matrix; keep extra room for the
    # font descenders so it remains visible in both PNG and tightly cropped PDF.
    bottom_margin = 110
    matrix_size = len(metric_names) * cell_size
    width = left_margin + matrix_size + right_margin
    height = top_margin + matrix_size + bottom_margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, Helvetica, sans-serif; }</style>',
    ]
    if show_sample_size:
        parts.append(
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            f'font-size="14" fill="#444">n = {analysis["num_observations"]}</text>'
        )

    for index, metric_name in enumerate(metric_names):
        center_x = left_margin + index * cell_size + cell_size / 2
        center_y = top_margin + index * cell_size + cell_size / 2
        parts.append(
            f'<text x="{left_margin - 10}" y="{center_y + 5:.1f}" text-anchor="end" '
            f'font-size="14">{metric_label_markup(metric_name)}</text>'
        )
        parts.append(
            f'<text x="{center_x:.1f}" y="{top_margin - 12}" text-anchor="start" '
            f'font-size="14" transform="rotate(-45 {center_x:.1f} {top_margin - 12})">'
            f'{metric_label_markup(metric_name)}</text>'
        )

    for row_index, row in enumerate(rho_matrix):
        for column_index, rho in enumerate(row):
            x_position = left_margin + column_index * cell_size
            y_position = top_margin + row_index * cell_size
            fill = rho_color(rho)
            parts.append(
                f'<rect x="{x_position}" y="{y_position}" width="{cell_size}" '
                f'height="{cell_size}" fill="{fill}" stroke="#eeeeee"/>'
            )
            label = "NA" if rho is None else f"{rho:.2f}"
            text_color = "white" if rho is not None and abs(rho) >= 0.65 else "#202020"
            parts.append(
                f'<text x="{x_position + cell_size / 2:.1f}" '
                f'y="{y_position + cell_size / 2 + 5:.1f}" text-anchor="middle" '
                f'font-size="13" font-weight="bold" fill="{text_color}">'
                f'{xml_escape(label)}</text>'
            )

    colorbar_x = left_margin
    colorbar_y = top_margin + matrix_size + 36
    colorbar_width = matrix_size
    segments = 100
    for segment in range(segments):
        segment_rho = -1.0 + 2.0 * segment / (segments - 1)
        x_position = colorbar_x + colorbar_width * segment / segments
        parts.append(
            f'<rect x="{x_position:.2f}" y="{colorbar_y}" '
            f'width="{colorbar_width / segments + 0.5:.2f}" height="14" '
            f'fill="{rho_color(segment_rho)}"/>'
        )
    for value, label in ((0.0, "−1"), (0.5, "0"), (1.0, "+1")):
        x_position = colorbar_x + colorbar_width * value
        parts.append(
            f'<text x="{x_position:.1f}" y="{colorbar_y + 33}" text-anchor="middle" '
            f'font-size="13">{label}</text>'
        )
    parts.append(
        f'<text x="{colorbar_x + colorbar_width / 2:.1f}" y="{colorbar_y + 55}" '
        f'text-anchor="middle" font-size="13">Spearman ρ</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts), width, height


def build_condition_comparison_markup(
    rag_analysis: dict,
    no_rag_analysis: dict,
) -> tuple[str, int, int]:
    if rag_analysis["metrics"] != no_rag_analysis["metrics"]:
        raise ValueError("RAG and NO RAG matrices must contain the same metrics")

    metric_names = rag_analysis["metrics"]
    cell_size = 66
    label_margin = 150
    panel_right_margin = 30
    panel_gap = 70
    top_margin = 135
    bottom_margin = 110
    matrix_size = len(metric_names) * cell_size
    panel_width = label_margin + matrix_size + panel_right_margin
    width = 2 * panel_width + panel_gap
    height = top_margin + matrix_size + bottom_margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, Helvetica, sans-serif; }</style>',
    ]

    panels = (
        ("RAG", rag_analysis, 0),
        ("NO RAG", no_rag_analysis, panel_width + panel_gap),
    )
    for panel_label, analysis, panel_x in panels:
        matrix_x = panel_x + label_margin
        parts.append(
            f'<text x="{matrix_x + matrix_size / 2:.1f}" y="28" '
            f'text-anchor="middle" font-size="18" font-weight="bold">'
            f'{panel_label}</text>'
        )

        for index, metric_name in enumerate(metric_names):
            center_x = matrix_x + index * cell_size + cell_size / 2
            center_y = top_margin + index * cell_size + cell_size / 2
            parts.append(
                f'<text x="{matrix_x - 10}" y="{center_y + 5:.1f}" '
                f'text-anchor="end" font-size="14">{metric_label_markup(metric_name)}</text>'
            )
            parts.append(
                f'<text x="{center_x:.1f}" y="{top_margin - 12}" '
                f'text-anchor="start" font-size="14" '
                f'transform="rotate(-45 {center_x:.1f} {top_margin - 12})">'
                f'{metric_label_markup(metric_name)}</text>'
            )

        for row_index, row in enumerate(analysis["rho"]):
            for column_index, rho in enumerate(row):
                x_position = matrix_x + column_index * cell_size
                y_position = top_margin + row_index * cell_size
                parts.append(
                    f'<rect x="{x_position}" y="{y_position}" width="{cell_size}" '
                    f'height="{cell_size}" fill="{rho_color(rho)}" '
                    f'stroke="#eeeeee"/>'
                )
                label = "NA" if rho is None else f"{rho:.2f}"
                text_color = (
                    "white"
                    if rho is not None and abs(rho) >= 0.65
                    else "#202020"
                )
                parts.append(
                    f'<text x="{x_position + cell_size / 2:.1f}" '
                    f'y="{y_position + cell_size / 2 + 5:.1f}" '
                    f'text-anchor="middle" font-size="13" font-weight="bold" '
                    f'fill="{text_color}">{xml_escape(label)}</text>'
                )

    colorbar_width = max(260, min(width - 200, matrix_size * 1.5))
    colorbar_x = (width - colorbar_width) / 2
    colorbar_y = top_margin + matrix_size + 36
    segments = 100
    for segment in range(segments):
        segment_rho = -1.0 + 2.0 * segment / (segments - 1)
        x_position = colorbar_x + colorbar_width * segment / segments
        parts.append(
            f'<rect x="{x_position:.2f}" y="{colorbar_y}" '
            f'width="{colorbar_width / segments + 0.5:.2f}" height="14" '
            f'fill="{rho_color(segment_rho)}"/>'
        )
    for value, label in ((0.0, "−1"), (0.5, "0"), (1.0, "+1")):
        x_position = colorbar_x + colorbar_width * value
        parts.append(
            f'<text x="{x_position:.1f}" y="{colorbar_y + 33}" '
            f'text-anchor="middle" font-size="13">{label}</text>'
        )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{colorbar_y + 55}" '
        f'text-anchor="middle" font-size="13">Spearman ρ</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts), width, height


EDGE_EXECUTABLES = (
    Path(
        "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    ),
    Path("/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"),
)


def find_edge_executable() -> Path:
    for candidate in EDGE_EXECUTABLES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Microsoft Edge was not found; it is required to render PNG/PDF heatmaps"
    )


def windows_path(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def wait_for_render(path: Path, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 1000:
            current_size = path.stat().st_size
            if current_size == previous_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return
            else:
                previous_size = current_size
                stable_checks = 0
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for rendered file: {path}")


def run_edge(command: list[str], expected_output: Path) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Heatmap rendering failed with exit code {result.returncode}:\n"
            f"{result.stdout}"
        )
    wait_for_render(expected_output)


def write_markup_images(
    output_dir: Path,
    stem: str,
    markup: str,
    width: int,
    height: int,
) -> tuple[Path, Path]:
    source_path = output_dir / f".{stem}.source.html"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    html = (
        "<!doctype html><meta charset=\"utf-8\">"
        f"<style>@page{{size:{width}px {height}px;margin:0}}"
        f"html,body{{width:{width}px;height:{height}px;"
        "margin:0;padding:0;background:white;overflow:hidden}}"
        "*{-webkit-print-color-adjust:exact;print-color-adjust:exact}"
        "svg{display:block}</style>"
        + markup
    )
    source_path.write_text(html, encoding="utf-8")

    for output_path in (pdf_path, png_path):
        if output_path.exists():
            output_path.unlink()

    edge = find_edge_executable()
    source_windows = windows_path(source_path)
    try:
        run_edge(
            [
                str(edge),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={windows_path(pdf_path)}",
                source_windows,
            ],
            pdf_path,
        )
        run_edge(
            [
                str(edge),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={width},{height}",
                f"--screenshot={windows_path(png_path)}",
                source_windows,
            ],
            png_path,
        )
    finally:
        source_path.unlink(missing_ok=True)
    return pdf_path, png_path


def write_heatmap_images(output_dir: Path, analysis: dict) -> tuple[Path, Path]:
    markup, width, height = build_heatmap_markup(analysis)
    prefix = f'{analysis["dimension"]}_' if "dimension" in analysis else ""
    return write_markup_images(
        output_dir,
        f'{prefix}spearman_heatmap_{analysis["scope"]}',
        markup,
        width,
        height,
    )


def write_condition_comparison_images(
    output_dir: Path,
    rag_analysis: dict,
    no_rag_analysis: dict,
) -> tuple[Path, Path]:
    if rag_analysis.get("dimension") != no_rag_analysis.get("dimension"):
        raise ValueError("RAG and NO RAG analyses must belong to the same dimension")
    prefix = (
        f'{rag_analysis["dimension"]}_'
        if "dimension" in rag_analysis
        else ""
    )
    markup, width, height = build_condition_comparison_markup(
        rag_analysis,
        no_rag_analysis,
    )
    return write_markup_images(
        output_dir,
        f"{prefix}spearman_heatmap_RAG_NO_RAG",
        markup,
        width,
        height,
    )


def serializable_analysis(analysis: dict) -> dict:
    return {
        **analysis,
        "num_pairwise_tests": len(analysis["pairs"]),
    }


def write_outputs(
    output_dir: Path,
    observations: Sequence[dict[str, object]],
    analyses: Sequence[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_pattern in ("p_values_*.csv", "spearman_heatmap_*.svg"):
        for stale_path in output_dir.glob(stale_pattern):
            stale_path.unlink()
    for stale_scope in ("RAG", "NO_RAG"):
        for extension in ("pdf", "png"):
            stale_path = output_dir / f"spearman_heatmap_{stale_scope}.{extension}"
            stale_path.unlink(missing_ok=True)
    write_observations_csv(output_dir / "observations.csv", observations)

    analyses_by_scope = {
        analysis["scope"]: analysis
        for analysis in analyses
    }
    dimension = analyses[0].get("dimension") if analyses else None
    if dimension is not None:
        for old_stem in (
            "spearman_heatmap_ALL",
            "spearman_heatmap_RAG_NO_RAG",
        ):
            for extension in ("pdf", "png"):
                (output_dir / f"{old_stem}.{extension}").unlink(missing_ok=True)
    for analysis in analyses:
        scope = analysis["scope"]
        metric_names = analysis["metrics"]
        write_matrix_csv(output_dir / f"spearman_rho_{scope}.csv", metric_names, analysis["rho"])
        write_matrix_csv(output_dir / f"sample_size_n_{scope}.csv", metric_names, analysis["n"])
        if scope == "ALL":
            write_heatmap_images(output_dir, analysis)

    if "RAG" not in analyses_by_scope or "NO_RAG" not in analyses_by_scope:
        raise ValueError("Both RAG and NO_RAG analyses are required")
    write_condition_comparison_images(
        output_dir,
        analyses_by_scope["RAG"],
        analyses_by_scope["NO_RAG"],
    )

    summary = {
        "method": "Spearman rank correlation",
        "unit_of_analysis": "newspaper x model x temperature x condition aggregated cell",
        "metric_definitions": [
            {
                "name": spec.name,
                "source": str(spec.path.relative_to(PROJECT_ROOT)),
                "field": spec.value_key,
            }
            for spec in METRICS
        ],
        "analyses": {
            analysis["scope"]: serializable_analysis(analysis)
            for analysis in analyses
        },
    }
    with (output_dir / "spearman_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False, allow_nan=False)


def print_summary(analyses: Sequence[dict], output_dir: Path) -> None:
    for analysis in analyses:
        print(
            f'{analysis["scope"]}: n={analysis["num_observations"]}, '
            f"metric pairs={len(analysis['pairs'])}"
        )
        for pair in analysis["pairs"][:5]:
            print(
                f'  {pair["metric_x"]} vs {pair["metric_y"]}: '
                f'rho={pair["rho"]:.3f}'
            )
    print(f"Results written to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    observations = build_observations()
    analyses = [analyze_scope(observations, scope) for scope in CONDITION_SCOPES]
    write_outputs(args.output_dir, observations, analyses)
    print_summary(analyses, args.output_dir)


if __name__ == "__main__":
    main()
