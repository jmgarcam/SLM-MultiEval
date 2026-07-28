"""Build per-news fluency/MTLD/FH JSON files from scored CSVs.

The authoritative inputs are the ``*_scored.csv`` files under
``baselines/files/fluency/tagged``. The generated JSON files are the only
inputs consumed by ``fluency_spearman_analysis.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lexicalrichness import LexicalRichness
import textstat


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
BASELINES_DIR = PROJECT_ROOT / "baselines"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.readability.text_processing import clean_metric_text


INPUT_OUTPUT_MAP = {
    "NO_RAG": {
        "input": BASELINES_DIR / "files" / "fluency" / "tagged" / "NO_RAG_news_csvs",
        "output": BASE_DIR / "fluency_NO_RAG_mtld_fh_by_news.json",
    },
    "RAG": {
        "input": BASELINES_DIR / "files" / "fluency" / "tagged" / "RAG_news_csvs",
        "output": BASE_DIR / "fluency_RAG_mtld_fh_by_news.json",
    },
}
REQUIRED_COLUMNS = {"id_news", "headline", "synthetic_description", "fluency"}
SCORED_FILENAME_RE = re.compile(
    r"^(?P<newspaper>.+)_(?P<llm_id>\d+)_(?P<feature_id>\d+)_"
    r"(?P<partition>[ab])_scored\.csv$"
)
MTLD_THRESHOLD = 0.72
textstat.set_lang("es")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mtld-threshold", type=float, default=MTLD_THRESHOLD,
        help=f"LexicalRichness MTLD threshold (default: {MTLD_THRESHOLD})",
    )
    return parser.parse_args()


def parse_file_identity(path: Path) -> tuple[str, str, str, str]:
    match = SCORED_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected scored CSV filename: {path.name}")
    return (
        match.group("newspaper"), match.group("llm_id"),
        match.group("feature_id"), match.group("partition"),
    )


def parse_fluency(value: object, context: str) -> int | float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid fluency value at {context}: {value!r}") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Non-finite fluency value at {context}: {value!r}")
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def clean_mtld_text(value: object) -> str:
    """Apply the MTLD preprocessing used by metrics/mtld."""

    if value is None:
        return ""
    text = str(value).replace("&nbsp;", " ")
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("Leer", "")
    return re.sub(r"\s+", " ", text).strip()


def calculate_mtld(clean_text: str, threshold: float) -> float:
    if not clean_text or len(clean_text.split()) <= 10:
        return 0.0
    try:
        value = LexicalRichness(clean_text).mtld(threshold=threshold)
    except (ValueError, ZeroDivisionError):
        return 0.0
    numeric_value = round(float(value), 2)
    if not math.isfinite(numeric_value):
        raise ValueError("LexicalRichness returned a non-finite MTLD value")
    return numeric_value


def calculate_fh(clean_text: str, context: str) -> float:
    if not clean_text:
        raise ValueError(f"Text is empty after FH cleaning at {context}")
    value = float(textstat.fernandez_huerta(clean_text))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite FH value at {context}")
    return value


def load_scored_rows(input_dir: Path, label: str) -> list[dict]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"{label}: input directory not found: {input_dir}")
    csv_paths = sorted(input_dir.glob("*_scored.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"{label}: no *_scored.csv files in {input_dir}")

    records: list[dict] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    partitions_by_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for csv_path in csv_paths:
        newspaper, llm_id, feature_id, partition = parse_file_identity(csv_path)
        partitions_by_group[(newspaper, llm_id, feature_id)].add(partition)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            if missing_columns:
                raise ValueError(f"{csv_path}: missing required columns {missing_columns}")
            for source_row, row in enumerate(reader, start=2):
                context = f"{csv_path.name}:{source_row}"
                news_id = str(row.get("id_news", "")).strip()
                description = str(row.get("synthetic_description", "")).strip()
                if not news_id:
                    raise ValueError(f"{context}: empty id_news")
                if not description:
                    raise ValueError(f"{context}: empty synthetic_description")
                key = (newspaper, llm_id, feature_id, news_id)
                if key in seen_keys:
                    raise ValueError(f"{context}: duplicate observation key {key}")
                seen_keys.add(key)
                records.append(
                    {
                        "newspaper": newspaper,
                        "id_llm": llm_id,
                        "id_feature": feature_id,
                        "id_news": news_id,
                        "headline": str(row.get("headline", "")).strip(),
                        "synthetic_description": description,
                        "fluency": parse_fluency(row.get("fluency"), context),
                        "source_file": csv_path.name,
                        "source_row": source_row,
                    }
                )

    for group, partitions in sorted(partitions_by_group.items()):
        if partitions != {"a", "b"}:
            print(
                f"[WARNING] {label}: {group} has partitions "
                f"{sorted(partitions)}; expected ['a', 'b']"
            )
    description_counts = Counter(record["synthetic_description"] for record in records)
    duplicated_rows = sum(count for count in description_counts.values() if count > 1)
    if duplicated_rows:
        print(
            f"[WARNING] {label}: {duplicated_rows} rows belong to "
            f"{sum(count > 1 for count in description_counts.values())} "
            "duplicated synthetic descriptions; rows are retained."
        )
    print(f"{label}: loaded {len(records)} rows from {len(csv_paths)} scored CSVs")
    return records


def enrich_rows(records: list[dict], threshold: float) -> None:
    for record in records:
        context = f"{record['source_file']}:{record['source_row']}"
        mtld_text = clean_mtld_text(record["synthetic_description"])
        fh_text = clean_metric_text(record["synthetic_description"])
        record["mtld_metric_text"] = mtld_text
        record["fh_metric_text"] = fh_text
        record["mtld"] = calculate_mtld(mtld_text, threshold)
        record["fh"] = calculate_fh(fh_text, context)


def nest_records(records: list[dict]) -> dict:
    nested: dict[str, dict[str, dict[str, list[dict]]]] = {}
    sorted_records = sorted(
        records,
        key=lambda record: (
            record["newspaper"], int(record["id_llm"]), int(record["id_feature"]),
            record["source_file"], record["source_row"],
        ),
    )
    for record in sorted_records:
        newspaper = record.pop("newspaper")
        llm_id = record["id_llm"]
        feature_id = record["id_feature"]
        nested.setdefault(newspaper, {}).setdefault(llm_id, {}).setdefault(
            feature_id, []
        ).append(record)
    return nested


def write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    temporary_path.replace(path)
    print(f"Saved intermediate results to {path}")


def main() -> None:
    args = parse_args()
    if not 0 < args.mtld_threshold < 1:
        raise ValueError("--mtld-threshold must be between 0 and 1")
    for label, paths in INPUT_OUTPUT_MAP.items():
        records = load_scored_rows(paths["input"], label)
        enrich_rows(records, args.mtld_threshold)
        write_json(paths["output"], nest_records(records))


if __name__ == "__main__":
    main()
