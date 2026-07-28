"""Build per-news coherence/FH/tree-depth JSON files from scored CSVs.

The authoritative inputs are the ``*_scored.csv`` files under
``baselines/files/coherence/tagged``. The generated JSON files are the only
inputs consumed by ``coherence_spearman_analysis.py``.
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

import spacy
import textstat


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
BASELINES_DIR = PROJECT_ROOT / "baselines"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.readability.text_processing import clean_metric_text


INPUT_OUTPUT_MAP = {
    "NO_RAG": {
        "input": BASELINES_DIR / "files" / "coherence" / "tagged" / "NO_RAG_news_csvs",
        "output": BASE_DIR / "coherence_NO_RAG_fh_td_by_news.json",
    },
    "RAG": {
        "input": BASELINES_DIR / "files" / "coherence" / "tagged" / "RAG_news_csvs",
        "output": BASE_DIR / "coherence_RAG_fh_td_by_news.json",
    },
}
REQUIRED_COLUMNS = {"id_news", "headline", "synthetic_description", "coherence"}
SCORED_FILENAME_RE = re.compile(
    r"^(?P<newspaper>.+)_(?P<llm_id>\d+)_(?P<feature_id>\d+)_"
    r"(?P<partition>[ab])_scored\.csv$"
)
textstat.set_lang("es")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacy-model", default="es_core_news_sm",
        help="Installed Spanish spaCy model (default: es_core_news_sm)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="spaCy pipeline batch size (default: 64)",
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


def parse_coherence(value: object, context: str) -> int | float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid coherence value at {context}: {value!r}") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Non-finite coherence value at {context}: {value!r}")
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def subtree_depth(root) -> int:
    max_depth = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node.children)
    return max_depth


def calculate_tree_depth(doc) -> float:
    depths = [subtree_depth(sent.root) for sent in doc.sents if sent.root is not None]
    return float(sum(depths) / len(depths)) if depths else 0.0


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
                        "coherence": parse_coherence(row.get("coherence"), context),
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


def enrich_rows(records: list[dict], nlp, batch_size: int) -> None:
    metric_texts = [clean_metric_text(record["synthetic_description"]) for record in records]
    if any(not text for text in metric_texts):
        empty_index = next(index for index, text in enumerate(metric_texts) if not text)
        record = records[empty_index]
        raise ValueError(
            "Text is empty after cleaning at "
            f"{record['source_file']}:{record['source_row']}"
        )
    fh_values = [float(textstat.fernandez_huerta(text)) for text in metric_texts]
    docs = nlp.pipe(metric_texts, batch_size=batch_size)
    for record, metric_text, fh_value, doc in zip(
        records, metric_texts, fh_values, docs, strict=True
    ):
        tree_depth = calculate_tree_depth(doc)
        if not math.isfinite(fh_value) or not math.isfinite(tree_depth):
            raise ValueError(
                "Non-finite derived metric at "
                f"{record['source_file']}:{record['source_row']}"
            )
        record["metric_text"] = metric_text
        record["fh"] = fh_value
        record["tree_depth"] = tree_depth


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


def load_spacy_model(model_name: str):
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. "
            "Install the project dependencies before running this script."
        ) from exc
    nlp.max_length = 2_000_000
    return nlp


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    print(f"Loading spaCy model {args.spacy_model!r}...")
    nlp = load_spacy_model(args.spacy_model)
    for label, paths in INPUT_OUTPUT_MAP.items():
        records = load_scored_rows(paths["input"], label)
        enrich_rows(records, nlp, args.batch_size)
        write_json(paths["output"], nest_records(records))


if __name__ == "__main__":
    main()
