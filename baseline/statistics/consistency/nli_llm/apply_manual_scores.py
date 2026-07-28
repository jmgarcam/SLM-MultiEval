#!/usr/bin/env python3
"""Validate and apply manually reviewed NLI scores to tagged CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
TAGGED = BASE / "tagged"
SCORES = BASE / "manual_scores"
KINDS = ("NO_RAG_news_csvs", "RAG_news_csvs")
SCORE_FIELDS = ("entailment", "contradiction", "no_mention")
LEGACY_FIELDS = set(SCORE_FIELDS) | {"no_neutral"}


def load_and_validate(csv_path: Path, json_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])
    with json_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    scores = payload.get("scores", [])
    if payload.get("source_file") != csv_path.name:
        raise ValueError(f"source_file incorrecto: {json_path}")
    if len(rows) != len(scores):
        raise ValueError(f"número de filas distinto: {csv_path}")
    for index, (row, score) in enumerate(zip(rows, scores), 1):
        if row.get("id_news") != score.get("id_news"):
            raise ValueError(f"id_news distinto: {csv_path}:{index}")
        for field in SCORE_FIELDS:
            value = score.get(field)
            if type(value) is not int or not 0 <= value <= 3:
                raise ValueError(f"valor inválido {field}: {json_path}:{index}")
    fields = [field for field in original_fields if field not in LEGACY_FIELDS]
    return rows, scores, fields + list(SCORE_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="reescribir los CSV validados")
    args = parser.parse_args()
    jobs = []
    total_rows = 0
    for kind in KINDS:
        csv_paths = sorted((TAGGED / kind).glob("*.csv"))
        json_paths = sorted((SCORES / kind).glob("*.json"))
        if len(csv_paths) != 36 or len(json_paths) != 36:
            raise ValueError(f"se esperaban 36 CSV y 36 JSON en {kind}")
        for csv_path in csv_paths:
            json_path = SCORES / kind / f"{csv_path.stem}.json"
            rows, scores, fields = load_and_validate(csv_path, json_path)
            jobs.append((csv_path, rows, scores, fields))
            total_rows += len(rows)
    if total_rows != 1798:
        raise ValueError(f"se esperaban 1798 filas, hay {total_rows}")
    if args.apply:
        for csv_path, rows, scores, fields in jobs:
            temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
            with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row, score in zip(rows, scores):
                    output = {field: row.get(field, "") for field in fields}
                    for field in SCORE_FIELDS:
                        output[field] = score[field]
                    writer.writerow(output)
            os.replace(temp_path, csv_path)
    action = "aplicadas" if args.apply else "validadas"
    print(f"72 archivos y {total_rows} filas {action} correctamente")


if __name__ == "__main__":
    main()
