#!/usr/bin/env python3
"""Combine consistency, NLI and factual evaluations into one JSON per condition."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

CONFIG = {
    "NO_RAG": {
        "news": PROJECT_ROOT
        / "metrics/selected_news_for_llm/sinteticas_NORAG_10_percent.json",
        "nli": BASE_DIR / "nli_llm/tagged/NO_RAG_news_csvs",
        "factual": BASE_DIR / "factual_llm/files/NO_RAG_news_csvs_tag",
        "consistency": PROJECT_ROOT
        / "baselines/files/consistency/tagged/NO_RAG_news_csvs",
        "output": BASE_DIR / "consistency_NO_RAG_by_news.json",
    },
    "RAG": {
        "news": PROJECT_ROOT
        / "metrics/selected_news_for_llm/sinteticas_RAG_10_percent.json",
        "nli": BASE_DIR / "nli_llm/tagged/RAG_news_csvs",
        "factual": BASE_DIR / "factual_llm/files/RAG_news_csvs_tag",
        "consistency": PROJECT_ROOT
        / "baselines/files/consistency/tagged/RAG_news_csvs",
        "output": BASE_DIR / "consistency_RAG_by_news.json",
    },
}

NLI_FIELDS = ("entailment", "contradiction", "no_mention")
FACTUAL_DECISIONS = {"correct", "partially_correct", "incorrect"}
Key = tuple[str, str, str, str]


def parse_filename(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    if stem.endswith("_scored"):
        stem = stem[: -len("_scored")]

    try:
        prefix, _batch = stem.rsplit("_", 1)
        newspaper, id_llm, id_feature = prefix.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Nombre de CSV no reconocido: {path.name}") from exc

    if not newspaper or not id_llm or not id_feature:
        raise ValueError(f"Nombre de CSV incompleto: {path.name}")
    return newspaper, id_llm, id_feature


def load_csv_index(
    directory: Path, required_fields: tuple[str, ...]
) -> dict[Key, dict[str, str]]:
    paths = sorted(directory.glob("*.csv"))
    if len(paths) != 36:
        raise ValueError(f"Se esperaban 36 CSV en {directory}, hay {len(paths)}")

    index: dict[Key, dict[str, str]] = {}
    for path in paths:
        newspaper, id_llm, id_feature = parse_filename(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing_fields = set(required_fields) - fields
            if missing_fields:
                raise ValueError(
                    f"Faltan columnas {sorted(missing_fields)} en {path}"
                )

            for row_number, row in enumerate(reader, 2):
                id_news = str(row.get("id_news", "")).strip()
                if not id_news:
                    raise ValueError(f"id_news vacío en {path}:{row_number}")
                key = (newspaper, id_llm, id_feature, id_news)
                if key in index:
                    raise ValueError(
                        "Muestra duplicada para periódico/LLM/feature/id_news: "
                        f"{key} ({path}:{row_number})"
                    )
                row["_source_file"] = path.name
                index[key] = row

    return index


def source_keys(data: dict[str, Any]) -> set[Key]:
    keys: set[Key] = set()
    for newspaper, llm_groups in data.items():
        for id_llm, feature_groups in llm_groups.items():
            for id_feature, news_items in feature_groups.items():
                for news in news_items:
                    id_news = str(news.get("id_news", "")).strip()
                    key = (newspaper, str(id_llm), str(id_feature), id_news)
                    if not id_news:
                        raise ValueError(f"id_news vacío en el JSON fuente: {key}")
                    if key in keys:
                        raise ValueError(f"Muestra duplicada en el JSON fuente: {key}")
                    keys.add(key)
    return keys


def parse_nli(row: dict[str, str], key: Key) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in NLI_FIELDS:
        raw_value = str(row.get(field, "")).strip()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field} no entero para {key}: {raw_value!r}") from exc
        if not 0 <= value <= 3:
            raise ValueError(f"{field} fuera de [0, 3] para {key}: {value}")
        result[field] = value
    return result


def parse_factual(row: dict[str, str], key: Key) -> dict[str, str]:
    decision = str(row.get("decision", "")).strip()
    if decision not in FACTUAL_DECISIONS:
        raise ValueError(f"Decisión factual inválida para {key}: {decision!r}")
    return {"decision": decision}


def parse_consistency(row: dict[str, str], key: Key) -> int:
    raw_value = str(row.get("consistency", "")).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"consistency no entero para {key}: {raw_value!r}"
        ) from exc
    if not 1 <= value <= 5:
        raise ValueError(f"consistency fuera de [1, 5] para {key}: {value}")
    return value


def normalized_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_same_keys(
    condition: str,
    expected: set[Key],
    nli_keys: set[Key],
    factual_keys: set[Key],
    consistency_keys: set[Key],
) -> None:
    for label, actual in (
        ("nli_llm", nli_keys),
        ("factual_llm", factual_keys),
        ("consistency", consistency_keys),
    ):
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            examples = {
                "missing": sorted(missing)[:3],
                "extra": sorted(extra)[:3],
            }
            raise ValueError(
                f"Claves distintas en {condition}/{label}: "
                f"missing={len(missing)}, extra={len(extra)}, ejemplos={examples}"
            )


def enrich_news(
    source: dict[str, Any],
    nli_index: dict[Key, dict[str, str]],
    factual_index: dict[Key, dict[str, str]],
    consistency_index: dict[Key, dict[str, str]],
) -> tuple[dict[str, Any], int, int]:
    enriched: dict[str, Any] = {}
    total = 0
    factual_description_matches = 0

    for newspaper, llm_groups in source.items():
        enriched[newspaper] = {}
        for id_llm, feature_groups in llm_groups.items():
            llm_key = str(id_llm)
            enriched[newspaper][llm_key] = {}
            for id_feature, news_items in feature_groups.items():
                feature_key = str(id_feature)
                output_items = []
                for news in news_items:
                    item = dict(news)
                    id_news = str(item["id_news"]).strip()
                    key = (newspaper, llm_key, feature_key, id_news)
                    nli_row = nli_index[key]
                    factual_row = factual_index[key]
                    consistency_row = consistency_index[key]

                    if str(item.get("id_llm", llm_key)) != llm_key:
                        raise ValueError(f"id_llm interno distinto para {key}")
                    if str(item.get("id_feature", feature_key)) != feature_key:
                        raise ValueError(f"id_feature interno distinto para {key}")

                    source_description = normalized_text(
                        item.get("synthetic_description", "")
                    )
                    nli_description = normalized_text(
                        nli_row.get("synthetic_description", "")
                    )
                    if source_description != nli_description:
                        raise ValueError(
                            "La descripción de NLI no coincide con el JSON fuente "
                            f"para {key} ({nli_row['_source_file']})"
                        )

                    consistency_description = normalized_text(
                        consistency_row.get("synthetic_description", "")
                    )
                    if source_description != consistency_description:
                        raise ValueError(
                            "La descripción de consistency no coincide con el JSON "
                            f"fuente para {key} "
                            f"({consistency_row['_source_file']})"
                        )

                    factual_description = normalized_text(
                        factual_row.get("synthetic_description", "")
                    )
                    if source_description == factual_description:
                        factual_description_matches += 1

                    item["nli_llm"] = parse_nli(nli_row, key)
                    item["factual_llm"] = parse_factual(factual_row, key)
                    item["consistency"] = parse_consistency(consistency_row, key)
                    output_items.append(item)
                    total += 1

                enriched[newspaper][llm_key][feature_key] = output_items

    return enriched, total, factual_description_matches


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def build_condition(condition: str, paths: dict[str, Path]) -> int:
    with paths["news"].open("r", encoding="utf-8-sig") as handle:
        source = json.load(handle)

    nli_index = load_csv_index(
        paths["nli"],
        ("id_news", "headline", "synthetic_description", *NLI_FIELDS),
    )
    factual_index = load_csv_index(
        paths["factual"], ("id_news", "synthetic_description", "decision")
    )
    consistency_index = load_csv_index(
        paths["consistency"],
        ("id_news", "headline", "synthetic_description", "consistency"),
    )
    expected = source_keys(source)
    validate_same_keys(
        condition,
        expected,
        set(nli_index),
        set(factual_index),
        set(consistency_index),
    )

    enriched, total, factual_description_matches = enrich_news(
        source, nli_index, factual_index, consistency_index
    )
    if total != 899:
        raise ValueError(f"Se esperaban 899 noticias en {condition}, hay {total}")

    write_json(paths["output"], enriched)
    print(
        f"{condition}: {total} noticias -> {paths['output'].name}; "
        "descripciones factuales idénticas al original: "
        f"{factual_description_matches}/{total}"
    )
    return total


def main() -> None:
    total = sum(build_condition(condition, paths) for condition, paths in CONFIG.items())
    if total != 1798:
        raise ValueError(f"Se esperaban 1798 noticias en total, hay {total}")
    print(
        f"Validación completa: {total} noticias con consistency, "
        "nli_llm y factual_llm"
    )


if __name__ == "__main__":
    main()
