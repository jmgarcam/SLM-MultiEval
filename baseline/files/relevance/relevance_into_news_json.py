import json
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
METRIC_NAME = "relevance"
def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
def parse_filename(filename):
    parts = filename.replace(".csv", "").split("_")
    newspaper = "_".join(parts[:-4])
    id_llm = parts[-4]
    id_feature = parts[-3]
    return newspaper, id_llm, id_feature
def parse_metric_value(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value
def build_score_map(tagged_dir):
    score_map = {}
    files = sorted(
        path for path in tagged_dir.iterdir()
        if path.name.endswith("_scored.csv")
    )
    for csv_path in files:
        dataframe = pd.read_csv(csv_path, engine="python")
        newspaper, id_llm, id_feature = parse_filename(csv_path.name)
        score_map.setdefault(newspaper, {}).setdefault(id_llm, {}).setdefault(
            id_feature, {}
        )
        for _, row in dataframe.iterrows():
            id_news = str(row.get("id_news", "")).strip()
            if not id_news:
                continue
            metric_value = parse_metric_value(row.get(METRIC_NAME))
            if metric_value is None:
                continue
            score_map[newspaper][id_llm][id_feature][id_news] = metric_value
    return score_map
def merge_scores_into_source(source_data, score_map):
    merged_data = {}
    for newspaper, llm_dict in source_data.items():
        merged_data[newspaper] = {}
        for id_llm, feature_dict in llm_dict.items():
            llm_id_str = str(id_llm)
            merged_data[newspaper][llm_id_str] = {}
            for id_feature, news_list in feature_dict.items():
                feature_id_str = str(id_feature)
                scores = (
                    score_map.get(newspaper, {})
                    .get(llm_id_str, {})
                    .get(feature_id_str, {})
                )
                enriched_news = []
                matched = 0
                for news_item in news_list:
                    enriched_item = dict(news_item)
                    news_id = str(news_item.get("id_news", "")).strip()
                    metric_value = scores.get(news_id)
                    if metric_value is not None:
                        matched += 1
                    enriched_item[METRIC_NAME] = metric_value
                    enriched_news.append(enriched_item)
                merged_data[newspaper][llm_id_str][feature_id_str] = enriched_news
                print(
                    f"{newspaper} | LLM {llm_id_str} | Feature {feature_id_str} "
                    f"-> matched {matched}/{len(news_list)}"
                )
                if matched != len(news_list):
                    print(
                        f"[WARNING] Missing {METRIC_NAME} rows for "
                        f"{newspaper} / {llm_id_str} / {feature_id_str}"
                    )
    return merged_data
def aggregate_tagged_csv_directory(input_dir, source_json, output_json):
    tagged_dir = BASE_DIR / input_dir
    source_json_path = BASE_DIR / source_json
    output_json_path = BASE_DIR / output_json
    source_data = load_json(source_json_path)
    score_map = build_score_map(tagged_dir)
    merged_data = merge_scores_into_source(source_data, score_map)
    try:
        with open(output_json_path, "w", encoding="utf-8") as file_obj:
            json.dump(merged_data, file_obj, indent=4, ensure_ascii=False)
        print(f"\nSaved results to {output_json_path}")
    except PermissionError:
        fallback_output_path = PROJECT_ROOT / output_json_path.name
        with open(fallback_output_path, "w", encoding="utf-8") as file_obj:
            json.dump(merged_data, file_obj, indent=4, ensure_ascii=False)
        print(
            f"\n[WARNING] Could not write to {output_json_path}. "
            f"Saved results to {fallback_output_path} instead."
        )
if __name__ == "__main__":
    raise SystemExit(
        "Use relevance_no_rag_into_json.py or relevance_rag_into_json.py."
    )