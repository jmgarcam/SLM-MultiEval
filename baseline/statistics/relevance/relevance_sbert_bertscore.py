import json
import os
import re
from pathlib import Path

import torch
from bert_score import score
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
INPUT_OUTPUT_MAP = {
    "NO_RAG": {
        "input": PROJECT_ROOT / "baselines" / "files" / "relevance" / "relevance_NO_RAG_by_news.json",
        "output": BASE_DIR / "relevance_NO_RAG_sbert_bertscore_by_news.json",
    },
    "RAG": {
        "input": PROJECT_ROOT / "baselines" / "files" / "relevance" / "relevance_RAG_by_news.json",
        "output": BASE_DIR / "relevance_RAG_sbert_bertscore_by_news.json",
    },
}
LOCAL_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
    "snapshots",
)
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_KEY = "synthetic_description"
REFERENCE_KEY = "headline"


def get_local_model_path():
    if not os.path.isdir(LOCAL_MODEL_CACHE_DIR):
        return None

    snapshots = [
        entry.path for entry in os.scandir(LOCAL_MODEL_CACHE_DIR)
        if entry.is_dir()
    ]
    snapshots.sort(key=os.path.getmtime, reverse=True)

    required_files = ["config.json", "modules.json", "sentence_bert_config.json"]
    for snapshot in snapshots:
        if all(os.path.exists(os.path.join(snapshot, name)) for name in required_files):
            return snapshot

    return None


print("Loading SBERT model...")
model_path = get_local_model_path() or MODEL_ID
print(f"Using model source: {model_path}")
model = SentenceTransformer(model_path)
print("SBERT model loaded successfully.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def remove_html(text):
    if not text:
        return ""
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("Leer", "")
    return re.sub(r"\s+", " ", text).strip()


def calculate_paired_sbert_scores(text_list_a, text_list_b):
    if not text_list_a or not text_list_b:
        return []

    emb_a = model.encode(text_list_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(text_list_b, convert_to_tensor=True, show_progress_bar=False)
    scores = torch.nn.functional.cosine_similarity(emb_a, emb_b)
    return [round(score_value.item(), 4) for score_value in scores]


def enrich_news_list(news_list):
    candidates = []
    references = []
    valid_indices = []

    for index, news in enumerate(news_list):
        candidate_text = remove_html(news.get(TEXT_KEY, ""))
        reference_text = remove_html(news.get(REFERENCE_KEY, ""))

        if candidate_text and reference_text:
            candidates.append(candidate_text)
            references.append(reference_text)
            valid_indices.append(index)
        else:
            news_list[index]["sbert_headline_raw"] = 0.0
            news_list[index]["bert_precision"] = 0.0
            news_list[index]["bert_recall"] = 0.0
            news_list[index]["bert_f1"] = 0.0

    if not candidates:
        return news_list

    sbert_scores = calculate_paired_sbert_scores(candidates, references)
    bert_precision, bert_recall, bert_f1 = score(candidates, references, lang="es", verbose=False)

    for idx, sbert_value, p_value, r_value, f1_value in zip(
        valid_indices,
        sbert_scores,
        bert_precision,
        bert_recall,
        bert_f1,
    ):
        news_list[idx]["sbert_headline_raw"] = sbert_value
        news_list[idx]["bert_precision"] = round(p_value.item(), 4)
        news_list[idx]["bert_recall"] = round(r_value.item(), 4)
        news_list[idx]["bert_f1"] = round(f1_value.item(), 4)

    return news_list


def enrich_nested_news(source_data, label):
    enriched_data = {}

    for newspaper, llm_dict in source_data.items():
        enriched_data[newspaper] = {}

        for llm_id, feature_dict in llm_dict.items():
            enriched_data[newspaper][str(llm_id)] = {}

            for feature_id, news_list in feature_dict.items():
                enriched_news = enrich_news_list([dict(news) for news in news_list])
                enriched_data[newspaper][str(llm_id)][str(feature_id)] = enriched_news
                print(
                    f"{label} | {newspaper} | LLM {llm_id} | Feature {feature_id} "
                    f"-> processed {len(enriched_news)} news"
                )

    return enriched_data


def save_json_with_fallback(output_path, data):
    try:
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=4, ensure_ascii=False)
        print(f"Saved results to {output_path}")
    except PermissionError:
        fallback_path = PROJECT_ROOT / output_path.name
        with open(fallback_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=4, ensure_ascii=False)
        print(
            f"[WARNING] Could not write to {output_path}. "
            f"Saved results to {fallback_path} instead."
        )


def main():
    for label, paths in INPUT_OUTPUT_MAP.items():
        source_data = load_json(paths["input"])
        enriched_data = enrich_nested_news(source_data, label)
        save_json_with_fallback(paths["output"], enriched_data)


if __name__ == "__main__":
    main()