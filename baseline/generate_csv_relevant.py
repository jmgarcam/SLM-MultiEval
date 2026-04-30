import json
import os
import csv
import re
import random

# =======================
# CONFIG
# =======================
INPUT_JSON = "../metrics/selected_news_for_llm/sinteticas_RAG_10_percent.json"
OUTPUT_DIR = "./files/relevance/RAG_news_csvs"
RANDOM_STATE = 42


def sanitize_filename(text):
    """Make strings safe to use as filenames."""
    text = str(text).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", text)


def write_csv(path, data_subset):
    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Cabecera correcta (4 columnas)
        writer.writerow([
            "id_news",
            "headline",
            "synthetic_description",
            "relevance"
        ])

        for news_item in data_subset:
            writer.writerow([
                news_item.get("id_news", ""),
                news_item.get("headline", ""),  # <-- headline incluido
                news_item.get("synthetic_description", ""),
                ""  # relevance vacío para rellenar luego
            ])


def export_csvs_by_feature(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(RANDOM_STATE)

    if not isinstance(data, dict):
        raise ValueError("The JSON root must be a dictionary.")

    print("Newspapers found:", list(data.keys()))

    total_files = 0

    for newspaper, llm_dict in data.items():
        if not isinstance(llm_dict, dict):
            print(f"[WARNING] Skipping newspaper '{newspaper}' because its value is not a dictionary.")
            continue

        print(f"\nProcessing newspaper: {newspaper}")

        for id_llm, feature_dict in llm_dict.items():
            if not isinstance(feature_dict, dict):
                print(f"[WARNING] Skipping model '{id_llm}' in '{newspaper}' because its value is not a dictionary.")
                continue

            for id_feature, news_list in feature_dict.items():
                if not isinstance(news_list, list):
                    print(f"[WARNING] Skipping feature '{id_feature}' in '{newspaper}/{id_llm}' because its value is not a list.")
                    continue

                n_news = len(news_list)

                if n_news == 0:
                    print(f"[WARNING] Empty list for {newspaper}/{id_llm}/{id_feature}. Skipping.")
                    continue

                # Take up to 50 rows
                sample_size = min(50, n_news)

                # Random sample without replacement
                sampled_news = rng.sample(news_list, sample_size)

                # Split into two halves
                split_point = (sample_size + 1) // 2
                sample_a = sampled_news[:split_point]
                sample_b = sampled_news[split_point:]

                safe_newspaper = sanitize_filename(newspaper)

                filename_a = f"{safe_newspaper}_{id_llm}_{id_feature}_a.csv"
                filename_b = f"{safe_newspaper}_{id_llm}_{id_feature}_b.csv"

                path_a = os.path.join(output_dir, filename_a)
                path_b = os.path.join(output_dir, filename_b)

                write_csv(path_a, sample_a)
                write_csv(path_b, sample_b)

                total_files += 2

                print(
                    f"Generated: {path_a} ({len(sample_a)} rows) | "
                    f"Generated: {path_b} ({len(sample_b)} rows)"
                )

    print(f"\nDone. Total CSV files generated: {total_files}")


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"File not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    export_csvs_by_feature(data, OUTPUT_DIR)


if __name__ == "__main__":
    main()