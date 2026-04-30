import json
import os
import csv
import re
import random

# =======================
# CONFIG
# =======================
INPUT_JSON = "../metrics/selected_news_for_llm/real_10_percent.json"
OUTPUT_DIR = "./files/relevance/real_news_csvs"
RANDOM_STATE = 42


def sanitize_filename(text):
    text = str(text).strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", text)


def clean_html(text):
    """Remove HTML tags from description."""
    if not text:
        return ""
    return re.sub(r"<.*?>", "", text)


def write_csv(path, data_subset):
    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "id_news",
            "headline",
            "synthetic_description",
            "relevance"
        ])

        for news_item in data_subset:
            writer.writerow([
                news_item.get("_id", ""),
                news_item.get("headline", ""),
                clean_html(news_item.get("description", "")),
                ""
            ])


def export_csvs(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(RANDOM_STATE)

    if not isinstance(data, dict):
        raise ValueError("The JSON root must be a dictionary.")

    total_files = 0

    for newspaper, news_list in data.items():

        if not isinstance(news_list, list):
            print(f"[WARNING] Skipping {newspaper}, not a list.")
            continue

        n_news = len(news_list)

        if n_news == 0:
            print(f"[WARNING] Empty list for {newspaper}. Skipping.")
            continue

        # sample up to 50
        sample_size = min(50, n_news)
        sampled_news = rng.sample(news_list, sample_size)

        # split
        split_point = (sample_size + 1) // 2
        sample_a = sampled_news[:split_point]
        sample_b = sampled_news[split_point:]

        safe_newspaper = sanitize_filename(newspaper)

        filename_a = f"{safe_newspaper}_a.csv"
        filename_b = f"{safe_newspaper}_b.csv"

        path_a = os.path.join(output_dir, filename_a)
        path_b = os.path.join(output_dir, filename_b)

        write_csv(path_a, sample_a)
        write_csv(path_b, sample_b)

        total_files += 2

        print(
            f"{newspaper}: {len(sample_a)} rows (A), {len(sample_b)} rows (B)"
        )

    print(f"\nDone. Total CSV files generated: {total_files}")


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"File not found: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    export_csvs(data, OUTPUT_DIR)


if __name__ == "__main__":
    main()