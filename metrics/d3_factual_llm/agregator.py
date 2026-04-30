import os
import pandas as pd
import json
from collections import defaultdict

INPUT_DIR = "./files/NO_RAG_news_csvs_tag"  # o RAG_news_csvs_tag
OUTPUT_JSON = "factual_llm_NO_RAG_results.json"

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
    "correct": 0,
    "partially_correct": 0,
    "incorrect": 0
})))

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file in files:
    path = os.path.join(INPUT_DIR, file)

    try:
        df = pd.read_csv(path, engine="python")
    except Exception as e:
        print(f"[ERROR] Could not read {file}: {e}")
        continue

    parts = file.replace(".csv", "").split("_")
    newspaper = "_".join(parts[:-3])
    id_llm = parts[-3]
    id_feature = parts[-2]

    if "decision" not in df.columns:
        print(f"[WARNING] Missing decision in {file}")
        print("Columns found:", df.columns.tolist())
        continue

    counts = df["decision"].value_counts()

    results[newspaper][id_llm][id_feature]["correct"] += int(counts.get("correct", 0))
    results[newspaper][id_llm][id_feature]["partially_correct"] += int(counts.get("partially_correct", 0))
    results[newspaper][id_llm][id_feature]["incorrect"] += int(counts.get("incorrect", 0))

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Saved results to {OUTPUT_JSON}")