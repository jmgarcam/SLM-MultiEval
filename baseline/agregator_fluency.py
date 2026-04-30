import os
import pandas as pd
import json
from collections import defaultdict

INPUT_DIR = "./files/fluency/tagged/NO_RAG_news_csvs"
OUTPUT_JSON = "fluency_NO_RAG_results.json"

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
    "avg_fluency": 0,
    "samples": 0,
    "dist": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
})))

files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_scored.csv")]

for file in files:
    path = os.path.join(INPUT_DIR, file)

    try:
        df = pd.read_csv(path, engine="python")
    except Exception as e:
        print(f"[ERROR] Could not read {file}: {e}")
        continue

    parts = file.replace(".csv", "").split("_")

    newspaper = "_".join(parts[:-4])
    id_llm = parts[-4]
    id_feature = parts[-3]

    if "fluency" not in df.columns:
        print(f"[WARNING] Missing fluency in {file}")
        continue

    values = pd.to_numeric(df["fluency"], errors="coerce").dropna()

    # acumuladores
    results[newspaper][id_llm][id_feature]["avg_fluency"] += values.sum()
    results[newspaper][id_llm][id_feature]["samples"] += len(values)

    # distribución 1–5
    counts = values.value_counts()

    for score in range(1, 6):
        results[newspaper][id_llm][id_feature]["dist"][score] += int(counts.get(score, 0))

# calcular media final
for newspaper in results:
    for id_llm in results[newspaper]:
        for id_feature in results[newspaper][id_llm]:
            data = results[newspaper][id_llm][id_feature]

            total = data["avg_fluency"]
            count = data["samples"]

            avg = total / count if count > 0 else 0
            data["avg_fluency"] = round(avg, 3)

# guardar JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Saved results to {OUTPUT_JSON}")