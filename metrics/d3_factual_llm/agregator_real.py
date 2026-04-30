import os
import pandas as pd
import json
from collections import defaultdict

INPUT_DIR = "./files/REAL_news_csvs_tag"
OUTPUT_JSON = "factual_llm_REAL_results.json"

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
    "correct": 0,
    "partially_correct": 0,
    "incorrect": 0
})))


def read_csv_safe(path):
    """
    Robust CSV reader:
    - skips malformed lines
    - handles encoding issues
    - removes duplicated headers
    """
    try:
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",   # 🔥 clave
            encoding="utf-8"
        )
    except:
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            encoding="latin-1"
        )

    # eliminar filas donde decision = "decision" (header duplicado)
    df = df[df["decision"] != "decision"]

    return df


files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for file in files:
    path = os.path.join(INPUT_DIR, file)

    try:
        df = read_csv_safe(path)
    except Exception as e:
        print(f"[ERROR] Could not read {file}: {e}")
        continue

    # ===============================
    # EXTRAER PERIÓDICO
    # ===============================
    parts = file.replace(".csv", "").split("_")
    newspaper = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    # ===============================
    # VALIDACIÓN
    # ===============================
    if "decision" not in df.columns:
        print(f"[WARNING] Missing decision in {file}")
        continue

    counts = df["decision"].value_counts()

    llm_id = "REAL"
    feature_id = "1"

    results[newspaper][llm_id][feature_id]["correct"] += int(counts.get("correct", 0))
    results[newspaper][llm_id][feature_id]["partially_correct"] += int(counts.get("partially_correct", 0))
    results[newspaper][llm_id][feature_id]["incorrect"] += int(counts.get("incorrect", 0))

    print(f"{file} → {newspaper} ({len(df)} rows OK)")

# ===============================
# SAVE
# ===============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nSaved results to {OUTPUT_JSON}")