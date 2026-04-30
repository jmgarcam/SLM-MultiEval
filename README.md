# SLM-MultiEval: A Multidimensional Evaluation Framework for Small Language Models


SLM-MultiEval is a multidimensional evaluation methodology designed to analyze the behavior of small language models (SLMs) beyond aggregate performance metrics. Instead of treating text quality as a single score, the framework disentangles three complementary dimensions: syntactic–lexical complexity, informational competence, and consistency.  

By evaluating model outputs across these dimensions and comparing them to human-written references, SLM-MultiEval enables a fine-grained diagnosis of model behavior, particularly under different generation paradigms such as Retrieval-Augmented Generation (RAG) and parametric-only generation (NO-RAG).

---

## Experimental Design

SLM-MultiEval evaluates models under controlled conditions to isolate different factors:

- **Models**: Ministral-3B, Qwen-2.5-0.5B, Llama-3.2-1B  
- **Temperatures**: 0.5, 0.75, 1.0  
- **Generation modes**:
  - **NO-RAG**: parametric generation only  
  - **RAG**: external context injected at inference time  

---

## Metrics (/metrics)

All metrics are organized by evaluation dimension (d1–d3) and implemented in dedicated modules following the structure:

---

### D1 — Syntactic–Lexical Complexity

Located in:

- `d1_tree_depth/`
  - `tree_depth.py`  
    → Measures syntactic complexity via dependency tree depth.

- `d1_fh/`
  - `fh.py`  
    → Computes the Fernández-Huerta readability index (Spanish-specific readability).

- `d1_mtld/`
  - `mtld.py`  
    → Measures lexical diversity using MTLD.

---

### D2 — Informational Competence

Located in:

- `d2_bert/`
  - `bert.py`  
    → Token-level semantic alignment between generated and reference text.

- `d2_sbert/`
  - `sbert.py`  
    → Sentence-level semantic similarity (global alignment).

- `d2_uer/`
  - `uer.py`  
    → Entity density (information concentration relative to text length).

- `d2_ete/`
  - `ete.py`  
    → Entity type entropy (categorical diversity of information).

---

### D3 — Consistency

Located in:

- `d3_nli_llm/`
  - `files/`  
    → Results from NLI-based evaluation (REAL, NO-RAG, and RAG descriptions)
  - `nli_average_calculation.py`  
    → Aggregates the source consistency results via NLI based on LLM (entailment, neutral, contradiction).

- `d3_factual_llm/`
  - `files/`  
    → Results from LLM as judge (Real, NO RAG and RAG descriptions)
  - `agregator.py`  
    → Agreggate the factual results from LLM as judge.
  

Note: LLM-based evaluation relies on prompt-based scoring. Prompts and configurations are provided to ensure reproducibility.

---
## Baselines (/baselines)

The baselines folder contains baseline evaluations for text quality metrics using G-Eval methodology.

- `agregator_coherence.py` 
  → Aggregates coherence scores.
- `agregator_consistency.py` 
  → Aggregates consistency scores.
- `agregator_fluency.py` 
  → Aggregates fluency scores.
- `agregator_relevance.py` 
  → Aggregates relevance scores.
- `generate_csv_coherence.py` 
  → Generates CSV for coherence.
- `generate_csv_consistency.py` 
  → Generates CSV for consistency.
- `generate_csv_fluency.py` 
  → Generates CSV for fluency.
- `generate_csv_relevance.py` 
  → Generates CSV for relevance.

Results are stored in JSON files such as `coherence_NO_RAG_results.json`, `consistency_RAG_results.json`, etc.

---
### Output and Visualization

Each metric module generates its results as a JSON file stored within the same directory as the corresponding metric:

- `dimension_metric/{metric}_results.json`

For the visualization:

- `dimension_metric/print_metric_results.py`
→ Reads the generated JSON file and produces plots or summaries for direct inspection without requiring re-computation.

- `radar/`  
  → Represents aggregated results per dimension based on the corresponding metric results.

  - `print_D1.py` 
  - `print_D2.py` 
  - `print_D3.py`   


---

## Data

reales.json  
→ Set of real news selected for the experiment.

sinteticas_NORAG.json  
→ Set of generated news based on NO-RAG configuration.

sinteticas_RAG.json  
→ Set of generated news based on RAG configuration.

---

## Pipeline

1. Generate synthetic texts (SLMs, multiple configs) based on: [controlled-news-generation-es](https://github.com/jmgarcam/controlled-news-generation-es)
2. Compute metrics independently per dimension
3. Aggregate results across:
   - models
   - temperatures
   - generation modes
4. Analyze trade-offs across dimensions

---

## Usage

```bash
pip install -r requirements.txt

# Example for d1 fh
python fh.py
>> fh_results.json
python print_fh_results.py

# Example of radar graph (radar folder)
# json file results are required for that.
python print_D1.py

```

---

## Project Structure

```  
  SLM-MultiEval/
  ├── README.md
  ├── requirements.txt
  ├── baselines/
  │   ├── agregator_coherence.py
  │   ├── agregator_consistency.py
  │   ├── agregator_fluency.py
  │   ├── agregator_relevance.py
  │   ├── generate_csv_coherence.py
  │   ├── generate_csv_consistency.py
  │   ├── generate_csv_fluency.py
  │   ├── generate_csv_relevance.py
  │   ├── prompts/
  ├── .gitignore
  ├── metrics/
  │   ├── reales.json
  │   ├── sinteticas_NORAG.json
  │   ├── sinteticas_RAG.json
  │   ├── d1_fh/
  │   │   ├── fh.py
  │   │   └── print_fh_results.py
  │   ├── d1_mtld/
  │   │   ├── mtld.py
  │   │   └── print_mtld_results.py
  │   ├── d1_tree_depth/
  │   │   ├── tree_depth.py
  │   │   └── print_tree_depth_results.py
  │   ├── d2_bert/
  │   │   ├── bert.py
  │   │   └── print_bert_results.py
  │   ├── d2_ete/
  │   │   ├── ete.py
  │   │   └── print_ete_results.py
  │   ├── d2_sbert/
  │   │   ├── sbert.py
  │   │   └── print_sbert_results.py
  │   ├── d2_uer/
  │   │   ├── uer.py
  │   │   └── print_uer_results.py
  │   ├── d3_factual_llm/
  │   │   ├── agregator_real.py
  │   │   ├── agregator.py
  │   │   ├── print_factual_llm_results.py
  │   │   ├── prompt.txt
  │   │   └── files/
  │   │       ├── NO_RAG_news_csvs_tag/
  │   │       │   └── [CSV files for NO-RAG]
  │   │       ├── RAG_news_csvs_tag/
  │   │       │   └── [CSV files for RAG]
  │   │       └── REAL_news_csvs_tag/
  │   │           └── [CSV files for REAL]
  │   ├── d3_nli_llm/
  │   │   ├── evaluation_no_rag_grouped.json
  │   │   ├── evaluation_rag_grouped.json
  │   │   ├── evaluation_real_description_grouped.json
  │   │   ├── nli_average_calculation.py
  │   │   ├── print_nli_llm.py
  │   │   ├── prompt.txt
  │   │   └── files/
  │   │       ├── evaluacion_rag.csv
  │   │       ├── evaluation_no_rag.csv
  │   │       └── evaluation_real_description.csv
  │   ├── radar/
  │   │   ├── print_D1.py
  │   │   ├── print_D2.py
  │   │   └── print_D3.py
  └── news_queries/
      └── news_data.py
```

---

## License

This project is licensed under the terms of the [GNU General Public License (GPL) version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).

**Authors:**

  - José Manuel García Campos
  - Agustín Walabonso Lara Delgado
  - Vicente Jesús Mayor Gallego
  - Jorge Calvillo Arbizu
