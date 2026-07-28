# SLM-MultiEval


SLM-MultiEval is a multidimensional evaluation methodology designed to analyze the behavior of small language models (SLMs) beyond aggregate performance metrics. Instead of treating text quality as a single score, the framework disentangles three complementary dimensions: syntactic–lexical complexity, informational competence, and consistency. 

By evaluating model outputs across these dimensions and comparing them to human-written references, SLM-MultiEval enables a fine-grained diagnosis of model behavior, particularly under different generation paradigms such as Retrieval-Augmented Generation (RAG) and parametric-only generation (NO-RAG).

In addition to metric means, the current pipeline reports sample sizes, two-sided 95% confidence intervals, paired RAG versus NO-RAG differences and, for the D1 and D2 metric drivers, computation times per news item.

---

## Experimental Design

The experiment controls model, temperature and generation condition.

### Models

| Internal ID | Model |
|---|---|
| `1` | Ministral-3B |
| `2` | Qwen-2.5-0.5B |
| `3` | Llama-3.2-1B |

### Temperatures

| Internal ID | Temperature |
|---|---:|
| `1` | 1.00 |
| `2` | 0.75 |
| `3` | 0.50 |

Results are organised by newspaper, model, temperature and generation
condition. The supplied datasets use the source identifiers `NEWSPAPER_A` and
`NEWSPAPER_B`.

---

## Evaluation Dimensions and Metrics

### D1 — Syntactic and Lexical Complexity

| Metric | Main script | Purpose | Main JSON fields |
|---|---|---|---|
| Fernández-Huerta | `metrics/d1_fh/fh.py` | Spanish readability | `mean_fh`, `ci95_fh`, `n_fh` |
| MTLD | `metrics/d1_mtld/mtld.py` | Lexical diversity | `mean_mtld`, `ci95_mtld`, `n_mtld` |
| Tree Depth | `metrics/d1_tree_depth/tree_depth.py` | Dependency-tree complexity | `mean_depth`, `ci95_depth`, `n_depth` |

### D2 — Informational Competence

| Metric | Main script | Purpose | Main JSON fields |
|---|---|---|---|
| BERTScore | `metrics/d2_bert/bert.py` | Token-level semantic alignment | `mean_precision`, `mean_recall`, `mean_f1` and their `ci95_*`/`n_*` fields |
| SBERT | `metrics/d2_sbert/sbert.py` | Sentence-level semantic similarity | `mean_sbert`, `ci95_sbert` |
| ETE | `metrics/d2_ete/ete.py` | Entropy of named-entity types | `mean_entity_entropy`, `ci95_entity_entropy` |
| UER | `metrics/d2_uer/uer.py` | Unique entities relative to token count | `mean_uer`, `ci95_uer`, `n_uer` |

### D3 — Logical and Factual Consistency

| Metric family | Main script | Purpose | Main JSON fields |
|---|---|---|---|
| NLI-LLM | `metrics/d3_nli_llm/nli_average_calculation.py` | Entailment and contradiction analysis | `mean_entailment_softmax`, `mean_contradiction_softmax`, `mean_no_neutral_softmax` and their `ci95_*` fields |
| Factual-LLM | `metrics/d3_factual_llm/factual_average_calculation.py` | LLM-as-a-judge factual assessment | proportions and 95% CIs for `correct`, `partially_correct` and `incorrect` |

The repository contains nine metric families. Some correlation analyses use
eleven scalar indicators because NLI-LLM and Factual-LLM contribute more than
one derived variable.

LLM-based evaluation depends on prompt-driven judgements. The corresponding
prompts are stored in the D3 metric directories and in `baseline/prompts/`.

---

## Statistical Methodology

### Aggregated metric results

For each experimental cell, metric result files can include:

- the sample mean;
- the number of valid observations;
- a two-sided 95% Student *t* confidence interval;
- a comparison with the human-reference baseline;
- metric computation timing information.

A typical metric block has the following form:

```json
{
  "mean_fh": 67.42,
  "ci95_fh": {
    "lower": 65.81,
    "upper": 69.03
  },
  "n_fh": 150
}
```

### Computation times

D1 and D2 result files include a `calculation_times` block grouped by
newspaper. Timing summaries use seconds and contain the total duration, mean
duration per news item and its 95% confidence interval:

```json
{
  "calculation_times": {
    "fernandez_huerta": {
      "unit": "seconds",
      "n_news": 9150,
      "total_seconds": 12.4,
      "mean_seconds_per_news": 0.00136,
      "ci95_seconds_per_news": {
        "lower": 0.00131,
        "upper": 0.00141
      }
    }
  }
}
```

Timing values are hardware- and environment-dependent. They should only be
compared when device, software versions and execution conditions are
controlled.

### Paired RAG versus NO-RAG comparisons

The `*_rag_no_rag_*difference.py` scripts compare the two generation
conditions at news-item level. Records are paired using:

```text
newspaper + id_llm + id_feature + id_news
```

All paired analyses use the sign convention:

```text
difference = RAG - NO_RAG
```

A positive difference therefore means that RAG has a higher metric value than
NO-RAG. Whether that represents an improvement depends on the interpretation
of the metric.

For each model-temperature group, these scripts write:

- a JSON statistical summary;
- a CSV containing the paired observations;
- the mean paired difference;
- a two-sided 95% Student *t* confidence interval over the paired differences.

### Spearman correlations

The repository includes scripts for building cross-metric Spearman correlation
matrices:

- `metrics/statistics/spearman_metric_matrix.py`;
- `metrics/statistics/spearman_nine_metrics.py`.

The intended unit of analysis is an aggregated experimental cell:

```text
newspaper × model × temperature × condition
```

With two newspapers, three models, three temperatures and two conditions, the
complete design contains 36 observations.

> **Current limitation:** the metric paths configured in the Spearman scripts
> still refer to an earlier directory layout such as `metrics/readability/`,
> `metrics/mtld/` and `metrics/bert/`. Update those paths to the current
> `metrics/d1_*`, `metrics/d2_*` and `metrics/d3_*` result locations before
> running the correlation pipeline.

---

## G-Eval Baselines

The `baseline/` directory contains G-Eval-based assessments for:

- coherence;
- consistency;
- fluency;
- relevance.

The main aggregators are:

```text
baseline/agregator_coherence.py
baseline/agregator_consistency.py
baseline/agregator_fluency.py
baseline/agregator_relevance.py
```

Their aggregated RAG and NO-RAG results are stored in JSON files such as:

```text
baseline/coherence_NO_RAG_results.json
baseline/coherence_RAG_results.json
baseline/consistency_NO_RAG_results.json
baseline/consistency_RAG_results.json
```

The plotting scripts in `baseline/metrics/` display the mean score and its 95%
confidence interval. If an aggregated JSON does not contain an explicit 95% CI,
the plotting code reconstructs it from the stored score distribution.

Additional utilities under `baseline/files/` and `baseline/statistics/`:

- merge tagged CSV results back into per-news JSON data;
- build aligned RAG and NO-RAG datasets;
- compare G-Eval scores using paired differences;
- relate G-Eval scores to the automatic metrics;
- calculate Spearman correlations;
- apply manually reviewed NLI/Factual-LLM scores.

> Some per-news G-Eval preparation scripts expect tagged CSV directories and
> selected 10% sample files that are experimental artefacts rather than
> top-level repository inputs. Check their configured paths before running
> this stage.

---

## Input Data

The main datasets are stored under `metrics/`:

```text
metrics/reales.json
metrics/sinteticas_NORAG.json
metrics/sinteticas_RAG.json
```

- `reales.json` contains the human-written reference news.
- `sinteticas_NORAG.json` contains generations without retrieval context.
- `sinteticas_RAG.json` contains generations produced with retrieval context.

Synthetic records are grouped by newspaper, model and temperature. The
per-news analyses require stable `id_news`, `id_llm` and `id_feature` values so
that RAG and NO-RAG records can be paired correctly.

The synthetic-news generation process is maintained in
[controlled-news-generation-es](https://github.com/jmgarcam/controlled-news-generation-es).

---

## Installation

Create and activate a virtual environment before installing the pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installation notes:

- `requirements.txt` currently pins a complete CUDA-enabled PyTorch stack.
  CPU-only environments may require a platform-specific PyTorch installation
  instead.
- BERTScore and SBERT download pretrained models on first use, so network
  access and additional disk space are required.
- The Spanish spaCy model is installed from the URL pinned in
  `requirements.txt`.
- Rendering Spearman heatmaps as PNG/PDF requires a supported local headless
  Chromium/Edge browser.

---

## End-to-End Workflow

```text
Real and synthetic news
          │
          ▼
Per-news metric calculation
          │
          ▼
Aggregation: mean + n + 95% CI + timing
          │
          ├──────────────► Metric plots with error bars
          │
          ▼
Paired RAG - NO-RAG analysis
          │
          ▼
Cross-metric Spearman analysis
          │
          ▼
Dimension-level radar and bar charts
```

The G-Eval branch follows a related process:

```text
Prompt-based G-Eval judgements
          │
          ▼
Aggregation and per-news enrichment
          │
          ▼
Paired RAG - NO-RAG analysis
          │
          ▼
Correlation with automatic metrics
```

---

## Usage

### Run the automatic metrics

The main metric drivers currently resolve their input files relative to the
working directory. Run them from their own metric directory:

```bash
(cd metrics/d1_fh && python fh.py)
(cd metrics/d1_mtld && python mtld.py)
(cd metrics/d1_tree_depth && python tree_depth.py)

(cd metrics/d2_bert && python bert.py)
(cd metrics/d2_ete && python ete.py)
(cd metrics/d2_sbert && python sbert.py)
(cd metrics/d2_uer && python uer.py)
```

BERTScore and SBERT can be significantly slower than the remaining metrics.
Their current implementations evaluate individual text pairs in order to
record per-news timings.

Generated result files are:

| Metric | Result file |
|---|---|
| Fernández-Huerta | `metrics/d1_fh/fh_results.json` |
| MTLD | `metrics/d1_mtld/mtld_results.json` |
| Tree Depth | `metrics/d1_tree_depth/tree_depth_results.json` |
| BERTScore | `metrics/d2_bert/bert_results.json` |
| ETE | `metrics/d2_ete/ete_results.json` |
| SBERT | `metrics/d2_sbert/sbert_results.json` |
| UER | `metrics/d2_uer/uer_results.json` |

### Run the D3 aggregations

```bash
(cd metrics/d3_nli_llm && python nli_average_calculation.py)
(cd metrics/d3_factual_llm && python factual_average_calculation.py)
```

These commands generate `nli_llm_results.json` and
`results_factual_one_graph/factual_means_by_newspaper.json`, respectively.

### Plot individual metrics

```bash
(cd metrics/d1_fh && python print_fh_results.py)
(cd metrics/d1_mtld && python print_mtld_results.py)
(cd metrics/d1_tree_depth && python print_tree_depth_results.py)

(cd metrics/d2_bert && python print_bert_results.py)
(cd metrics/d2_ete && python print_ete_results.py)
(cd metrics/d2_sbert && python print_sbert_results.py)
(cd metrics/d2_uer && python print_uer_results.py)
```

The figures show RAG and NO-RAG means with 95% CI error bars. When available,
the human-reference mean is displayed as a horizontal line and its 95% CI as a
shaded band.

### Run paired RAG versus NO-RAG analyses

The paired-comparison scripts use paths derived from their own location and can
be run from the repository root:

```bash
python metrics/d1_fh/fh_rag_no_rag_difference.py
python metrics/d1_mtld/mtld_rag_no_rag_mean_difference.py
python metrics/d1_tree_depth/td_rag_no_rag_difference.py

python metrics/d2_bert/bert_f1_rag_no_rag_mean_difference.py
python metrics/d2_ete/ete_rag_no_rag_mean_difference.py
python metrics/d2_sbert/sbert_rag_no_rag_mean_difference.py
python metrics/d2_uer/uer_rag_no_rag_mean_difference.py

python metrics/d3_nli_llm/nli_llm_rag_no_rag_mean_difference.py
python metrics/d3_factual_llm/factual_llm_rag_no_rag_mean_difference.py
```

### Plot the aggregated dimensions

The original radar scripts resolve their inputs relative to the working
directory. Run them from `metrics/radar/`:

```bash
(cd metrics/radar && python print_D1.py)
(cd metrics/radar && python print_D2.py)
(cd metrics/radar && python print_D3.py)
```

The `print_D*_barras.py` scripts provide grouped-bar alternatives:

```text
metrics/radar/print_D1_barras.py
metrics/radar/print_D2_barras.py
metrics/radar/print_D3_barras.py
```

> **Current limitation:** these three bar-chart scripts still point to legacy
> result locations such as `metrics/readability/`, `metrics/mtld/` and
> `metrics/bert/`. Update their input constants to the current D1/D2/D3 result
> files before running them.

Figures are written to dimension-specific directories under `metrics/radar/`.

### Plot G-Eval results

After the G-Eval result JSON files have been generated:

```bash
(cd baseline/metrics && python print_coherence.py)
(cd baseline/metrics && python print_consistency.py)
(cd baseline/metrics && python print_fluency.py)
(cd baseline/metrics && python print_relevance.py)
```

The revised plots are saved with a `_V2` suffix.

After preparing the required per-news G-Eval JSON files, paired analyses can be
run with:

```bash
python baseline/statistics/geval_rag_no_rag_mean_difference.py
```

---

## Generated Outputs

Depending on the script, the pipeline writes:

- JSON metric summaries;
- JSON RAG versus NO-RAG statistical summaries;
- CSV files with paired observations;
- PNG and PDF metric plots;
- radar and grouped bar charts;
- Spearman coefficient and sample-size matrices;
- machine-readable Spearman summaries.

The repository `.gitignore` excludes generated `*.csv`, `*.png` and `*.pdf`
files. JSON outputs are not globally ignored.

---

## Project Structure

```text
SLM-MultiEval/
├── README.md
├── requirements.txt
├── baseline/
│   ├── agregator_*.py
│   ├── generate_csv_*.py
│   ├── files/
│   │   ├── coherence/
│   │   ├── consistency/
│   │   ├── fluency/
│   │   └── relevance/
│   ├── metrics/
│   │   └── print_*.py
│   ├── prompts/
│   └── statistics/
│       ├── coherence/
│       ├── consistency/
│       ├── fluency/
│       ├── relevance/
│       └── geval_rag_no_rag_mean_difference.py
├── metrics/
│   ├── reales.json
│   ├── sinteticas_NORAG.json
│   ├── sinteticas_RAG.json
│   ├── d1_fh/
│   ├── d1_mtld/
│   ├── d1_tree_depth/
│   ├── d2_bert/
│   ├── d2_ete/
│   ├── d2_sbert/
│   ├── d2_uer/
│   ├── d3_factual_llm/
│   ├── d3_nli_llm/
│   ├── radar/
│   └── statistics/
└── news_queries/
    └── news_data.py
```

Each D1 and D2 metric directory contains:

1. the main metric implementation;
2. a result-plotting script;
3. a paired RAG versus NO-RAG analysis script.

Generated data directories and result artefacts are omitted from the tree
above for clarity.

---

## Reproducibility Notes

- Record the Python, PyTorch, CUDA, spaCy and transformer-model versions used
  for an experiment.
- Report the execution device when comparing computation times.
- Preserve `id_news`, `id_llm` and `id_feature` across RAG and NO-RAG datasets.
- Do not interpret a paired result when one condition has unmatched records;
  inspect the generated pairing summary first.
- Use the same text-cleaning rules when comparing results produced by the main
  metric drivers and the standalone paired-analysis scripts.
- Store the prompts and manual scoring criteria alongside LLM-based D3 and
  G-Eval results.

---

## License

This project is licensed under the terms of the
[GNU General Public License version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).
