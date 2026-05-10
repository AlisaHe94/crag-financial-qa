# Beyond Naive RAG: Probabilistic Corrective RAG for Financial Document QA

[![tests](https://github.com/AlisaHe94/crag-financial-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/AlisaHe94/crag-financial-qa/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**STAT 5293 Final Project** — Dishen Yang, Siwen Chen, Jiayi He

A Corrective RAG (CRAG) pipeline for question answering over SEC financial filings. Features a probabilistic retrieval evaluator, modality-aware FAISS indexes (text + table), and a three-tier fallback: internal text → table → live web search.

This repository contains **three labeled versions** of the system, organized as sibling folders so the architectural progression is visible at a glance:

| Folder | Version | What's in it |
|---|---|---|
| [`v1/`](./v1/) | **v1** | Original Probabilistic CRAG (3-tier text/table/web fallback). No decomposition, no consistency checks. The starting point of the iteration. |
| [`v2/`](./v2/) | **v2** | v1 + modality-aware decomposition + per-sub-question coverage check + numerical fidelity check + α-ensemble + bootstrap CIs + 44-question eval + corrected LLM-as-judge rubric. **Web noise included** (sub-questions can trigger their own web fallback). |
| [`v3/`](./v3/) | **v3** | v2 + the `_suppress_web=True` fix for decomposed sub-questions. Decomposed sub-questions are now corpus-bound; top-level OOC queries still use web normally. **Recommended version.** |

Each version is self-contained and independently runnable. v2 and v3 share most of their code — the architectural difference is one keyword argument (`_suppress_web=True`) in `_query_with_decomposition`. Diff [`v2/crag_pipeline.py`](./v2/crag_pipeline.py) against [`v3/crag_pipeline.py`](./v3/crag_pipeline.py) to see exactly what the v2→v3 fix changes.

## Headline results

### v2 / v3 ablation (44-question expanded eval bank)

| Condition | KHR | KHR 95% CI | Routing precision | Latency (s) |
|---|---|---|---|---|
| `baseline_no_tables` | 0.668 | [0.609, 0.727] | — | 5.7 |
| `baseline_tables` | 0.641 | [0.554, 0.723] | — | 4.9 |
| `crag_no_tables` (v3) | **0.764** | **[0.709, 0.814]** | **0.886** | 15.3 |
| `crag_tables` (v3) | 0.723 | [0.654, 0.791] | 0.864 | 17.8 |

**Key takeaways:**

- **CRAG (v3) outperforms baseline by ~0.07 KHR points** with non-overlapping bootstrap CIs. Headline ablation result.
- **Routing precision of 0.886** demonstrates the probabilistic evaluator is well-calibrated (CRAG-only metric; baseline does not have routing).
- **Numerical fidelity 0.73** (crag_no_tables, v3) — DANA-inspired post-hoc check showing ~27% of dollar/percentage values in CRAG answers are not directly grounded in retrieved chunks. A failure mode KHR cannot detect.
- **Sub-question coverage 0.90** on both CRAG conditions — modality-aware decomposition + the per-sub-question coverage check are doing real work.

**v2 → v3 deltas** (impact of the web-suppression fix alone):

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| Routing precision (crag_no_tables) | 0.841 | 0.886 | +0.04 |
| Routing precision (crag_tables) | 0.795 | 0.864 | +0.07 |
| Numerical fidelity (crag_no_tables) | 0.683 | 0.729 | +0.05 |
| Sub-question coverage (crag_tables) | 0.818 | 0.900 | +0.08 |

The v3 fix improved every secondary metric while keeping KHR essentially flat — confirming that web noise was contributing to lower routing precision and fidelity, not to the headline accuracy result.

### Eval bank evolution (22 → 44 questions)

The headline numbers above are the **canonical results** for this project, measured on the 44-question eval bank.

The eval bank started at **22 questions**, and v1 was originally evaluated on that bank — those are the numbers shown in our class presentation and the live demo. Inspecting the 22-question results made it clear that 22 questions was not enough: failure modes we cared about (multimodal queries mixing narrative and tabular evidence, partial-out-of-corpus questions, dollar/percentage hallucination patterns) were underrepresented or missing entirely, and several question-type cells had too few examples to draw any conclusion at all. That observation is what motivated **expanding the eval bank to 44 questions** with explicit edge-case categories (n ≥ 2 per category) before the v2 → v3 iteration was run.

The 22-question v1 run is preserved in [`v1/`](./v1/) as a historical first pass — it is the artifact that surfaced the eval-bank limitation in the first place, not a number we still defend. v1 was not re-run on the 44-question bank because the v2 → v3 iteration ablation is end-to-end on the same 44-question denominator and that is what the secondary metrics (routing precision, numerical fidelity, sub-question coverage) need a consistent denominator for. Going forward, treat **44q as the eval bank for this project**.

---

## Project Structure

```
STAT 5293 Proj Proposal/
├── README.md              # this file (project overview + headline results)
├── AI_USAGE.md            # GenAI tool disclosure
├── prompts/               # per-component prompt summaries (one file per code module)
├── requirements.txt       # Python dependencies (shared across all three versions)
├── .env.example           # template for API keys + threading-safety env vars
├── .github/
│   └── workflows/         # GitHub Actions CI: pytest matrix + ruff lint + schema validation
├── tests/                 # pytest unit tests for shared helpers (bootstrap CI, KHR,
│   │                      #   numerical fidelity, multi-score parser, eval-questions schema)
│   └── README.md
├── notebooks/
│   └── results_analysis.ipynb   # v1/v2/v3 ablation analysis: KHR + bootstrap CIs,
│                                #   v2→v3 deltas, edge-case drill-down, latency analysis
├── data/                  # SEC filings + FAISS indexes + eval result CSVs (large files
│   │                      #   gitignored; committed CSVs hold the headline ablation results)
│   ├── eval_questions.json
│   ├── eval_results_v1_22q.csv
│   ├── eval_results_v1_44q.csv (+ _judged.csv)
│   ├── eval_results_v2_websnoise.csv
│   ├── eval_results_v3_judged.csv
│   └── backups/
├── v1/                    # original Probabilistic CRAG (baseline)
│   ├── crag_pipeline.py
│   ├── rag_baseline.py
│   ├── evaluate.py
│   ├── app.py + _styles.py + pages/   # Streamlit demo
│   ├── document_processor.py
│   ├── data_fetcher.py
│   ├── judge_results.py + smoke_test.py
│   └── README.md
├── v2/                    # iteration with web-noise (preserved for ablation comparison)
│   ├── (same Python files as v1, with crag_pipeline.py extended)
│   ├── scripts/           # alpha_sweep, alpha_ensemble, citation_correctness, compare_answers
│   └── README.md
└── v3/                    # iteration with web-noise removed (recommended)
    ├── (same Python files as v2 with one-line _suppress_web=True fix)
    ├── scripts/
    └── README.md
```

Each version folder is self-contained and runnable from its own directory. Shared infrastructure (corpus + dependencies + tests + CI + analysis notebook + AI documentation) lives at the repository root and is referenced by all three.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. GPU recommended but not required.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_...           # default LLM provider (Llama-3.1-8B-Instant)
TAVILY_API_KEY=tvly-...        # for web search fallback
LLM_PROVIDER=groq              # "groq" | "openai" | "anthropic"
LLM_MODEL=llama-3.1-8b-instant
```

**Why Groq + Llama-3.1-8B?** Groq's LPU endpoint serves Llama-3.1-8B-Instant
at ~500 tokens/sec, which keeps the full ablation study (4 conditions × ~30
questions × retries) under a few minutes instead of hours. 8B-Instant is the
smallest production-supported open Llama on Groq today (the 3B preview slot
was decommissioned in late 2025) and is well-suited to the "small open model"
baseline established by MultiFinRAG (arXiv:2506.20821).

> **Switching providers:** To use a commercial API instead, set
> `LLM_PROVIDER=openai` (with `OPENAI_API_KEY` and e.g. `LLM_MODEL=gpt-4o-mini`)
> or `LLM_PROVIDER=anthropic`. To run a true 3B Llama locally, swap to Ollama
> with `meta-llama/Llama-3.2-3B-Instruct` — see Groq's model list at
> [console.groq.com/docs/models](https://console.groq.com/docs/models) for
> current hosted ids.

---

## Step-by-Step Usage

> All commands below assume you've `cd`'d into one of the version folders (`v1/`, `v2/`, or `v3/`). Each version is independently runnable; the example commands work identically across all three. Pick `v3/` for the recommended (web-noise-removed) version, `v2/` for the iteration-with-web-noise comparison, or `v1/` for the original baseline.

### Step 1 — Download SEC filings

Downloads 10-K and 10-Q filings for AAPL, MSFT, GOOGL, AMZN, META from SEC EDGAR into the shared `../data/sec_filings/` folder at the repo root.

```bash
cd v3/        # or v1/ or v2/
python data_fetcher.py
```

**Options** (edit defaults at the top of `data_fetcher.py`):

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_TICKERS` | `["AAPL", "MSFT", "GOOGL", "AMZN", "META"]` | Companies to fetch |
| `DEFAULT_FILING_TYPES` | `["10-K", "10-Q"]` | Filing types |
| `num_filings` | `3` | Most recent filings per type |

---

### Step 2 — Build the vector index

Parses all downloaded filings and builds FAISS indexes (separate indexes for text and table chunks).

```bash
python rag_baseline.py data/sec_filings
```

This saves indexes to `data/vectordb/` by default. Two variants are built during the ablation study (with and without table-aware parsing); you can build them manually:

```bash
# Table-aware index (full system)
python rag_baseline.py data/sec_filings

# Naive index (baseline — edit table_aware=False in __main__ block)
python rag_baseline.py data/sec_filings
```

---

### Step 3 — Run a single query (interactive)

**Baseline RAG:**

```python
from rag_baseline import VectorStore, BaselineRAG

vs = VectorStore.load("data/vectordb")
rag = BaselineRAG(vs)
result = rag.query("What was Apple's gross margin in fiscal year 2023?")
print(result["answer"])
```

**Probabilistic CRAG:**

```python
from rag_baseline import VectorStore
from crag_pipeline import CorrectedRAG

vs = VectorStore.load("data/vectordb")
crag = CorrectedRAG(vs)
result = crag.query("What was Apple's gross margin in fiscal year 2023?")

print(result["answer"])
print(f"Routing decision : {result['routing_decision']}")   # correct / ambiguous / incorrect
print(f"Confidence score : {result['confidence_score']}")
print(f"Tier used        : {result['tier_used']}")          # text / text+table / web
```

---

### Step 4 — Run the ablation study

Runs all four experimental conditions and saves results to `data/eval_results.csv`:

```bash
python evaluate.py data/sec_filings
```

**Four conditions:**

| Condition | Chunking | CRAG evaluator |
|---|---|---|
| `baseline_no_tables` | Naive fixed-size | No |
| `baseline_tables` | Semantic + table-aware | No |
| `crag_no_tables` | Naive fixed-size | Yes |
| `crag_tables` | Semantic + table-aware | Yes |

**Output:** `data/eval_results.csv` with per-question rows and a printed summary table broken down by condition and question type (Types 1–4).

To use a custom question set instead of the built-in samples, create `data/eval_questions.json`:

```json
[
  {
    "id": "q1",
    "type": 3,
    "question": "What was Microsoft's total revenue for fiscal year 2023?",
    "expected_keywords": ["211", "billion", "revenue"],
    "ground_truth_in_corpus": true
  }
]
```

---

### Step 5 — Launch the Streamlit demo

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The demo provides:

- **Side-by-side comparison** of Baseline RAG vs. CRAG answers
- **Confidence score gauge** showing P(Relevant | q, d)
- **Routing decision badge** (CORRECT / AMBIGUOUS / INCORRECT)
- **Ablation results viewer** (reads `data/eval_results.csv` if present)

**Tip for the demo:** Use the out-of-corpus trick questions (e.g., *"What is the current federal funds rate?"*) to show the CRAG system recognize it cannot answer from the corpus and pivot to a live web search.

---

## Tuning CRAG Thresholds

Thresholds are set in `.env` and can be adjusted without code changes:

| Variable | Default | Meaning |
|---|---|---|
| `CRAG_THRESHOLD_HIGH` | `0.75` | Score ≥ this → CORRECT, answer directly |
| `CRAG_THRESHOLD_LOW` | `0.40` | Score < this → INCORRECT, trigger web search |
| `THRESHOLD_TEXT` | `0.70` | Min cosine sim for text chunk acceptance |
| `THRESHOLD_TABLE` | `0.65` | Min cosine sim for table chunk acceptance |
| `N_TEXT_MIN` | `6` | Min text hits before table fallback triggers |
| `M_TABLE` | `4` | Table chunks fetched on fallback |

---

## Architecture Overview

### v1 — Probabilistic CRAG baseline (3-tier fallback)

```
Query
  │
  ▼
[Tier 1] FAISS text index (θ_text = 0.70)
  │  ≥ 6 hits? ──Yes──► Probabilistic Evaluator
  │  < 6 hits?
  ▼
[Tier 2] FAISS table index (θ_table = 0.65)
  │  combine text + table chunks
  ▼
Probabilistic Evaluator
  score = α·cosine + (1−α)·sigmoid(cross_encoder_logit)   (α = 0.4)
  │
  ├─ score ≥ τ_high  →  CORRECT   → Generate answer
  ├─ τ_low ≤ score   →  AMBIGUOUS → Rewrite query → Re-retrieve → Generate
  └─ score < τ_low   →  INCORRECT
                            │
                        [Tier 3] Tavily web search → Generate answer
```


<img width="935" height="467" alt="image" src="https://github.com/user-attachments/assets/8e41f87f-efa6-4d91-8eb9-8522786093da" />


### v2 / v3 — Consistency interventions on top of v1

v2 and v3 wrap the v1 three-tier flow with a decomposition layer (for multimodal queries) and two post-hoc consistency checks. The architectural difference between v2 and v3 is **one keyword argument**: `_suppress_web=True` on the inner CRAG calls inside the decomposition loop, which prevents sub-questions from triggering their own web fallback.

```
Query
  │
  ▼
[Modality classifier] — text-only / table-only / multimodal
  │
  ├─ multimodal? ──Yes──► [Modality-aware decomposition]
  │                          • split into text-side and table-side sub-questions
  │                          • run each sub-q through the v1 three-tier flow
  │                              ├── v2: sub-q can hit Tier 3 (Tavily web)   ← web noise
  │                              └── v3: _suppress_web=True (corpus-bound)   ← the v2→v3 fix
  │                          • fuse sub-answers into a single response
  │
  └─ single-modality? ──► v1 three-tier flow directly
  ▼
Draft answer
  │
  ▼
[Per-sub-question coverage check]    (FinVet-style claim verification)
  │  ├─ does the answer address every sub-question?
  │  └─ flag any sub-q whose claims aren't grounded in retrieved chunks
  ▼
[Numerical fidelity check]           (DANA-inspired post-hoc verification)
  │  ├─ extract every $ / % / ratio in the draft answer
  │  ├─ normalize ($ + commas + units) and search retrieved chunks
  │  └─ flag any number not directly grounded
  ▼
Final answer + telemetry
  (routing_decision, confidence_score, tier_used,
   sub_q_coverage, numerical_fidelity, α used)
```

**v2-only / v3-only details:** v2 retains web fallback for decomposed sub-questions (web noise can leak into multimodal answers); v3 suppresses it. Both versions also ship an **α-ensemble inference** mode (`v2/scripts/alpha_ensemble.py`, same in v3) that runs the evaluator across α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and reports stability buckets — useful for understanding how robust each routing decision is to the cosine ↔ cross-encoder weighting choice. See [`v2/README.md`](./v2/README.md) and [`v3/README.md`](./v3/README.md) for run-time details and the corresponding scripts.

---

## Error Handling

This project uses a mix of **fail-fast checks**, **logged degradation**, and **optional failover** so that transient external failures do not always abort the full pipeline.

- **Configuration / secrets.** Missing required credentials are surfaced early where practical: e.g. `data_fetcher.py` raises if `SEC_EDGAR_USER_AGENT` is unset; `rag_baseline._build_single_client` raises if the primary provider's API key (e.g. `GROQ_API_KEY`) is missing. When an optional `LLM_FALLBACK_PROVIDER` is configured but cannot be initialized, the code logs a warning and continues with the primary client only.

- **LLM calls.** The OpenAI-compatible wrapper detects output truncation (`finish_reason == "length"`) and retries once with a larger `max_tokens`; if the retry fails, the partial answer is kept and a warning is logged. When a fallback provider is enabled, **rate limits, timeouts, and common 5xx-class errors** on the primary route trigger an automatic switch to the fallback; authentication, unknown model, and other non-transient errors are **re-raised** so they are not silently masked.

- **CRAG routing and auxiliary LLM steps.** Query-type classification (`_classify_query`) wraps the classifier call: on any exception or unparseable label it **defaults to** `multimodal`. HyDE generation (`_generate_hyde`) falls back to the **original question** on failure or suspiciously short output. The multimodal completeness audit (`_answer_addresses_all_parts`) uses a **fail-open** policy: if the audit LLM call errors, the pipeline assumes the answer is complete so a flaky judge does not worsen the user-facing result; the broader completeness block is also wrapped so failures **log a warning and preserve the previous answer**.

- **Probabilistic retrieval scorer.** The cross-encoder path guards against **NaN / non-finite** scores by substituting a neutral value so aggregate confidence and routing remain well-defined.

- **Document ingestion and indexing.** `build_index_from_dir` processes filings **per file** inside `try / except`: a corrupt or unsupported file logs a warning and is skipped instead of failing the entire index build. HTML table extraction via `pandas.read_html` returns an empty string on error so the parser can fall back to simpler paths. Missing `beautifulsoup4` for HTML filings raises a clear `ImportError` with install instructions.

- **Web search tier.** If `TAVILY_API_KEY` is unset, `WebSearchFallback` logs a warning and returns **no web passages** (corpus-only behavior) instead of crashing at initialization.

- **Batch evaluation.** `evaluate.run_condition` catches **per-question** query failures, records an empty answer for that row, and continues the ablation. `run_ablation` wraps each **condition** so one failed configuration does not prevent writing partial results; results are flushed to CSV incrementally where possible.

- **Smoke test.** `smoke_test.py` validates Groq connectivity with explicit exit codes and hints when the model id is deprecated or unavailable.

Together, these behaviors prioritize **continued operation with degraded features** (skip bad filings, fall back to simpler retrieval text, optional second LLM provider) while still **failing loudly** on missing mandatory configuration for the task at hand (e.g. SEC user-agent, primary LLM key).

---

## Tests & CI/CD

Unit tests for the project's pure-Python helpers (`_bootstrap_ci`, `keyword_hit_rate`, `_check_numerical_fidelity`, `_parse_multi_score`, eval-questions schema):

```bash
pip install pytest
pytest tests/ -v
```

48 tests across 5 files. Tests that depend on heavy ML dependencies (`sentence_transformers`, `faiss`, `torch`) skip cleanly via `pytest.importorskip` when those aren't installed — useful for minimal CI environments. See [`tests/README.md`](./tests/README.md) for details.

### Continuous Integration

A GitHub Actions workflow at [`.github/workflows/tests.yml`](./.github/workflows/tests.yml) runs three jobs on every push and pull request to `main`:

| Job | What it runs | Why |
|---|---|---|
| **pytest** | Full test suite × Python 3.10 / 3.11 / 3.12 matrix, with coverage report | Catches version-specific regressions; coverage XML uploaded as artifact |
| **lint** | `ruff check tests/` (E, W, F rule families) | Surfaces style and unused-import issues without blocking the build |
| **validate-data** | Schema-only test subset against `data/eval_questions.json` | Fast guard against eval-bank regressions (no ML deps required, runs in <10s) |

Concurrency control cancels in-flight workflow runs when a new commit is pushed to the same branch (saves CI minutes during rapid iteration). Pip cache is keyed on Python version to avoid redundant downloads. Manual triggering enabled via `workflow_dispatch`.

Two production bugs were originally surfaced by this test suite — a `None`-handling crash in `keyword_hit_rate` and a normalization gap in `_check_numerical_fidelity` — both fixed before final results were reported, demonstrating the test-driven feedback loop.

---

## Analysis notebook

The Jupyter notebook at [`notebooks/results_analysis.ipynb`](./notebooks/results_analysis.ipynb) walks through the v1/v2/v3 ablation CSVs end-to-end: regenerates the headline KHR table, computes bootstrap 95% CIs, plots KHR by question type, surfaces v2→v3 deltas, and shows the edge-case category breakdown. Run from the repo root:

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/results_analysis.ipynb
```

---

## Troubleshooting Guide

### Environment & dependencies

- **`Warning: camelot-py … does not provide the extra 'cv'`**
  This comes from the `[cv]` extra in `requirements.txt` on some `camelot-py` versions. It is usually harmless if installs completed; if table extraction still fails, reinstall `camelot-py` per its current docs.

### API keys & LLM

- **`RuntimeError: GROQ_API_KEY is not set`** (or similar for other providers)
  Copy `.env.example` → `.env` and fill in real keys. Restart the terminal / IDE so `python-dotenv` picks up changes.

- **429 / rate limit / timeouts during `evaluate.py` or the Streamlit demo**
  Increase sleep between questions in `evaluate.py` if needed, or set `LLM_FALLBACK_PROVIDER=gemini` (with `GEMINI_API_KEY` / `GOOGLE_API_KEY`) so transient limits on the primary provider fail over automatically.

### SEC data & indexing

- **`RuntimeError: SEC_EDGAR_USER_AGENT is not set`**
  SEC requires a real contact email in the User-Agent. Set `SEC_EDGAR_USER_AGENT=you@example.com` in `.env` before `python data_fetcher.py`.

- **Downloads warn `Failed to download …` for some tickers**
  Check network, SEC availability, and that you are not blocked for excessive requests. The script logs warnings and continues; verify files under `data/sec_filings/…`.

### CRAG / web / demo

- **Web search never triggers or always empty context**
  Set `TAVILY_API_KEY` in `.env`. Without it, `WebSearchFallback` logs a warning and returns no web passages (corpus-only behavior).

- **Streamlit crashes or hangs on macOS / Python 3.13 with PyTorch**
  Set threading-related variables as in `.env.example` (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, etc.). On Unix, `run.sh` exports them before launching Streamlit; on Windows, add the same keys to `.env` or set them in the shell before `streamlit run app.py`.

### Evaluation & judging

- **`evaluate.py` exits early: `expected pre-built indexes …`**
  Build both indexes with `rag_baseline.py` (see above). Paths must exist: `data/vectordb_baseline` and `data/vectordb_crag_tables`.

- **`judge_results.py` fails or skips rows**
  Ensure the judge provider's API key and model access match your `.env` configuration; the script is designed to skip already-scored rows and write incrementally — check logs for parse/API errors.

---

## References

- Gondhalekar, Patel, Yeh (2025). *MultiFinRAG*. arXiv:2506.20821
- Yan et al. (2024). *Corrective Retrieval Augmented Generation*. arXiv:2401.15884
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 33
