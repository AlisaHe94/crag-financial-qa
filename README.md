# Beyond Naive RAG: Probabilistic Corrective RAG for Financial Document QA

**STAT 5293 Final Project** — Dishen Yang, Siwen Chen, Jiayi He

A Corrective RAG (CRAG) pipeline for question answering over SEC financial filings. Features a probabilistic retrieval evaluator, modality-aware FAISS indexes (text + table), and a three-tier fallback: internal text → table → live web search.

---

## Project Structure

```
FinalProject/
├── data_fetcher.py        # Download SEC filings from EDGAR
├── document_processor.py  # PDF parsing: semantic chunking + table extraction
├── rag_baseline.py        # FAISS vector store + baseline RAG
├── crag_pipeline.py       # Probabilistic CRAG with tiered fallback
├── evaluate.py            # Ablation study runner (4 conditions × 4 question types)
├── app.py                 # Streamlit demo interface
└── requirements.txt       # Python dependencies
```

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

### Step 1 — Download SEC filings

Downloads 10-K and 10-Q filings for AAPL, MSFT, GOOGL, AMZN, META from SEC EDGAR into `data/sec_filings/`.

```bash
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
  score = α·cosine + (1−α)·sigmoid(cross_encoder_logit)
  │
  ├─ score ≥ τ_high  →  CORRECT   → Generate answer
  ├─ τ_low ≤ score   →  AMBIGUOUS → Rewrite query → Re-retrieve → Generate
  └─ score < τ_low   →  INCORRECT
                            │
                        [Tier 3] Tavily web search → Generate answer
```


<img width="935" height="467" alt="image" src="https://github.com/user-attachments/assets/8e41f87f-efa6-4d91-8eb9-8522786093da" />


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

## References

- Gondhalekar, Patel, Yeh (2025). *MultiFinRAG*. arXiv:2506.20821
- Yan et al. (2024). *Corrective Retrieval Augmented Generation*. arXiv:2401.15884
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 33
