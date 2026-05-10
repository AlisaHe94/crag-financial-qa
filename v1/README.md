# v1 — Original Probabilistic CRAG (baseline)

This is **v1** — the original Probabilistic CRAG system as initially
submitted, before any of the consistency-focused interventions in
[`../v2/`](../v2/) and [`../v3/`](../v3/) were added. v1 is the
"baseline" of the iteration trajectory and the starting point of the
ablation comparison.

## What this contains

| File | Purpose |
|---|---|
| `crag_pipeline.py` | Probabilistic CRAG with 3-tier text/table/web fallback. No decomposition, no consistency checks. |
| `rag_baseline.py` | FAISS vector store + baseline RAG + LLM provider abstraction (Groq/Gemini/OpenAI/Anthropic). |
| `evaluate.py` | 4-condition ablation runner (baseline_no_tables, baseline_tables, crag_no_tables, crag_tables). KHR + routing precision + latency. |
| `document_processor.py` | PDF/HTML parsing with semantic chunking and HTML-native table extraction. |
| `data_fetcher.py` | SEC EDGAR downloader (10-K and 10-Q for AAPL, MSFT, GOOGL, AMZN, META). |
| `app.py` | Streamlit demo interface with side-by-side baseline vs. CRAG comparison. |
| `_styles.py` | Streamlit styling. |
| `judge_results.py` | Original LLM-as-judge scorer (single-criterion 0.0–1.0 score). |
| `smoke_test.py` | Quick connectivity/sanity check for Groq + retrieval. |
| `requirements.txt` | Python dependencies. |
| `run.sh` | Streamlit launcher with thread-safety env vars set for macOS + PyTorch. |
| `pages/` | Streamlit subpages (Ablation Results viewer). |

## What's NOT in v1 (added in v2 and v3)

The following are documented in [`../v2/README.md`](../v2/README.md) and
[`../v3/README.md`](../v3/README.md):

- Modality-aware query decomposition for multimodal queries (v2)
- Per-sub-question coverage check (v2)
- Numerical fidelity check, DANA-inspired (v2)
- α-sensitivity sweep + α-ensemble inference (v2)
- Citation-correctness post-hoc metric (v2)
- LLM-as-judge with corrected 3-criterion rubric (v2)
- Parallel judge runner with `ThreadPoolExecutor` (v2)
- 44-question expanded eval bank with edge-case categories (v2)
- Bootstrap 95% CIs on KHR / routing precision / latency (v2)
- Qualitative answer comparison (v2)
- **Web suppression for decomposed sub-questions (v3)** — the
  `_suppress_web=True` fix that prevents off-topic web content from
  polluting decomposed-query answers. The single-line architectural
  change distinguishing v3 from v2.

## Running v1 in isolation

```bash
# from this folder
cd v1/
ln -s ../data data 2>/dev/null   # symlink shared corpus + indexes
python evaluate.py
```

For Streamlit demo:

```bash
cd v1/
bash run.sh
```

Each version folder is independently runnable — `v1/`, `v2/`, `v3/` all
have their own entry points and their own README. Shared infrastructure
(`requirements.txt`, `data/`, `prompts/`, `AI_USAGE.md`, top-level
`README.md`) lives at the repository root.

## Relationship to v2 and v3

```
STAT 5293 Proj Proposal/
├── README.md                # repo overview + v1/v2/v3 table + headline results
├── AI_USAGE.md
├── prompts/
├── requirements.txt
├── data/                    # shared SEC corpus + FAISS indexes (gitignored)
├── v1/                      ← YOU ARE HERE: original baseline
├── v2/                      # iteration with web noise included
└── v3/                      # iteration with web noise removed (recommended)
```

The architectural diff between v1 → v2 → v3 is documented at the
project root README under "Headline results" and in the per-version
READMEs. v1 is the unmodified baseline; v2 and v3 share most of their
code and differ by one keyword argument (`_suppress_web=True`).
