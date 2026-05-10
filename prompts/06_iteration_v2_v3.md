# Prompt Summary — Consistency-Interventions Iteration (v2 → v3)

## Goal

Layer three consistency-focused architectural interventions on top of the
v1 system documented in `03_crag_pipeline.md` and `04_evaluation.md`,
informed by qualitative inspection of v1 failure modes and current
financial-RAG literature (DANA, FinVet, MultiFinRAG, Self-RAG).

The iterated code lives in `../v2/` (with web-noise) and `../v3/` (web
denoised). The v1 baseline is preserved unchanged in `../baseline_crag/`
for direct comparison. The v2-vs-v3 diff is the architectural change in
`_query_with_decomposition` (one keyword argument: `_suppress_web=True`)
plus the `_suppress_web` parameter wired through `query()`.

## Summary of AI assistance

The team identified the failure modes through qualitative inspection of
v1 answers (cross-company contamination, fabricated numbers, web-noise
pollution in decomposed queries) and chose which industry techniques to
adapt — DANA's deterministic-value-lookup principle, FinVet's claim-
verification staging, modality-aware decomposition. AI was used as a
coding assistant for implementation. All architectural decisions and
final calibrations came from team review of system behavior.

## High-level prompts used

### CRAG pipeline extensions (`crag_pipeline.py`)

- *Add a modality-aware query decomposition function. For multimodal
  queries that mix a quantitative figure with narrative MD&A about the
  same entity, split into modality-specific sub-questions before
  retrieval. Each sub-question routes through the existing classifier
  independently. Include the full entity name in each sub-question so
  retrieval has standalone context. Keep the function env-gated via
  `QUERY_DECOMPOSITION=1`.*
- *Add a per-sub-question coverage check using a fresh LLM call. After
  each decomposed sub-question is independently answered, verify
  whether the answer addresses the sub-question. Surface uncovered
  sub-questions as `incomplete_sub_questions` and
  `sub_question_coverage_rate` in the result dict. Fail-safe on LLM
  errors (assume covered).*
- *Add a numerical fidelity check after generation. Regex-extract every
  dollar amount, percentage, and unit-prefixed number from the answer.
  Verify each appears verbatim (with light comma/$ normalization) in a
  retrieved chunk. Return `numerical_fidelity` with `fidelity_score`,
  `verified_numbers`, `unverified_numbers`. This is a CHECK only, not a
  correction — flagged numbers are surfaced in metadata.*
- *Add a `_suppress_web` parameter to `query()` that disables all four
  web-firing paths (OOC bypass, AMBIGUOUS web fallback, OOC refusal
  rescue, multimodal completeness augment). Wire it through
  `_query_with_decomposition` so decomposed sub-questions never trigger
  web fallback — only top-level OOC queries do. This is the v3 fix; the
  v2 folder preserves the prior behavior without this parameter so the
  ablation comparison is auditable as a folder diff.*

### Evaluation harness (`evaluate.py`)

- *Add a `_bootstrap_ci(values, n_resamples=1000, ci=0.95)` helper using
  the percentile method with seed=42. Apply it to KHR, routing
  precision, and latency in the printed summary tables. Report 95%
  CIs alongside means in the overall and by-question-type tables.*
- *Wire the CRAG-only consistency-check telemetry into the per-row
  CSV: `fidelity_score`, `n_numbers_in_answer`, `n_unverified_numbers`,
  `unverified_numbers`, `sub_question_coverage_rate`,
  `n_incomplete_sub_questions`, `incomplete_sub_questions`. Add two
  new summary tables: Consistency Checks (CRAG-only) and Edge-case
  KHR by category (with bootstrap CI per category).*

### Judge runner parallelization (`judge_results.py`)

- *Parallelize the judge runner using `ThreadPoolExecutor` with 8
  workers. Remove the legacy 2-second sleep (was for Groq free tier;
  paid Gemini supports much higher RPM). Per-future error handling
  with one retry on transient failures, thread-safe DataFrame writes
  via `threading.Lock`, periodic CSV flush every 10 completions for
  interruption safety.*

### Inference-time α-ensemble (`scripts/alpha_sweep.py`, `scripts/alpha_ensemble.py`, `scripts/alpha_ensemble_ci.py`)

- *Build an offline α-sensitivity sweep that runs the full eval set
  through CRAG-tables at α ∈ {0.0, 0.4, 0.6, 1.0}. Output per-row KHR
  and latency to `alpha_sweep_results.csv` plus three summary tables:
  KHR by α × question_type, latency by α, latency by α × type.*
- *Build an inference-time α-ensemble that runs each query at three
  α values simultaneously and uses the agreement pattern across the
  three routing decisions as a confidence signal. Use α ∈ {0.0, 0.4,
  1.0} (skip 0.6, confirmed worst from offline sweep). α=0.4 is the
  primary; the other two are confidence probes only. Stability label:
  "stable" (3-way agreement), "majority" (2-of-3), "disagree" (all
  different). Save results CSV and report KHR conditional on
  stability bucket.*
- *Build a bootstrap CI script for the α-ensemble stability buckets.
  Reuse the `_bootstrap_ci` helper from `evaluate.py` so the
  methodology stays consistent. Report whether CIs overlap (informs
  whether report claims "directional evidence" or "significant
  difference").*

### Novel evaluation metric (`scripts/citation_correctness.py`)

- *Build a post-hoc citation-correctness metric that scores whether
  each answer cites the correct company's filing. Map tickers to
  company-specific patterns (AAPL → Apple/iPhone/iPad/Mac,
  MSFT → Microsoft/Azure/Office, GOOGL → Alphabet/Google/YouTube,
  AMZN → Amazon/AWS, META → Meta/Facebook/Instagram). For each
  (question, answer) pair, score 1.0 if the expected company is
  cited and no others; 0.5 if mixed; 0.0 if wrong. Report
  cross-company contamination rate as a separate metric. Read
  existing eval_results.csv — no new eval runs needed.*

### Qualitative comparison (`scripts/compare_answers.py`)

- *Build a qualitative side-by-side answer comparison script. Read
  the existing eval_results.csv and pick a curated set of 8 questions
  spanning three categories: big CRAG wins on multimodal, KHR ties
  where content may differ, and failure modes (balance-sheet
  lookup, OOC). For each question, print all four conditions'
  answers with a per-row metrics summary (KHR, fidelity, coverage,
  unverified numbers). Output as markdown so it can be saved or
  pasted into the report.*
