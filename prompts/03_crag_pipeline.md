# Prompt Summary — CRAG Pipeline (`crag_pipeline.py`)

## Goal

The full Probabilistic CRAG pipeline: query classification, hybrid
retrieval, probabilistic evaluator, three-way routing, tiered fallback,
and a self-reflective completeness check for multi-part questions.

## AI assistance

This was the most-iterated component of the project. The team
designed the architecture (continuous evaluator, three-way routing
structure, MultiFinRAG-style tiered fallback) and the engineering
choices that came out of testing — the strong-corpus-match guard, the
completeness-check pattern, the cosine-ordering preference for
multimodal queries. AI was used as a coding assistant for
implementation; the team reviewed, modified, and validated all
generated code, and most architectural refinements came from
team-observed demo behavior.

## High-level prompts used

- *Implement a `RetrievalEvaluator` that scores each (query, chunk)
  pair as `α · cosine + (1 − α) · sigmoid(cross_encoder_logit)` with
  α = 0.4, using `BAAI/bge-reranker-base` as the cross-encoder.
  Three-way routing thresholds at τ_high = 0.75 and τ_low = 0.40.*
- *Implement tiered retrieval (text → table → web fallback via
  Tavily), per MultiFinRAG §3.3.4.*
- *Add a query classifier as Tier 0: ask the LLM to label each query
  as `table_lookup`, `narrative`, `multimodal`, or `out_of_corpus`,
  and use a per-type retrieval composition.*
- *Add hybrid retrieval: use the naive baseline index for text
  retrieval (so flat-text numerical mentions are recoverable) and
  the table-aware index only for tables.*
- *Add a strong-corpus-match guard: if `max_cosine ≥ 0.65`, stay on
  corpus even when mean confidence is below τ_low, to prevent web
  fallback from over-firing on questions the corpus actually answers.*
- *Add a multimodal completeness check that re-prompts on the same
  context with a sharper instruction; only augment with web search
  if the second pass still misses parts and corpus signal is weak.*
- *For multimodal queries, use cosine ordering instead of the
  reranker-blended ordering for context selection.*
- *Add demo controls (web-fallback toggle, query-type override, α
  slider) to the constructor for the live UI to use.*
