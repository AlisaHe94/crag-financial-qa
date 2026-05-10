# Prompt Summary — Baseline RAG (`rag_baseline.py`)

## Goal

A standard RAG baseline as the control arm in the ablation: same
corpus, same generator, same chunks as CRAG, but no probabilistic
evaluator, no router, no fallback.

## AI assistance

The team designed the baseline architecture (FAISS + bge-base + Groq
Llama-3.1-8B with Gemini fallback) and chose the engineering
decisions: rate-limit handling strategy, bge query-prefix usage, and
length-truncation detection. AI was used as a coding assistant for
the implementation; the team reviewed and modified all generated code
before integration.

## High-level prompts used

- *Implement a `BaselineRAG` class with a `query(question)` method.
  Use FAISS cosine over `BAAI/bge-base-en-v1.5` embeddings, top-K = 5.
  Return the answer text plus retrieved chunks with cosine scores.*
- *Add an LLM client factory that selects provider via env var
  (`LLM_PROVIDER`) and returns a callable. Default to Groq with
  `llama-3.1-8b-instant`.*
- *Wrap the LLM call: if the primary provider hits a rate limit or
  transient 5xx, retry on `LLM_FALLBACK_PROVIDER` (default `gemini`).*
- *Apply the bge query prefix
  ('Represent this sentence for searching relevant passages: ') only
  when embedding queries, not chunks.*
- *Support modality-aware indexing: separate FAISS indexes for text
  and tables, with a `search_modality()` method.*
- *Detect length-truncated answers via `finish_reason == 'length'`.
  Auto-retry with doubled max_tokens up to a cap.*
