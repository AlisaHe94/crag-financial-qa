"""
Modality-aware vector store and baseline RAG pipeline.

Following MultiFinRAG (Gondhalekar et al., 2025):
  - Embedding model: BAAI/bge-base-en-v1.5
  - Separate FAISS indexes per modality: text / table
  - Modality-specific similarity thresholds: θ_text=0.70, θ_table=0.65
"""

from __future__ import annotations

import os
import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from document_processor import DocumentChunk, Modality, load_document, EMBED_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBED_MODEL)
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "data/vectordb"))

# Modality-specific similarity thresholds (calibrated in MultiFinRAG §3.3.5)
THRESHOLDS: dict[Modality, float] = {
    "text": float(os.getenv("THRESHOLD_TEXT", 0.70)),
    "table": float(os.getenv("THRESHOLD_TABLE", 0.65)),
    "image": float(os.getenv("THRESHOLD_IMAGE", 0.55)),
}


class ModalityIndex:
    """FAISS inner-product index for one modality."""

    def __init__(self, modality: Modality, model: SentenceTransformer):
        self.modality = modality
        self.model = model
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = [c for c in chunks if c.modality == self.modality]
        if not self.chunks:
            logger.info(f"No {self.modality} chunks to index")
            return
        texts = [c.text for c in self.chunks]
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs.astype(np.float32))
        logger.info(f"Built {self.modality} index: {len(self.chunks)} chunks, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if self.index is None or not self.chunks:
            return []
        thresh = threshold if threshold is not None else THRESHOLDS[self.modality]
        # bge-base-en-v1.5 was trained with this query prefix; documents are
        # encoded without it. Adding it materially boosts retrieval scores
        # for short retrieval-style queries. (BAAI Hugging Face card §Usage.)
        if "bge" in (EMBEDDING_MODEL or "").lower():
            encoded_query = "Represent this sentence for searching relevant passages: " + query
        else:
            encoded_query = query
        q_vec = self.model.encode([encoded_query], normalize_embeddings=True).astype(np.float32)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0 and float(score) >= thresh
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.index:
            faiss.write_index(self.index, str(path / f"index_{self.modality}.faiss"))
        with open(path / f"chunks_{self.modality}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: Path) -> None:
        idx_path = path / f"index_{self.modality}.faiss"
        chunk_path = path / f"chunks_{self.modality}.pkl"
        if idx_path.exists():
            self.index = faiss.read_index(str(idx_path))
        if chunk_path.exists():
            with open(chunk_path, "rb") as f:
                self.chunks = pickle.load(f)
        logger.info(f"Loaded {self.modality} index: {len(self.chunks)} chunks from {path}")


class VectorStore:
    """Multi-modality vector store with separate FAISS indexes."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Default to CPU. Apple Silicon MPS + sentence-transformers + faiss
        # has been observed to segfault intermittently under streamlit's
        # hot-reload (see e.g. pytorch/pytorch issue tracker on MPS bge-base).
        # Set EMBEDDING_DEVICE=mps in .env to opt back in to GPU acceleration
        # if your machine handles it; CPU is fast enough for query-time work.
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        self.indexes: dict[Modality, ModalityIndex] = {
            m: ModalityIndex(m, self.model) for m in ("text", "table", "image")
        }

    def build(self, chunks: list[DocumentChunk]) -> None:
        for idx in self.indexes.values():
            idx.build(chunks)

    def search_modality(
        self,
        query: str,
        modality: Modality,
        top_k: int = 6,
        threshold: float | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        return self.indexes[modality].search(query, top_k, threshold)

    def search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Flat search across text index only (used by baseline RAG)."""
        return self.indexes["text"].search(query, top_k, threshold=0.0)

    def save(self, path: Path = VECTOR_DB_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for idx in self.indexes.values():
            idx.save(path)
        logger.info(f"Saved vector store to {path}")

    @classmethod
    def load(cls, path: Path = VECTOR_DB_DIR, model_name: str = EMBEDDING_MODEL) -> "VectorStore":
        vs = cls(model_name)
        for idx in vs.indexes.values():
            idx.load(path)
        return vs


# ---------------------------------------------------------------------------
# LLM wrapper (Groq / Gemini / OpenAI / Anthropic) with optional fallback
# ---------------------------------------------------------------------------
# Default provider is Groq, serving Llama-3.1-8B-Instant via its
# OpenAI-compatible endpoint. Set LLM_FALLBACK_PROVIDER=gemini in .env to
# automatically retry on Groq's free-tier rate limit (6000 TPM) using
# Gemini 2.5 Flash. Both are free; Gemini fills in when Groq is throttled.
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Per-provider sensible default models (used when LLM_MODEL is not set
# explicitly OR when falling back from a provider whose model id is foreign
# to the fallback provider, e.g. "llama-3.1-8b-instant" on Gemini).
_DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def _build_single_client(provider: str, model: Optional[str] = None):
    """Build a callable `call(system, user) -> str` for one provider."""
    provider = provider.lower()
    model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider, "")

    # If model id doesn't fit this provider, fall back to that provider's default.
    # (e.g., LLM_MODEL=llama-3.1-8b-instant + provider=gemini → use gemini-2.5-flash)
    if provider == "gemini" and not model.startswith(("gemini", "models/")):
        model = _DEFAULT_MODELS["gemini"]
    if provider == "groq" and not model.startswith(("llama", "mixtral", "openai/", "gemma")):
        model = _DEFAULT_MODELS["groq"]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        def call(system: str, user: str) -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        return call

    def _make_openai_call(client, model_name: str):
        """Shared call wrapper for OpenAI-compatible endpoints (Groq/Gemini/OpenAI).

        Detects truncation via `finish_reason == "length"` and auto-retries
        once with double the max_tokens. Without this, the LLM occasionally
        stops mid-word ("$416," or "fiscal year 202") when context is large
        and the raw response gets returned to the user looking broken.
        """
        def call(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2048,
                # Near-deterministic outputs for the demo. Default
                # temperature (~1.0) makes CRAG appear inconsistent
                # across runs because each query involves multiple LLM
                # calls (classifier, generator, completeness check,
                # optional re-prompt). 0.05 keeps outputs effectively
                # deterministic without the rare "empty response"
                # edge case some providers exhibit at exactly 0.0.
                temperature=0.05,
            )
            answer = resp.choices[0].message.content or ""
            try:
                finish = resp.choices[0].finish_reason
            except Exception:
                finish = None
            # If the model stopped because it hit the token cap, retry with
            # 2x the budget (one shot only — don't loop).
            if finish == "length":
                logger.info(f"Output truncated (finish_reason=length); retrying with 4096 tokens")
                try:
                    resp2 = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=4096,
                        temperature=0.05,
                    )
                    answer = resp2.choices[0].message.content or answer
                except Exception as e:
                    logger.warning(f"Retry on length-truncation failed ({e}); keeping partial answer")
            return answer
        return call

    if provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        # max_retries=0: surface 429s immediately so our outer fallback wrapper
        # can route to Gemini, instead of the OpenAI SDK silently sleeping
        # 20-30 seconds per retry and inflating per-call latency.
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=0)
        return _make_openai_call(client, model)

    if provider == "gemini":
        from openai import OpenAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in .env. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        return _make_openai_call(client, model)

    # Default: openai
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _make_openai_call(client, model)


# Substring markers we treat as "this is a transient capacity / rate-limit
# error worth retrying on the fallback provider, not a real failure".
_FALLBACK_TRIGGERS = (
    "rate_limit", "rate limit", "tokens per minute", "tpm",
    "request too large", "413", "429", "503", "504",
    "timeout", "timed out", "overloaded", "service unavailable",
    "ratelimiterror", "too many requests", "exceeded",
)


# Exception classes from the openai SDK that we always treat as fallback-worthy
# regardless of message text — covers all 429 paths even if the message string
# differs across SDK versions.
def _is_transient_openai_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ("ratelimit", "apitimeout", "apiconnection", "internalserver"))


def _build_llm_client():
    """Build the primary LLM caller, optionally wrapped with a fallback.

    LLM_PROVIDER (default: groq) — primary provider.
    LLM_FALLBACK_PROVIDER (optional) — if set, triggered on rate-limit / 5xx
    errors. Recommended: gemini (free, generous quota, no card needed).
    """
    primary_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "").lower().strip()

    primary = _build_single_client(primary_provider)
    if not fallback_provider or fallback_provider == primary_provider:
        return primary

    # Build the fallback eagerly so a missing/bad key is surfaced at startup
    # rather than the first time it's needed.
    try:
        fallback = _build_single_client(fallback_provider)
    except Exception as e:
        logger.warning(
            f"LLM_FALLBACK_PROVIDER={fallback_provider} could not be initialized "
            f"({type(e).__name__}: {e}). Continuing without a fallback."
        )
        return primary

    logger.info(
        f"LLM client: primary={primary_provider}, fallback={fallback_provider}"
    )

    def call_with_fallback(system: str, user: str) -> str:
        try:
            return primary(system, user)
        except Exception as e:
            err = str(e).lower()
            # Two ways to detect a fallback-worthy error:
            #   1. Exception class name matches an "expected transient" type
            #      (RateLimitError, APITimeoutError, etc.)
            #   2. Error message contains any rate-limit / 5xx / timeout marker
            if _is_transient_openai_error(e) or any(t in err for t in _FALLBACK_TRIGGERS):
                logger.warning(
                    f"Primary ({primary_provider}) hit transient error "
                    f"[{type(e).__name__}]; falling back to {fallback_provider}."
                )
                return fallback(system, user)
            raise  # auth / model-not-found / other real errors propagate

    return call_with_fallback


SYSTEM_PROMPT = (
    "You are a precise financial analyst assistant. "
    "Answer the user's question using the provided context from SEC filings. "
    "ALWAYS provide complete answers — never stop mid-sentence or mid-number. "
    "When citing a financial figure, include the FULL value, the unit (e.g. "
    "'million' or 'billion'), and the fiscal period. For example, write "
    "'$416,161 million for fiscal year 2025', not '$416,'. "
    "Cite specific section names and dates when present in the context. "
    "If the context contains relevant but incomplete information, summarize what is "
    "available and note what is missing — do not refuse to answer just because the "
    "context is partial. Only respond with 'Insufficient information.' if the context "
    "is genuinely irrelevant or contains nothing about the topic. "
    "Do not invent numbers that are not in the context."
)


# ---------------------------------------------------------------------------
# Baseline RAG (text-only retrieval, no quality check)
# ---------------------------------------------------------------------------

class BaselineRAG:
    """Standard RAG: retrieve from text index → generate. No evaluator."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vs = vector_store
        self.top_k = top_k
        self.llm = _build_llm_client()

    def query(self, question: str) -> dict:
        retrieved = self.vs.search(question, self.top_k)
        # Cap each chunk to 4000 chars so 5 huge tables don't blow up the LLM
        # input. Groq's Llama-3.1-8B handles 128k context, but pushing 200kB
        # of tokenized table soup adds 10-30s of latency for marginal gain.
        context = "\n\n---\n\n".join(
            f"[{c.source}]\n{c.text[:2000]}" for c, _ in retrieved
        )
        answer = self.llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        return {
            "answer": answer,
            "retrieved_chunks": [
                {"text": c.text[:200], "score": s, "source": c.source, "modality": c.modality}
                for c, s in retrieved
            ],
            "mode": "baseline_rag",
        }


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index_from_dir(
    filing_dir: str | Path,
    table_aware: bool = True,
    save_path: Path = VECTOR_DB_DIR,
    embedder=None,
) -> VectorStore:
    """Build a FAISS index over all filings in `filing_dir`.

    If `embedder` is provided, it's reused (saves ~30s of model load when
    building both ablation variants back-to-back). Otherwise a fresh
    SentenceTransformer is loaded.
    """
    from sentence_transformers import SentenceTransformer as ST
    filing_dir = Path(filing_dir)
    if embedder is None:
        embedder = ST(EMBEDDING_MODEL)

    all_chunks: list[DocumentChunk] = []
    files = (
        list(filing_dir.rglob("*.pdf"))
        + list(filing_dir.rglob("*.htm"))
        + list(filing_dir.rglob("*.txt"))
    )

    # Note: sec-edgar-downloader v5 only saves `full-submission.txt` (a multi-
    # part SEC submission archive containing the primary 10-K/10-Q HTML plus
    # all exhibits). The parser in document_processor.py knows how to extract
    # the primary document out of that envelope, so we DO want to process it.
    logger.info(f"Processing {len(files)} files from {filing_dir}  (table_aware={table_aware})")
    for f in files:
        try:
            chunks = load_document(f, table_aware=table_aware, _embedder=embedder)
            all_chunks.extend(chunks)
            logger.info(f"  {f.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to process {f}: {e}")

    vs = VectorStore(EMBEDDING_MODEL)
    vs.model = embedder  # reuse already-loaded model
    vs.indexes = {m: ModalityIndex(m, embedder) for m in ("text", "table", "image")}
    vs.build(all_chunks)
    vs.save(save_path)
    logger.info(f"Saved index to {save_path}  ({len(all_chunks)} chunks total)")
    return vs


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Build FAISS indexes from SEC filings.")
    p.add_argument("directory", nargs="?",
                   default=os.getenv("DATA_DIR", "data/sec_filings"),
                   help="Path to directory of filings (default: data/sec_filings).")
    p.add_argument("--variant", choices=["both", "table-aware", "baseline"], default="both",
                   help="Which index to build (default: both).")
    p.add_argument("--crag-path", default="data/vectordb_crag_tables",
                   help="Save path for the table-aware (full system) index.")
    p.add_argument("--baseline-path", default="data/vectordb_baseline",
                   help="Save path for the naive (no-table-awareness) baseline index.")
    args = p.parse_args()

    # Load embedder once and share across both builds.
    from sentence_transformers import SentenceTransformer as ST
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    shared_embedder = ST(EMBEDDING_MODEL)

    if args.variant in ("both", "table-aware"):
        logger.info("=== Building table-aware (full system) index ===")
        build_index_from_dir(args.directory, table_aware=True,
                             save_path=Path(args.crag_path), embedder=shared_embedder)
    if args.variant in ("both", "baseline"):
        logger.info("=== Building baseline (no-table-awareness) index ===")
        build_index_from_dir(args.directory, table_aware=False,
                             save_path=Path(args.baseline_path), embedder=shared_embedder)
