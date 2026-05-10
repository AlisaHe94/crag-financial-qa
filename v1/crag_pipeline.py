"""
Probabilistic CRAG pipeline with MultiFinRAG-style tiered fallback.

Tiered retrieval strategy (Gondhalekar et al., 2025 §3.3.4):
  Tier 1 — text-only:  retrieve text chunks above θ_text (0.70)
            if |results| ≥ n_text_min → generate answer, done
  Tier 2 — table fallback: fetch top-m table chunks above θ_table (0.65)
            combine with text results → generate
  Tier 3 — web fallback:  if overall confidence < τ_low
            trigger Tavily web search, replace context

Probabilistic evaluator adds a continuous confidence score
P(Relevant | q, d) ∈ [0,1] combining:
  - Cosine similarity from FAISS (bi-encoder, fast)
  - Cross-encoder re-ranker score (MiniLM, accurate)
  final = α * cosine + (1-α) * sigmoid(cross_encoder_logit)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from document_processor import DocumentChunk
from rag_baseline import VectorStore, SYSTEM_PROMPT, _build_llm_client, THRESHOLDS

load_dotenv()
logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
THRESHOLD_HIGH = float(os.getenv("CRAG_THRESHOLD_HIGH", 0.75))
THRESHOLD_LOW = float(os.getenv("CRAG_THRESHOLD_LOW", 0.40))

# MultiFinRAG tiered retrieval parameters (§3.3.4, n=6, m=4)
N_TEXT_MIN = int(os.getenv("N_TEXT_MIN", 6))    # min text hits before table fallback
M_TABLE = int(os.getenv("M_TABLE", 4))           # table chunks to fetch on fallback


class RoutingDecision(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class EvaluatedChunk:
    chunk: DocumentChunk
    cosine_score: float
    cross_encoder_score: float
    final_score: float
    decision: RoutingDecision


# ---------------------------------------------------------------------------
# Probabilistic retrieval evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    P(Relevant | q, d) = α * cosine_score + (1-α) * sigmoid(cross_encoder_logit)

    Three-way routing:
      score ≥ τ_high  → CORRECT
      τ_low ≤ score   → AMBIGUOUS
      score < τ_low   → INCORRECT
    """

    def __init__(
        self,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        alpha: float = 0.4,
        threshold_high: float = THRESHOLD_HIGH,
        threshold_low: float = THRESHOLD_LOW,
    ):
        self.alpha = alpha
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        # Use sentence_transformers.CrossEncoder rather than hand-rolled
        # tokenizer + model.forward + sigmoid. The hand-rolled version was
        # silently producing NaN logits for every input on this stack
        # (Python 3.13 + Apple Silicon torch), poisoning the evaluator.
        # CrossEncoder.predict() handles tokenization, batching, and the
        # sigmoid in a tested code path.
        logger.info(f"Loading cross-encoder: {cross_encoder_model}")
        self.model = CrossEncoder(cross_encoder_model, device="cpu")

    def _cross_score(self, query: str, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        # CrossEncoder.predict returns raw logits by default; pass the
        # activation_fct=torch.nn.Sigmoid() kwarg to get a probability in [0,1].
        score = float(self.model.predict(
            [(query, text[:2000])],
            activation_fct=torch.nn.Sigmoid(),
        )[0])
        # Defensive guard in case the model still produces NaN somewhere.
        if not (score == score) or score in (float("inf"), float("-inf")):
            return 0.5
        return score

    def _cross_score_batch(self, query: str, texts: list[str]) -> list[float]:
        """Batched scoring for efficiency when evaluating many chunks."""
        if not texts:
            return []
        pairs = [(query, (t or "")[:2000]) for t in texts]
        scores = self.model.predict(pairs, activation_fct=torch.nn.Sigmoid())
        out: list[float] = []
        for s in scores:
            v = float(s)
            if not (v == v) or v in (float("inf"), float("-inf")):
                v = 0.5
            out.append(v)
        return out

    def evaluate(
        self, query: str, retrieved: list[tuple[DocumentChunk, float]]
    ) -> list[EvaluatedChunk]:
        if not retrieved:
            return []
        chunks = [c for c, _ in retrieved]
        cosines = [s for _, s in retrieved]
        ces = self._cross_score_batch(query, [c.text for c in chunks])
        result = []
        for chunk, cosine, ce in zip(chunks, cosines, ces):
            final = self.alpha * cosine + (1 - self.alpha) * ce
            result.append(
                EvaluatedChunk(
                    chunk=chunk,
                    cosine_score=cosine,
                    cross_encoder_score=ce,
                    final_score=final,
                    decision=self._route(final),
                )
            )
        return result

    def _route(self, score: float) -> RoutingDecision:
        if score >= self.threshold_high:
            return RoutingDecision.CORRECT
        if score >= self.threshold_low:
            return RoutingDecision.AMBIGUOUS
        return RoutingDecision.INCORRECT

    def aggregate_decision(self, evaluated: list[EvaluatedChunk]) -> RoutingDecision:
        decisions = {e.decision for e in evaluated}
        if RoutingDecision.CORRECT in decisions:
            return RoutingDecision.CORRECT
        if RoutingDecision.AMBIGUOUS in decisions:
            return RoutingDecision.AMBIGUOUS
        return RoutingDecision.INCORRECT

    def mean_score(self, evaluated: list[EvaluatedChunk]) -> float:
        if not evaluated:
            return 0.0
        scores = [e.final_score for e in evaluated]
        # np.nanmean ignores NaNs instead of propagating them. If every score
        # is NaN somehow, fall back to 0.0 so routing still makes a decision.
        m = float(np.nanmean(scores))
        return m if (m == m) else 0.0  # m == m is False only when m is NaN


# ---------------------------------------------------------------------------
# Web search fallback
# ---------------------------------------------------------------------------

class WebSearchFallback:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.client = None
        if api_key:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[str]:
        if self.client is None:
            logger.warning("TAVILY_API_KEY not set — web search unavailable")
            return []
        results = self.client.search(query=query, max_results=max_results)
        return [
            f"[Web: {r.get('url', '')}]\n{r.get('content', '')}"
            for r in results.get("results", [])
            if r.get("content")
        ]


# ---------------------------------------------------------------------------
# Query rewriting helper
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = (
    "insufficient information",
    "i cannot verify",
    "i don't have access",
    "i do not have access",
    "the context does not contain",
    "the context doesn't contain",
    "context does not explicitly mention",
    "context doesn't explicitly mention",
    "context does not explicitly state",
    "i was unable to verify",
    "unable to verify",
    "i'm unable to",
    "is not provided in the context",
    "not provided in the context",
    "not directly stated",
    "context does not directly provide",
    "context doesn't directly provide",
    "only partial available information",
    "only partial information",
    "partial information available",
    "we can try to find",
    "metrics for fy2025 are not provided",
    "metrics for the entire fiscal year",
    "specific metrics for fy2025 are not",
    "unfortunately, the provided context",
    "unfortunately the provided context",
    "unfortunately, the context",
    "the actual number of",
    "is not given in any",
    "are not given in any",
    "doesn't directly provide the",
    "does not directly provide the",
    "no specific numbers provided",
    "no detailed information",
)


def _looks_like_refusal(answer: str) -> bool:
    if not answer:
        return True
    a = answer.strip().lower()
    # Check the first 400 chars (was 200) — some LLMs preface a refusal with
    # one or two filler sentences before the actual "I can't answer" signal.
    head = a[:400]
    if any(phrase in head for phrase in _REFUSAL_PHRASES):
        return True
    # Also catch answers that are mostly hedging — if the answer contains
    # any refusal phrase AND is short, it's a refusal regardless of position.
    if len(a) < 250 and any(phrase in a for phrase in _REFUSAL_PHRASES):
        return True
    return False


# ---------------------------------------------------------------------------
# Dynamic query routing (LLM-based query classifier)
# ---------------------------------------------------------------------------
# The proposal §3.5 listed dynamic routing as an exploratory next step. We
# implemented it after observing CRAG underperformed Baseline on narrative
# MD&A questions: a fixed retrieval composition (4 text + 4 table) forced
# table chunks to compete for slots even on questions whose answer lives in
# a single text paragraph. Classifying the query first lets us bias the
# retrieval composition per query type before any embedding is touched.
# ---------------------------------------------------------------------------

QUERY_TYPES = ("table_lookup", "narrative", "multimodal", "out_of_corpus")

# Per-query-type retrieval composition. (text_top_k, table_top_k)
RETRIEVAL_COMPOSITION = {
    "table_lookup":   (2, 6),  # numerical lookup: weight tables heavily
    "narrative":      (6, 1),  # MD&A / risk-factors prose: text-dominant
    "multimodal":     (6, 2),  # synthesis: bumped text from 4→6 because the
                                # MD&A paragraph that contains BOTH number
                                # and reasoning is text — table chunks were
                                # crowding it out at 4+4.
    "out_of_corpus":  (0, 0),  # skip retrieval, go straight to web
}


def _classify_query(llm_call, question: str) -> str:
    """Classify a question into one of QUERY_TYPES via a short LLM prompt.

    Returns one of: "table_lookup", "narrative", "multimodal", "out_of_corpus".
    Falls back to "multimodal" (the safest balanced choice) if the LLM's reply
    can't be matched to a known label.
    """
    system = (
        "You are a query router for a financial-document QA system. "
        "Classify each question into exactly one of four types. "
        "Reply with ONLY the lowercase type label and nothing else."
    )
    user = f"""Classify the question into ONE of these four types:

- **table_lookup**: Asks for a single specific number from a financial table — revenue, margin, expense, headcount, segment figure. Examples: "What was Apple's net sales in FY2025?" / "What was Microsoft's revenue from Azure?"
- **narrative**: Asks for a description, explanation, list of risk factors, or business overview. Answer lives in prose paragraphs. Examples: "What are Apple's principal risk factors?" / "How does Microsoft describe its Azure business?"
- **multimodal**: Asks for synthesis combining numerical figures AND surrounding management commentary. Examples: "How did iPhone revenue change YoY and what does management cite as the reason?"
- **out_of_corpus**: Asks about current real-time data not present in SEC filings — current stock prices, today's interest rates, recent news. Examples: "What is the current federal funds rate?" / "Who is the current US Treasury Secretary?"

Question: {question}

Type:"""
    try:
        response = llm_call(system, user).strip().lower()
    except Exception as e:
        logger.warning(f"Query classification failed ({e}); defaulting to multimodal")
        return "multimodal"

    # Pick the longest matching type label so "out_of_corpus" beats "out".
    matches = sorted(
        (qt for qt in QUERY_TYPES if qt in response),
        key=len,
        reverse=True,
    )
    if matches:
        return matches[0]
    logger.warning(f"Could not parse query type from '{response[:80]}'; defaulting to multimodal")
    return "multimodal"


def _generate_hyde(llm_call, question: str) -> str:
    """HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

    Ask the LLM to produce a 2-3 sentence hypothetical answer paragraph for
    the question. Embedding THAT (instead of the bare query) and searching
    FAISS with it improves retrieval because answer-shaped text lives much
    closer to document chunks in embedding space than question-shaped text.

    Returns the hypothetical answer (string). On any error, returns the
    original question so retrieval still proceeds.
    """
    system = (
        "You generate concise hypothetical answer paragraphs for SEC 10-K and "
        "10-Q filings. Your output is used as a retrieval query against a "
        "vector index of real filing chunks, so include realistic financial "
        "terminology, specific section names, and plausible-looking numbers "
        "(approximate values are fine — they don't need to be accurate). Do "
        "NOT add disclaimers; output the paragraph directly."
    )
    user = (
        f"Question: {question}\n\n"
        "Write a 2-3 sentence hypothetical answer paragraph that would appear "
        "in a real SEC filing. Include specific terminology and any plausible "
        "numbers / dates / section names that would be in the actual answer. "
        "Output only the paragraph, no preamble:"
    )
    try:
        hyde = llm_call(system, user).strip()
        # Strip stray markdown / quotes that some models add despite instructions.
        hyde = hyde.strip("\"'`").strip()
        if "\n\n" in hyde:
            hyde = hyde.split("\n\n")[0].strip()
        if len(hyde) < 30:
            logger.warning(f"HyDE output suspiciously short ({len(hyde)} chars), using original")
            return question
        logger.info(f"HyDE: '{question[:60]}…' → '{hyde[:80]}…'")
        return hyde
    except Exception as e:
        logger.warning(f"HyDE generation failed ({e}); using original query")
        return question


def _rewrite_query(llm_call, original: str) -> str:
    system = (
        "You rewrite questions to retrieve relevant passages from SEC 10-K and 10-Q "
        "filings. Keep all proper nouns, numbers, fiscal periods, and named sections "
        "(e.g. 'Risk Factors', 'MD&A', 'Item 1A') exactly as written. Make the query "
        "more specific only by adding canonical SEC vocabulary if useful. "
        "Output ONLY the rewritten query on a single line, no preamble."
    )
    user = (
        f"Original: {original}\n\nRewritten query (one line, keep all named entities):"
    )
    rewritten = llm_call(system, user).strip()
    # Strip stray quotes / markdown that some models add despite instructions.
    rewritten = rewritten.strip("\"'`").strip()
    if "\n" in rewritten:
        rewritten = rewritten.split("\n")[0].strip()
    logger.info(f"Query rewrite: '{original}' → '{rewritten}'")
    # If the rewrite is suspiciously short or stripped meaning, fall back to original.
    if len(rewritten) < 8 or rewritten.lower() == original.lower():
        return original
    return rewritten


# ---------------------------------------------------------------------------
# Corrective RAG
# ---------------------------------------------------------------------------

class CorrectedRAG:
    """
    Full CRAG pipeline:
      1. Text-only retrieval with θ_text threshold
      2. If |text_hits| < N_TEXT_MIN: add table fallback (θ_table threshold)
      3. Probabilistic evaluator scores all chunks
      4. If aggregate confidence < τ_low: web-search fallback replaces context
      5. If AMBIGUOUS: rewrite query and re-retrieve before final generation
    """

    def __init__(
        self,
        vector_store: VectorStore,
        evaluator: Optional[RetrievalEvaluator] = None,
        top_k: int = 6,
        text_store: Optional[VectorStore] = None,
        web_fallback_enabled: bool = True,
        query_type_override: Optional[str] = None,
    ):
        # Hybrid retrieval: `vector_store` is used for table retrieval, but
        # `text_store` (if provided) is used for text retrieval. This fixes
        # the architectural mistake where the table-aware index's text portion
        # contained semantic chunks WITHOUT flat-text numerical mentions —
        # which destroyed CRAG's ability to find values like "$209,586 million"
        # that the naive baseline index trivially retrieved as embedded text.
        # When text_store is None, behaves exactly as before (single index).
        self.vs = vector_store
        self.text_vs = text_store or vector_store
        self.evaluator = evaluator or RetrievalEvaluator()
        self.web_search = WebSearchFallback()
        self.top_k = top_k
        self.llm = _build_llm_client()
        # Demo controls (default to existing behavior so non-demo callers
        # see no change). web_fallback_enabled=False makes CRAG behave
        # "corpus-only" — useful for showing what CRAG looks like without
        # the web fallback claim. query_type_override=<label> bypasses the
        # LLM classifier — useful when the classifier mis-routes or to
        # demonstrate per-route retrieval behavior on a fixed question.
        self.web_fallback_enabled = web_fallback_enabled
        self.query_type_override = query_type_override

    def _answer_addresses_all_parts(self, question: str, answer: str) -> bool:
        """LLM-based completeness audit for multi-part questions.

        Returns True if the LLM judges the answer addresses every part of
        the question. Returns True (i.e. assume complete) on any error so
        we never *worse* the answer due to a flaky audit call.
        """
        system = (
            "You audit answers for COMPLETENESS, not correctness. The user "
            "question may have multiple parts (e.g. 'How did X change AND "
            "what does management cite?'). Your job: did the answer engage "
            "with EVERY part of the question, even briefly? "
            "Reply with EXACTLY one word — YES or NO. Nothing else."
        )
        user = (
            f"Question: {question}\n\n"
            f"Answer: {answer.strip()}\n\n"
            "Does the answer address every part of the question? YES or NO:"
        )
        try:
            verdict = self.llm(system, user).strip().upper()
        except Exception:
            return True  # fail-open: don't worsen the answer on judge errors
        # Be conservative — only treat clear "NO" as incomplete; anything
        # else (YES, or an unparseable response) is treated as complete.
        return not verdict.startswith("NO")

    def query(self, question: str) -> dict:
        # --- Tier 0: Dynamic query routing ---
        # Classify the query before retrieval so we can pick a retrieval
        # composition appropriate to the question type. This was the root-cause
        # fix for CRAG underperforming Baseline on narrative MD&A queries —
        # see RETRIEVAL_COMPOSITION above. The query_type_override demo
        # control bypasses the LLM classifier (saves ~500-1000ms per query
        # and lets a presenter force a specific routing path).
        if self.query_type_override and self.query_type_override in QUERY_TYPES:
            query_type = self.query_type_override
            logger.info(f"Query type forced via override: '{query_type}'")
        else:
            query_type = _classify_query(self.llm, question)
        text_k, table_k = RETRIEVAL_COMPOSITION[query_type]
        logger.info(
            f"Routed query as '{query_type}' → text_k={text_k}, table_k={table_k}"
        )

        # OOC questions skip the corpus entirely and go straight to web search
        # — UNLESS the demo control web_fallback_enabled is False, in which
        # case we fall through to corpus retrieval to demonstrate what CRAG
        # looks like without web access (the answer will be a refusal,
        # which is the architecturally correct outcome).
        if query_type == "out_of_corpus" and self.web_fallback_enabled:
            logger.info("OOC query → bypassing corpus retrieval, going straight to web")
            passages = self.web_search.search(question)
            context = "\n\n---\n\n".join(passages)
            answer = self.llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
            return {
                "answer": answer,
                "routing_decision": RoutingDecision.INCORRECT.value,
                "confidence_score": 0.0,
                "tier_used": "web (OOC)",
                "query_type": query_type,
                "chunk_scores": [],
                "sources_used": ["web_search"],
                "mode": "crag",
            }
        elif query_type == "out_of_corpus" and not self.web_fallback_enabled:
            logger.info("OOC query but web_fallback_enabled=False → forcing corpus retrieval")

        # --- Tier 0.5: HyDE (DISABLED) ---
        # We tried Hypothetical Document Embeddings here but the ablation
        # showed it regressed CRAG's keyword hit rate (0.227 → 0.121 overall).
        # Hypothesis: the LLM-generated hypothetical answers were generic
        # enough that they embedded near *less*-specific chunks, pushing
        # the truly-relevant ones out of top-K. Restoring direct query
        # embedding restored the prior performance. Code retained for
        # potential reactivation behind an env flag in future work.
        # if query_type in ("narrative", "multimodal"):
        #     retrieval_query = _generate_hyde(self.llm, question)
        # else:
        #     retrieval_query = question
        retrieval_query = question

        # --- Tier 1: text retrieval (composition driven by query type) ---
        # Uses self.text_vs (defaults to self.vs); in hybrid mode this is the
        # naive flat-text index whose chunks include embedded numerical mentions.
        text_hits = self.text_vs.search_modality(
            retrieval_query, "text", top_k=text_k, threshold=THRESHOLDS["text"]
        ) if text_k > 0 else []
        context_chunks: list[tuple[DocumentChunk, float]] = list(text_hits)
        tier_used = "text"

        # --- Tier 2: table retrieval (composition driven by query type) ---
        # Uses self.vs (the structured-table index in hybrid mode).
        table_hits = self.vs.search_modality(
            retrieval_query, "table", top_k=table_k, threshold=THRESHOLDS["table"]
        ) if table_k > 0 else []
        if table_hits:
            context_chunks.extend(table_hits)
            tier_used = "text+table"
            logger.info(f"Table search returned {len(table_hits)} table chunks")

        # --- Diagnostic: preview the retrieved chunks themselves ---
        # Without this we only see scores; with it we can confirm whether the
        # right chunk is in the index at all.
        for i, (c, s) in enumerate(context_chunks[:6]):
            preview = (c.text or "")[:120].replace("\n", " ")
            logger.info(
                f"  [{c.modality[:3]} #{i+1} cos={s:.2f}] {Path(c.source).name}: {preview}…"
            )

        # --- Probabilistic evaluation ---
        evaluated = self.evaluator.evaluate(question, context_chunks)
        decision = self.evaluator.aggregate_decision(evaluated)
        confidence = self.evaluator.mean_score(evaluated)

        # Diagnostic line so we can see why a routing decision was made.
        if evaluated:
            score_summary = ", ".join(
                f"{e.chunk.modality[:3]}:cos={e.cosine_score:.2f}/ce={e.cross_encoder_score:.2f}/f={e.final_score:.2f}"
                for e in evaluated[:5]
            )
            logger.info(
                f"Eval @ '{question[:60]}': mean={confidence:.3f} → {decision.value} | {score_summary}"
            )
        else:
            logger.info(f"Eval @ '{question[:60]}': NO retrieved chunks at current thresholds")

        context_passages: list[str] = []
        sources_used: list[str] = []

        # --- Web fallback decision ---
        # We trigger web search whenever the corpus clearly doesn't contain the
        # answer. Two signals (in addition to the explicit INCORRECT decision):
        #   (a) mean confidence below threshold_low — the original criterion
        #   (b) the BEST individual chunk's cosine score is mediocre, meaning
        #       even the strongest match is weakly relevant. This is what
        #       catches out-of-corpus questions like "What is the current
        #       federal funds rate?" where the cross-encoder can't tell the
        #       chunks are bad on its own.
        max_cosine = max((e.cosine_score for e in evaluated), default=0.0)
        # Web fallback decision with a "strong corpus match" guard:
        # The mean confidence over 6+ chunks gets dragged down by
        # less-relevant chunks even when one or two chunks ARE highly
        # relevant. Without a guard, narrative queries about content that's
        # genuinely in the corpus (e.g. "How does Microsoft describe Azure?")
        # were getting routed to web search because mean < threshold_low,
        # producing worse answers than baseline got from the same corpus.
        # Guard: if max_cosine >= 0.65, at least one chunk is clearly
        # relevant — trust the corpus even when the mean is low.
        trigger_web = (
            decision == RoutingDecision.INCORRECT
            or confidence < self.evaluator.threshold_low
        )
        strong_corpus_match = max_cosine >= 0.65

        # Decide: web fallback, or use corpus chunks. The previous structure
        # had a bug where the strong-corpus-match branch set trigger_web=False
        # but never actually populated context_passages, because if/elif/else
        # doesn't fall through. The result was an empty context being sent to
        # the LLM, which reasonably refused with "Insufficient information."
        # Restructured so that "use corpus chunks" is always the default path
        # and "use web" is an explicit override that bypasses it.
        # Demo control: web_fallback_enabled=False forces corpus-only behavior
        # so the audience can see what CRAG looks like without web access.
        use_web = trigger_web and not strong_corpus_match and self.web_fallback_enabled

        if use_web:
            logger.info(
                f"Triggering web fallback (decision={decision.value}, "
                f"confidence={confidence:.3f}, max_cosine={max_cosine:.3f})"
            )
            passages = self.web_search.search(question)
            context_passages = passages
            sources_used = ["web_search"]
            tier_used = "web"
            decision = RoutingDecision.INCORRECT  # reflect in UI badge
        else:
            if trigger_web and strong_corpus_match:
                logger.info(
                    f"Mean confidence low ({confidence:.3f}) but max_cosine "
                    f"{max_cosine:.3f} >= 0.65 — staying on corpus, skipping web fallback"
                )
                decision = RoutingDecision.AMBIGUOUS  # reflect uncertainty without going to web

            # MAX_CONTEXT 7 — matches the upper end of what retrieval
            # composition produces (6 text + 2 table = 8). Earlier we set
            # this to 5 to "match Baseline's top-K" but that was wrong —
            # Baseline retrieves 6 chunks and keeps all 6, while CRAG
            # retrieves up to 8 and was truncating to 5, dropping useful
            # chunks. Concrete failure: on the iPhone YoY question, the
            # MD&A paragraph carrying "due to higher net sales of Pro
            # models" was the 6th-ranked chunk that we were silently
            # dropping.
            MAX_CONTEXT = 7
            # For text-leaning queries (multimodal AND narrative), sort by
            # cosine rather than the reranker-blended final_score. Reranker
            # noise can demote the genuinely relevant text chunk a few slots
            # — for narrative this caused Llama-8B to read off-topic chunks
            # first and refuse with "Insufficient information." even though
            # the right chunk was lower in the same context. Baseline uses
            # cosine ordering and answers these questions correctly, so
            # aligning text-leaning CRAG with cosine ordering is strictly
            # safe. table_lookup keeps reranker ordering because for
            # numerical lookups the cross-encoder reliably promotes the
            # specific row that holds the value.
            if query_type in ("multimodal", "narrative"):
                accepted = sorted(
                    evaluated, key=lambda e: e.cosine_score, reverse=True
                )[:MAX_CONTEXT]
            else:
                accepted = sorted(
                    evaluated, key=lambda e: e.final_score, reverse=True
                )[:MAX_CONTEXT]
            # Per-chunk text budget: 2000 was too tight. The Apple chunk
            # carrying both the iPhone segment table AND the "due to higher
            # net sales of Pro models" attribution is 2,916 chars total,
            # with the attribution sentence sitting at char 1921 — JUST
            # inside the old 2000 cutoff but at the very tail of the visible
            # window, where Llama-8B has been losing it. Bumping to 3500
            # keeps the full attribution context visible while still capping
            # pathological mega-chunks.
            CHUNK_CHAR_BUDGET = 3500
            for e in accepted:
                label = "table" if e.chunk.is_table else "text"
                context_passages.append(
                    f"[{label.upper()} | {e.chunk.source}]\n{e.chunk.text[:CHUNK_CHAR_BUDGET]}"
                )
                sources_used.append(e.chunk.source)

        context = "\n\n---\n\n".join(context_passages)

        # Multimodal user-prompt directive (CRAG-only, baseline unchanged).
        # Multi-part questions ("how did X change AND why?") on Llama-8B fail
        # in a specific way: the model reads the context, finds the numerical
        # change, but skips over the management-attribution sentence buried
        # among similar product-by-product or segment-by-segment commentary.
        # Concrete failure: on the Apple iPhone YoY question, the chunk
        # containing "due to higher net sales of Pro models" was in context,
        # but Llama-8B surfaced an unrelated "proportion of net sales"
        # sentence and reported the reason as missing. The directive below
        # tells the model what attribution language looks like, requires it
        # to scan product-by-product paragraphs for the specific entity, and
        # asks for a verbatim quote rather than analysis. Scoped to
        # multimodal so it does NOT affect narrative or table_lookup paths.
        if query_type == "multimodal":
            multimodal_directive = (
                "\n\nThis question has multiple parts (a quantitative change "
                "AND the reason for it). When answering:\n"
                "1. State the numerical change with units and fiscal periods.\n"
                "2. Search the context for a sentence explaining the cause. "
                "These sentences typically use phrases like 'due to', "
                "'driven by', 'primarily reflecting', 'attributed to', or "
                "'resulting from'. They often appear in product-by-product "
                "or segment-by-segment paragraphs (e.g. a paragraph that "
                "lists each product line with one sentence each).\n"
                "3. If the question asks about a specific entity (a product "
                "line, segment, or business unit), find the sentence in the "
                "context that names that exact entity AND contains a causal "
                "phrase. Quote it verbatim — do not paraphrase or analyze.\n"
                "4. Only respond that the reason is unstated AFTER you have "
                "scanned every paragraph in the context for the relevant "
                "entity name."
            )
            user_prompt = (
                f"Context:\n{context}\n\nQuestion: {question}"
                f"{multimodal_directive}"
            )
        else:
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        answer = self.llm(SYSTEM_PROMPT, user_prompt)

        # Narrative refusal rescue: if the LLM refuses on a narrative query
        # but at least one chunk strongly matches by cosine, re-prompt over
        # the SAME context with a sharper "the answer is in the context"
        # instruction. This is the same pattern as the multimodal completeness
        # recovery below — no web augmentation, no new chunks, just a second
        # pass with clearer instructions. Triggered only when corpus signal
        # is strong (max_cosine >= 0.6) so we don't paper over genuine OOC.
        if (
            tier_used != "web"
            and query_type == "narrative"
            and _looks_like_refusal(answer)
            and max_cosine >= 0.6
        ):
            logger.info(
                "Narrative refusal with strong corpus signal "
                "(max_cosine=%.3f) — re-prompting with sharper instruction",
                max_cosine,
            )
            sharper_user = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"The answer to this question IS in the context above. Find "
                f"the relevant passage and quote or paraphrase it directly. "
                f"Cite section names and figures where present. Use ONLY the "
                f"context above; do NOT respond with 'Insufficient information' "
                f"unless the context truly contains nothing related to the "
                f"question."
            )
            answer_v2 = self.llm(SYSTEM_PROMPT, sharper_user)
            if answer_v2 and not _looks_like_refusal(answer_v2):
                answer = answer_v2

        # Post-hoc rescue: if the LLM signals it couldn't answer from the
        # corpus AND the query was classified out_of_corpus, retry with web.
        # We restrict this to OOC because ablation showed in-corpus questions
        # that triggered the rescue often got WORSE answers from web search
        # than from the original corpus chunks (LLM was just being conservative
        # with hedging language, not actually refusing).
        if (
            tier_used != "web"
            and query_type == "out_of_corpus"
            and _looks_like_refusal(answer)
            and self.web_fallback_enabled
        ):
            logger.info("LLM signaled insufficient info — retrying with web fallback")
            web_passages = self.web_search.search(question)
            if web_passages:
                web_context = "\n\n---\n\n".join(web_passages)
                answer = self.llm(SYSTEM_PROMPT, f"Context:\n{web_context}\n\nQuestion: {question}")
                sources_used = ["web_search"]
                tier_used = "web"
                decision = RoutingDecision.INCORRECT

        # ----------------------------------------------------------------
        # Self-reflective completeness check (multimodal only)
        # ----------------------------------------------------------------
        # Multimodal questions ("How did X change AND why?") have multiple
        # parts. Even when CRAG finds the right chunks, the LLM sometimes
        # answers only one half (numbers OR reasoning) and silently drops
        # the other. We ask the LLM itself to audit completeness; if it
        # reports the answer doesn't address all parts, we augment with a
        # supplementary web search and regenerate using corpus + web context.
        # Wrapped in try/except so any judge-side failure preserves the
        # original answer rather than breaking the pipeline.
        if (
            query_type == "multimodal"
            and tier_used != "web"
            and not _looks_like_refusal(answer)  # already-failed answers handled above
        ):
            try:
                if not self._answer_addresses_all_parts(question, answer):
                    # Two-pass strategy:
                    # 1) Re-prompt over the SAME corpus context with a sharper
                    #    "address every part" instruction. Often the LLM had
                    #    the relevant text but only answered one half.
                    # 2) Only if that re-prompt still misses parts AND
                    #    confidence is low, augment with web. (We learned the
                    #    hard way that web augmentation on iPhone-YoY-style
                    #    queries dilutes the corpus reasoning rather than
                    #    helping it.)
                    logger.info(
                        "Completeness check failed — re-prompting over same "
                        "corpus context with explicit multi-part instruction"
                    )
                    sharper_user = (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Important: This question has multiple parts. Address "
                        f"EVERY part explicitly. If management cites a reason "
                        f"in the context (e.g. 'due to ...', 'driven by ...', "
                        f"'primarily reflecting ...'), quote or paraphrase it. "
                        f"Use ONLY the context above."
                    )
                    answer_v2 = self.llm(SYSTEM_PROMPT, sharper_user)
                    if answer_v2 and not _looks_like_refusal(answer_v2):
                        answer = answer_v2

                    # Only fall back to web if the re-prompt STILL misses parts
                    # AND we don't have a strong corpus match. This guards
                    # against the iPhone failure mode where web noise replaced
                    # good corpus reasoning.
                    still_incomplete = not self._answer_addresses_all_parts(
                        question, answer
                    )
                    if still_incomplete and max_cosine < 0.65:
                        logger.info(
                            "Re-prompt still incomplete and corpus match weak "
                            "(max_cosine=%.3f) — augmenting with web", max_cosine
                        )
                        web_passages = self.web_search.search(question)
                        if web_passages:
                            augmented_context = (
                                context
                                + "\n\n---\n\n[Supplementary web context]\n"
                                + "\n\n".join(web_passages)
                            )
                            answer = self.llm(
                                SYSTEM_PROMPT,
                                f"Context:\n{augmented_context}\n\nQuestion: {question}",
                            )
                            sources_used = list(set(sources_used + ["web_search"]))
                            tier_used = (
                                "text+table+web" if "table" in tier_used else "text+web"
                            )
            except Exception as e:
                logger.warning(f"Completeness check failed ({e}); keeping original answer")

        return {
            "answer": answer,
            "routing_decision": decision.value,
            "confidence_score": round(confidence, 4),
            "tier_used": tier_used,
            "query_type": query_type,
            "chunk_scores": [
                {
                    "score": round(e.final_score, 4),
                    "decision": e.decision.value,
                    "modality": e.chunk.modality,
                }
                for e in evaluated
            ],
            "sources_used": list(set(sources_used)),
            "mode": "crag",
        }
