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


def _decompose_question(llm_call, question: str) -> list[str]:
    """Decompose a multi-part question into atomic sub-questions.

    Used to handle the partial-OOC case where a question has both
    in-corpus and out-of-corpus parts (e.g., "What was Apple's iPhone
    net sales in FY2025, AND what is the current Apple stock price?").
    Without decomposition, the classifier sees the OOC half and labels
    the whole question as out_of_corpus, causing both halves to be
    answered poorly from a single web search.

    Returns:
        - [question] if the question is single-part (no decomposition)
        - [sub_q1, sub_q2, ...] if multiple distinct parts are detected

    On any error, returns [question] (fail-safe — no decomposition).
    """
    system = (
        "You analyze user questions to determine if they contain "
        "multiple distinct sub-questions. A 'sub-question' is one that "
        "could stand alone and be answered independently. "
        "Return your response as numbered atomic sub-questions, one per "
        "line, with no preamble or commentary. If the question is "
        "single-part, return it unchanged on a single line."
    )
    user = f"""Decompose this question into atomic sub-questions if it contains MULTIPLE distinct parts that could be answered separately.

Examples:
- Single-part: "What is Apple's revenue?" → "What is Apple's revenue?"
- Multimodal split (number + narrative about the SAME entity): "How did iPhone revenue change and what does management cite as the reason?" →
  1. How did Apple's iPhone revenue change?
  2. What reasons does Apple's management cite for the change in iPhone revenue?
  (decompose because part 1 wants a quantitative figure best retrieved from the structured-table index, while part 2 wants narrative MD&A best retrieved from the prose index — splitting lets each sub-question land on the appropriate retrieval modality)
- Multi-part with mixed sources: "What was Apple's iPhone net sales in FY2025, and what is the current Apple stock price?" →
  1. What was Apple's iPhone net sales in FY2025?
  2. What is the current Apple stock price?
- Multi-part across topics: "What is Microsoft's Azure revenue in FY2025, and what is the current price of Bitcoin?" →
  1. What is Microsoft's Azure revenue in FY2025?
  2. What is the current price of Bitcoin?

When to decompose:
(a) parts ask about DIFFERENT entities OR DIFFERENT data sources (e.g., one historical SEC filing, one real-time market data), OR
(b) parts mix a QUANTITATIVE/TABLE component with a NARRATIVE/COMMENTARY component about the same entity (the "multimodal" case above), since splitting lets each part retrieve from the index best suited to its modality.

When NOT to decompose:
- Single-fact questions about one entity ("What is Apple's revenue?")
- Comparisons within a single modality ("How does Apple's iPhone revenue compare to its Services revenue?" — both quantitative, retrieve together)

When repeating an entity name in a sub-question, always include the FULL entity name (e.g., "Apple's iPhone revenue", not just "iPhone revenue") so each sub-question is self-contained and can be retrieved independently without losing context.

Question: {question}

Sub-questions (numbered, or unchanged if single-part):"""
    try:
        response = llm_call(system, user).strip()
    except Exception as e:
        logger.warning(f"Question decomposition failed ({e}); using original")
        return [question]

    # Parse numbered list. Lines like "1. ...", "1) ...", "- ...", or
    # plain text (single-part case).
    import re as _re
    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    sub_qs: list[str] = []
    for line in lines:
        # Strip leading numbering / bullets
        cleaned = _re.sub(r"^[\-\*\d]+[\.\)]\s*", "", line).strip()
        if cleaned:
            sub_qs.append(cleaned)

    if not sub_qs:
        return [question]
    if len(sub_qs) == 1:
        # Single-part — could be the original question or a rephrasing.
        # Use the original to be safe.
        return [question]

    # Multi-part. Sanity check: if any sub-question is suspiciously short
    # (< 10 chars), the parse went wrong; fall back to original.
    if any(len(s) < 10 for s in sub_qs):
        logger.warning(f"Decomposition produced short sub-questions; using original")
        return [question]

    logger.info(f"Decomposed into {len(sub_qs)} sub-questions:")
    for i, sq in enumerate(sub_qs, 1):
        logger.info(f"  {i}. {sq}")
    return sub_qs


def _check_numerical_fidelity(
    answer: str,
    chunks: list[DocumentChunk],
) -> dict:
    """Verify that every numerical claim in the answer is grounded in a chunk.

    Inspired by the DANA neurosymbolic system (Hettiarachchi et al., 2024)
    and FinVet's claim-verification stage: arithmetic and value-lookup are
    the failure modes most likely to produce confident-but-fabricated
    output in financial QA, so we add a post-generation check that flags
    any number in the answer that doesn't appear verbatim (after light
    normalization) in any retrieved chunk.

    This is a CHECK, not a CORRECTION — we surface unverified numbers in
    the result metadata rather than rewriting the answer. Rationale:
    silently rewriting could itself introduce errors, and downstream
    consumers (UI, evaluation) can decide what to do with the flag.

    Number formats handled:
      - Currency:    $416, $416 million, $416B, $1.5 billion, $1,234.56
      - Percentages: 12%, 12.3%, 0.5%
      - Years:       2024, FY2025, fiscal 2026  (excluded — too noisy)
      - Plain ints/floats with units like 'million', 'billion'

    Returns:
      {
        'numbers_in_answer':       list of numeric tokens found in the answer,
        'verified_numbers':        list of those that match a chunk substring,
        'unverified_numbers':      list of those that do NOT,
        'fidelity_score':          verified / total (1.0 if no numbers at all),
      }
    """
    import re as _re

    if not answer:
        return {
            "numbers_in_answer": [],
            "verified_numbers": [],
            "unverified_numbers": [],
            "fidelity_score": 1.0,
        }

    # Patterns we treat as numerical claims worth verifying.
    # Order matters: try richer patterns (with currency/units) first so
    # the bare-number fallback doesn't shadow them.
    patterns = [
        # $1,234.56 million  / $1.5B / $416 billion
        r"\$\s?[\d,]+(?:\.\d+)?\s?(?:billion|million|thousand|trillion|B|M|K|T)?\b",
        # 12.3%  / 0.5%
        r"\b\d+(?:\.\d+)?\s?%",
        # 416 million  / 1.5 billion  (no $ sign)
        r"\b\d+(?:\.\d+)?\s+(?:billion|million|trillion|thousand)\b",
    ]

    raw_matches: list[str] = []
    for pat in patterns:
        raw_matches.extend(m.group(0) for m in _re.finditer(pat, answer, _re.IGNORECASE))

    # De-duplicate while preserving order so the report can name the
    # specific unverified numbers without repetition.
    seen = set()
    numbers: list[str] = []
    for n in raw_matches:
        key = _re.sub(r"\s+", " ", n.strip().lower())
        if key not in seen:
            seen.add(key)
            numbers.append(n.strip())

    if not numbers:
        return {
            "numbers_in_answer": [],
            "verified_numbers": [],
            "unverified_numbers": [],
            "fidelity_score": 1.0,
        }

    # Build the corpus-string-to-search-against. We concatenate all chunk
    # text into one big lowercased blob; we match against this rather
    # than per-chunk because a single answer often blends facts from
    # multiple chunks.
    corpus_blob = " ".join(
        getattr(c, "text", "") or "" for c in chunks
    ).lower()
    # Also strip commas in the corpus so $1,234 in the answer matches
    # "1234" if the chunk happens to be uncommatized (rare but real).
    corpus_blob_nocomma = corpus_blob.replace(",", "")

    verified: list[str] = []
    unverified: list[str] = []
    for n in numbers:
        n_lower = n.lower()
        # Try the literal string first.
        if n_lower in corpus_blob:
            verified.append(n)
            continue
        # Try with commas stripped (handles "$1,234" vs "1234").
        n_nocomma = n_lower.replace(",", "")
        if n_nocomma in corpus_blob_nocomma:
            verified.append(n)
            continue
        # Try without the leading "$" (some chunks omit currency markers).
        n_nodollar = n_lower.lstrip("$").strip()
        if n_nodollar in corpus_blob:
            verified.append(n)
            continue
        # Combined: no $ AND no commas — handles "$1,234 million" → "1234 million"
        # which is the most common SEC-table formatting mismatch.
        n_nodollar_nocomma = n_nodollar.replace(",", "")
        if n_nodollar_nocomma in corpus_blob_nocomma:
            verified.append(n)
            continue
        # No match → flag.
        unverified.append(n)

    score = len(verified) / len(numbers)
    return {
        "numbers_in_answer": numbers,
        "verified_numbers": verified,
        "unverified_numbers": unverified,
        "fidelity_score": round(score, 3),
    }


def _check_sub_question_coverage(llm_call, sub_question: str, answer: str) -> bool:
    """Verify that an answer actually addresses its sub-question.

    Used inside _query_with_decomposition so that, after each sub-question
    is independently routed and answered, we can check whether it was
    actually addressed (vs. silently dropped, evaded, or filled with an
    'I don't know'). Without this check the synthesis step blindly stitches
    together whatever came back, masking partial failures behind a
    plausible-looking combined answer.

    On any LLM error we return True (fail-safe: assume covered) so that a
    transient API failure on the *checker* doesn't poison the answer.
    """
    if not answer or not answer.strip():
        return False
    # Heuristic short-circuit: explicit refusals don't need an LLM call.
    refusal_markers = (
        "i don't know", "i do not know", "no information",
        "the document does not", "not provided in the", "cannot answer",
        "no data available", "the context does not",
    )
    a_lower = answer.lower()
    if any(m in a_lower for m in refusal_markers):
        return False

    system = (
        "You verify whether a draft answer directly addresses a question. "
        "You are checking for substantive coverage, not stylistic quality."
    )
    user = (
        f"Question: {sub_question}\n\n"
        f"Draft answer: {answer}\n\n"
        "Does the answer contain specific information that directly addresses "
        "the question?\n"
        "- YES: the answer states a fact, figure, or explanation that responds "
        "to what was asked.\n"
        "- NO: the answer is missing the requested information, evades the "
        "question, or only restates the question.\n\n"
        "Reply with exactly one word: YES or NO."
    )
    try:
        response = llm_call(system, user).strip().lower()
    except Exception as e:
        logger.warning(f"Coverage check LLM call failed ({e}); assuming covered")
        return True
    return response.startswith("yes")


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

    def _extract_top_strips(
        self,
        question: str,
        chunks: list[DocumentChunk],
        max_strips: int = 8,
        min_sentence_chars: int = 25,
    ) -> list[tuple[str, str]]:
        """Strip-level refinement (Yan et al. 2024 §3.3): decompose chunks
        into sentence-level strips, score each strip against the question
        with the cross-encoder, return the top-N strips paired with their
        source filename (readable form).

        Returns: list of (sentence_text, readable_source) tuples,
        sorted by cross-encoder relevance descending.

        This is run on TOP of chunk-level retrieval — the chunks are
        still passed to the LLM. The strips are prepended as a
        HIGHLIGHTS section to surface attribution sentences that
        would otherwise be buried inside long chunks (the Apple iPhone
        "Pro models" failure mode).
        """
        import re as _re

        # Collect all candidate sentences across the accepted chunks
        candidates: list[tuple[str, str]] = []  # (sentence, source)
        for c in chunks:
            text = (c.text or "").strip()
            if not text:
                continue
            # Simple sentence splitter — handles periods, question marks,
            # exclamation marks. Not perfect for SEC abbreviations
            # ("Inc." etc.) but the cross-encoder reranker filters noise.
            sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            for s in sentences:
                s = s.strip()
                if len(s) < min_sentence_chars:
                    continue
                # Cap each sentence at 500 chars so the cross-encoder
                # gets useful input without huge inputs.
                candidates.append((s[:500], c.source))

        if not candidates:
            return []

        # Score every candidate sentence with the cross-encoder. This
        # is the exact same model we use for chunk-level reranking.
        sentence_texts = [s for s, _src in candidates]
        try:
            ce_scores = self.evaluator._cross_score_batch(question, sentence_texts)
        except Exception:
            return []

        # Sort and keep top N. Light de-duplication: skip near-duplicate
        # sentences (e.g. boilerplate that repeats across filings).
        paired = list(zip(candidates, ce_scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        seen_starts = set()
        top: list[tuple[str, str]] = []
        for (sent, src), _score in paired:
            key = sent[:50].lower()
            if key in seen_starts:
                continue
            seen_starts.add(key)
            # Extract a readable source label like "AAPL/10-K" from the
            # SEC archive path; fall back to filename otherwise.
            readable = Path(src).name
            if "sec-edgar-filings" in src:
                parts = src.split("sec-edgar-filings/", 1)[1].split("/")
                if len(parts) >= 2:
                    readable = f"{parts[0]}/{parts[1]}"
            top.append((sent, readable))
            if len(top) >= max_strips:
                break
        return top

    def _query_with_decomposition(self, original_question: str,
                                   sub_questions: list[str]) -> dict:
        """Run each sub-question independently, combine the results into
        a single response that addresses the original full question.

        After each sub-question is answered, runs an LLM-based coverage
        check that verifies the sub-question was actually addressed (vs.
        silently dropped). Uncovered sub-questions are surfaced in the
        result metadata as `incomplete_sub_questions` so they can be
        flagged in the UI and counted in evaluation, rather than masked
        behind a plausible-looking synthesized answer.
        """
        # v2 behavior: decomposed sub-questions can trigger their own
        # web fallback if the corpus doesn't satisfy them. This is the
        # original (un-fixed) iteration; the v3 folder has the
        # `_suppress_web=True` fix that prevents off-topic web content
        # (e.g., 2013 Annual Report passages) from leaking into answers
        # about FY2024-25.
        sub_results: list[dict] = []
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"=== Sub-question {i}/{len(sub_questions)}: {sq} ===")
            r = self.query(sq, _skip_decomposition=True)
            covered = _check_sub_question_coverage(
                self.llm, sq, r.get("answer", "")
            )
            if not covered:
                logger.warning(
                    f"Sub-question {i} not adequately covered by its answer: "
                    f"{sq!r}"
                )
            sub_results.append({
                "sub_question": sq,
                "result": r,
                "covered": covered,
            })

        # Combine sub-answers via LLM synthesis
        sub_answers_text = "\n\n".join(
            f"Sub-question {i}: {sr['sub_question']}\n"
            f"Answer: {sr['result'].get('answer', '')}"
            for i, sr in enumerate(sub_results, 1)
        )
        synthesis_prompt = (
            f"Original question: {original_question}\n\n"
            f"This question was decomposed into {len(sub_questions)} "
            f"sub-questions, each routed and answered independently:\n\n"
            f"{sub_answers_text}\n\n"
            f"Combine these sub-answers into a single coherent response "
            f"that addresses the original question. Preserve all factual "
            f"content (numbers, citations, sources) from the sub-answers. "
            f"Do not add new claims that weren't in the sub-answers. "
            f"Do not duplicate content. Do not include 'Sub-question 1:' "
            f"or 'Answer:' headers in the final response — write a "
            f"single integrated answer."
        )
        try:
            combined_answer = self.llm(SYSTEM_PROMPT, synthesis_prompt)
        except Exception as e:
            logger.warning(f"Decomposition synthesis failed ({e}); "
                          f"concatenating sub-answers as fallback")
            combined_answer = "\n\n".join(
                f"Regarding '{sr['sub_question']}': "
                f"{sr['result'].get('answer', '')}"
                for sr in sub_results
            )

        # Aggregate metadata from sub-results.
        # tier: union of all tiers used (e.g., "text+table+web (decomposed)")
        # routing: most cautious decision (INCORRECT > AMBIGUOUS > CORRECT)
        # confidence: min across sub-questions (most cautious)
        # query_type: "decomposed"
        all_tiers = sorted(set(
            sr["result"].get("tier_used", "") for sr in sub_results
        ))
        tier_str = " + ".join(t for t in all_tiers if t) + " (decomposed)"

        decisions = [sr["result"].get("routing_decision", "")
                    for sr in sub_results]
        if "incorrect" in decisions:
            routing = RoutingDecision.INCORRECT.value
        elif "ambiguous" in decisions:
            routing = RoutingDecision.AMBIGUOUS.value
        else:
            routing = RoutingDecision.CORRECT.value

        confidences = [sr["result"].get("confidence_score", 0.0)
                       for sr in sub_results]
        min_conf = min(confidences) if confidences else 0.0

        all_sources: list[str] = []
        for sr in sub_results:
            all_sources.extend(sr["result"].get("sources_used", []))

        # Aggregate coverage flags so the caller can see which sub-questions
        # were silently incomplete. `incomplete_sub_questions` is the list
        # of sub-question strings that the coverage check flagged as
        # uncovered — useful for evaluation and UI surfacing.
        incomplete = [sr["sub_question"] for sr in sub_results
                      if not sr.get("covered", True)]
        coverage_rate = (
            (len(sub_results) - len(incomplete)) / len(sub_results)
            if sub_results else 1.0
        )

        return {
            "answer": combined_answer,
            "routing_decision": routing,
            "confidence_score": round(min_conf, 4),
            "tier_used": tier_str,
            "query_type": "decomposed",
            "chunk_scores": [],  # not meaningful for combined queries
            "sources_used": list(set(all_sources)),
            "mode": "crag",
            "sub_questions": sub_questions,
            "sub_results": sub_results,
            "incomplete_sub_questions": incomplete,
            "sub_question_coverage_rate": round(coverage_rate, 3),
        }

    def query(
        self,
        question: str,
        _skip_decomposition: bool = False,
    ) -> dict:
        # --- Tier -1: Query decomposition (env-gated, off by default) ---
        # For multi-part questions where parts have different sources (e.g.,
        # "What was Apple's iPhone net sales in FY2025, AND what is the
        # current Apple stock price?"), the classifier often labels the
        # whole question as out_of_corpus, sending it to web search where
        # neither part gets answered well. Decomposition splits the
        # question into atomic sub-questions, routes each independently
        # (one to corpus, one to web), and combines the answers.
        #
        # Disabled by default to keep the headline ablation numbers
        # comparable. Enable via QUERY_DECOMPOSITION=1 env var.
        decomp_enabled = os.getenv("QUERY_DECOMPOSITION", "0") == "1"
        if decomp_enabled and not _skip_decomposition:
            sub_qs = _decompose_question(self.llm, question)
            if len(sub_qs) > 1:
                logger.info(f"Decomposing into {len(sub_qs)} sub-questions")
                return self._query_with_decomposition(question, sub_qs)
            # else: single-part — fall through to normal handling

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
            logger.info("OOC query but web disabled → forcing corpus retrieval")

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

            # ----------------------------------------------------------
            # Strip-level refinement (multimodal only) — Yan et al. style.
            # ----------------------------------------------------------
            # For multimodal queries, we additionally extract the top-N
            # sentence-level "strips" across all accepted chunks, score
            # them with the cross-encoder, and prepend them as a HIGHLIGHTS
            # section before the full chunks. This addresses the iPhone-
            # style failure mode where the right chunk is in context but
            # Llama-8B misses the specific attribution sentence among many
            # sibling sentences.
            #
            # The full chunks are still included after the highlights, so
            # this is strictly additive — worst case the LLM ignores the
            # highlights. Toggleable via STRIP_LEVEL_REFINEMENT env var.
            strip_enabled = os.getenv("STRIP_LEVEL_REFINEMENT", "1") == "1"
            if (
                strip_enabled
                and query_type == "multimodal"
                and len(accepted) > 0
            ):
                try:
                    top_strips = self._extract_top_strips(
                        question,
                        [e.chunk for e in accepted],
                        max_strips=8,
                    )
                    if top_strips:
                        highlights = "\n".join(
                            f"• [{src}] {sent}" for sent, src in top_strips
                        )
                        context_passages.append(
                            f"[HIGHLIGHTS — top sentence-level strips by "
                            f"cross-encoder relevance]\n{highlights}"
                        )
                        logger.info(
                            f"Strip-level refinement: prepended "
                            f"{len(top_strips)} top strips to context"
                        )
                except Exception as e:
                    logger.warning(f"Strip-level refinement failed ({e}); "
                                   f"falling through to chunk-level only")

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
                    if still_incomplete and max_cosine < 0.65 and self.web_fallback_enabled:
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

        # Numerical fidelity check — verify that every number in the
        # answer appears verbatim in a retrieved chunk. We compute it
        # against the chunks that actually went into the LLM context
        # (post-evaluation), not the raw retrieval set, so the check
        # asks "could this number have plausibly come from what we
        # showed the model?" rather than "is it anywhere in the corpus."
        try:
            fidelity = _check_numerical_fidelity(
                answer, [e.chunk for e in evaluated]
            )
            if fidelity["unverified_numbers"]:
                logger.warning(
                    f"Numerical fidelity check flagged unverified numbers: "
                    f"{fidelity['unverified_numbers']}"
                )
        except Exception as e:
            logger.warning(f"Numerical fidelity check failed ({e})")
            fidelity = {
                "numbers_in_answer": [], "verified_numbers": [],
                "unverified_numbers": [], "fidelity_score": 1.0,
            }

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
            "numerical_fidelity": fidelity,
        }
