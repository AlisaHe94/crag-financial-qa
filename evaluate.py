"""
Evaluation harness and ablation study runner.

Four experimental conditions (ablation):
  1. baseline_no_tables   — naive chunking, text-only retrieval, no CRAG
  2. baseline_tables      — table-aware chunking, text-only retrieval, no CRAG
  3. crag_no_tables       — naive chunking, CRAG probabilistic router
  4. crag_tables          — semantic chunking + table-aware + CRAG  ← full system

Question taxonomy (MultiFinRAG §4.1):
  Type 1: text-based
  Type 2: image-based      (our system: partially handled via table fallback)
  Type 3: table-based
  Type 4: text + table/image combined

Metrics:
  - Keyword hit rate (proxy accuracy — exact LLM-eval requires API budget)
  - Routing precision (CRAG only): correct routing / total
  - Average latency (ms)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from document_processor import load_document, DocumentChunk
from rag_baseline import VectorStore, BaselineRAG, build_index_from_dir
from crag_pipeline import CorrectedRAG, RetrievalEvaluator

load_dotenv()
logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = Path("data/eval_questions.json")
RESULTS_PATH = Path("data/eval_results.csv")


# ---------------------------------------------------------------------------
# Evaluation question set — 4-type taxonomy (MultiFinRAG §4.1)
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS: list[dict] = [
    # Type 1: text-based
    {
        "id": "t1", "type": 1,
        "question": "What were Apple's total net sales for fiscal year 2023?",
        "expected_keywords": ["383", "billion", "net sales"],
        "ground_truth_in_corpus": True,
    },
    {
        "id": "t2", "type": 1,
        "question": "What are the main risk factors described in Microsoft's most recent 10-K?",
        "expected_keywords": ["competition", "regulation", "cybersecurity"],
        "ground_truth_in_corpus": True,
    },
    {
        "id": "t3", "type": 1,
        "question": "What business segment generates the most revenue for Amazon?",
        "expected_keywords": ["aws", "cloud", "north america"],
        "ground_truth_in_corpus": True,
    },
    # Type 3: table-based
    {
        "id": "tab1", "type": 3,
        "question": "What was Apple's gross margin percentage in fiscal year 2023?",
        "expected_keywords": ["44", "gross margin", "%"],
        "ground_truth_in_corpus": True,
    },
    {
        "id": "tab2", "type": 3,
        "question": "What was Amazon's operating income in Q4 2023?",
        "expected_keywords": ["operating income", "billion", "quarter"],
        "ground_truth_in_corpus": True,
    },
    {
        "id": "tab3", "type": 3,
        "question": "What was Microsoft's total revenue for the fiscal year ending June 2023?",
        "expected_keywords": ["211", "billion", "revenue"],
        "ground_truth_in_corpus": True,
    },
    # Type 4: text + table combined
    {
        "id": "comb1", "type": 4,
        "question": (
            "Which product segment had the highest revenue growth rate for Apple in fiscal 2023, "
            "and what was that rate?"
        ),
        "expected_keywords": ["services", "growth", "%"],
        "ground_truth_in_corpus": True,
    },
    {
        "id": "comb2", "type": 4,
        "question": (
            "What was Meta's advertising revenue in 2023, and how did it compare "
            "to the prior year according to management commentary?"
        ),
        "expected_keywords": ["advertising", "revenue", "increase", "billion"],
        "ground_truth_in_corpus": True,
    },
    # Out-of-corpus (should trigger CRAG web fallback — Type 1 framing but no corpus match)
    {
        "id": "oc1", "type": 1,
        "question": "What is the current federal funds rate set by the Fed this week?",
        "expected_keywords": ["federal funds", "rate", "%"],
        "ground_truth_in_corpus": False,
    },
    {
        "id": "oc2", "type": 1,
        "question": "What was Nvidia's stock price at market close yesterday?",
        "expected_keywords": ["nvidia", "price", "$"],
        "ground_truth_in_corpus": False,
    },
]


def load_eval_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    logger.info("Using built-in sample questions")
    return SAMPLE_QUESTIONS


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)


def routing_correct(result: dict, question: dict) -> bool | None:
    """True if CRAG routing decision matches whether corpus has the answer."""
    if result.get("mode") != "crag":
        return None
    decision = result.get("routing_decision", "")
    in_corpus = question.get("ground_truth_in_corpus", True)
    return (decision in ("correct", "ambiguous")) if in_corpus else (decision == "incorrect")


# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------

def run_condition(system, questions: list[dict], name: str, sleep_between: float = 1.5) -> list[dict]:
    rows = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        try:
            result = system.query(q["question"])
        except Exception as e:
            logger.warning(f"[{name}] query failed for {q['id']}: {e}")
            result = {"answer": "", "mode": name}
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = result.get("answer", "")
        khr = keyword_hit_rate(answer, q.get("expected_keywords", []))
        rc = routing_correct(result, q)

        rows.append({
            "condition": name,
            "question_id": q["id"],
            "question_type": q.get("type"),
            # Save full answer (capped at 2000 chars to keep CSV manageable)
            # rather than the previous 120-char snippet. The 120-char cut was
            # chopping off the actual numerical answer in CRAG's verbose
            # responses, making the LLM-as-judge metric meaningless.
            "answer_snippet": answer[:2000].replace("\n", " "),
            "keyword_hit_rate": round(khr, 3),
            "routing_correct": rc,
            "confidence_score": result.get("confidence_score"),
            "routing_decision": result.get("routing_decision"),
            "tier_used": result.get("tier_used"),
            "query_type": result.get("query_type"),
            "latency_ms": round(latency_ms, 1),
        })
        logger.info(f"[{name}] {q['id']} (type {q.get('type')}) khr={khr:.2f} {latency_ms:.0f}ms")
        # Pace requests so we don't slam Groq's 6k TPM / 30 RPM free-tier limits.
        # Skip the pause after the last question of each condition.
        if i < len(questions) - 1:
            time.sleep(sleep_between)
    return rows


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    filing_dir: str | Path,
    questions: list[dict] | None = None,
    results_path: Path = RESULTS_PATH,
) -> pd.DataFrame:
    if questions is None:
        questions = load_eval_questions()

    filing_dir = Path(filing_dir)
    evaluator = RetrievalEvaluator()
    all_rows: list[dict] = []

    # Reuse existing pre-built indexes — saves ~10 minutes per condition that
    # would otherwise be spent re-embedding all 10 filings from scratch.
    # vectordb_baseline = naive 512-char chunking, no table extraction
    # vectordb_crag_tables = semantic chunking + structured table extraction
    BASELINE_INDEX = Path("data/vectordb_baseline")
    TABLES_INDEX   = Path("data/vectordb_crag_tables")

    if not BASELINE_INDEX.exists() or not TABLES_INDEX.exists():
        logger.error(
            f"Expected pre-built indexes at {BASELINE_INDEX} and {TABLES_INDEX}. "
            "Run `python rag_baseline.py` first to build them."
        )
        return pd.DataFrame()

    logger.info(f"Loading existing indexes (skipping rebuild)…")
    vs_baseline = VectorStore.load(BASELINE_INDEX)
    vs_tables   = VectorStore.load(TABLES_INDEX)

    # (condition_name, primary_vs, use_crag, hybrid_text_vs)
    # `hybrid_text_vs` is only used by CorrectedRAG: when set, text retrieval
    # uses that store while table retrieval uses primary_vs. This is the
    # hybrid retrieval architecture — naive flat-text for text (baseline-style
    # recall on numerical mentions), structured tables for table modality.
    conditions = [
        ("baseline_no_tables", vs_baseline, False, None),
        ("baseline_tables",    vs_tables,   False, None),
        ("crag_no_tables",     vs_baseline, True,  None),
        ("crag_tables",        vs_tables,   True,  vs_baseline),  # ← HYBRID
    ]

    for cond_name, vs, use_crag, hybrid_text_vs in conditions:
        logger.info(f"\n{'='*60}\nCondition: {cond_name}\n{'='*60}")
        try:
            if use_crag:
                system = CorrectedRAG(vs, evaluator=evaluator, text_store=hybrid_text_vs)
            else:
                system = BaselineRAG(vs)
            all_rows.extend(run_condition(system, questions, cond_name))
            del system  # release LLM client + cross-encoder before next condition
            # Persist incrementally so a mid-run failure doesn't lose data.
            partial = pd.DataFrame(all_rows)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            partial.to_csv(results_path, index=False)
            logger.info(f"  → wrote {len(all_rows)} rows so far to {results_path}")
        except Exception as e:
            logger.error(f"Condition {cond_name} failed: {e}")

    df = pd.DataFrame(all_rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== Ablation Study — Overall ===")
    overall = (
        df.groupby("condition")
        .agg(
            avg_khr=("keyword_hit_rate", "mean"),
            routing_precision=("routing_correct", lambda x: x.dropna().mean() if x.notna().any() else None),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .round(3)
    )
    print(overall.to_string())

    print("\n=== Ablation Study — By Question Type ===")
    by_type = (
        df.groupby(["condition", "question_type"])
        .agg(avg_khr=("keyword_hit_rate", "mean"), n=("question_id", "count"))
        .round(3)
    )
    print(by_type.to_string())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/sec_filings"
    run_ablation(directory)
