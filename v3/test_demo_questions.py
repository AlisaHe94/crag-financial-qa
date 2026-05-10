"""Quick smoke test for the four demo questions, with and without
strip-level refinement enabled.

Usage:
    # With strip-level (default):
    python test_demo_questions.py

    # Without strip-level (for comparison):
    STRIP_LEVEL_REFINEMENT=0 python test_demo_questions.py
"""

from __future__ import annotations
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_baseline import VectorStore, BaselineRAG
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


DEMO_QUESTIONS = [
    {
        "label": "Multimodal — Meta DAU + Reality Labs",
        "question": "How did Meta's family of apps daily active users change "
                    "in 2025, and what does management say about Reality Labs "
                    "spending?",
        "expected_keywords": ["3.58", "billion", "reality labs"],
    },
    {
        "label": "iPhone YoY (the canary — strip-level should help here)",
        "question": "How did Apple's iPhone revenue change between fiscal "
                    "year 2024 and 2025, and what does management cite as "
                    "the reason?",
        "expected_keywords": ["pro models", "increase", "209"],
    },
    {
        "label": "Narrative — Microsoft Azure",
        "question": "How does Microsoft describe its Azure business in the "
                    "fiscal year 2025 10-K?",
        "expected_keywords": ["azure", "cloud", "intelligent"],
    },
    {
        "label": "Pure OOC — Bitcoin price",
        "question": "What is the current price of Bitcoin?",
        "expected_keywords": ["bitcoin", "$"],
    },
]


def keyword_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    a = (answer or "").lower()
    return sum(1 for k in expected if k.lower() in a) / len(expected)


def main():
    print(f"STRIP_LEVEL_REFINEMENT = {os.getenv('STRIP_LEVEL_REFINEMENT', '1')}")
    print("Loading vector stores...")
    vs_crag = VectorStore.load(Path("data/vectordb_crag_tables"))
    vs_text = VectorStore.load(Path("data/vectordb_baseline"))
    crag = CorrectedRAG(
        vector_store=vs_crag,
        text_store=vs_text,
        evaluator=RetrievalEvaluator(),
    )

    for q in DEMO_QUESTIONS:
        print(f"\n{'=' * 70}\n{q['label']}\nQuestion: {q['question']}\n{'-' * 70}")
        t0 = time.perf_counter()
        try:
            r = crag.query(q["question"])
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        ans = r.get("answer", "")
        khr = keyword_hit_rate(ans, q["expected_keywords"])
        print(f"  Routing:    {r.get('routing_decision')}")
        print(f"  Tier:       {r.get('tier_used')}")
        print(f"  Confidence: {r.get('confidence_score'):.3f}")
        print(f"  Latency:    {elapsed:.0f} ms")
        print(f"  KHR:        {khr:.2f}  (expected: {q['expected_keywords']})")
        print(f"  Answer (first 400 chars):")
        print(f"    {(ans or '')[:400]}")


if __name__ == "__main__":
    main()
