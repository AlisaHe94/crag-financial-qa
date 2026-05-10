"""Quick α sensitivity check on the 4 demo questions only.

Tests CRAG-tables at 3 α values, prints a summary table + saves details.
Runtime: ~2-3 min on Gemini.
"""
from __future__ import annotations
import time
from pathlib import Path
from collections import defaultdict
from statistics import mean

from dotenv import load_dotenv
load_dotenv()

from rag_baseline import VectorStore
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


DEMO_QUESTIONS = [
    {
        "id": "Q1",
        "label": "Meta DAU + Reality Labs",
        "question": "How did Meta's family of apps daily active users "
                    "change in 2025, and what does management say about "
                    "Reality Labs spending?",
        "expected_keywords": ["3.58", "billion", "reality labs"],
    },
    {
        "id": "Q2",
        "label": "iPhone YoY",
        "question": "How did Apple's iPhone revenue change between fiscal "
                    "year 2024 and 2025, and what does management cite as "
                    "the reason?",
        "expected_keywords": ["pro models", "increase", "209"],
    },
    {
        "id": "Q3",
        "label": "Microsoft Azure",
        "question": "How does Microsoft describe its Azure business in the "
                    "fiscal year 2025 10-K?",
        "expected_keywords": ["azure", "cloud", "intelligent"],
    },
    {
        "id": "Q4",
        "label": "Bitcoin OOC",
        "question": "What is the current price of Bitcoin?",
        "expected_keywords": ["bitcoin", "$"],
    },
]

ALPHAS = [0.2, 0.4, 0.6]


def keyword_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    a = (answer or "").lower()
    return sum(1 for k in expected if k.lower() in a) / len(expected)


def main():
    print("Loading vector stores...")
    vs_crag = VectorStore.load(Path("data/vectordb_crag_tables"))
    vs_text = VectorStore.load(Path("data/vectordb_baseline"))
    print()

    # results[alpha][q_id] = dict
    results = defaultdict(dict)

    for alpha in ALPHAS:
        print(f"\n=== α = {alpha} ===")
        evaluator = RetrievalEvaluator(alpha=alpha)
        crag = CorrectedRAG(
            vector_store=vs_crag,
            text_store=vs_text,
            evaluator=evaluator,
        )
        for q in DEMO_QUESTIONS:
            t0 = time.perf_counter()
            try:
                r = crag.query(q["question"])
                ans = r.get("answer", "")
                meta = {
                    "tier": r.get("tier_used", ""),
                    "routing": r.get("routing_decision", ""),
                    "confidence": r.get("confidence_score", 0.0),
                    "query_type": r.get("query_type", ""),
                }
            except Exception as e:
                ans = ""
                meta = {"error": str(e)}
            khr = keyword_hit_rate(ans, q["expected_keywords"])
            elapsed = (time.perf_counter() - t0) * 1000
            results[alpha][q["id"]] = {
                "khr": khr,
                "answer": ans,
                "meta": meta,
                "latency_ms": elapsed,
            }
            print(f"  [{q['id']}] KHR={khr:.2f}  tier={meta.get('tier','?')}  "
                  f"routing={meta.get('routing','?')}  "
                  f"conf={meta.get('confidence',0):.2f}  ({elapsed:.0f}ms)")

    # Summary table: KHR by α × question
    print(f"\n\n{'=' * 70}\nSUMMARY: KHR by α × question\n{'=' * 70}")
    print(f"{'α':<8}", end="")
    for q in DEMO_QUESTIONS:
        print(f"{q['id']:<8}", end="")
    print("MEAN")
    for alpha in ALPHAS:
        print(f"{alpha:<8}", end="")
        khrs = []
        for q in DEMO_QUESTIONS:
            k = results[alpha][q["id"]]["khr"]
            khrs.append(k)
            print(f"{k:<8.2f}", end="")
        print(f"{mean(khrs):.3f}")

    # Tier consistency: did α changes shift routing decisions?
    print(f"\n{'=' * 70}\nTier / routing changes across α\n{'=' * 70}")
    for q in DEMO_QUESTIONS:
        print(f"\n{q['id']} ({q['label']})")
        for alpha in ALPHAS:
            m = results[alpha][q["id"]]["meta"]
            print(f"  α={alpha}: tier={m.get('tier','?'):<14s}  "
                  f"routing={m.get('routing','?'):<10s}  "
                  f"conf={m.get('confidence',0):.2f}")

    # Save full answers for manual review
    out_path = Path("alpha_sensitivity_answers.md")
    lines = ["# α Sensitivity — Full Answers\n"]
    for q in DEMO_QUESTIONS:
        lines.append(f"\n## {q['id']}: {q['label']}\n")
        lines.append(f"**Question:** {q['question']}\n")
        for alpha in ALPHAS:
            r = results[alpha][q["id"]]
            m = r["meta"]
            lines.append(f"\n### α = {alpha}  (KHR={r['khr']:.2f}, "
                         f"tier={m.get('tier','?')}, "
                         f"conf={m.get('confidence',0):.2f}, "
                         f"{r['latency_ms']:.0f} ms)\n")
            lines.append(f"\n{r['answer'].strip()}\n")
    out_path.write_text("\n".join(lines))
    print(f"\nFull answers saved to: {out_path}")


if __name__ == "__main__":
    main()
