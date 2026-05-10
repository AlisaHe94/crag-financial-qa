"""α sensitivity sweep for the probabilistic evaluator.

Runs the existing 22-question eval set through CRAG-tables for several α
values, records keyword hit rate per α, and writes a CSV summary.

Time budget: ~8-12 min depending on Groq response time.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag_baseline import VectorStore
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


def load_questions() -> list[dict]:
    return json.loads(Path("data/eval_questions.json").read_text())


def keyword_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    a = (answer or "").lower()
    return sum(1 for k in expected if k.lower() in a) / len(expected)


def run_alpha(alpha: float, questions: list[dict], crag_vs, text_vs) -> list[dict]:
    """Run all questions through CRAG-tables with the given α."""
    evaluator = RetrievalEvaluator(alpha=alpha)
    crag = CorrectedRAG(
        vector_store=crag_vs,
        text_store=text_vs,
        evaluator=evaluator,
    )
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            r = crag.query(q["question"])
            ans = r.get("answer", "")
            err = None
        except Exception as e:
            ans = ""
            err = str(e)
        rows.append({
            "alpha": alpha,
            "question_id": q["id"],
            "question_type": q["type"],
            "khr": keyword_hit_rate(ans, q.get("expected_keywords", [])),
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "ground_truth_in_corpus": q.get("ground_truth_in_corpus", True),
            "error": err,
            "answer_snippet": (ans or "")[:200],
        })
    return rows


def main():
    print("Loading vector stores...")
    crag_vs = VectorStore.load(Path("data/vectordb_crag_tables"))
    text_vs = VectorStore.load(Path("data/vectordb_baseline"))
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.\n")

    # 4 alpha values is enough to see the curve shape and stays under
    # ~10 min total runtime. If we want fuller coverage later, add 0.2/0.8.
    alphas = [0.0, 0.4, 0.6, 1.0]

    all_rows: list[dict] = []
    for a in alphas:
        print(f"\n=== α = {a} ===")
        t0 = time.perf_counter()
        rows = run_alpha(a, questions, crag_vs, text_vs)
        all_rows.extend(rows)
        elapsed = time.perf_counter() - t0
        df_a = pd.DataFrame(rows)
        mean_khr = df_a["khr"].mean()
        print(f"α={a}  mean KHR={mean_khr:.3f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(all_rows)
    out_csv = Path("alpha_sweep_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Pivot summary: KHR by α x question_type
    summary = df.groupby(["alpha", "question_type"])["khr"].mean().unstack().round(3)
    summary["overall_mean"] = df.groupby("alpha")["khr"].mean().round(3)
    print("\n=== Summary (KHR by α × question_type) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
