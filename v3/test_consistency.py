"""Head-to-head consistency test.

Runs each demo question N times against BOTH BaselineRAG and CorrectedRAG.
Measures:
  - Did each system produce the SAME answer across runs? (consistency)
  - Did CRAG outperform Baseline by KHR on each run?
  - Latency variance per system

Goal: confirm temperature=0.05 makes the systems deterministic AND
that CRAG consistently >= Baseline on KHR.

Time budget: ~5-7 min depending on question count and Groq response time.
"""

from __future__ import annotations
import time
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

from dotenv import load_dotenv
load_dotenv()

from rag_baseline import VectorStore, BaselineRAG
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


# Use a 3-question subset to keep total runtime under ~5 min on Groq's
# free tier. Pick one of each interesting type.
DEMO_QUESTIONS = [
    {
        "label": "Multimodal — Meta DAU + Reality Labs",
        "question": "How did Meta's family of apps daily active users "
                    "change in 2025, and what does management say about "
                    "Reality Labs spending?",
        "expected_keywords": ["3.58", "billion", "reality labs"],
    },
    {
        "label": "iPhone YoY (canary)",
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
]

N_RUNS = 3   # times to run each question


def keyword_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    a = (answer or "").lower()
    return sum(1 for k in expected if k.lower() in a) / len(expected)


def run_query(system, question: str):
    t0 = time.perf_counter()
    try:
        result = system.query(question)
        ans = result.get("answer", "")
        err = None
    except Exception as e:
        ans = ""
        err = str(e)
    return {
        "answer": ans,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "error": err,
    }


def main():
    print("Loading vector stores...")
    vs_crag_idx = VectorStore.load(Path("data/vectordb_crag_tables"))
    vs_text_idx = VectorStore.load(Path("data/vectordb_baseline"))

    crag = CorrectedRAG(
        vector_store=vs_crag_idx,
        text_store=vs_text_idx,
        evaluator=RetrievalEvaluator(),
    )
    baseline = BaselineRAG(vs_text_idx)

    # results[label][system_name] = list of run dicts
    results = defaultdict(lambda: defaultdict(list))

    for q in DEMO_QUESTIONS:
        print(f"\n{'=' * 70}\n{q['label']}\n{'-' * 70}")
        for system_name, system in [("CRAG", crag), ("Baseline", baseline)]:
            print(f"\n  --- {system_name} (running {N_RUNS}×) ---")
            for i in range(N_RUNS):
                r = run_query(system, q["question"])
                khr = keyword_hit_rate(r["answer"], q["expected_keywords"])
                r["khr"] = khr
                results[q["label"]][system_name].append(r)
                snippet = (r["answer"] or "")[:80].replace("\n", " ")
                print(f"  run {i+1}: KHR={khr:.2f}  lat={r['latency_ms']:.0f}ms"
                      f"  | {snippet}…")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for label in results:
        print(f"\n{label}")
        for system_name in ["CRAG", "Baseline"]:
            runs = results[label][system_name]
            khrs = [r["khr"] for r in runs]
            lats = [r["latency_ms"] for r in runs]
            answers = [r["answer"] for r in runs]
            unique_answers = len(set(a[:200] for a in answers))
            khr_str = f"{mean(khrs):.2f}"
            if len(khrs) > 1:
                khr_str += f" ± {stdev(khrs):.2f}"
            lat_str = f"{mean(lats):.0f} ms"
            if len(lats) > 1:
                lat_str += f" (σ={stdev(lats):.0f})"
            consistency = "✓ identical" if unique_answers == 1 else f"✗ {unique_answers} variants"
            print(f"  {system_name:9s}  KHR={khr_str:14s}  lat={lat_str:25s}  answers: {consistency}")

    # Head-to-head: per question, per run, did CRAG >= Baseline on KHR?
    print(f"\n\n{'=' * 70}\nHEAD-TO-HEAD: CRAG vs Baseline KHR per run\n{'=' * 70}")
    crag_wins = 0
    crag_ties = 0
    baseline_wins = 0
    for label in results:
        crag_khrs = [r["khr"] for r in results[label]["CRAG"]]
        base_khrs = [r["khr"] for r in results[label]["Baseline"]]
        print(f"\n{label}")
        for i, (c, b) in enumerate(zip(crag_khrs, base_khrs)):
            verdict = "CRAG wins" if c > b else ("BASELINE wins" if b > c else "tie")
            if c > b:
                crag_wins += 1
            elif c < b:
                baseline_wins += 1
            else:
                crag_ties += 1
            print(f"  run {i+1}: CRAG {c:.2f}  vs  Baseline {b:.2f}   → {verdict}")
    total = crag_wins + crag_ties + baseline_wins
    print(f"\nOVERALL: CRAG wins {crag_wins}/{total}, ties {crag_ties}/{total}, "
          f"baseline wins {baseline_wins}/{total}")


if __name__ == "__main__":
    main()
