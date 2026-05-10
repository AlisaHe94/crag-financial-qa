"""Test query decomposition on the partial-OOC question.

Runs the partial-OOC question with decomposition OFF (current behavior)
and ON (new), so we can compare side-by-side.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_baseline import VectorStore
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


PARTIAL_OOC_QUESTION = (
    "What was Apple's iPhone net sales in fiscal year 2025, "
    "and what is the current price of Apple stock?"
)


def run(label: str, decomp_on: bool, crag) -> dict:
    if decomp_on:
        os.environ["QUERY_DECOMPOSITION"] = "1"
    else:
        os.environ["QUERY_DECOMPOSITION"] = "0"
    print(f"\n{'=' * 70}\n{label}  (QUERY_DECOMPOSITION={os.environ['QUERY_DECOMPOSITION']})\n{'=' * 70}")
    t0 = time.perf_counter()
    r = crag.query(PARTIAL_OOC_QUESTION)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\nrouting:    {r.get('routing_decision')}")
    print(f"tier:       {r.get('tier_used')}")
    print(f"confidence: {r.get('confidence_score'):.2f}")
    print(f"query_type: {r.get('query_type')}")
    print(f"latency:    {elapsed:.0f} ms")
    if "sub_questions" in r:
        print(f"\nsub-questions ({len(r['sub_questions'])}):")
        for i, sq in enumerate(r["sub_questions"], 1):
            print(f"  {i}. {sq}")
        print(f"\nsub-results:")
        for i, sr in enumerate(r["sub_results"], 1):
            sub_r = sr["result"]
            print(f"  Sub {i}: tier={sub_r.get('tier_used')}, "
                  f"routing={sub_r.get('routing_decision')}, "
                  f"conf={sub_r.get('confidence_score'):.2f}")
    print(f"\nFinal answer:\n{r.get('answer', '')}")
    return r


def main():
    print("Loading vector stores...")
    vs_crag = VectorStore.load(Path("data/vectordb_crag_tables"))
    vs_text = VectorStore.load(Path("data/vectordb_baseline"))
    crag = CorrectedRAG(
        vector_store=vs_crag,
        text_store=vs_text,
        evaluator=RetrievalEvaluator(),
    )

    r_off = run("WITHOUT decomposition (current)", False, crag)
    r_on = run("WITH decomposition (new)", True, crag)

    # Quick comparison
    print(f"\n\n{'=' * 70}\nCOMPARISON\n{'=' * 70}")
    print(f"{'metric':<20} {'OFF':<25} {'ON':<25}")
    print(f"{'-' * 70}")
    for k in ["tier_used", "routing_decision", "query_type"]:
        v_off = r_off.get(k, "")
        v_on = r_on.get(k, "")
        print(f"{k:<20} {str(v_off):<25} {str(v_on):<25}")

    # Manual quality check (to be evaluated by reading the answers)
    print(f"\n\nMANUAL CHECK — compare the full answers above to see whether")
    print(f"decomposition produces a better answer that addresses BOTH")
    print(f"halves of the question.")


if __name__ == "__main__":
    main()
