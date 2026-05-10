"""Dump full CRAG and Baseline answers for the 4 demo questions, in a
clean markdown format that's easy to paste into a chat for human/LLM
qualitative evaluation.

Output: writes to qualitative_eval_output.md AND prints to stdout.

Time budget: ~2-3 min on Gemini-as-primary (no 429 issues).
"""

from __future__ import annotations
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_baseline import VectorStore, BaselineRAG
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


DEMO_QUESTIONS = [
    {
        "id": "Q1",
        "label": "Multimodal — Meta DAU + Reality Labs",
        "question": "How did Meta's family of apps daily active users "
                    "change in 2025, and what does management say about "
                    "Reality Labs spending?",
    },
    {
        "id": "Q2",
        "label": "Multimodal canary — iPhone YoY + reason",
        "question": "How did Apple's iPhone revenue change between fiscal "
                    "year 2024 and 2025, and what does management cite as "
                    "the reason?",
    },
    {
        "id": "Q3",
        "label": "Narrative — Microsoft Azure",
        "question": "How does Microsoft describe its Azure business in the "
                    "fiscal year 2025 10-K?",
    },
    {
        "id": "Q4",
        "label": "Pure OOC — current Bitcoin price",
        "question": "What is the current price of Bitcoin?",
    },
]


def run_query(system, question: str):
    t0 = time.perf_counter()
    try:
        result = system.query(question)
        ans = result.get("answer", "")
        meta = {
            "tier": result.get("tier_used", ""),
            "routing": result.get("routing_decision", ""),
            "confidence": result.get("confidence_score", 0.0),
            "query_type": result.get("query_type", ""),
        }
        err = None
    except Exception as e:
        ans = ""
        meta = {}
        err = str(e)
    return {
        "answer": ans,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "meta": meta,
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

    out_lines: list[str] = ["# Qualitative Evaluation — CRAG vs Baseline\n"]

    for q in DEMO_QUESTIONS:
        out_lines.append(f"\n## {q['id']}: {q['label']}\n")
        out_lines.append(f"**Question:** {q['question']}\n")

        for system_name, system in [("CRAG", crag), ("Baseline", baseline)]:
            print(f"\n[{q['id']}] running {system_name}...", flush=True)
            r = run_query(system, q["question"])
            if r["error"]:
                out_lines.append(f"\n### {system_name}\n\n```\nERROR: {r['error']}\n```\n")
                continue

            meta_line = ""
            if system_name == "CRAG" and r["meta"]:
                m = r["meta"]
                conf = m.get("confidence", 0.0)
                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                meta_line = (
                    f"_query_type={m.get('query_type', '')} | "
                    f"routing={m.get('routing', '')} | "
                    f"tier={m.get('tier', '')} | "
                    f"confidence={conf_str}_\n"
                )
            else:
                meta_line = "_tier=text only_\n"

            out_lines.append(f"\n### {system_name} ({r['latency_ms']:.0f} ms)\n")
            out_lines.append(meta_line)
            out_lines.append(f"\n{r['answer'].strip()}\n")

    output = "\n".join(out_lines)
    out_path = Path("qualitative_eval_output.md")
    out_path.write_text(output)

    print(f"\n\n{'=' * 70}")
    print(f"Saved to: {out_path}")
    print(f"{'=' * 70}\n")
    print(output)


if __name__ == "__main__":
    main()
