"""α-ensemble inference: hyperparameter perturbation as a confidence signal.

Standard CRAG picks one α offline (we found α=0.4 in alpha_sweep.py) and uses
it to blend cosine + cross-encoder scores into a single retrieval evaluator.
This script treats α as an *ensemble axis* at inference time instead of a fixed
hyperparameter.

For each question, three retrieval evaluators with different α values cast
independent routing votes (CORRECT / AMBIGUOUS / INCORRECT). The agreement
pattern is itself a confidence signal:
- 3-way agreement ("stable") → high confidence, return primary answer
- 2-way agreement ("majority") → moderate confidence, primary answer flagged
- full disagreement ("disagree") → low confidence, primary answer flagged

α values picked from the alpha_sweep.py results:
- α=0.0  cross-encoder only (semantic signal)        — KHR 0.712
- α=0.4  empirically best blend (sweet spot)         — KHR 0.803  ← primary
- α=1.0  cosine only (lexical/embedding signal)      — KHR 0.758

We deliberately skip α=0.6 (the worst-performing α in our sweep, also 0.712)
and pick three structurally different scoring strategies that span the range.
The point of the ensemble is *disagreement* on borderline queries; clustering
all three values near 0.4 would defeat the purpose.

Why this is novel:
- Yan et al. (2024) CRAG has no α — they use one discriminative T5 model.
- Self-RAG generates self-reflection tokens but does NOT perturb its own
  hyperparameters at inference time.
- MultiFinRAG retrieves multimodally but uses a single fixed scorer.
- Hyperparameter perturbation as an inference-time confidence proxy reframes
  α from "knob to tune offline" to "ensemble axis exposed at runtime."

Runtime: 3× alpha_sweep.py per question, but only across one α-set rather
than four — net ~50% the runtime of alpha_sweep.py (~10–15 min on Gemini).
"""

from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from rag_baseline import VectorStore
from crag_pipeline import CorrectedRAG, RetrievalEvaluator


# Alpha values for the ensemble. See module docstring for rationale.
ALPHAS = [0.0, 0.4, 1.0]
PRIMARY_ALPHA = 0.4  # The answer returned to the user comes from this α.


def load_questions() -> list[dict]:
    return json.loads(Path("data/eval_questions.json").read_text())


def keyword_hit_rate(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    a = (answer or "").lower()
    return sum(1 for k in expected if k.lower() in a) / len(expected)


def run_one_alpha(alpha: float, q: dict, crag_vs, text_vs) -> dict:
    """Run one question through CRAG at the given α; capture route + answer."""
    evaluator = RetrievalEvaluator(alpha=alpha)
    crag = CorrectedRAG(
        vector_store=crag_vs,
        text_store=text_vs,
        evaluator=evaluator,
    )
    t0 = time.perf_counter()
    try:
        r = crag.query(q["question"])
        return {
            "alpha": alpha,
            "answer": r.get("answer", ""),
            "route": r.get("routing_decision", "UNKNOWN"),
            "confidence": r.get("confidence_score", 0.0),
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": None,
        }
    except Exception as e:
        return {
            "alpha": alpha,
            "answer": "",
            "route": "ERROR",
            "confidence": 0.0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": str(e),
        }


def stability_label(routes: list[str]) -> str:
    """3-way agreement, 2-way majority, or full disagreement.

    Returns one of: 'stable', 'majority', 'disagree'.
    """
    counts = Counter(routes).most_common()
    top_count = counts[0][1]
    if top_count == 3:
        return "stable"
    if top_count == 2:
        return "majority"
    return "disagree"


def main() -> None:
    print("Loading vector stores...")
    crag_vs = VectorStore.load(Path("data/vectordb_crag_tables"))
    text_vs = VectorStore.load(Path("data/vectordb_baseline"))
    questions = load_questions()
    print(f"Loaded {len(questions)} questions; running ensemble at α ∈ {ALPHAS}")
    print(f"Primary α (answer source) = {PRIMARY_ALPHA}\n")

    rows: list[dict] = []
    t_start = time.perf_counter()
    for q in questions:
        per_alpha = [run_one_alpha(a, q, crag_vs, text_vs) for a in ALPHAS]
        routes = [r["route"] for r in per_alpha]
        stability = stability_label(routes)

        # Headline answer comes from the primary α; other α values are used
        # purely as a confidence probe.
        primary = next(r for r in per_alpha if r["alpha"] == PRIMARY_ALPHA)
        khr = keyword_hit_rate(primary["answer"], q.get("expected_keywords", []))

        rows.append({
            "question_id": q["id"],
            "question_type": q["type"],
            "ground_truth_in_corpus": q.get("ground_truth_in_corpus", True),
            "khr": khr,
            "stability": stability,
            "route_a0": routes[0],
            "route_a4": routes[1],
            "route_a10": routes[2],
            "primary_latency_ms": primary["latency_ms"],
            "ensemble_latency_ms": sum(r["latency_ms"] for r in per_alpha),
            "primary_confidence": primary["confidence"],
            "answer_snippet": (primary["answer"] or "")[:200],
        })
        print(f"  Q{q['id']:>2}  {stability:>9}  routes={routes}  KHR={khr:.2f}")

    elapsed = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    out_csv = Path("alpha_ensemble_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}  (total wall time: {elapsed:.1f}s)")

    # ---------- Summary 1: stability distribution ----------
    print("\n=== Routing-stability distribution ===")
    print(df["stability"].value_counts().to_string())

    # ---------- Summary 2: KHR conditional on stability ----------
    # If routing-stability is a real confidence signal, KHR should be higher
    # on 'stable' rows than on 'disagree' rows. This is the headline plot.
    print("\n=== Mean KHR by stability bucket ===")
    print(
        df.groupby("stability")["khr"]
        .agg(["mean", "count"])
        .round(3)
        .to_string()
    )

    # ---------- Summary 3: stability × question_type ----------
    print("\n=== Stability distribution by question_type ===")
    pivot = (
        df.groupby(["question_type", "stability"]).size().unstack(fill_value=0)
    )
    print(pivot.to_string())

    # ---------- Summary 4: latency overhead ----------
    primary_mean = df["primary_latency_ms"].mean()
    ensemble_mean = df["ensemble_latency_ms"].mean()
    overhead = ensemble_mean / primary_mean if primary_mean else float("nan")
    print(
        f"\nLatency: primary={primary_mean:.0f}ms  "
        f"ensemble={ensemble_mean:.0f}ms  ({overhead:.2f}× overhead)"
    )

    # ---------- Summary 5: where disagreement concentrates ----------
    # Helpful for the report: surface the actual questions that triggered
    # routing disagreement, since those are the cases where the ensemble
    # is doing real work vs. wasting compute.
    disagreed = df[df["stability"] != "stable"]
    if len(disagreed):
        print(f"\n=== {len(disagreed)} unstable rows ===")
        print(
            disagreed[
                ["question_id", "question_type", "stability",
                 "route_a0", "route_a4", "route_a10", "khr"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
