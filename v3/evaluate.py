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
    """Fraction of expected keywords that appear (substring, case-insensitive)
    in the answer. Returns 0.0 if either side is empty/None — robust against
    LLM call failures that produce a None answer."""
    if not keywords:
        return 0.0
    if not answer:  # handles "" and None gracefully
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

        # Pull the consistency-check metadata that CRAG attaches when it
        # runs (numerical fidelity check + per-sub-question coverage).
        # Baseline conditions don't produce these — default to neutral
        # values (1.0 / empty list) so the columns are uniform across rows.
        fidelity = result.get("numerical_fidelity") or {}
        incomplete_subqs = result.get("incomplete_sub_questions") or []
        coverage_rate = result.get("sub_question_coverage_rate")
        edge_case = q.get("edge_case", "")

        rows.append({
            "condition": name,
            "question_id": q["id"],
            "question_type": q.get("type"),
            "edge_case": edge_case,
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
            # --- Consistency-check telemetry ---
            "fidelity_score": fidelity.get("fidelity_score"),
            "n_numbers_in_answer": len(fidelity.get("numbers_in_answer", [])),
            "n_unverified_numbers": len(fidelity.get("unverified_numbers", [])),
            "unverified_numbers": "; ".join(fidelity.get("unverified_numbers", [])),
            "sub_question_coverage_rate": coverage_rate,
            "n_incomplete_sub_questions": len(incomplete_subqs),
            "incomplete_sub_questions": "; ".join(incomplete_subqs),
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

    # ----- Resume support -----
    # If results_path already exists with rows for some conditions, preserve
    # those rows and skip those conditions on this run. Useful when a long
    # ablation gets interrupted (Ctrl-C, network blip, etc.) — re-running
    # without --no-resume picks up where the previous run stopped instead of
    # redoing everything. To force a clean rerun from scratch, delete the
    # CSV first.
    completed_conditions: set[str] = set()
    if results_path.exists():
        try:
            existing = pd.read_csv(results_path)
            # A condition is "complete" only if it has rows for all questions
            # (handles the mid-condition crash case where partial rows
            # might get written to the CSV).
            n_questions = len(questions)
            for cond_name in [c[0] for c in conditions]:
                cond_rows = existing[existing["condition"] == cond_name]
                if len(cond_rows) == n_questions:
                    completed_conditions.add(cond_name)
            if completed_conditions:
                logger.info(
                    f"Resume mode: found {len(completed_conditions)} completed "
                    f"conditions in {results_path}: {sorted(completed_conditions)}. "
                    f"Will preserve those rows and skip those conditions."
                )
                # Preserve the completed rows so the final CSV has them.
                preserved = existing[
                    existing["condition"].isin(completed_conditions)
                ]
                all_rows.extend(preserved.to_dict("records"))
        except Exception as e:
            logger.warning(
                f"Could not parse existing {results_path} for resume "
                f"(starting clean): {e}"
            )

    for cond_name, vs, use_crag, hybrid_text_vs in conditions:
        if cond_name in completed_conditions:
            logger.info(
                f"\n{'='*60}\nSkipping completed condition: {cond_name}\n{'='*60}"
            )
            continue
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


def _bootstrap_ci(values, n_resamples: int = 1000, ci: float = 0.95):
    """Compute the (lower, upper) bootstrap percentile CI of the mean.

    Returns (mean, lower, upper). With our N=22 eval set the bounds are
    wide — that is the honest picture.
    """
    import numpy as np
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean()
                  for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return (float(arr.mean()), float(lo), float(hi))


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== Ablation Study — Overall (with 95% bootstrap CIs) ===")
    rows = []
    for condition, group in df.groupby("condition"):
        khr_mean, khr_lo, khr_hi = _bootstrap_ci(group["keyword_hit_rate"])
        routing_vals = group["routing_correct"].dropna()
        rp_mean = routing_vals.mean() if len(routing_vals) else None
        rp_lo, rp_hi = (None, None)
        if rp_mean is not None:
            _, rp_lo, rp_hi = _bootstrap_ci(routing_vals)
        # Bootstrap CI on latency too — small per-cell N means latency
        # variance is meaningful. Reported in milliseconds.
        lat_mean, lat_lo, lat_hi = _bootstrap_ci(group["latency_ms"])
        rows.append({
            "condition": condition,
            "avg_khr": round(khr_mean, 3),
            "khr_95_ci": f"[{khr_lo:.3f}, {khr_hi:.3f}]",
            "routing_precision": (None if rp_mean is None else round(rp_mean, 3)),
            "rp_95_ci": (None if rp_lo is None
                         else f"[{rp_lo:.3f}, {rp_hi:.3f}]"),
            "avg_latency_ms": round(lat_mean, 0),
            "lat_95_ci_ms": f"[{lat_lo:.0f}, {lat_hi:.0f}]",
            "n": len(group),
        })
    overall = pd.DataFrame(rows).set_index("condition")
    print(overall.to_string())

    print("\n=== Ablation Study — By Question Type (with 95% bootstrap CIs) ===")
    by_type_rows = []
    for (condition, qtype), group in df.groupby(["condition", "question_type"]):
        khr_mean, khr_lo, khr_hi = _bootstrap_ci(group["keyword_hit_rate"])
        lat_mean, lat_lo, lat_hi = _bootstrap_ci(group["latency_ms"])
        by_type_rows.append({
            "condition": condition,
            "question_type": qtype,
            "n": len(group),
            "avg_khr": round(khr_mean, 3),
            "khr_95_ci": f"[{khr_lo:.3f}, {khr_hi:.3f}]",
            "avg_latency_ms": round(lat_mean, 0),
            "lat_95_ci_ms": f"[{lat_lo:.0f}, {lat_hi:.0f}]",
        })
    by_type = pd.DataFrame(by_type_rows).set_index(["condition", "question_type"])
    print(by_type.to_string())

    # ---------- Consistency-check summary (CRAG conditions only) ----------
    # Numerical fidelity + sub-question coverage are CRAG-only metrics
    # (the baselines don't produce them). Filter to crag_* conditions and
    # report mean fidelity + count of unverified-number rows.
    crag_df = df[df["condition"].astype(str).str.startswith("crag")]
    if len(crag_df):
        print("\n=== Consistency Checks (CRAG conditions only) ===")
        cc_rows = []
        for condition, group in crag_df.groupby("condition"):
            fidelity_vals = group["fidelity_score"].dropna()
            mean_fidelity = fidelity_vals.mean() if len(fidelity_vals) else None
            n_with_numbers = int((group["n_numbers_in_answer"].fillna(0) > 0).sum())
            n_with_unverified = int((group["n_unverified_numbers"].fillna(0) > 0).sum())
            coverage_vals = group["sub_question_coverage_rate"].dropna()
            n_decomposed = len(coverage_vals)
            mean_coverage = coverage_vals.mean() if n_decomposed else None
            n_with_incomplete = int(
                (group["n_incomplete_sub_questions"].fillna(0) > 0).sum()
            )
            cc_rows.append({
                "condition": condition,
                "mean_fidelity": (None if mean_fidelity is None
                                   else round(mean_fidelity, 3)),
                "n_qs_with_numbers": n_with_numbers,
                "n_qs_with_unverified_nums": n_with_unverified,
                "n_qs_decomposed": n_decomposed,
                "mean_subq_coverage": (None if mean_coverage is None
                                        else round(mean_coverage, 3)),
                "n_qs_with_incomplete_subqs": n_with_incomplete,
            })
        cc = pd.DataFrame(cc_rows).set_index("condition")
        print(cc.to_string())

    # ---------- Edge-case breakdown (CRAG conditions only) ----------
    # Pivot KHR by edge_case category for the failure-mode taxonomy in
    # the report. Rows without an edge_case tag are excluded.
    ec_df = crag_df[crag_df["edge_case"].astype(str).ne("")] if len(crag_df) else crag_df
    if len(ec_df):
        print("\n=== Edge-case KHR by category (CRAG conditions only) ===")
        ec_rows = []
        for (condition, ec), group in ec_df.groupby(["condition", "edge_case"]):
            khr_mean, khr_lo, khr_hi = _bootstrap_ci(group["keyword_hit_rate"])
            ec_rows.append({
                "condition": condition,
                "edge_case": ec,
                "n": len(group),
                "avg_khr": round(khr_mean, 3),
                "khr_95_ci": f"[{khr_lo:.3f}, {khr_hi:.3f}]",
            })
        ec_summary = pd.DataFrame(ec_rows).set_index(["condition", "edge_case"])
        print(ec_summary.to_string())

    print("\nNote: 95% confidence intervals computed via 1000-sample percentile "
          "bootstrap (seed=42). With per-cell N as low as 4-8 questions, intervals "
          "are wide; this is the honest picture of small-eval variance.")
    print("Latency includes all LLM calls (classifier + generator + completeness "
          "check + optional re-prompt) plus retrieval and reranker time.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/sec_filings"
    run_ablation(directory)
