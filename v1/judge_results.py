"""
LLM-as-judge scorer for ablation results.

The keyword-hit-rate metric in `evaluate.py` is a phrasing-sensitive proxy
that rewards refusals containing common keywords ("rate", "federal" appearing
in "I cannot verify the current rate") and penalizes correct answers phrased
differently from the expected keyword list. This script re-scores every row
in `data/eval_results.csv` using Gemini-Flash as a judge — same answers,
fairer metric.

Usage:
    python judge_results.py [path/to/eval_results.csv]

Behavior:
    * Idempotent: rows that already have a non-null `llm_judge_score` are skipped.
    * Writes back to the same CSV after every row (safe to interrupt).
    * Adds two new columns: `llm_judge_score` (0.0–1.0) and `llm_judge_reason`
      (one-line explanation; helpful for debugging the judge's decisions).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rag_baseline import _build_single_client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path("data/eval_results.csv")
EVAL_QUESTIONS_PATH = Path("data/eval_questions.json")


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of question-answering systems for "
    "financial documents (SEC 10-K and 10-Q filings).\n\n"
    "CONTEXT YOU MUST ASSUME:\n"
    "- Today's date is May 9, 2026. Fiscal year 2025 and 2026 SEC filings "
    "have been filed and exist publicly. Apple's FY2025 10-K (year ended "
    "September 27, 2025), Microsoft's FY2025 10-K (year ended June 30, "
    "2025), Meta/Alphabet/Amazon FY2025 10-Ks (years ended December 31, "
    "2025) are all real. Do NOT flag specific 2025 or 2026 fiscal-year "
    "figures as 'hallucinated future content' — they refer to filings the "
    "system has retrieved from a local corpus.\n"
    "- The corpus consists of HTML SEC filings. HTML filings have no real "
    "page boundaries. Cited 'page numbers' in answers are FABRICATED — do "
    "NOT reward citation of page numbers as faithfulness signal. Reward "
    "citation of section names ('Risk Factors', 'Management's Discussion "
    "and Analysis'), statement names ('Consolidated Statements of "
    "Operations'), or fiscal periods.\n"
    "- For financial QA, refusing to answer when the corpus genuinely "
    "doesn't contain the data is the CORRECT behavior. A confident wrong "
    "answer is much worse than an honest 'insufficient information.' Score "
    "honest refusal accordingly.\n\n"
    "Output ONLY a JSON object with four keys, scored on these THREE "
    "dimensions (NOT on phrasing or word choice):\n\n"
    "- `correctness` (0.0 to 1.0): does the answer factually answer the "
    "question?\n"
    "    1.0 = correct and complete\n"
    "    0.7 = honest refusal because the corpus does not contain the data "
    "(this is the right behavior — score is high but capped because the "
    "user still didn't get an answer)\n"
    "    0.5 = partially correct, or correct but missing context\n"
    "    0.2 = clearly wrong but the system flagged its uncertainty\n"
    "    0.0 = confidently wrong / hallucinated facts presented as truth\n\n"
    "- `faithfulness` (0.0 to 1.0): is the answer grounded in identifiable "
    "sources?\n"
    "    1.0 = explicit grounding in section/statement names with hedges "
    "like 'according to' or 'as stated in'\n"
    "    0.7 = mostly grounded but some claims uncited\n"
    "    0.5 = honest refusal (no claims to verify — neutral)\n"
    "    0.3 = confidently asserts numbers/facts without grounding\n"
    "    0.0 = answer contradicts what the cited source actually says, OR "
    "fabricates citations (page numbers, made-up section names)\n\n"
    "- `helpfulness` (0.0 to 1.0): does the answer ADDRESS the question?\n"
    "    1.0 = substantively engages and answers\n"
    "    0.7 = honest refusal due to corpus silence on the question — this "
    "is GOOD behavior, not a punt; the system is preventing fabrication\n"
    "    0.4 = answers but with significant gaps or off-topic content\n"
    "    0.2 = pure punt with no engagement (e.g., 'I don't know' with "
    "no explanation of WHY the corpus is silent)\n"
    "    0.0 = ignores the question entirely\n\n"
    "  - `reason` (one short sentence justifying the lowest score).\n"
    "Do not include markdown."
)


def _build_judge_user(
    question: str,
    answer: str,
    expected_keywords: list[str],
    in_corpus: bool,
) -> str:
    hint_line = (
        f"\nHints (the answer should reference at least some of these "
        f"concepts, though exact wording is not required): {expected_keywords}"
        if expected_keywords else ""
    )

    corpus_note = (
        "\nThis question asks about LIVE/CURRENT data NOT in SEC filings. "
        "A correct answer either provides up-to-date information (via web "
        "search) OR explicitly admits the system cannot verify current data."
        if not in_corpus else
        "\nThis question's answer should be present in the SEC filings. A "
        "correct answer extracts and reports the relevant fact."
    )

    if not answer or not answer.strip():
        # Empty answer is automatically 0.0 — skip the LLM call.
        return ""

    return (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer.strip()}"
        f"{hint_line}"
        f"{corpus_note}\n\n"
        "Score on the THREE dimensions described in the system prompt. "
        "Output JSON ONLY, e.g.:\n"
        '{"correctness": 0.8, "faithfulness": 0.9, "helpfulness": 1.0, '
        '"reason": "Correct number, well-cited."}'
    )


# ---------------------------------------------------------------------------
# JSON-extraction helper
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r'(?:^|[^0-9])([01](?:\.\d+)?)')


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse_multi_score(raw: str) -> tuple[float, float, float, str]:
    """Extract (correctness, faithfulness, helpfulness, reason) from judge response.

    Defaults a missing/unparseable dimension to 0.0 so noise can't accidentally
    score broken responses high. The judge being unable to parse its own
    output is itself a signal that something is wrong.
    """
    if not raw:
        return 0.0, 0.0, 0.0, "judge returned empty"
    raw = raw.strip()

    # Strip markdown code fences if the model added them anyway.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    # Try strict JSON first.
    try:
        parsed = json.loads(raw)
        c = _clip(parsed.get("correctness", 0.0))
        f = _clip(parsed.get("faithfulness", 0.0))
        h = _clip(parsed.get("helpfulness", 0.0))
        reason = str(parsed.get("reason", ""))[:200]
        return c, f, h, reason
    except Exception:
        pass

    # Fallback: regex-extract each named field.
    def _grab(name: str) -> float:
        m = re.search(rf'"{name}"\s*:\s*([0-9]*\.?[0-9]+)', raw)
        if m:
            try:
                return _clip(float(m.group(1)))
            except ValueError:
                return 0.0
        return 0.0

    c = _grab("correctness")
    f = _grab("faithfulness")
    h = _grab("helpfulness")
    if c + f + h == 0:
        return 0.0, 0.0, 0.0, f"unparseable: {raw[:120]}"
    return c, f, h, f"regex-recovered: {raw[:120]}"


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def main(results_path: Path = DEFAULT_RESULTS_PATH) -> None:
    if not results_path.exists():
        logger.error(f"Results CSV not found at {results_path}")
        sys.exit(1)

    df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(df)} rows from {results_path}")

    # Ensure judge columns exist (multi-criteria: 3 separate scores).
    for col in ("judge_correctness", "judge_faithfulness", "judge_helpfulness"):
        if col not in df.columns:
            df[col] = pd.NA
    if "llm_judge_reason" not in df.columns:
        df["llm_judge_reason"] = ""
    # Backwards-compat alias — overall judge score = mean of three dimensions.
    if "llm_judge_score" not in df.columns:
        df["llm_judge_score"] = pd.NA

    # Build a question lookup so we have access to question text + expected
    # keywords + ground_truth_in_corpus from the eval_questions.json file.
    if EVAL_QUESTIONS_PATH.exists():
        questions = {q["id"]: q for q in json.loads(EVAL_QUESTIONS_PATH.read_text())}
        logger.info(f"Loaded {len(questions)} questions from {EVAL_QUESTIONS_PATH}")
    else:
        logger.error(f"Question set not found at {EVAL_QUESTIONS_PATH}")
        sys.exit(1)

    # Judge: Gemini-2.5-Flash on paid tier (user added $10 quota).
    # Why Gemini over Groq's Llama: Groq's free-tier daily quotas were
    # exhausted across both 70B (100K limit) and partially 8B during today's
    # iterations. With paid Gemini we get a single consistent judge model
    # across all 88 rows, removing the mixed-model bias concern that was
    # going to require methodology caveats.
    import os
    os.environ.setdefault("__JUDGE_MODE__", "1")
    judge = _build_single_client("gemini", model="gemini-2.5-flash")
    logger.info("Judge: gemini-2.5-flash (paid tier — single consistent judge across all rows)")

    todo = df[df["judge_correctness"].isna()].index.tolist()
    logger.info(f"{len(todo)} rows need scoring; {len(df) - len(todo)} already done")

    # ----- Concurrent scoring -----
    # Paid Gemini gives us much higher RPM than the free tier (1000+ RPM at
    # paid tier 1), so we can run requests concurrently. ThreadPoolExecutor
    # with 8 workers gives ~8× throughput while staying well under quota.
    # Errors back off via per-future try/except; on rate-limit errors we
    # sleep briefly and retry once before giving up on that row.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    NUM_WORKERS = 8
    _df_lock = threading.Lock()

    def _score_one(idx: int) -> tuple[int, dict]:
        """Worker: score a single row. Returns (idx, fields_to_write)."""
        row = df.loc[idx]
        qid = row["question_id"]
        q = questions.get(qid, {})
        question = q.get("question", "")
        keywords = q.get("expected_keywords", [])
        in_corpus = q.get("ground_truth_in_corpus", True)
        answer = str(row.get("answer_snippet", "") or "")

        if not answer.strip():
            return idx, {
                "judge_correctness": 0.0,
                "judge_faithfulness": 0.0,
                "judge_helpfulness": 0.0,
                "llm_judge_score": 0.0,
                "llm_judge_reason": "empty answer",
            }

        user_prompt = _build_judge_user(question, answer, keywords, in_corpus)
        # One retry on transient errors (rate limit, timeout). If retry also
        # fails, surface the error in llm_judge_reason instead of crashing
        # the whole batch.
        for attempt in (1, 2):
            try:
                raw = judge(JUDGE_SYSTEM, user_prompt)
                break
            except Exception as e:
                if attempt == 1:
                    time.sleep(3.0)
                    continue
                return idx, {"llm_judge_reason": f"judge_error: {e}"}

        c, f, h, reason = _parse_multi_score(raw)
        return idx, {
            "judge_correctness": c,
            "judge_faithfulness": f,
            "judge_helpfulness": h,
            "llm_judge_score": round((c + f + h) / 3, 3),
            "llm_judge_reason": reason,
        }

    completed_count = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_idx = {executor.submit(_score_one, idx): idx for idx in todo}
        for future in as_completed(future_to_idx):
            idx, fields = future.result()
            with _df_lock:
                for col, val in fields.items():
                    df.at[idx, col] = val
                completed_count += 1
                row = df.loc[idx]
                if "judge_correctness" in fields:
                    logger.info(
                        f"  [{completed_count}/{len(todo)}] "
                        f"{row['condition']}/{row['question_id']} "
                        f"(type {row['question_type']}): "
                        f"C={fields.get('judge_correctness', 0):.2f} "
                        f"F={fields.get('judge_faithfulness', 0):.2f} "
                        f"H={fields.get('judge_helpfulness', 0):.2f} — "
                        f"{str(fields.get('llm_judge_reason', ''))[:60]}"
                    )
                # Periodic flush so an interruption doesn't lose progress.
                if completed_count % 10 == 0:
                    df.to_csv(results_path, index=False)

    df.to_csv(results_path, index=False)
    logger.info(f"Wrote {len(df)} rows back to {results_path}")
    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== LLM-as-Judge — Overall (3 dimensions) ===")
    overall = (
        df.groupby("condition")
        .agg(
            khr=("keyword_hit_rate", "mean"),
            correctness=("judge_correctness", "mean"),
            faithfulness=("judge_faithfulness", "mean"),
            helpfulness=("judge_helpfulness", "mean"),
            n=("question_id", "count"),
        )
        .round(3)
    )
    print(overall.to_string())

    print("\n=== By Question Type (judge dimensions) ===")
    by_type = (
        df.groupby(["condition", "question_type"])
        .agg(
            correctness=("judge_correctness", "mean"),
            faithfulness=("judge_faithfulness", "mean"),
            helpfulness=("judge_helpfulness", "mean"),
            n=("question_id", "count"),
        )
        .round(3)
    )
    print(by_type.to_string())


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH
    main(path)
