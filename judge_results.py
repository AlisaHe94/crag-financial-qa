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
    "financial documents (SEC 10-K and 10-Q filings). You score answers on "
    "THREE separate dimensions, NOT on phrasing or word choice. Output ONLY a "
    "JSON object with four keys:\n"
    "  - `correctness` (0.0 to 1.0): does the answer factually answer the "
    "question? 1.0 = correct and complete, 0.5 = partially correct, 0.0 = "
    "wrong, hallucinated, or refused without justification.\n"
    "  - `faithfulness` (0.0 to 1.0): is the answer grounded in identifiable "
    "sources? Cites section names, page numbers, filing references, or hedges "
    "with 'according to' = high. Confidently asserts numbers without grounding "
    "= low. Refuses = neutral 0.5 (no claims to verify).\n"
    "  - `helpfulness` (0.0 to 1.0): does the answer ADDRESS the question or "
    "just punt? Substantively engages with the question = 1.0. Says "
    "'insufficient information' = 0.2 (technically a non-answer). Provides "
    "current/live data when asked = 1.0.\n"
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

    for n, idx in enumerate(todo, start=1):
        row = df.loc[idx]
        qid = row["question_id"]
        q = questions.get(qid, {})
        question = q.get("question", "")
        keywords = q.get("expected_keywords", [])
        in_corpus = q.get("ground_truth_in_corpus", True)
        answer = str(row.get("answer_snippet", "") or "")

        if not answer.strip():
            df.at[idx, "judge_correctness"] = 0.0
            df.at[idx, "judge_faithfulness"] = 0.0
            df.at[idx, "judge_helpfulness"] = 0.0
            df.at[idx, "llm_judge_score"] = 0.0
            df.at[idx, "llm_judge_reason"] = "empty answer"
            logger.info(f"  [{n}/{len(todo)}] {row['condition']}/{qid}: 0.00 (empty)")
            continue

        user_prompt = _build_judge_user(question, answer, keywords, in_corpus)
        try:
            raw = judge(JUDGE_SYSTEM, user_prompt)
        except Exception as e:
            logger.warning(f"  [{n}/{len(todo)}] {row['condition']}/{qid}: judge failed ({e})")
            df.at[idx, "llm_judge_reason"] = f"judge_error: {e}"
            time.sleep(2.0)  # back off on errors
            continue

        c, f, h, reason = _parse_multi_score(raw)
        df.at[idx, "judge_correctness"] = c
        df.at[idx, "judge_faithfulness"] = f
        df.at[idx, "judge_helpfulness"] = h
        df.at[idx, "llm_judge_score"] = round((c + f + h) / 3, 3)  # mean = legacy column
        df.at[idx, "llm_judge_reason"] = reason
        logger.info(
            f"  [{n}/{len(todo)}] {row['condition']}/{qid} (type {row['question_type']}): "
            f"C={c:.2f} F={f:.2f} H={h:.2f} — {reason[:60]}"
        )

        # Write back after every row so an interruption doesn't lose progress.
        if n % 5 == 0 or n == len(todo):
            df.to_csv(results_path, index=False)

        # Pacing for Groq free tier (6k TPM, 30 RPM). 2s per call ≈ 30 RPM
        # exactly. Slow but predictable; with 88 rows we finish in ~3-5 min.
        time.sleep(2.0)

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
