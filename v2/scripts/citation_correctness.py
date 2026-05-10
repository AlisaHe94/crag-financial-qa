"""Citation-correctness post-hoc analysis.

Measures whether each answer cites the *right company's filing* for the
question it was asked. For an Apple question, the answer should mention
Apple's filings (10-K or 10-Q); citations of Amazon, Microsoft, etc. are
*cross-company contamination* signals.

This complements KHR by capturing whether the LLM grounds its answer in
the correct company's filing — a metric especially relevant for
financial QA where cross-company contamination is a real failure mode
that KHR can't detect.

Output: per-condition citation-correctness score and per-question
breakdown.

Reads: data/eval_results.csv (already produced by evaluate.py)
Writes: scripts/citation_correctness_results.csv
"""

from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


# Map ticker → expected company name patterns in answer text
# (a question's expected company is identified by its ticker, parsed
# from the question text or eval_questions.json's question_id pattern).
COMPANY_PATTERNS: dict[str, list[str]] = {
    "AAPL":  [r"\bApple\b", r"\bAAPL\b", r"\biPhone\b", r"\biPad\b"],
    "MSFT":  [r"\bMicrosoft\b", r"\bMSFT\b", r"\bAzure\b"],
    "GOOGL": [r"\bAlphabet\b", r"\bGOOGL\b", r"\bGoogle Search\b", r"\bYouTube\b"],
    "AMZN":  [r"\bAmazon\b", r"\bAMZN\b", r"\bAWS\b"],
    "META":  [r"\bMeta\b", r"\bMETA\b", r"\bReality Labs\b",
              r"\bFamily of Apps\b"],
}


def expected_ticker_for_question(question: str) -> str | None:
    """Identify which company a question is asking about from its text."""
    for ticker, patterns in COMPANY_PATTERNS.items():
        if any(re.search(p, question, flags=re.IGNORECASE) for p in patterns):
            return ticker
    return None


def cited_companies_in_answer(answer: str) -> set[str]:
    """Return the set of tickers whose company is mentioned in an answer."""
    if not answer:
        return set()
    cited: set[str] = set()
    for ticker, patterns in COMPANY_PATTERNS.items():
        if any(re.search(p, answer, flags=re.IGNORECASE) for p in patterns):
            cited.add(ticker)
    return cited


def score_row(question: str, answer: str, in_corpus: bool) -> dict:
    """Score a single (question, answer) pair on citation correctness.

    Returns:
        - expected_ticker: which company the question asks about (None if OOC)
        - cited_companies: which companies the answer mentions
        - citation_correct: 1.0 if expected ticker is mentioned and no
          OTHER company is mentioned (no contamination), 0.5 if expected
          + others (partial), 0.0 if expected absent or wrong company only
    """
    expected = expected_ticker_for_question(question)
    cited = cited_companies_in_answer(answer)

    # OOC questions don't have a single "right" company — exclude from scoring.
    if not in_corpus or expected is None:
        return {
            "expected_ticker": expected,
            "cited_companies": "|".join(sorted(cited)),
            "citation_correct": None,
            "contamination": None,
        }

    if expected in cited:
        # Right company cited. Check for cross-company contamination.
        others = cited - {expected}
        if not others:
            score = 1.0
            contamination = 0
        else:
            # Partial: cited the right one but also others (could be
            # legitimate comparison or contamination)
            score = 0.5
            contamination = len(others)
    else:
        score = 0.0
        contamination = len(cited)  # all citations are "wrong"
    return {
        "expected_ticker": expected,
        "cited_companies": "|".join(sorted(cited)),
        "citation_correct": score,
        "contamination": contamination,
    }


def main():
    results_path = Path("data/eval_results.csv")
    questions_path = Path("data/eval_questions.json")

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")

    # Load original questions to get ground_truth_in_corpus and question text
    import json
    questions = json.loads(questions_path.read_text())
    qmap = {q["id"]: q for q in questions}

    rows = []
    for _, r in df.iterrows():
        q = qmap.get(r["question_id"], {})
        question_text = q.get("question", "")
        in_corpus = bool(q.get("ground_truth_in_corpus", True))
        scoring = score_row(question_text, str(r.get("answer_snippet", "")), in_corpus)
        rows.append({
            "condition": r["condition"],
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "in_corpus": in_corpus,
            **scoring,
        })

    out = pd.DataFrame(rows)
    out_path = Path("scripts/citation_correctness_results.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved per-question scores to {out_path}")

    print("\n=== Citation Correctness — Overall (in-corpus questions only) ===")
    in_corpus_df = out[out["citation_correct"].notna()]
    by_cond = (
        in_corpus_df.groupby("condition")
        .agg(
            citation_correct=("citation_correct", "mean"),
            cross_company_contamination=("contamination", "mean"),
            n=("citation_correct", "count"),
        )
        .round(3)
    )
    print(by_cond.to_string())

    print("\n=== Citation Correctness — By Question Type ===")
    by_type = (
        in_corpus_df.groupby(["condition", "question_type"])
        .agg(
            citation_correct=("citation_correct", "mean"),
            n=("citation_correct", "count"),
        )
        .round(3)
    )
    print(by_type.to_string())

    print("\nNote: citation_correct = 1.0 if expected company cited and no "
          "others; 0.5 if expected + others (partial); 0.0 if expected absent. "
          "cross_company_contamination = mean number of OTHER companies "
          "mentioned per answer (lower = more focused).")


if __name__ == "__main__":
    main()
