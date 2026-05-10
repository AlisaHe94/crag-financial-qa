"""Side-by-side answer comparison for qualitative spot-checks.

KHR can mask real improvement (or fake it) when answers are similar in
keyword density but differ in correctness or grounding. This script picks
a small set of high-signal questions and prints all four conditions'
answers next to each other so you can visually decide whether the
quantitative win is also a qualitative win.

Picks questions across three categories:
  1. Big CRAG wins — questions where CRAG's KHR is substantially higher
     than baseline. Sanity-check that the gap is real, not just keyword luck.
  2. Tied or close — questions where conditions converge. Useful for
     seeing whether CRAG's longer/richer answers are actually better
     even when KHR doesn't reflect it.
  3. Failure modes — edge-case categories where ALL conditions struggle
     (e.g., balance_sheet_lookup at 0.3-0.4 KHR). Inspect the failure to
     write the Limitations section's worked-examples paragraph.

Usage:
    python scripts/compare_answers.py
    python scripts/compare_answers.py --csv data/eval_results.csv

Output is markdown — pipe it to a file so you can paste sections into
the report verbatim:
    python scripts/compare_answers.py > data/answer_comparison.md
"""

from __future__ import annotations
import argparse
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

# Question IDs to inspect. Curated to cover three categories.
# Adjust this list if you want to focus on different cases.
QUESTIONS_TO_INSPECT = [
    # --- Big CRAG wins on multimodal (Type 4) ---
    ("m1", "CRAG win — multimodal: iPhone YoY change + management reasoning"),
    ("m4", "CRAG win — multimodal: Meta DAU + Reality Labs spending"),
    # --- Mixed: KHR ties but content may differ ---
    ("t3", "KHR tie — table: Microsoft Intelligent Cloud revenue"),
    ("ec3", "Keyword fix worked — year-over-year arithmetic"),
    # --- Failure modes (low KHR across the board) ---
    ("ec7", "Failure mode — balance_sheet_lookup (Apple cash)"),
    ("ec17", "Failure mode — balance_sheet_lookup (Microsoft total assets)"),
    ("ec4", "Long-form summary — China/Asia risks"),
    # --- Out-of-corpus: should refuse or use web ---
    ("oc1", "OOC — federal funds rate (current)"),
]

CONDITION_ORDER = [
    "baseline_no_tables",
    "baseline_tables",
    "crag_no_tables",
    "crag_tables",
]


def load_questions(path: Path) -> dict[str, dict]:
    return {q["id"]: q for q in json.loads(path.read_text())}


def truncate(s: str, n: int = 1500) -> str:
    if not s or pd.isna(s):
        return "_(no answer)_"
    s = str(s).strip()
    if len(s) <= n:
        return s
    return s[:n] + "...[truncated]"


def format_question_block(qid: str, label: str, q_meta: dict,
                          rows_by_condition: dict[str, dict]) -> str:
    out: list[str] = []
    out.append(f"## {qid} — {label}\n")
    out.append(f"**Question:** {q_meta.get('question', '(missing)')}\n")
    out.append(
        f"**Type:** {q_meta.get('type', '?')}  |  "
        f"**Edge case:** `{q_meta.get('edge_case', '—')}`  |  "
        f"**Ground truth in corpus:** {q_meta.get('ground_truth_in_corpus', '?')}\n"
    )
    out.append(
        f"**Expected keywords:** "
        f"{', '.join('`' + k + '`' for k in q_meta.get('expected_keywords', []))}\n"
    )

    # KHR table per condition
    out.append("\n### KHR / fidelity / coverage per condition\n")
    out.append("| condition | KHR | fidelity | coverage | unverified |")
    out.append("|---|---|---|---|---|")
    for cond in CONDITION_ORDER:
        row = rows_by_condition.get(cond, {})
        khr = row.get("keyword_hit_rate")
        fidelity = row.get("fidelity_score")
        coverage = row.get("sub_question_coverage_rate")
        unverified = row.get("unverified_numbers", "")
        out.append(
            f"| {cond} "
            f"| {khr if khr is not None else '—'} "
            f"| {fidelity if fidelity is not None and not pd.isna(fidelity) else '—'} "
            f"| {coverage if coverage is not None and not pd.isna(coverage) else '—'} "
            f"| {(unverified[:60] + '…') if isinstance(unverified, str) and len(unverified) > 60 else (unverified or '—')} |"
        )

    # Answers per condition
    out.append("\n### Answers\n")
    for cond in CONDITION_ORDER:
        row = rows_by_condition.get(cond, {})
        ans = truncate(row.get("answer_snippet", ""))
        # Wrap long lines for readability when piped to a terminal.
        wrapped = "\n".join(
            textwrap.wrap(ans, width=100, replace_whitespace=False)
            if "\n" not in ans
            else ans.splitlines()
        )
        out.append(f"#### `{cond}`")
        out.append(f"```\n{wrapped}\n```\n")

    out.append("\n---\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval_results.csv",
                    help="Path to eval_results.csv (default: data/eval_results.csv)")
    ap.add_argument("--questions", default="data/eval_questions.json",
                    help="Path to eval_questions.json")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run evaluate.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    qmeta = load_questions(Path(args.questions))

    print("# Side-by-side answer comparison\n")
    print(
        "Generated by `scripts/compare_answers.py`. Use this to visually "
        "verify whether KHR-level improvements correspond to genuinely "
        "better answers, or to extract worked failure-mode examples for "
        "the report's Limitations section.\n"
    )

    for qid, label in QUESTIONS_TO_INSPECT:
        q_meta = qmeta.get(qid)
        if q_meta is None:
            print(f"WARNING: question id {qid} not found in eval_questions.json",
                  file=sys.stderr)
            continue
        sub = df[df["question_id"] == qid]
        if sub.empty:
            print(f"WARNING: no rows for {qid} in {csv_path}", file=sys.stderr)
            continue
        rows_by_condition = {row["condition"]: row.to_dict()
                             for _, row in sub.iterrows()}
        print(format_question_block(qid, label, q_meta, rows_by_condition))


if __name__ == "__main__":
    main()
