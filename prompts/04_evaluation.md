# Evaluation Harness (`evaluate.py`, `judge_results.py`)

## Goal

Run each evaluation question through 4 ablation conditions and record
keyword hit rate, routing precision, latency, and an optional
multi-criteria LLM-judge score.

## Summary of AI assistance

The team designed the four ablation conditions
(`baseline_no_tables`, `baseline_tables`, `crag_no_tables`,
`crag_tables`) and the metric set. AI was used to draft the harness,
draft an initial 22-question evaluation set (which the team
reviewed), and implement the LLM-as-judge scorer.

## High-level prompts used

- *Generate an initial 22-question evaluation set covering the four
  MultiFinRAG question types. Each question needs an id, the
  question text, expected keywords, and a `ground_truth_in_corpus`
  flag. Stratify across the five companies. Save as
  `data/eval_questions.json`.*
- *Build `evaluate.py` that iterates over each question × each of
  4 conditions, records condition / question_id / question_type /
  tier_used / routing_decision / confidence / KHR / routing_correct /
  latency / answer_snippet, and saves to `data/eval_results.csv`.*
- *Make the harness idempotent: skip rows that already have a
  keyword_hit_rate so we can iterate on scoring without re-running.*
- *Build `judge_results.py`: load the eval results CSV, run an
  LLM-as-judge on each row (multi-criteria: correctness, faithfulness,
  helpfulness), append the scores and reasoning. Use Gemini-2.5-Flash
  via OpenAI-compatible endpoint as the judge.*
