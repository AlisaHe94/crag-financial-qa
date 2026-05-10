# v2 — CRAG iteration with web noise (preserved for ablation comparison)

This is **v2** of the iterated CRAG system. v2 layers four consistency-
focused interventions on top of the v1 baseline (`../baseline_crag/`)
but **does not** include the web-suppression fix that distinguishes
v3 (`../v3/`). Sub-questions in this version can trigger their own web
fallback, which qualitatively introduces off-topic content (e.g., a 2013
Apple Annual Report passage in answers about FY2024–25). v2 is preserved
here as the "before" half of the v2-vs-v3 ablation comparison.

## What's in v2 (vs. baseline_crag)

- Modality-aware query decomposition for multimodal queries
- Per-sub-question coverage check
- Numerical fidelity check (DANA-inspired post-generation verification)
- α-sensitivity sweep + α-ensemble (`scripts/alpha_sweep.py`,
  `scripts/alpha_ensemble.py`, `scripts/alpha_ensemble_ci.py`)
- 44-question eval bank with 12 edge-case categories
- Bootstrap 95% CIs on KHR / routing precision / latency
- Citation-correctness post-hoc metric (`scripts/citation_correctness.py`)
- Qualitative answer comparison (`scripts/compare_answers.py`)
- LLM-as-judge with corrected 3-criterion rubric

## What's NOT in v2 (added in v3)

- The `_suppress_web=True` flag passed from `_query_with_decomposition`
  to `query()` for sub-questions. The single-line difference is the
  whole story of v3 — diff `v2/crag_pipeline.py` against
  `../v3/crag_pipeline.py` to see it.

## Running v2

Standard reproduction sequence (same as v3 but with v2's call pattern):

```bash
# from this folder
python evaluate.py 2>&1 | tee data/eval_results_v2.log
python judge_results.py data/eval_results.csv
python scripts/alpha_sweep.py
python scripts/alpha_ensemble.py
python scripts/alpha_ensemble_ci.py
python scripts/citation_correctness.py
python scripts/compare_answers.py > data/answer_comparison.md
```

## Reference

Full project documentation, results, and the v2-vs-v3 trade-off
discussion live at the parent folder's `../README.md`.
