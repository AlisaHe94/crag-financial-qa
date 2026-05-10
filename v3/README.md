# v3 — CRAG iteration with web noise removed (recommended)

This is **v3** of the iterated CRAG system, the recommended version.
v3 inherits everything from v2 (`../v2/`) and adds one architectural
fix: decomposed sub-questions are corpus-bound. They no longer trigger
their own web fallback, which had been pulling in off-topic content
(e.g., a 2013 Apple Annual Report passage in answers about FY2024–25)
and tanking the LLM-as-judge faithfulness metric.

The single-line difference is `_suppress_web=True` passed from
`_query_with_decomposition` to `query()`, plus the `_suppress_web`
parameter wired through `query()` to gate four web-firing paths.
Diff `v2/crag_pipeline.py` against `crag_pipeline.py` in this folder
to see exactly what changed.

## What v3 adds vs. v2

- `_suppress_web` parameter in `query()` that disables four web-firing
  paths (OOC bypass, AMBIGUOUS web fallback, OOC refusal rescue,
  multimodal completeness augment) when set to True
- `_query_with_decomposition` passes `_suppress_web=True` for every
  sub-question call, while top-level OOC queries still go to web normally

## What v3 inherits from v2 (unchanged)

- Modality-aware query decomposition
- Per-sub-question coverage check
- Numerical fidelity check (DANA-inspired)
- α-sensitivity sweep + α-ensemble
- 44-question eval bank with edge cases
- Bootstrap 95% CIs
- Citation-correctness metric
- Qualitative answer comparison
- Corrected 3-criterion LLM-as-judge rubric

## Running v3

```bash
# from this folder
python evaluate.py 2>&1 | tee data/eval_results_v3.log
python judge_results.py data/eval_results.csv
python scripts/alpha_sweep.py
python scripts/alpha_ensemble.py
python scripts/alpha_ensemble_ci.py
python scripts/citation_correctness.py
python scripts/compare_answers.py > data/answer_comparison.md
```

## Empirical impact of the v3 fix (v2 → v3 deltas, n=44)

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| Routing precision (crag_no_tables) | 0.841 | 0.886 | **+0.04** |
| Routing precision (crag_tables) | 0.795 | 0.864 | **+0.07** |
| Numerical fidelity (crag_no_tables) | 0.683 | 0.729 | **+0.05** |
| Sub-question coverage (crag_tables) | 0.818 | 0.900 | **+0.08** |
| KHR (crag_no_tables) | 0.750 | 0.764 | +0.01 |
| KHR (crag_tables) | 0.727 | 0.723 | −0.00 |

The v3 fix improved every secondary metric while keeping KHR essentially
flat — confirming the qualitative noise hypothesis without trading off
the headline metric.

## Reference

Full project documentation, results tables, and the v2-vs-v3 trade-off
discussion live at the parent folder's `../README.md`.
