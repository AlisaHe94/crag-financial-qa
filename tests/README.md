# tests/

Unit tests for the project's pure-Python helpers. Tests target v3 as the
canonical version (v1/v2/v3 share these helpers' interfaces).

## Run

From the repo root:

```bash
pip install pytest
pytest tests/ -v
```

Tests gracefully skip via `pytest.importorskip` when heavy dependencies
(`sentence_transformers`, `faiss`, `torch`) aren't installed — useful for
running the schema tests on a minimal CI environment without GPU support.

## What's tested

| File | Function under test | What it verifies |
|---|---|---|
| `test_bootstrap_ci.py` | `evaluate._bootstrap_ci` | Seeded reproducibility (seed=42), arithmetic mean correctness, CI bounds containment, NaN handling, single-sample degeneracy |
| `test_keyword_hit_rate.py` | `evaluate.keyword_hit_rate` | Full / partial / no match, case insensitivity, empty inputs, substring matching |
| `test_numerical_fidelity.py` | `crag_pipeline._check_numerical_fidelity` | Dollar / percentage / unit-prefixed number extraction; verification against retrieved chunks; comma + $ normalization |
| `test_parse_multi_score.py` | `judge_results._parse_multi_score` | Strict JSON, markdown fence stripping, regex fallback, score clipping to [0,1] |
| `test_eval_questions_schema.py` | `data/eval_questions.json` | Schema integrity, ID uniqueness, keyword count = 5, type validity, edge-case category coverage |

Total: 48 tests across 5 files.

## What's NOT tested (and why)

- **End-to-end pipeline runs** (`crag.query()`, `evaluate.run_ablation()`) — these require API keys, a built FAISS index, and the SEC corpus. Out of scope for unit tests; covered by the smoke test (`smoke_test.py`) and the live ablation runs.
- **LLM responses** — non-deterministic; covered indirectly by the resume-aware ablation harness which records per-row results to CSV.
- **Streamlit UI behavior** — manual visual testing in the demo flow.
