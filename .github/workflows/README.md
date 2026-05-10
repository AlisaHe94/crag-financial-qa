# CI/CD Workflows

This directory contains GitHub Actions workflows that automatically validate
the codebase on every push and pull request to `main`.

## Workflows

### `tests.yml` — Test Suite + Lint + Schema Validation

Runs three independent jobs in parallel on every push:

| Job | Purpose | Runtime |
|---|---|---|
| `pytest` | Full unit test suite across Python 3.10/3.11/3.12 with coverage | ~30s × 3 = ~1.5 min total |
| `lint` | `ruff` static analysis on `tests/` (E, W, F rule families) | ~10s |
| `validate-data` | Schema-only test subset (validates `data/eval_questions.json`) | ~5s |

**Triggers:** push to `main`, pull request targeting `main`, manual via Actions UI.

**Concurrency:** in-flight runs of the same workflow on the same branch are
cancelled when a new commit lands. This saves CI minutes during rapid
iteration without losing the most recent result.

**Caching:** pip downloads are cached per Python version, so the second and
later runs skip redundant package downloads.

**Coverage:** the `pytest` job uploads `coverage.xml` as a workflow artifact
(retained for 14 days) — downloadable from the Actions tab for offline
inspection.

## What does NOT run in CI

- **Full ablation runs** (`evaluate.py`) — these require `sentence_transformers`,
  `faiss`, `torch`, `tavily-python`, API keys for Gemini/Groq/Tavily, and a
  pre-built FAISS index over SEC filings. Combined ~2 GB of dependencies plus
  paid API quota; out of scope for unit-test CI. Tests that depend on these
  heavy modules skip cleanly via `pytest.importorskip`.
- **Streamlit demo** — manual visual verification.
- **LLM-as-judge runs** — non-deterministic and require API keys.

## How to add a new test

1. Add a test file under `../../tests/` following the `test_*.py` convention.
2. Tests are auto-discovered by pytest. No CI configuration changes needed.
3. If the test imports a heavy module, use `pytest.importorskip("module_name")`
   at the top of each test function so it skips gracefully on minimal CI
   environments.

## Status badges

The repository's main `README.md` displays status badges that link to this
workflow. A green check on the badge means all three jobs are passing on
the latest commit to `main`.
