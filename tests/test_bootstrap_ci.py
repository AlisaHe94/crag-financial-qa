"""Tests for the bootstrap CI helper used in evaluate.py summaries.

The helper drives every "95% CI" cell in the headline ablation tables, so
seeded reproducibility and edge-case correctness matter directly for the
report's numbers.
"""
from __future__ import annotations
import math

import pytest

# Use lazy import + importorskip so the test file can be collected even when
# heavy deps (faiss, torch, sentence-transformers) aren't installed in the
# test environment. The actual import happens inside each test function.

def _get_bootstrap_ci():
    # evaluate.py transitively imports rag_baseline → document_processor →
    # sentence_transformers + faiss. Skip cleanly if those heavy deps aren't
    # installed (e.g., on a minimal CI runner).
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from evaluate import _bootstrap_ci
    return _bootstrap_ci


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_seeded_and_reproducible():
    """Two calls on the same input must return identical CIs."""
    bootstrap = _get_bootstrap_ci()
    values = [0.5, 0.7, 0.6, 0.8, 0.4, 0.65, 0.55, 0.75, 0.45, 0.85]
    a = bootstrap(values)
    b = bootstrap(values)
    assert a == b, f"Expected reproducible CI, got {a} != {b}"


def test_bootstrap_ci_returns_three_floats():
    bootstrap = _get_bootstrap_ci()
    mean, lo, hi = bootstrap([0.1, 0.2, 0.3, 0.4, 0.5])
    assert isinstance(mean, float)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean_matches_arithmetic_mean():
    """The mean returned should equal the arithmetic mean of the input."""
    bootstrap = _get_bootstrap_ci()
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean, lo, hi = bootstrap(values)
    assert mean == pytest.approx(0.3, abs=1e-9)


def test_bootstrap_ci_bounds_contain_mean():
    """For any non-degenerate sample, lo <= mean <= hi."""
    bootstrap = _get_bootstrap_ci()
    values = [0.0, 0.5, 1.0, 0.3, 0.7, 0.9, 0.1]
    mean, lo, hi = bootstrap(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_bounds_are_within_data_range():
    """CI bounds should not exceed [min, max] of the data (percentile method)."""
    bootstrap = _get_bootstrap_ci()
    values = [0.2, 0.4, 0.6, 0.8]
    mean, lo, hi = bootstrap(values)
    assert lo >= min(values) - 1e-9
    assert hi <= max(values) + 1e-9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_bootstrap_ci_empty_returns_nan_triple():
    bootstrap = _get_bootstrap_ci()
    mean, lo, hi = bootstrap([])
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_ci_single_value_is_degenerate():
    """One sample → mean = lo = hi (degenerate CI)."""
    bootstrap = _get_bootstrap_ci()
    mean, lo, hi = bootstrap([0.42])
    assert mean == pytest.approx(0.42)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def test_bootstrap_ci_filters_nans():
    """None and NaN values should be excluded from the resample population."""
    import numpy as np
    bootstrap = _get_bootstrap_ci()
    clean = [0.1, 0.2, 0.3, 0.4, 0.5]
    noisy = clean + [None, float("nan"), np.nan]
    mean_clean, _, _ = bootstrap(clean)
    mean_noisy, _, _ = bootstrap(noisy)
    assert mean_clean == pytest.approx(mean_noisy)
