"""Tests for the KHR primary metric used across all eval result tables."""
from __future__ import annotations

import pytest


def _get_khr():
    pytest.importorskip("pandas")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from evaluate import keyword_hit_rate
    return keyword_hit_rate


def test_khr_full_match():
    khr = _get_khr()
    score = khr("Apple iPhone net sales were 209 billion in fiscal year 2025",
                ["iphone", "209", "fiscal"])
    assert score == pytest.approx(1.0)


def test_khr_partial_match():
    khr = _get_khr()
    score = khr("Apple revenue grew",
                ["apple", "revenue", "iphone", "fiscal"])
    # 2 of 4 keywords matched → 0.5
    assert score == pytest.approx(0.5)


def test_khr_no_match():
    khr = _get_khr()
    score = khr("Microsoft cloud growth",
                ["apple", "iphone", "ipad"])
    assert score == pytest.approx(0.0)


def test_khr_case_insensitive():
    """KHR should ignore case so 'IPHONE' in expected matches 'iphone' in answer."""
    khr = _get_khr()
    score_lower = khr("apple iphone revenue", ["IPHONE", "APPLE"])
    score_upper = khr("APPLE IPHONE REVENUE", ["iphone", "apple"])
    assert score_lower == pytest.approx(1.0)
    assert score_upper == pytest.approx(1.0)


def test_khr_empty_keywords_returns_zero():
    """Empty expected_keywords should not crash; returns 0.0."""
    khr = _get_khr()
    assert khr("any answer", []) == pytest.approx(0.0)


def test_khr_empty_answer_returns_zero():
    khr = _get_khr()
    assert khr("", ["apple", "revenue"]) == pytest.approx(0.0)


def test_khr_handles_none_answer():
    """A None answer (which can occur on LLM failure) should not crash."""
    khr = _get_khr()
    assert khr(None, ["apple"]) == pytest.approx(0.0)


def test_khr_substring_matching():
    """Keywords match anywhere in the answer (substring), not as whole words.

    This is intentional and documented — a question expecting '209' will
    match an answer containing '209,586'. The trade-off is that some
    keywords are legitimately substrings of larger numbers, which is
    the correct behavior for financial QA.
    """
    khr = _get_khr()
    score = khr("Apple iPhone net sales were 209,586 million", ["209"])
    assert score == pytest.approx(1.0)
