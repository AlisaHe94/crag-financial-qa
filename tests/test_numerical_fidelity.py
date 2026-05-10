"""Tests for the DANA-inspired numerical fidelity check in crag_pipeline.py.

The check extracts dollar amounts, percentages, and unit-prefixed numbers
from generated answers, then verifies each appears verbatim (with light
normalization) in the retrieved chunk text. This is the post-hoc
verification metric reported in `result["numerical_fidelity"]`.
"""
from __future__ import annotations
from dataclasses import dataclass

import pytest


# Lightweight stand-in for DocumentChunk so tests don't need the heavy
# document_processor module loaded. The fidelity checker only reads `.text`.
@dataclass
class FakeChunk:
    text: str


def _get_fidelity_check():
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from crag_pipeline import _check_numerical_fidelity
    return _check_numerical_fidelity


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

def test_extracts_dollar_amounts():
    check = _get_fidelity_check()
    answer = "Apple net sales were $416 billion in fiscal 2025."
    chunks = [FakeChunk(text="Net sales: $416 billion. iPhone: $209 million.")]
    result = check(answer, chunks)
    assert any("416" in n for n in result["numbers_in_answer"])


def test_extracts_percentages():
    check = _get_fidelity_check()
    answer = "Gross margin was 38.5% this quarter."
    chunks = [FakeChunk(text="Gross margin: 38.5% (vs 37.0% prior).")]
    result = check(answer, chunks)
    assert any("38.5%" in n.replace(" ", "") for n in result["numbers_in_answer"])


def test_extracts_unit_prefixed_numbers():
    """Numbers like '209 million' (no $) should be extracted."""
    check = _get_fidelity_check()
    answer = "Revenue was 209 million."
    chunks = [FakeChunk(text="Revenue: 209 million.")]
    result = check(answer, chunks)
    assert len(result["numbers_in_answer"]) >= 1


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------

def test_verified_numbers_when_present_in_chunks():
    check = _get_fidelity_check()
    answer = "Apple iPhone revenue was $209 billion."
    chunks = [FakeChunk(text="Apple iPhone net sales: $209 billion in FY25.")]
    result = check(answer, chunks)
    assert result["fidelity_score"] == pytest.approx(1.0)
    assert len(result["unverified_numbers"]) == 0


def test_unverified_numbers_when_absent_from_chunks():
    """A number in the answer that doesn't appear in any chunk is flagged."""
    check = _get_fidelity_check()
    answer = "Apple revenue was $999 billion (fabricated number)."
    chunks = [FakeChunk(text="Apple net sales: $416 billion. iPhone $209 million.")]
    result = check(answer, chunks)
    assert result["fidelity_score"] < 1.0
    # The fabricated $999 should be flagged
    assert any("999" in n for n in result["unverified_numbers"])


def test_partial_fidelity_score():
    """Mix of verified + unverified numbers → fractional fidelity score."""
    check = _get_fidelity_check()
    answer = "Apple revenue was $416 billion and iPhone was $999 million."
    chunks = [FakeChunk(text="Apple net sales: $416 billion in FY25.")]
    result = check(answer, chunks)
    # 1 of 2 numbers verified → 0.5
    assert 0.0 < result["fidelity_score"] < 1.0


# ---------------------------------------------------------------------------
# Normalization (commas, $ sign)
# ---------------------------------------------------------------------------

def test_comma_normalization():
    """'$1,234' in answer should match '1234' in chunks (or vice versa)."""
    check = _get_fidelity_check()
    answer = "Apple revenue was $1,234 million."
    chunks = [FakeChunk(text="Apple revenue: 1234 million dollars")]
    result = check(answer, chunks)
    assert result["fidelity_score"] == pytest.approx(1.0)


def test_dollar_sign_optional():
    """An answer with '$209' should match a chunk with '209' (no $)."""
    check = _get_fidelity_check()
    answer = "iPhone revenue: $209 million."
    chunks = [FakeChunk(text="iPhone net sales 209 million")]
    result = check(answer, chunks)
    assert result["fidelity_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_answer_returns_perfect_score():
    check = _get_fidelity_check()
    result = check("", [FakeChunk(text="some text")])
    assert result["fidelity_score"] == pytest.approx(1.0)
    assert result["numbers_in_answer"] == []


def test_answer_with_no_numbers_returns_perfect_score():
    check = _get_fidelity_check()
    result = check("Apple has cloud services.", [FakeChunk(text="Apple cloud.")])
    assert result["fidelity_score"] == pytest.approx(1.0)


def test_no_chunks_flags_all_numbers():
    """If no chunks were retrieved, every number in the answer is unverified."""
    check = _get_fidelity_check()
    result = check("Apple revenue was $416 billion.", [])
    assert len(result["unverified_numbers"]) >= 1
    assert result["fidelity_score"] == pytest.approx(0.0)
