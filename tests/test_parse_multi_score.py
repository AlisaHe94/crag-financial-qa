"""Tests for the LLM-as-judge response parser in judge_results.py.

The judge returns a JSON object with correctness/faithfulness/helpfulness
scores. The parser handles strict JSON, JSON wrapped in markdown code
fences, and regex-fallback extraction when the model produces malformed
output. Reliable parsing is the difference between a usable judge metric
and silent zeros across the eval CSV.
"""
from __future__ import annotations

import pytest


def _get_parser():
    pytest.importorskip("pandas")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from judge_results import _parse_multi_score
    return _parse_multi_score


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------

def test_strict_json_parsing():
    parse = _get_parser()
    raw = '{"correctness": 0.8, "faithfulness": 0.9, "helpfulness": 1.0, "reason": "Good answer."}'
    c, f, h, reason = parse(raw)
    assert c == pytest.approx(0.8)
    assert f == pytest.approx(0.9)
    assert h == pytest.approx(1.0)
    assert "Good answer" in reason


def test_strict_json_zero_scores():
    parse = _get_parser()
    raw = '{"correctness": 0.0, "faithfulness": 0.0, "helpfulness": 0.0, "reason": "Wrong."}'
    c, f, h, _ = parse(raw)
    assert c == pytest.approx(0.0)
    assert f == pytest.approx(0.0)
    assert h == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Markdown code-fence handling
# ---------------------------------------------------------------------------

def test_strips_markdown_json_fence():
    """Some models wrap JSON in ```json ... ``` despite the prompt saying not to."""
    parse = _get_parser()
    raw = '```json\n{"correctness": 0.7, "faithfulness": 0.6, "helpfulness": 0.8, "reason": "ok"}\n```'
    c, f, h, _ = parse(raw)
    assert c == pytest.approx(0.7)
    assert f == pytest.approx(0.6)
    assert h == pytest.approx(0.8)


def test_strips_plain_code_fence():
    parse = _get_parser()
    raw = '```\n{"correctness": 0.5, "faithfulness": 0.5, "helpfulness": 0.5, "reason": "mid"}\n```'
    c, f, h, _ = parse(raw)
    assert c == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Regex fallback when JSON is malformed
# ---------------------------------------------------------------------------

def test_regex_recovers_from_malformed_json():
    """Trailing comma or other JSON glitches should be regex-recovered."""
    parse = _get_parser()
    # Trailing comma → strict json.loads fails
    raw = '{"correctness": 0.8, "faithfulness": 0.7, "helpfulness": 0.9,}'
    c, f, h, reason = parse(raw)
    assert c == pytest.approx(0.8)
    assert f == pytest.approx(0.7)
    assert h == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Score clipping
# ---------------------------------------------------------------------------

def test_scores_above_one_are_clipped():
    parse = _get_parser()
    raw = '{"correctness": 1.5, "faithfulness": 2.0, "helpfulness": 0.9, "reason": "x"}'
    c, f, h, _ = parse(raw)
    assert c == pytest.approx(1.0)
    assert f == pytest.approx(1.0)


def test_negative_scores_are_clipped():
    parse = _get_parser()
    raw = '{"correctness": -0.5, "faithfulness": 0.5, "helpfulness": 0.5, "reason": "x"}'
    c, f, h, _ = parse(raw)
    assert c == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Empty / unparseable inputs
# ---------------------------------------------------------------------------

def test_empty_response_returns_zeros():
    parse = _get_parser()
    c, f, h, reason = parse("")
    assert c == pytest.approx(0.0)
    assert f == pytest.approx(0.0)
    assert h == pytest.approx(0.0)
    assert "empty" in reason.lower()


def test_unparseable_response_returns_zeros_with_reason():
    parse = _get_parser()
    c, f, h, reason = parse("this is not JSON at all, just prose")
    assert c == pytest.approx(0.0)
    assert "unparseable" in reason.lower() or "regex" in reason.lower()
