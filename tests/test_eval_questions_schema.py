"""Schema integrity tests for the canonical eval_questions.json bank.

These tests don't import any heavy modules — they just validate the JSON
file's structure. Useful as a guard against future edits that accidentally
break a required field.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


@pytest.fixture(scope="module")
def questions():
    if not EVAL_PATH.exists():
        pytest.skip(f"eval_questions.json not found at {EVAL_PATH}")
    return json.loads(EVAL_PATH.read_text())


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------

def test_eval_set_is_a_list(questions):
    assert isinstance(questions, list)


def test_eval_set_size_is_44(questions):
    """The expanded eval bank should have 44 questions."""
    assert len(questions) == 44


def test_every_question_has_required_fields(questions):
    required = {"id", "type", "question", "expected_keywords", "ground_truth_in_corpus"}
    for q in questions:
        missing = required - set(q.keys())
        assert not missing, f"Question {q.get('id', '?')} missing fields: {missing}"


def test_question_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


def test_question_types_are_valid(questions):
    """Question type should be one of MultiFinRAG's defined types."""
    valid_types = {1, 2, 3, 4}
    for q in questions:
        assert q["type"] in valid_types, \
            f"Question {q['id']} has invalid type {q['type']}"


def test_expected_keywords_is_nonempty_list(questions):
    for q in questions:
        kws = q["expected_keywords"]
        assert isinstance(kws, list), \
            f"Question {q['id']}: expected_keywords must be a list"
        assert len(kws) > 0, \
            f"Question {q['id']}: expected_keywords is empty"


def test_keywords_are_5_per_question(questions):
    """The 44q bank should use 5 keywords per question (KHR granularity = 0.2)."""
    for q in questions:
        n = len(q["expected_keywords"])
        assert n == 5, f"Question {q['id']} has {n} keywords (expected 5)"


def test_ground_truth_flag_is_boolean(questions):
    for q in questions:
        assert isinstance(q["ground_truth_in_corpus"], bool), \
            f"Question {q['id']}: ground_truth_in_corpus must be bool"


# ---------------------------------------------------------------------------
# Content composition (sanity checks against the eval design)
# ---------------------------------------------------------------------------

def test_eval_has_at_least_5_narrative_questions(questions):
    """Type 1 narrative questions should be present (n1-n5 + edge cases)."""
    n_narrative = sum(1 for q in questions if q["type"] == 1)
    assert n_narrative >= 5


def test_eval_has_at_least_4_ooc_questions(questions):
    """Out-of-corpus questions should be tagged ground_truth_in_corpus=False."""
    ooc = [q for q in questions if not q["ground_truth_in_corpus"]]
    assert len(ooc) >= 4, f"Found only {len(ooc)} OOC questions, expected >= 4"


def test_edge_case_questions_have_category_tag(questions):
    """Edge-case questions (ec*) should have an `edge_case` tag for grouping."""
    edge_qs = [q for q in questions if q["id"].startswith("ec")]
    assert len(edge_qs) > 0, "No edge-case questions found"
    for q in edge_qs:
        assert "edge_case" in q and q["edge_case"], \
            f"Edge case {q['id']} missing edge_case tag"


def test_most_edge_case_categories_have_n_2(questions):
    """Most edge_case categories should have n>=2 for non-degenerate per-category CIs.

    Two categories (`balance_sheet_plus_narrative`, `ambiguous_time_reference_latest_filing`)
    are intentionally left at n=1 — they're tagged for tracking the failure mode
    even though we don't have enough sample volume to compute per-category CIs
    on them. The majority of categories must reach n>=2.
    """
    from collections import Counter
    cats = Counter(q.get("edge_case", "") for q in questions if q["id"].startswith("ec"))
    cats.pop("", None)
    n2_or_more = sum(1 for n in cats.values() if n >= 2)
    n_total = len(cats)
    # Require at least 75% of edge-case categories to reach n>=2
    assert n2_or_more / n_total >= 0.75, \
        f"Only {n2_or_more}/{n_total} edge-case categories have n>=2; expected >=75%."
