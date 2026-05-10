"""pytest configuration: make v3/ importable from tests.

Tests target v3 as the canonical version. v3/, v2/, v1/ have identical
public interfaces (rag_baseline, evaluate, judge_results, document_processor),
so passing tests against v3 imply equivalent behavior on v1/v2.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "v3"))
