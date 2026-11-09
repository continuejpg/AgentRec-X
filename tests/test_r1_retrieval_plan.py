"""R1 memory-guided retrieval plan unit tests (experiment only)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.preference_construction import TaskPreference
from recommendation.demo.retrieval_plan import (
    dynamic_retrieval_enabled,
    plan_dynamic_retrieval,
)
from tests.agent_reranking_fixture import build_index


def _records():
    return dict(build_index().records)


def _pref(attribute: str, value, action: str = "value") -> TaskPreference:
    return TaskPreference(
        attribute=attribute,
        action=action,
        value=value,
        option_id=f"pref:{attribute}:{action}:{value}",
        label=str(value),
    )


def test_numeric_preference_is_rerank_only_and_does_not_rewrite_query() -> None:
    plan = plan_dynamic_retrieval(
        "camping cookware under a budget",
        ("cand-red", "cand-blue", "cand-black"),
        _records(),
        changed_preferences=(_pref("price", None, "lower"),),
    )
    assert plan.action == "RERANK"
    assert "numeric" in plan.reason
    assert plan.query_terms == ()


def test_missing_trusted_categorical_value_requests_one_retrieve() -> None:
    plan = plan_dynamic_retrieval(
        "camping cookware",
        ("cand-red", "cand-blue", "cand-black"),
        _records(),
        active_preferences=(_pref("color", "green"),),
        changed_preferences=(_pref("color", "green"),),
        sources=("history",),
        available_sources=("history", "catalog_search"),
    )
    assert plan.action == "RETRIEVE"
    assert "not_covered" in plan.reason
    assert plan.sources == ("history", "catalog_search")
    assert "green" in {term.casefold() for term in plan.query_terms}
    assert plan.matched_candidates == 0


def test_covered_categorical_value_prefers_rerank() -> None:
    records = _records()
    plan = plan_dynamic_retrieval(
        "red gear",
        ("cand-red", "cand-blue"),
        records,
        active_preferences=(_pref("color", "red"),),
        changed_preferences=(_pref("color", "red"),),
    )
    assert plan.action == "RERANK"
    assert plan.matched_candidates == 1
    assert plan.reason == "changed_preference_already_covered"


def test_one_covered_candidate_is_sufficient_without_recall() -> None:
    plan = plan_dynamic_retrieval(
        "red gear",
        ("cand-red", "cand-blue"),
        _records(),
        changed_preferences=(_pref("color", "red"),),
    )
    assert plan.action == "RERANK"
    assert plan.reason == "changed_preference_already_covered"


def test_retrieval_budget_guard_prevents_repeat_recall() -> None:
    plan = plan_dynamic_retrieval(
        "camping cookware",
        ("cand-red", "cand-blue"),
        _records(),
        changed_preferences=(_pref("color", "green"),),
        already_retrieved=True,
    )
    assert plan.action == "RERANK"
    assert plan.reason == "retrieval_budget_used"


def test_r1_switch_defaults_off() -> None:
    assert dynamic_retrieval_enabled(None) is False
    assert dynamic_retrieval_enabled("off") is False
    assert dynamic_retrieval_enabled("on") is True
