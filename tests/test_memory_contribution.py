"""Targeted regression tests for the Memory Contribution Phase 1 benchmark.

These tests cover the pieces the benchmark depends on and that are easy to get
silently wrong:

* the Memory-OFF input keeps current-query preferences and drops only the
  historical channel;
* the conservative query-soft-preference extractor does not invent a preference
  from a product request;
* observed-acceptable metrics treat unlabelled candidates as unjudged;
* the legal M9 write/read path handles positive, update, withdrawal and conflict
  states with the documented semantics;
* the fixed-weight Scorer treats missing catalogue evidence as UNKNOWN: no match,
  no conflict and no bonus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.memory_contribution import common  # noqa: E402
from recommendation.preference.history import HistoryStateReader  # noqa: E402
from recommendation.preference.scorer import PreferenceAwareScorer  # noqa: E402
from recommendation.shopping_companion.memory import (  # noqa: E402
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    ShoppingCompanionMemory,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def read_memory(statements: list[str], *, user_key: str = "test_user") -> dict:
    sessions = [
        {
            "session_id": f"{user_key}_s{index}",
            "date": f"2025-01-{index + 1:02d}",
            "turns": [{"role": "user", "content": statement}],
        }
        for index, statement in enumerate(statements)
    ]
    with ShoppingCompanionMemory() as memory:
        memory.ingest_sessions(
            user_key,
            sessions,
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref="tests:memory_contribution",
        )
        return HistoryStateReader(memory.preference_service).read(
            user_key, conversation_id=user_key, current_query="headphones"
        )


def active_pairs(history: dict) -> list[tuple[str, str]]:
    canonical = common.canonical_prefs(history["active_preferences"])
    return [(item["value"], item["polarity"]) for item in canonical]


# --------------------------------------------------------------------------- #
# Memory-OFF input control
# --------------------------------------------------------------------------- #


def test_memory_off_keeps_query_preferences_and_drops_history():
    query_preferences = [
        {"value": "microphone", "polarity": "positive", "origin": "current_query"}
    ]
    history_preferences = [
        {"value": "over_ear", "polarity": "negative", "origin": "history"}
    ]
    off = common.combined_preferences(
        query_preferences=query_preferences,
        history_preferences=history_preferences,
        memory_enabled=False,
    )
    on = common.combined_preferences(
        query_preferences=query_preferences,
        history_preferences=history_preferences,
        memory_enabled=True,
    )
    assert [(item["value"], item["polarity"]) for item in off] == [
        ("microphone", "positive")
    ]
    assert [item["origin"] for item in on] == ["current_query", "history"]


def test_query_soft_preference_does_not_invent_from_product_request():
    assert common.query_soft_preferences("I need earbuds I can use while working out.") == []
    extracted = common.query_soft_preferences("I prefer a built-in microphone.")
    assert [(item["value"], item["polarity"], item["origin"]) for item in extracted] == [
        ("microphone", "positive", "current_query")
    ]


# --------------------------------------------------------------------------- #
# Metrics / eligibility
# --------------------------------------------------------------------------- #


def test_observed_acceptable_metrics_leave_unjudged_unlabelled():
    metrics = common.hit_metrics(["x", "acceptable", "y"], {"acceptable"})
    assert metrics["hit@1"] is False
    assert metrics["hit@3"] is True
    assert metrics["hit@5"] is True
    assert metrics["first_acceptable_rank"] == 2
    assert metrics["reciprocal_rank@10"] == 0.5
    # x and y are not negatives; they are simply not labelled acceptable.
    assert metrics["acceptable_in_ranking"] == 1


def test_eligibility_uses_existing_hard_tier_and_role_only():
    row = {
        "candidate_pool": [
            {"item_id": "ok", "scorer": {"hard_tier": 0, "role_status": "MATCH"}},
            {"item_id": "no_role", "scorer": {"hard_tier": 0, "role_status": "MISMATCH"}},
            {"item_id": "bad_hard", "scorer": {"hard_tier": 2, "role_status": "MATCH"}},
            {"item_id": "not_required", "scorer": {"hard_tier": 0, "role_status": "NOT_REQUIRED"}},
        ]
    }
    eligible, excluded = common.eligibility(row)
    assert eligible == ["ok", "not_required"]
    assert excluded == {"no_role": "role_MISMATCH", "bad_hard": "hard_tier_2"}


# --------------------------------------------------------------------------- #
# Legal M9 state semantics
# --------------------------------------------------------------------------- #


def test_memory_update_supersedes_previous_value():
    history = read_memory(
        ["I prefer over-ear.", "Actually, I prefer rechargeable instead of over-ear."]
    )
    assert active_pairs(history) == [("rechargeable", "positive")]
    inactive = [
        (item["value"], item["polarity"], item["status"])
        for item in history["inactive_preferences"]
    ]
    assert ("over-ear", "prefer", "superseded") in inactive


def test_memory_withdrawal_removes_preference():
    history = read_memory(["I prefer microphone.", "Forget my microphone preference."])
    assert active_pairs(history) == []
    inactive = [
        (item["value"], item["polarity"], item["status"])
        for item in history["inactive_preferences"]
    ]
    assert ("microphone", "prefer", "removed") in inactive


def test_memory_conflict_is_reported_not_silently_resolved():
    history = read_memory(["I prefer over-ear headphones.", "I prefer in-ear headphones."])
    assert sorted(active_pairs(history)) == [("in_ear", "positive"), ("over_ear", "positive")]
    assert common.conflicts(common.canonical_prefs(history["active_preferences"]))


# --------------------------------------------------------------------------- #
# Scorer: UNKNOWN evidence is neither match nor conflict
# --------------------------------------------------------------------------- #


def test_unknown_positive_evidence_earns_no_bonus_and_is_recorded():
    scorer = PreferenceAwareScorer()
    rows = scorer.score(
        preferences=[{"value": "microphone", "polarity": "positive"}],
        current_query="headphones",
        candidates=[
            {
                "product_ref": {"namespace": "ns", "item_id": "unknown"},
                "query_rank": 1,
                "attributes": {},
                "form_factor": None,
            },
            {
                "product_ref": {"namespace": "ns", "item_id": "known"},
                "query_rank": 2,
                "attributes": {"microphone": {"evidence": "built-in mic"}},
                "form_factor": "in_ear",
            },
        ],
        namespace="ns",
    )
    by_id = {row["product_ref"]["item_id"]: row for row in rows}
    assert by_id["unknown"]["preference_match_score"] == 0.0
    assert by_id["unknown"]["preference_conflict_score"] == 0.0
    assert [item["value"] for item in by_id["unknown"]["unknown_preferences"]] == ["microphone"]
    assert by_id["known"]["preference_match_score"] == 0.2
    assert by_id["known"]["matched_preferences"][0]["value"] == "microphone"


def test_negative_preference_conflicts_only_with_present_evidence():
    scorer = PreferenceAwareScorer()
    rows = scorer.score(
        preferences=[{"value": "over_ear", "polarity": "negative"}],
        current_query="headphones",
        candidates=[
            {
                "product_ref": {"namespace": "ns", "item_id": "in_ear"},
                "query_rank": 1,
                "attributes": {"in_ear": {"evidence": "in-ear design"}},
                "form_factor": "in_ear",
            },
            {
                "product_ref": {"namespace": "ns", "item_id": "over_ear"},
                "query_rank": 2,
                "attributes": {"over_ear": {"evidence": "over-ear design"}},
                "form_factor": "over_ear",
            },
        ],
        namespace="ns",
    )
    by_id = {row["product_ref"]["item_id"]: row for row in rows}
    assert by_id["in_ear"]["preference_conflict_score"] == 0.0
    assert [item["value"] for item in by_id["in_ear"]["unknown_preferences"]] == ["over_ear"]
    assert by_id["over_ear"]["preference_conflict_score"] == 0.35
    assert by_id["over_ear"]["conflicting_preferences"][0]["value"] == "over_ear"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
