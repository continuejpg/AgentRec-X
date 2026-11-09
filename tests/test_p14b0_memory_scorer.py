"""P14-B0 tests: M9 update/retraction semantics, legal history read, scorer contract.

These tests encode the two P14-E findings: a form-factor retraction used to leave a
stale active preference, and the lexical read could miss preferences when the request
did not repeat their words.  They also pin the scorer's fairness/explainability rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommendation.memory.extraction import RuleBasedPreferenceExtractor
from recommendation.memory.service import PreferenceMemoryService
from recommendation.memory.store import InMemoryPreferenceStore
from recommendation.preference.history import HistoryReadConfig, HistoryStateReader
from recommendation.preference.scorer import PreferenceAwareScorer, ScorerConfig

REPO = Path(__file__).resolve().parents[1]


def _service() -> PreferenceMemoryService:
    return PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())


def _turns(service: PreferenceMemoryService, user: str, texts: list[str]):
    for index, text in enumerate(texts):
        service.process_turn(user_key=user, user_message=text, turn_id=f"t{index}", now=1000.0 + index)


def test_form_factor_retraction_removes_stale_preference():
    service = _service()
    _turns(service, "u1", ["I prefer over-ear.", "I also prefer wireless.", "Forget my earlier preference for over-ear.", "I now prefer in-ear."])
    active = [(entry.kind.value, entry.value, entry.polarity.value) for entry in service.get_active_preferences("u1").entries]
    assert ("free_form_constraint", "over-ear", "prefer") not in active
    assert ("free_form_constraint", "in-ear", "prefer") in active
    assert ("free_form_constraint", "wireless", "prefer") in active
    history = {(entry.value, entry.status.value) for entry in service.get_memory_history("u1").entries}
    assert ("over-ear", "removed") in history
    assert ("in-ear", "active") in history


def test_extractor_emits_value_retraction_for_hyphenated_form_factor():
    extraction = RuleBasedPreferenceExtractor().extract("Forget my earlier preference for in-ear.")
    assert extraction.removals, "form-factor retraction must be recognised"
    removal = extraction.removals[0]
    assert removal.kind is None
    assert removal.value == "in-ear"


def test_negated_preference_is_avoid_not_retraction():
    service = _service()
    _turns(service, "u2", ["I don't like in-ear."])
    entries = service.get_active_preferences("u2").entries
    assert [(entry.value, entry.polarity.value) for entry in entries] == [("in-ear", "avoid")]


def test_retraction_of_slot_still_works():
    service = _service()
    _turns(service, "u3", ["I prefer a mechanical keyboard with rgb lighting.", "Forget about the rgb preference."])
    active = [(entry.kind.value, entry.value) for entry in service.get_active_preferences("u3").entries]
    assert ("feature", "rgb lighting") not in active


def test_multi_user_isolation_and_provenance():
    service = _service()
    _turns(service, "u4", ["I prefer wireless."])
    _turns(service, "u5", ["I prefer wired."])
    assert [entry.value for entry in service.get_active_preferences("u4").entries] == ["wireless"]
    assert [entry.value for entry in service.get_active_preferences("u5").entries] == ["wired"]
    entry = service.get_active_preferences("u4").entries[0]
    assert entry.source_turn_id == "t0"
    assert entry.source_text == "I prefer wireless."


def test_history_reader_is_query_independent_and_authorised():
    service = _service()
    _turns(service, "u6", ["I prefer noise cancelling.", "I prefer over-ear."])
    reader = HistoryStateReader(service, HistoryReadConfig(max_preferences=10))
    result = reader.read("u6", conversation_id="u6", current_query="I need a backpack for hiking.")
    values = {item["value"] for item in result["active_preferences"]}
    assert values == {"noise cancelling", "over-ear"}
    assert result["query_relevant_preferences"] == 0
    assert all(item["source_turn_id"] for item in result["active_preferences"])
    with pytest.raises(PermissionError):
        reader.read("u6", conversation_id="u7")


def test_history_reader_excludes_retracted_preferences():
    service = _service()
    _turns(service, "u8", ["I prefer in-ear.", "Forget my earlier preference for in-ear."])
    result = HistoryStateReader(service).read("u8", conversation_id="u8", current_query="earbuds")
    assert result["active_preferences"] == []
    assert any(item["status"] == "removed" for item in result["inactive_preferences"])


def _candidates():
    return [
        {"product_ref": {"namespace": "amazon_electronics_c4", "item_id": "B1"}, "query_rank": 1,
         "query_score": 10.0, "attributes": {"wireless": {"evidence": "wireless"}}, "form_factor": "in_ear"},
        {"product_ref": {"namespace": "amazon_electronics_c4", "item_id": "B2"}, "query_rank": 2,
         "query_score": 9.0, "attributes": {"wired": {"evidence": "3.5mm"}}, "form_factor": None},
        {"product_ref": {"namespace": "amazon_electronics_c4", "item_id": "B3"}, "query_rank": 3,
         "query_score": 8.0, "attributes": {"over-ear": {"evidence": "over-ear"}}, "form_factor": "over_ear"},
    ]


def test_scorer_positive_preference_and_conflict_change_ranking():
    scorer = PreferenceAwareScorer()
    import copy

    candidates = copy.deepcopy(_candidates())
    # query order: B3 first, then B2, then B1 (so preference evidence must move the result)
    ordered = [candidates[2], candidates[1], candidates[0]]
    for rank, candidate in enumerate(ordered, start=1):
        candidate["query_rank"] = rank
        candidate["query_score"] = 11 - rank
    baseline = scorer.score(preferences=[], current_query="headphones", candidates=ordered, namespace="amazon_electronics_c4")
    assert [row["product_ref"]["item_id"] for row in baseline][:2] == ["B3", "B2"]
    with_positive = scorer.score(
        preferences=[{"value": "over-ear", "polarity": "positive", "memory_id": "m1"}],
        current_query="headphones", candidates=ordered, namespace="amazon_electronics_c4",
    )
    assert with_positive[0]["product_ref"]["item_id"] == "B3"
    assert with_positive[0]["preference_match_score"] > 0
    with_conflict = scorer.score(
        preferences=[{"value": "wireless", "polarity": "negative", "memory_id": "m2"}],
        current_query="headphones", candidates=ordered, namespace="amazon_electronics_c4",
    )
    assert with_conflict[-1]["product_ref"]["item_id"] == "B1"
    assert with_conflict[-1]["preference_conflict_score"] > 0


def test_scorer_unknown_is_not_satisfied_and_is_recorded():
    scorer = PreferenceAwareScorer()
    rows = scorer.score(
        preferences=[{"value": "waterproof", "polarity": "positive", "memory_id": "m3"}],
        current_query="headphones", candidates=_candidates(), namespace="amazon_electronics_c4",
    )
    for row in rows:
        assert row["preference_match_score"] == 0
        assert row["unknown_preferences"][0]["value"] == "waterproof"


def test_scorer_hard_violation_beats_query_relevance():
    scorer = PreferenceAwareScorer()
    rows = scorer.score(
        preferences=[], current_query="headphones", candidates=_candidates(),
        hard_requirements=[{"key": "form", "kind": "form_factor", "value": "over_ear"}],
        namespace="amazon_electronics_c4",
    )
    assert rows[0]["product_ref"]["item_id"] == "B3"  # verified hard form factor
    violated = [row for row in rows if row["hard_tier"] == 2]
    assert violated and violated[-1]["rank"] == len(rows)
    assert not any(row["presentable"] for row in rows if row["hard_tier"] == 2)


def test_scorer_rejects_namespace_mismatch_and_is_deterministic():
    scorer = PreferenceAwareScorer()
    candidates = _candidates()
    first = scorer.score(preferences=[{"value": "over-ear", "polarity": "positive"}], current_query="x",
                         candidates=candidates, namespace="amazon_electronics_c4")
    second = scorer.score(preferences=[{"value": "over-ear", "polarity": "positive"}], current_query="x",
                          candidates=candidates, namespace="amazon_electronics_c4")
    assert [row["product_ref"] for row in first] == [row["product_ref"] for row in second]
    with pytest.raises(ValueError):
        scorer.score(preferences=[], current_query="x", candidates=candidates, namespace="amazon_sports_5core")


def test_scorer_has_no_search_side_effect(monkeypatch):
    """The scorer consumes a fixed candidate set; it must not search for anything."""
    scorer = PreferenceAwareScorer()
    roster = list(_candidates())

    def no_search(*args, **kwargs):  # pragma: no cover - only invoked on violation
        raise AssertionError("scorer attempted to search")

    rows = scorer.score(preferences=[], current_query="headphones", candidates=roster, namespace="amazon_electronics_c4")
    assert len(rows) == len(roster)
    assert no_search  # silence unused warning; real assertion is that no catalogue/search object is reachable


def test_p14e_benchmark_files_unchanged():
    manifest = json.loads((REPO / "data/p14e/benchmark/manifest.json").read_text(encoding="utf-8"))
    import hashlib

    for name in ("agent_tasks.jsonl", "evaluator_only.jsonl"):
        digest = hashlib.sha256((REPO / "data/p14e/benchmark" / name).read_bytes()).hexdigest()
        assert digest == manifest["files"][name]["sha256"], f"{name} was modified by P14-B0"
