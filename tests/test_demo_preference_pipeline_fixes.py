"""Interview-demo preference-pipeline fixes: conjunction extraction and value-scoped removal.

Reproduces the two reported live-demo failures and locks in the fixes:

* ``"I prefer lightweight and durable products."`` used to store ONE atomic
  ``feature="lightweight and durable"``.  The M10A matcher requires *every* token of the value
  to appear in the candidate text (``matcher._covers``), so a compound value could never be
  supported by any candidate and evidence was UNKNOWN for all of them.
* ``"Remove my lightweight preference."`` used to emit a ``kind=free_form_constraint`` removal
  while the stored entry was a ``feature``, so the retraction matched nothing and no mutation
  occurred.

These tests also state the LIMITS honestly: there is no synonym handling, so ``lightweight``
is NOT supported by ``ultralight``, and whole-token matching means ``durable`` is not
supported by ``durability``.  Those are asserted as current behaviour rather than papered over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recommendation.catalog.metadata import normalize_product_record
from recommendation.memory import (
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    PreferencePolarity,
    RuleBasedPreferenceExtractor,
)
from recommendation.memory.schemas import PreferenceKind
from recommendation.preference_matching import match_candidates
from tests.preference_matching_fixture import make_candidate, make_entry, make_snapshot


_TURN = [0]


def _n() -> int:
    """A distinct turn id per call, so turns are not treated as duplicates."""
    _TURN[0] += 1
    return _TURN[0]


def _service() -> PreferenceMemoryService:
    return PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())


def _values(service: PreferenceMemoryService, user_key: str = "alice") -> set[tuple[str, str]]:
    snapshot = service.get_active_preferences(user_key)
    return {(entry.kind.value, entry.value) for entry in snapshot.active_entries}


def _durable_metadata(*, text: str, parent_asin: str = "cand-lifestraw") -> Any:
    return normalize_product_record(
        {
            "parent_asin": parent_asin,
            "title": "LifeStraw Personal Water Filter",
            "description": [text],
            "features": [],
            "categories": ["Sports & Outdoors"],
            "store": "LifeStraw",
            "details": {},
        }
    )


# --------------------------------------------------------------------------- #
# 1. conjunction extraction
# --------------------------------------------------------------------------- #


def test_compound_preference_splits_into_independent_features() -> None:
    extraction = RuleBasedPreferenceExtractor().extract(
        "I prefer lightweight and durable products."
    )

    pairs = [(p.kind, p.value, p.polarity.value) for p in extraction.preferences]
    assert (PreferenceKind.FEATURE, "lightweight", "prefer") in pairs
    assert (PreferenceKind.FEATURE, "durable", "prefer") in pairs
    assert len(pairs) == 2, pairs


def test_split_parts_keep_the_resolved_feature_kind() -> None:
    """A split part must NOT be downgraded to free_form_constraint (which is always UNKNOWN)."""
    extraction = RuleBasedPreferenceExtractor().extract(
        "I prefer lightweight and durable products."
    )
    for candidate in extraction.preferences:
        assert candidate.kind is PreferenceKind.FEATURE, candidate


def test_single_word_preference_is_unchanged() -> None:
    """Pre-existing behaviour for a bare adjective is preserved, not silently re-kinded."""
    extraction = RuleBasedPreferenceExtractor().extract("I prefer lightweight")
    assert [(p.kind, p.value) for p in extraction.preferences] == [
        (PreferenceKind.FREE_FORM_CONSTRAINT, "lightweight")
    ]


def test_brand_and_category_values_are_not_split() -> None:
    """Only attribute lists split; a brand or category phrase stays whole."""
    extraction = RuleBasedPreferenceExtractor().extract("I prefer the brand Johnson and Johnson")
    assert [p.value for p in extraction.preferences] == ["Johnson and Johnson"]


def test_three_term_conjunction_splits() -> None:
    extraction = RuleBasedPreferenceExtractor().extract("I prefer compact and waterproof or durable")
    assert sorted(p.value for p in extraction.preferences) == ["compact", "durable", "waterproof"]


def test_comma_separated_lists_are_still_not_captured() -> None:
    """Pre-existing limitation, stated rather than hidden: the value pattern excludes commas.

    ``_POSITIVE`` captures ``[a-z0-9 '\\-]`` only, so a comma ends the value and the rest of the
    list is never seen.  Fixing that means widening a shared capture pattern, which is a larger
    change than this demo blocker warrants.
    """
    extraction = RuleBasedPreferenceExtractor().extract("I prefer compact, waterproof and durable")
    assert [p.value for p in extraction.preferences] == ["compact"]


# --------------------------------------------------------------------------- #
# 2. persistence across turns
# --------------------------------------------------------------------------- #


def test_persistence_across_turns() -> None:
    service = _service()
    service.process_turn(user_message="I prefer lightweight and durable products.", user_key="alice", turn_id=f"t{_n()}")

    assert _values(service) == {("feature", "lightweight"), ("feature", "durable")}

    # a following, unrelated turn must not disturb them
    service.process_turn(user_message="Recommend some hiking gear.", user_key="alice", turn_id=f"t{_n()}")
    assert _values(service) == {("feature", "lightweight"), ("feature", "durable")}


# --------------------------------------------------------------------------- #
# 3. evidence for an exact supported term
# --------------------------------------------------------------------------- #


def test_exact_feature_term_is_supported_by_metadata_text() -> None:
    """`durable` must MATCH metadata that contains the token `durable`."""
    entry = make_entry(kind=PreferenceKind.FEATURE, value="durable",
                       polarity=PreferencePolarity.PREFER)
    candidate = make_candidate(
        "cand-lifestraw", rank=1, item_id=1,
        metadata=_durable_metadata(text="Ultralight and durable, the LifeStraw weighs less than 2 ounces"),
    )

    report = match_candidates(candidates=[candidate], preferences=make_snapshot(entry))

    evidence = report.candidates[0].evidence[0]
    assert evidence.status.value == "match", evidence


def test_compound_value_is_unsupported_which_is_why_the_split_is_required() -> None:
    """The old representation could never match: every token of the value is required."""
    from recommendation.preference_matching.matcher import _covers

    text = "Ultralight and durable, the LifeStraw weighs less than 2 ounces"
    assert _covers(text, "durable") is True
    assert _covers(text, "lightweight and durable") is False, (
        "a compound value requires all of its tokens; this is the reported Repro A root cause"
    )


# --------------------------------------------------------------------------- #
# 4. no synonym handling (asserted, not hidden)
# --------------------------------------------------------------------------- #


def test_no_synonym_support_lightweight_is_not_ultralight() -> None:
    from recommendation.preference_matching.matcher import _covers

    text = "Ultralight and durable, the LifeStraw weighs less than 2 ounces"
    assert _covers(text, "ultralight") is True
    assert _covers(text, "lightweight") is False, (
        "there is no synonym/stemming layer; `lightweight` is NOT supported by `ultralight`"
    )


def test_no_stemming_durable_is_not_durability() -> None:
    from recommendation.preference_matching.matcher import _covers

    assert _covers("high durability fishing line", "durable") is False, (
        "matching is whole-token; `durable` is NOT supported by `durability`"
    )


def test_ultralight_preference_is_supported_by_ultralight_text() -> None:
    """The honest positive case: the term the user actually types is the term that matches."""
    entry = make_entry(kind=PreferenceKind.FEATURE, value="ultralight",
                       polarity=PreferencePolarity.PREFER)
    candidate = make_candidate("cand-lifestraw", rank=1, item_id=1,
                               metadata=_durable_metadata(text="Ultralight and durable, weighs 2 ounces"))
    report = match_candidates(candidates=[candidate], preferences=make_snapshot(entry))
    evidence = report.candidates[0].evidence[0]
    assert evidence.status.value == "match", evidence


# --------------------------------------------------------------------------- #
# 5. removal
# --------------------------------------------------------------------------- #


def test_removal_by_value_targets_the_stored_feature() -> None:
    extraction = RuleBasedPreferenceExtractor().extract("Remove my lightweight preference.")
    assert len(extraction.removals) == 1
    removal = extraction.removals[0]
    assert removal.value == "lightweight"
    assert removal.kind is None, "a value target must not also claim a kind"


def test_removal_actually_mutates_the_store() -> None:
    service = _service()
    service.process_turn(user_message="I prefer lightweight and durable products.", user_key="alice", turn_id=f"t{_n()}")
    service.process_turn(user_message="Remove my lightweight preference.", user_key="alice", turn_id=f"t{_n()}")

    values = _values(service)
    assert ("feature", "lightweight") not in values, "the removal must take effect"
    assert ("feature", "durable") in values, "and must not remove the other term"


def test_slot_removal_is_unchanged() -> None:
    """Regression guard: retracting a whole SLOT still removes by kind."""
    extraction = RuleBasedPreferenceExtractor().extract("I don't care about color anymore")
    assert len(extraction.removals) == 1
    assert extraction.removals[0].kind is PreferenceKind.COLOR
    assert extraction.removals[0].value is None


# --------------------------------------------------------------------------- #
# 6. combined remove + add
# --------------------------------------------------------------------------- #


def test_combined_remove_and_reaffirm() -> None:
    service = _service()
    service.process_turn(user_message="I prefer lightweight and durable products.", user_key="alice", turn_id=f"t{_n()}")
    service.process_turn(
        user_message="Remove my lightweight preference. I care more about durability.",
        user_key="alice",
        turn_id=f"t{_n()}",
    )

    values = _values(service)
    assert ("feature", "lightweight") not in values
    assert ("feature", "durable") in values, (
        "durability must survive: it was stored in the first turn and this turn does not retract it"
    )


def test_reaffirming_does_not_duplicate() -> None:
    service = _service()
    service.process_turn(user_message="I prefer lightweight and durable products.", user_key="alice", turn_id=f"t{_n()}")
    service.process_turn(user_message="I prefer durable", user_key="alice", turn_id=f"t{_n()}")

    durable = [v for v in _values(service) if v[1] == "durable"]
    assert durable, "durable is still present"


# --------------------------------------------------------------------------- #
# 7. candidate order is untouched
# --------------------------------------------------------------------------- #


def test_matching_preserves_candidate_order_and_is_deterministic() -> None:
    """The fix is extraction-side only: it must not reorder candidates."""
    entry = make_entry(kind=PreferenceKind.FEATURE, value="durable",
                       polarity=PreferencePolarity.PREFER)
    candidates = [
        make_candidate("cand-a", rank=1, item_id=1, metadata=_durable_metadata(text="durable", parent_asin="cand-a")),
        make_candidate("cand-b", rank=2, item_id=2, metadata=_durable_metadata(text="nothing here", parent_asin="cand-b")),
        make_candidate("cand-c", rank=3, item_id=3, metadata=_durable_metadata(text="durable too", parent_asin="cand-c")),
    ]
    first = match_candidates(candidates=candidates, preferences=make_snapshot(entry))
    second = match_candidates(candidates=candidates, preferences=make_snapshot(entry))

    order = [c.recommendation.parent_asin for c in candidates]
    assert order == ["cand-a", "cand-b", "cand-c"], "input order is the Tool's order, untouched"
    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(second.as_dict(), sort_keys=True), (
        "evidence is deterministic for a fixed input"
    )


def test_extraction_does_not_touch_candidates() -> None:
    """Structural: the extractor's output types cannot carry an item id or a score."""
    extraction = RuleBasedPreferenceExtractor().extract("I prefer lightweight and durable products.")
    for candidate in extraction.preferences:
        fields = set(type(candidate).model_fields)
        assert "parent_asin" not in fields and "item_id" not in fields and "score" not in fields


# --------------------------------------------------------------------------- #
# 8. live-demo / API integration path
# --------------------------------------------------------------------------- #


def test_demo_api_end_to_end(tmp_path: Path) -> None:
    """The two reported reproductions, over the real demo HTTP surface."""
    from tests.demo_fixture import build_harness

    harness = build_harness(tmp_path)
    session_id = harness.create_session()["session_id"]

    first = harness.chat(session_id, "Recommend some products. I prefer lightweight and durable products.")
    assert first["route"] == "recommend", first

    values = harness.active_values(session_id)
    assert ("feature", "prefer", "lightweight") in values, values
    assert ("feature", "prefer", "durable") in values, values

    harness.chat(session_id, "Remove my lightweight preference. I care more about durability.")

    after = harness.active_values(session_id)
    assert ("feature", "prefer", "lightweight") not in after, after
    assert ("feature", "prefer", "durable") in after, after
