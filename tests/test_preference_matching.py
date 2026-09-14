"""Milestone 10A tests: preference-candidate evidence matching.

Fully offline and deterministic.  Synthetic M8 metadata and synthetic M9 preferences, so
the suite needs no checkpoint, no metadata artifact and no database.

Coverage follows the milestone's required list: active-only semantics, the three
statuses, missing/malformed/unsupported handling, categorical and numeric rules,
candidate preservation (identity, count, rank, score, order), the absence of ranking
artefacts, provenance, determinism and input immutability.
"""

from __future__ import annotations

import inspect
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.preference_matching_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    candidates_from_rows,
    make_candidate,
    make_entry,
    make_preference_candidate,
    make_snapshot,
    metadata_for,
)
from recommendation.catalog.metadata import normalize_product_record  # noqa: E402
from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    PreferenceMode,
    RuleBasedPreferenceExtractor,
)
from recommendation.memory.schemas import (  # noqa: E402
    PreferenceKind,
    PreferencePolarity,
    PreferenceStatus,
)
from recommendation.preference_matching import (  # noqa: E402
    MATCHING_SUPPORT,
    EvidenceStatus,
    MatchingSupport,
    PreferenceCandidateMatcher,
    ReasonCode,
    active_preferences,
    match_candidates,
)


def only(report) -> object:
    """Return the single candidate's evidence tuple."""
    assert len(report.candidates) == 1
    return report.candidates[0].evidence


def status_for(evidence, *, kind=None, value=None):
    """Select one evidence record by preference kind/value."""
    matches = [
        record
        for record in evidence
        if (kind is None or record.preference_kind is kind)
        and (value is None or record.preference_value == value)
    ]
    assert len(matches) == 1, f"expected exactly one record, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# 1-6. Active-only semantics
# --------------------------------------------------------------------------- #


def test_only_active_preferences_are_used() -> None:
    active = make_entry(memory_id="a", value="red", logical_seq=1)
    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(active),
    )
    assert report.active_preference_count == 1
    assert all(len(c.evidence) == 1 for c in report.candidates)
    assert all(c.evidence[0].preference_id == "a" for c in report.candidates)


def test_superseded_preference_is_ignored() -> None:
    superseded = make_entry(
        memory_id="old",
        value="black",
        polarity=PreferencePolarity.PREFER,
        status=PreferenceStatus.SUPERSEDED,
        superseded_by="new",
        logical_seq=1,
    )
    active = make_entry(
        memory_id="new",
        value="blue",
        polarity=PreferencePolarity.PREFER,
        logical_seq=2,
        supersedes="old",
    )
    report = match_candidates(
        candidates=candidates_from_rows(), preferences=make_snapshot(superseded, active)
    )
    assert report.active_preference_count == 1
    ids = {record.preference_id for c in report.candidates for record in c.evidence}
    assert ids == {"new"}
    assert "black" not in {r.preference_value for c in report.candidates for r in c.evidence}


def test_removed_preference_is_ignored() -> None:
    removed = make_entry(
        memory_id="gone",
        value="red",
        status=PreferenceStatus.REMOVED,
        logical_seq=1,
    )
    report = match_candidates(
        candidates=candidates_from_rows(), preferences=make_snapshot(removed)
    )
    assert report.active_preference_count == 0
    assert all(c.evidence == () for c in report.candidates)


def test_inactive_entries_passed_as_a_plain_sequence_are_still_filtered() -> None:
    """Even a raw sequence cannot smuggle history into matching."""
    history = (
        make_entry(memory_id="sup", value="black", status=PreferenceStatus.SUPERSEDED, logical_seq=1),
        make_entry(memory_id="rem", value="red", status=PreferenceStatus.REMOVED, logical_seq=2),
        make_entry(memory_id="act", value="blue", logical_seq=3),
    )
    report = match_candidates(candidates=candidates_from_rows(), preferences=history)
    assert report.active_preference_count == 1
    assert {r.preference_id for c in report.candidates for r in c.evidence} == {"act"}


def test_active_preferences_helper_excludes_inactive_entries() -> None:
    snapshot = make_snapshot(
        make_entry(memory_id="a", logical_seq=1),
        make_entry(memory_id="b", status=PreferenceStatus.REMOVED, logical_seq=2),
        make_entry(memory_id="c", status=PreferenceStatus.SUPERSEDED, logical_seq=3),
    )
    assert [entry.memory_id for entry in active_preferences(snapshot)] == ["a"]


def test_add_coexistence_reaches_matching_as_two_records() -> None:
    """M9 keeps independent avoidances; matching must not merge them into one slot."""
    red = make_entry(memory_id="m-red", value="red", logical_seq=1)
    blue = make_entry(memory_id="m-blue", value="blue", logical_seq=2)
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(red, blue),
    )
    evidence = only(report)
    assert len(evidence) == 2
    assert status_for(evidence, value="red").status is EvidenceStatus.VIOLATION
    assert status_for(evidence, value="blue").status is EvidenceStatus.UNKNOWN


def test_no_single_valued_kind_assumption_is_reintroduced() -> None:
    """Guard against reviving the deleted singleton-slot behaviour."""
    source = Path(
        __import__(
            "recommendation.preference_matching.matcher", fromlist=["x"]
        ).__file__
    ).read_text(encoding="utf-8")
    assert "SINGLE_VALUED_KINDS" not in source


def test_explicit_replacement_final_state_is_respected_end_to_end() -> None:
    """Tied to the M9 replacement semantics: only the active value is matched."""
    extractor = RuleBasedPreferenceExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    service.process_turn(user_key="u", user_message="I prefer black.", turn_id="t1", now=1.0)
    service.process_turn(
        user_key="u", user_message="Actually, I prefer red instead.", turn_id="t2", now=2.0
    )
    snapshot = service.get_active_preferences("u")
    # The active snapshot holds only the surviving preference...
    assert [(e.value, e.status.value) for e in snapshot.entries] == [("red", "active")]
    # ...while the audit trail still shows the superseded entry.
    assert [(e.value, e.status.value) for e in service.get_memory_history("u").entries] == [
        ("black", "superseded"),
        ("red", "active"),
    ]

    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=snapshot,
    )
    evidence = only(report)
    # Exactly one record: the active red preference.  Black produces nothing.
    assert len(evidence) == 1
    assert evidence[0].preference_value == "red"
    assert evidence[0].status is EvidenceStatus.MATCH
    assert all(record.preference_value != "black" for record in evidence)


def test_category_wide_removal_final_state_is_respected_end_to_end() -> None:
    extractor = RuleBasedPreferenceExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    service.process_turn(user_key="u", user_message="I don't want blue.", turn_id="t2", now=2.0)
    service.process_turn(
        user_key="u", user_message="I don't care about color anymore.", turn_id="t3", now=3.0
    )
    snapshot = service.get_active_preferences("u")
    assert snapshot.active_entries == ()

    report = match_candidates(candidates=candidates_from_rows(), preferences=snapshot)
    assert report.active_preference_count == 0
    assert all(c.evidence == () for c in report.candidates)
    # The REMOVE directive must not itself become an evidence-producing preference.
    assert all(record.preference_kind is not PreferenceKind.FREE_FORM_CONSTRAINT
               for c in report.candidates for record in c.evidence)


# --------------------------------------------------------------------------- #
# 7-11. Statuses, missing, malformed, unsupported
# --------------------------------------------------------------------------- #


def test_negative_categorical_exact_violation() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.AVOID)),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.VIOLATION
    assert record.reason_code is ReasonCode.EXPLICIT_VALUE_CONFLICT
    assert record.metadata_field == "details.Color"
    assert record.metadata_value == "red"


def test_positive_categorical_exact_match() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.PREFER)),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.MATCH
    assert record.reason_code is ReasonCode.EXACT_VALUE_MATCH


def test_positive_categorical_nonmatch_is_not_a_hard_violation() -> None:
    """A different listed colour does not prove the wanted one is absent."""
    report = match_candidates(
        candidates=[make_candidate("cand-blue", metadata=metadata_for(CANDIDATE_ROWS[1]))],
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.PREFER)),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.PREFERRED_VALUE_ABSENT
    assert record.metadata_field == "details.Color"
    assert record.metadata_value == "blue"


def test_avoid_with_missing_attribute_is_unknown_not_match() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-nocolor", metadata=metadata_for(CANDIDATE_ROWS[2]))],
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.AVOID)),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.INSUFFICIENT_METADATA
    assert record.metadata_field is None


def test_prefer_with_missing_attribute_is_unknown() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-nocolor", metadata=metadata_for(CANDIDATE_ROWS[2]))],
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.PREFER)),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.INSUFFICIENT_METADATA


def test_candidate_without_metadata_is_unknown_for_every_preference() -> None:
    preferences = make_snapshot(
        make_entry(memory_id="a", kind=PreferenceKind.COLOR, value="red", logical_seq=1),
        make_entry(memory_id="b", kind=PreferenceKind.PRICE_MAX, value="100",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
        make_entry(memory_id="c", kind=PreferenceKind.FEATURE, value="waterproof",
                   polarity=PreferencePolarity.PREFER, logical_seq=3),
    )
    report = match_candidates(
        candidates=[make_candidate("cand-nometa", metadata=None)],
        preferences=preferences,
    )
    evidence = only(report)
    assert len(evidence) == 3
    assert all(record.status is EvidenceStatus.UNKNOWN for record in evidence)
    assert all(record.reason_code is ReasonCode.METADATA_MISSING for record in evidence)
    assert all(record.metadata_present is False for record in evidence)


def test_malformed_price_is_unknown_not_zero() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-badprice", metadata=metadata_for(CANDIDATE_ROWS[3]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.PRICE_MAX, value="100", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.METADATA_UNPARSEABLE
    assert record.metadata_value == "—"


@pytest.mark.parametrize("kind", [PreferenceKind.FREE_FORM_CONSTRAINT])
def test_unsupported_preference_kind_is_unknown_and_retained(kind: PreferenceKind) -> None:
    """An unsupported kind is reported as such even when metadata is absent.

    The reason is structural: no metadata could ever resolve this kind, so reporting
    "metadata_missing" would wrongly suggest that better data could.
    """
    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(make_entry(kind=kind, value="something", polarity=PreferencePolarity.PREFER)),
    )
    assert report.active_preference_count == 1
    for candidate in report.candidates:
        assert len(candidate.evidence) == 1
        record = candidate.evidence[0]
        assert record.status is EvidenceStatus.UNKNOWN
        assert record.reason_code is ReasonCode.UNSUPPORTED_PREFERENCE_KIND
        assert record.support is MatchingSupport.UNSUPPORTED_FOR_MATCHING


def test_unsupported_kind_does_not_crash_and_stays_visible() -> None:
    """A later stage must be able to see that evidence was unavailable."""
    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(
            make_entry(
                kind=PreferenceKind.FREE_FORM_CONSTRAINT,
                value="lightweight",
                polarity=PreferencePolarity.PREFER,
            )
        ),
    )
    assert report.counts.unknown_count == len(report.candidates)
    assert report.counts.match_count == 0
    assert report.counts.violation_count == 0


# --------------------------------------------------------------------------- #
# 12-15. Categorical and numeric rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "value", "polarity", "expected"),
    [
        (PreferenceKind.COLOR, "red", PreferencePolarity.AVOID, EvidenceStatus.VIOLATION),
        (PreferenceKind.COLOR, "red", PreferencePolarity.PREFER, EvidenceStatus.MATCH),
        (PreferenceKind.MATERIAL, "leather", PreferencePolarity.AVOID, EvidenceStatus.VIOLATION),
        (PreferenceKind.MATERIAL, "leather", PreferencePolarity.PREFER, EvidenceStatus.MATCH),
        (PreferenceKind.BRAND, "Acme", PreferencePolarity.PREFER, EvidenceStatus.MATCH),
        (PreferenceKind.BRAND, "Acme", PreferencePolarity.AVOID, EvidenceStatus.VIOLATION),
    ],
)
def test_supported_categorical_kinds(kind, value, polarity, expected) -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(make_entry(kind=kind, value=value, polarity=polarity)),
    )
    assert only(report)[0].status is expected


def test_material_from_a_different_candidate_is_unknown_for_avoid() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-blue", metadata=metadata_for(CANDIDATE_ROWS[1]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.MATERIAL, value="leather", polarity=PreferencePolarity.AVOID)
        ),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.metadata_value == "foam"


def test_brand_uses_the_structured_store_field() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.BRAND, value="Acme", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.metadata_field == "store"
    assert record.status is EvidenceStatus.MATCH


def test_partial_kinds_are_marked_partially_supported() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.FEATURE, value="waterproof", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.support is MatchingSupport.PARTIALLY_SUPPORTED
    assert record.status is EvidenceStatus.MATCH


def test_partial_kind_absence_is_unknown_not_violation() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-blue", metadata=metadata_for(CANDIDATE_ROWS[1]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.FEATURE, value="waterproof", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.INSUFFICIENT_METADATA


def test_category_partial_match_over_categories() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.CATEGORY, value="Camping", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.MATCH
    assert record.metadata_field == "categories"


def test_token_matching_is_whole_word_not_substring() -> None:
    """``red`` must not match ``hundred``; ``blue`` must not match ``blueberry``."""
    metadata = normalize_product_record(
        {"parent_asin": "s", "details": {"Color": "hundred"}, "title": "blueberry"}
    )
    report = match_candidates(
        candidates=[make_candidate("s", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.COLOR, value="red", polarity=PreferencePolarity.AVOID)
        ),
    )
    assert only(report)[0].status is EvidenceStatus.UNKNOWN


def test_multi_token_preference_requires_all_tokens() -> None:
    metadata = normalize_product_record(
        {"parent_asin": "s", "features": ["lightweight waterproof shell"]}
    )
    matching = match_candidates(
        candidates=[make_candidate("s", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.FEATURE, value="lightweight waterproof",
                       polarity=PreferencePolarity.PREFER)
        ),
    )
    assert only(matching)[0].status is EvidenceStatus.MATCH

    partial = match_candidates(
        candidates=[make_candidate("s", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.FEATURE, value="lightweight insulated",
                       polarity=PreferencePolarity.PREFER)
        ),
    )
    assert only(partial)[0].status is EvidenceStatus.UNKNOWN


@pytest.mark.parametrize(
    ("price", "expected", "reason"),
    [
        (99.99, EvidenceStatus.MATCH, ReasonCode.NUMERIC_WITHIN_LIMIT),
        (100.0, EvidenceStatus.MATCH, ReasonCode.NUMERIC_WITHIN_LIMIT),
        (100.01, EvidenceStatus.VIOLATION, ReasonCode.NUMERIC_EXCEEDS_LIMIT),
    ],
)
def test_numeric_max_boundary(price, expected, reason) -> None:
    metadata = normalize_product_record({"parent_asin": "p", "price": price})
    report = match_candidates(
        candidates=[make_candidate("p", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.PRICE_MAX, value="100", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is expected
    assert record.reason_code is reason


@pytest.mark.parametrize(
    ("price", "expected", "reason"),
    [
        (49.99, EvidenceStatus.VIOLATION, ReasonCode.NUMERIC_BELOW_MINIMUM),
        (50.0, EvidenceStatus.MATCH, ReasonCode.NUMERIC_WITHIN_LIMIT),
        (50.01, EvidenceStatus.MATCH, ReasonCode.NUMERIC_WITHIN_LIMIT),
    ],
)
def test_numeric_min_boundary(price, expected, reason) -> None:
    metadata = normalize_product_record({"parent_asin": "p", "price": price})
    report = match_candidates(
        candidates=[make_candidate("p", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.PRICE_MIN, value="50", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is expected
    assert record.reason_code is reason


def test_numeric_missing_value_is_unknown() -> None:
    metadata = normalize_product_record({"parent_asin": "p"})
    report = match_candidates(
        candidates=[make_candidate("p", metadata=metadata)],
        preferences=make_snapshot(
            make_entry(kind=PreferenceKind.PRICE_MAX, value="100", polarity=PreferencePolarity.PREFER)
        ),
    )
    record = only(report)[0]
    assert record.status is EvidenceStatus.UNKNOWN
    assert record.reason_code is ReasonCode.INSUFFICIENT_METADATA


# --------------------------------------------------------------------------- #
# Opposite polarity (M9 final active semantics)
# --------------------------------------------------------------------------- #


def test_active_avoidance_violates_while_active_preference_matches() -> None:
    candidate = [make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))]
    avoided = match_candidates(
        candidates=candidate,
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.AVOID)),
    )
    preferred = match_candidates(
        candidates=candidate,
        preferences=make_snapshot(make_entry(value="red", polarity=PreferencePolarity.PREFER)),
    )
    assert only(avoided)[0].status is EvidenceStatus.VIOLATION
    assert only(preferred)[0].status is EvidenceStatus.MATCH


def test_inactive_opposite_polarity_history_has_no_effect() -> None:
    historical = make_entry(
        memory_id="old",
        value="red",
        polarity=PreferencePolarity.PREFER,
        status=PreferenceStatus.SUPERSEDED,
        superseded_by="new",
        logical_seq=1,
    )
    active = make_entry(
        memory_id="new",
        value="red",
        polarity=PreferencePolarity.AVOID,
        logical_seq=2,
        supersedes="old",
    )
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(historical, active),
    )
    evidence = only(report)
    assert len(evidence) == 1
    assert evidence[0].polarity if False else evidence[0].preference_polarity is PreferencePolarity.AVOID
    assert evidence[0].status is EvidenceStatus.VIOLATION


# --------------------------------------------------------------------------- #
# 33. Multi-preference evidence
# --------------------------------------------------------------------------- #


def test_one_candidate_can_carry_match_violation_and_unknown_together() -> None:
    metadata = normalize_product_record(
        {
            "parent_asin": "m",
            "price": 150.0,
            "features": ["waterproof membrane"],
            "details": {"Color": "red"},
        }
    )
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.FEATURE, value="waterproof",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
        make_entry(memory_id="u", kind=PreferenceKind.FREE_FORM_CONSTRAINT, value="lightweight",
                   polarity=PreferencePolarity.PREFER, logical_seq=3),
        make_entry(memory_id="p", kind=PreferenceKind.PRICE_MAX, value="100",
                   polarity=PreferencePolarity.PREFER, logical_seq=4),
    )
    report = match_candidates(
        candidates=[make_candidate("m", metadata=metadata)], preferences=preferences
    )
    evidence = only(report)
    assert len(evidence) == 4
    by_id = {record.preference_id: record for record in evidence}
    assert by_id["v"].status is EvidenceStatus.VIOLATION
    assert by_id["m"].status is EvidenceStatus.MATCH
    assert by_id["u"].status is EvidenceStatus.UNKNOWN
    assert by_id["p"].status is EvidenceStatus.VIOLATION

    counts = report.candidates[0]
    assert counts.count(EvidenceStatus.MATCH) == 1
    assert counts.count(EvidenceStatus.VIOLATION) == 2
    assert counts.count(EvidenceStatus.UNKNOWN) == 1
    assert report.counts.model_dump() == {
        "match_count": 1,
        "violation_count": 2,
        "unknown_count": 1,
    }


# --------------------------------------------------------------------------- #
# 20-25. Candidate preservation
# --------------------------------------------------------------------------- #


def test_candidate_identity_count_and_order_are_preserved() -> None:
    candidates = candidates_from_rows()
    preferences = make_snapshot(
        make_entry(kind=PreferenceKind.COLOR, value="red", polarity=PreferencePolarity.AVOID),
        make_entry(memory_id="b", kind=PreferenceKind.PRICE_MAX, value="100",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    report = match_candidates(candidates=candidates, preferences=preferences)

    assert report.parent_asins == tuple(row["parent_asin"] for row in CANDIDATE_ROWS)
    assert [c.item_id for c in report.candidates] == [row["item_id"] for row in CANDIDATE_ROWS]
    assert len(report.candidates) == len(candidates)
    assert report.ranks == tuple(range(1, len(candidates) + 1))


def test_original_rank_and_sasrec_score_are_copied_exactly() -> None:
    candidates = candidates_from_rows()
    report = match_candidates(candidates=candidates, preferences=make_snapshot())
    for original, preserved in zip(candidates, report.candidates):
        assert preserved.original_rank == original.rank
        assert preserved.sasrec_score == original.score
        assert preserved.sasrec_score == original.recommendation.score
        assert preserved.item_id == original.recommendation.item_id
        assert preserved.parent_asin == original.parent_asin


def test_order_is_unchanged_when_rank_one_violates_and_rank_five_matches() -> None:
    """The explicit required test: evidence must not reorder anything."""
    candidates = [
        make_candidate(
            "ranks-first-violates",
            rank=1,
            item_id=1,
            score=9.0,
            metadata=normalize_product_record({"parent_asin": "ranks-first-violates",
                                               "details": {"Color": "red"}}),
        ),
        make_candidate("filler-2", rank=2, item_id=2, score=8.0, metadata=None),
        make_candidate("filler-3", rank=3, item_id=3, score=7.0, metadata=None),
        make_candidate("filler-4", rank=4, item_id=4, score=6.0, metadata=None),
        make_candidate(
            "ranks-last-matches",
            rank=5,
            item_id=5,
            score=1.0,
            metadata=normalize_product_record({"parent_asin": "ranks-last-matches",
                                               "details": {"Color": "blue"}}),
        ),
    ]
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    report = match_candidates(candidates=candidates, preferences=preferences)

    assert report.parent_asins == (
        "ranks-first-violates",
        "filler-2",
        "filler-3",
        "filler-4",
        "ranks-last-matches",
    )
    assert report.ranks == (1, 2, 3, 4, 5)
    assert [c.sasrec_score for c in report.candidates] == [9.0, 8.0, 7.0, 6.0, 1.0]
    # The evidence really is in the "wrong" order relative to ranking quality.
    assert report.candidates[0].count(EvidenceStatus.VIOLATION) == 1
    assert report.candidates[4].count(EvidenceStatus.MATCH) == 1


def test_report_contains_no_ranking_artefacts() -> None:
    """No reranked rank, final score, critic score or combined preference score."""
    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(make_entry(value="red")),
    )
    dumped = report.as_dict()
    assert set(dumped) == {"candidates", "active_preference_count", "counts"}
    for candidate in dumped["candidates"]:
        assert set(candidate) == {
            "original_rank",
            "item_id",
            "parent_asin",
            "sasrec_score",
            "evidence",
        }
        for forbidden in ("reranked_rank", "final_score", "critic_score", "preference_score"):
            assert forbidden not in candidate


def test_counts_are_descriptive_only_and_never_weighted() -> None:
    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(make_entry(value="red")),
    )
    counts = report.counts.model_dump()
    assert set(counts) == {"match_count", "violation_count", "unknown_count"}
    assert all(isinstance(value, int) for value in counts.values())


def test_candidate_universe_is_not_expanded_or_filtered() -> None:
    """A preference naming an unknown product cannot add a candidate."""
    preferences = make_snapshot(
        make_entry(memory_id="x", kind=PreferenceKind.BRAND, value="NotACandidate",
                   polarity=PreferencePolarity.PREFER)
    )
    candidates = candidates_from_rows()
    report = match_candidates(candidates=candidates, preferences=preferences)
    assert len(report.candidates) == len(candidates)
    assert "NotACandidate" not in report.parent_asins
    assert report.parent_asins == tuple(row["parent_asin"] for row in CANDIDATE_ROWS)


def test_matcher_has_no_retrieval_or_store_dependency() -> None:
    """Structural proof of the RAG boundary: no catalogue, retriever or store handle."""
    import recommendation.preference_matching.matcher as matcher_module

    source = Path(matcher_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "MetadataIndex",
        "retrieve_evidence",
        "RecommendationTool",
        "SASRecInferenceEngine",
        "sqlite",
        "PreferenceMemoryService",
    ):
        assert forbidden not in source, f"matcher must not reference {forbidden}"

    parameters = set(inspect.signature(PreferenceCandidateMatcher.match).parameters)
    assert parameters == {"self", "candidates", "preferences"}
    assert "user_key" not in parameters


# --------------------------------------------------------------------------- #
# 29. Determinism and immutability
# --------------------------------------------------------------------------- #


def test_matching_is_deterministic_across_repeated_calls() -> None:
    def run() -> str:
        report = match_candidates(
            candidates=candidates_from_rows(),
            preferences=make_snapshot(
                make_entry(memory_id="a", value="red", logical_seq=1),
                make_entry(memory_id="b", kind=PreferenceKind.PRICE_MAX, value="100",
                           polarity=PreferencePolarity.PREFER, logical_seq=2),
            ),
        )
        return report.model_dump_json()

    baseline = run()
    for _ in range(5):
        assert run() == baseline


def test_repeated_matches_on_the_same_objects_are_structurally_identical() -> None:
    matcher = PreferenceCandidateMatcher()
    candidates = candidates_from_rows()
    preferences = make_snapshot(make_entry(value="red"))
    first = matcher.match(candidates=candidates, preferences=preferences)
    second = matcher.match(candidates=candidates, preferences=preferences)
    assert first == second


def test_matching_does_not_mutate_its_inputs() -> None:
    candidates = candidates_from_rows()
    candidates_before = [c.model_dump() for c in candidates]
    preferences = make_snapshot(make_entry(value="red"))
    preferences_before = preferences.model_dump()
    entry_before = preferences.entries[0].model_dump()
    metadata_before = [
        None if c.metadata is None else c.metadata.model_dump() for c in candidates
    ]

    match_candidates(candidates=candidates, preferences=preferences)

    assert [c.model_dump() for c in candidates] == candidates_before
    assert preferences.model_dump() == preferences_before
    assert preferences.entries[0].model_dump() == entry_before
    assert [
        None if c.metadata is None else c.metadata.model_dump() for c in candidates
    ] == metadata_before


def test_matching_does_not_mutate_the_candidate_list_order() -> None:
    candidates = candidates_from_rows()
    snapshot = list(candidates)
    match_candidates(candidates=candidates, preferences=make_snapshot(make_entry(value="red")))
    assert candidates == snapshot


# --------------------------------------------------------------------------- #
# 35. Offline
# --------------------------------------------------------------------------- #


def test_matching_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("preference matching must not touch the network")

    monkeypatch.setattr(socket, "socket", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)

    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(make_entry(value="red")),
    )
    assert len(report.candidates) == len(CANDIDATE_ROWS)


def test_matcher_module_imports_no_provider_sdk() -> None:
    import importlib

    for module in (
        "recommendation.preference_matching",
        "recommendation.preference_matching.matcher",
        "recommendation.preference_matching.schemas",
    ):
        importlib.import_module(module)
    for forbidden in ("openai", "anthropic", "google.generativeai", "cohere", "vertexai"):
        assert forbidden not in sys.modules


# --------------------------------------------------------------------------- #
# Support matrix
# --------------------------------------------------------------------------- #


def test_support_matrix_covers_every_m9_preference_kind() -> None:
    """The matrix must classify the accepted ontology exhaustively."""
    assert set(MATCHING_SUPPORT) == set(PreferenceKind)


def test_declared_support_matches_observed_behaviour() -> None:
    """A SUPPORTED kind must be able to produce MATCH or VIOLATION from metadata."""
    metadata = normalize_product_record(
        {
            "parent_asin": "s",
            "store": "Acme",
            "price": 90.0,
            "details": {"Color": "red", "Material": "leather"},
        }
    )
    candidates = [make_candidate("s", metadata=metadata)]
    cases = {
        PreferenceKind.COLOR: ("red", EvidenceStatus.MATCH),
        PreferenceKind.MATERIAL: ("leather", EvidenceStatus.MATCH),
        PreferenceKind.BRAND: ("Acme", EvidenceStatus.MATCH),
        PreferenceKind.PRICE_MAX: ("100", EvidenceStatus.MATCH),
        PreferenceKind.PRICE_MIN: ("50", EvidenceStatus.MATCH),
    }
    for kind, (value, expected) in cases.items():
        assert MATCHING_SUPPORT[kind] is MatchingSupport.SUPPORTED
        report = match_candidates(
            candidates=candidates,
            preferences=make_snapshot(
                make_entry(kind=kind, value=value, polarity=PreferencePolarity.PREFER)
            ),
        )
        assert only(report)[0].status is expected, kind


def test_unsupported_kinds_never_produce_match_or_violation() -> None:
    for kind, support in MATCHING_SUPPORT.items():
        if support is not MatchingSupport.UNSUPPORTED_FOR_MATCHING:
            continue
        report = match_candidates(
            candidates=candidates_from_rows(),
            preferences=make_snapshot(
                make_entry(kind=kind, value="anything", polarity=PreferencePolarity.PREFER)
            ),
        )
        for candidate in report.candidates:
            assert candidate.evidence[0].status is EvidenceStatus.UNKNOWN


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_evidence_retains_full_preference_provenance() -> None:
    entry = make_entry(
        memory_id="mem-xyz",
        kind=PreferenceKind.COLOR,
        value="red",
        polarity=PreferencePolarity.AVOID,
        source_text="I don't want red.",
        source_turn_id="turn-7",
        logical_seq=4,
    )
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(entry),
    )
    record = only(report)[0]
    assert record.preference_id == "mem-xyz"
    assert record.preference_kind is PreferenceKind.COLOR
    assert record.preference_polarity is PreferencePolarity.AVOID
    assert record.preference_value == "red"
    assert record.preference_source_text == "I don't want red."
    assert record.preference_source_turn_id == "turn-7"
    assert record.preference_logical_seq == 4


def test_evidence_retains_the_metadata_linkage_it_used() -> None:
    report = match_candidates(
        candidates=[make_candidate("cand-red", metadata=metadata_for(CANDIDATE_ROWS[0]))],
        preferences=make_snapshot(make_entry(value="red")),
    )
    record = only(report)[0]
    assert record.metadata_field == "details.Color"
    assert record.metadata_value == "red"
    assert record.metadata_present is True


def test_evidence_is_serialisable_and_round_trips() -> None:
    import json

    report = match_candidates(
        candidates=candidates_from_rows(),
        preferences=make_snapshot(make_entry(value="red")),
    )
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["active_preference_count"] == 1
    assert len(payload["candidates"]) == len(CANDIDATE_ROWS)


def test_evidence_schema_forbids_extra_fields() -> None:
    from pydantic import ValidationError

    from recommendation.preference_matching import PreferenceEvidence

    with pytest.raises(ValidationError):
        PreferenceEvidence(
            preference_id="x",
            preference_kind="color",
            preference_polarity="avoid",
            preference_value="red",
            preference_source_text="s",
            preference_source_turn_id="t",
            preference_logical_seq=1,
            status="violation",
            reason_code="explicit_value_conflict",
            support="supported",
            fabricated_score=0.5,  # type: ignore[call-arg]
        )


def test_matcher_api_requires_explicit_dependencies() -> None:
    """No user id, no store handle: the caller loads active preferences."""
    signature = inspect.signature(PreferenceCandidateMatcher.match)
    assert set(signature.parameters) == {"self", "candidates", "preferences"}
    assert "user_key" not in signature.parameters
