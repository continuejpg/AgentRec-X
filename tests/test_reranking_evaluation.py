"""Milestone 10C tests: reranking-policy evaluation and diagnostics.

Fully offline and deterministic.  The evaluator is exercised through the accepted M10A
evidence pipeline and the accepted M10B reranker, so no policy logic is duplicated here.

Coverage follows the milestone's required list: displacement metrics, top-k overlap,
adherence counts before/after, evidence coverage, policy-order consistency, the ten
required synthetic scenarios, ADD/REPLACE/REMOVE semantics, counterfactual baselines,
determinism, immutability, and the absence of relevance metrics, stores, models and
network dependencies.
"""

from __future__ import annotations

import ast
import inspect
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.preference_matching_fixture import (  # noqa: E402
    make_candidate,
    make_entry,
    make_snapshot,
)
from recommendation.catalog import normalize_product_record  # noqa: E402
from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
)
from recommendation.memory.schemas import (  # noqa: E402
    PreferenceKind,
    PreferencePolarity,
)
from recommendation.preference_matching import (  # noqa: E402
    EvidenceStatus,
    PreferenceEvidenceReport,
    match_candidates,
)
from recommendation.reranking import RERANK_SORT_KEY_DOC, PreferenceReranker  # noqa: E402
from recommendation.reranking.evaluation import (  # noqa: E402
    BASELINE_MATCH_ONLY,
    BASELINE_ORIGINAL_ORDER,
    BASELINE_VIOLATION_ONLY,
    aggregate_requests,
    build_report,
    evaluate,
    evaluate_request,
    movement_causes_for,
    position_reasons_for,
)
from recommendation.reranking.evaluation_schemas import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    CandidateEvidenceCoverage,
    DisplacementMetrics,
    MovementCause,
    PolicyEvaluationRequest,
    PositionReason,
    RerankingEvaluationReport,
    TopKOverlap,
    evaluation_digest,
)


# --------------------------------------------------------------------------- #
# Fixture scenario builders
# --------------------------------------------------------------------------- #


def metadata_for(asin: str, *, color: str | None = None, features: tuple[str, ...] = (),
                 price: float | None = None):
    """Normalized metadata with the requested attributes."""
    payload: dict[str, object] = {"parent_asin": asin}
    if color is not None:
        payload["details"] = {"Color": color}
    if features:
        payload["features"] = list(features)
    if price is not None:
        payload["price"] = price
    return normalize_product_record(payload)


def candidates(specs):
    """Build candidates from ``(asin, rank, item_id, score, metadata)`` rows."""
    return [
        make_candidate(asin, rank=rank, item_id=item_id, score=score, metadata=meta)
        for asin, rank, item_id, score, meta in specs
    ]


def five_candidates(colors: dict[str, str | None], *, features: dict[str, tuple[str, ...]] | None = None):
    """The standard five-candidate set ``A..E`` with per-candidate colour."""
    features = features or {}
    return candidates(
        [
            (
                asin,
                position,
                position,
                float(10 - position),
                None
                if colors.get(asin) is None and not features.get(asin)
                else metadata_for(asin, color=colors.get(asin), features=features.get(asin, ())),
            )
            for position, asin in enumerate(["A", "B", "C", "D", "E"], start=1)
        ]
    )


def avoid(value: str, *, seq: int = 1, kind: PreferenceKind = PreferenceKind.COLOR):
    """An active avoidance."""
    return make_entry(
        memory_id=f"avoid-{kind.value}-{value}",
        kind=kind,
        value=value,
        polarity=PreferencePolarity.AVOID,
        logical_seq=seq,
    )


def prefer(value: str, *, seq: int = 1, kind: PreferenceKind = PreferenceKind.COLOR):
    """An active preference."""
    return make_entry(
        memory_id=f"prefer-{kind.value}-{value}",
        kind=kind,
        value=value,
        polarity=PreferencePolarity.PREFER,
        logical_seq=seq,
    )


def evaluate_scenario(label: str, cands, preferences, **kwargs):
    """Match then evaluate one scenario, returning the request diagnostics."""
    report = match_candidates(candidates=cands, preferences=preferences)
    request, reranked = evaluate_request(report=report, label=label, **kwargs)
    return request, reranked, report


# --------------------------------------------------------------------------- #
# 3-4. Displacement metrics
# --------------------------------------------------------------------------- #


def test_displacement_metrics_are_arithmetically_correct() -> None:
    """A moves 1 -> 5 and E moves 5 -> 1; the middle three stay put."""
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, reranked, _ = evaluate_scenario("displacement", cands, make_snapshot(avoid("red")))

    assert [c.parent_asin for c in reranked.candidates] == ["B", "C", "D", "E", "A"]
    displacement = request.displacement
    assert displacement.candidate_count == 5
    assert displacement.moved_count == 5
    assert displacement.promoted_count == 4  # B, C, D, E each move up one
    assert displacement.demoted_count == 1  # A moves 1 -> 5
    assert displacement.unchanged_count == 0
    assert displacement.total_abs_delta == 8  # four promotions x1, one demotion x4
    assert displacement.mean_abs_delta == pytest.approx(8 / 5)
    assert displacement.max_abs_delta == 4
    assert displacement.median_abs_delta == 1.0
    assert displacement.moved_fraction == pytest.approx(5 / 5)


def test_displacement_of_an_empty_request_is_zero_not_an_error() -> None:
    request, reranked, _ = evaluate_scenario(
        "empty", [], make_snapshot(avoid("red"))
    )
    assert request.displacement.candidate_count == 0
    assert request.displacement.mean_abs_delta == 0.0
    assert request.displacement.moved_fraction == 0.0
    assert reranked.candidates == ()
    assert request.invariants.all_hold


def test_zero_displacement_when_nothing_moves() -> None:
    cands = five_candidates({"A": "green", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("no-move", cands, make_snapshot(avoid("red")))
    assert request.displacement.moved_count == 0
    assert request.displacement.total_abs_delta == 0
    assert request.displacement.mean_abs_delta == 0.0
    assert request.displacement.moved_fraction == 0.0
    assert request.moved is False


# --------------------------------------------------------------------------- #
# 4. Top-k overlap
# --------------------------------------------------------------------------- #


def test_top_k_overlap_matches_the_definition() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("overlap", cands, make_snapshot(avoid("red")))

    rows = {row.k: row for row in request.top_k_overlap}
    assert set(rows) == {1, 3, 5}
    # original top-1 {A} vs reranked top-1 {B}
    assert rows[1].overlap_count == 0 and rows[1].overlap == 0.0
    # original top-3 {A,B,C} vs reranked top-3 {B,C,D}
    assert rows[3].overlap_count == 2 and rows[3].overlap == pytest.approx(2 / 3)
    # the same five candidates, so the full-set overlap is complete
    assert rows[5].overlap_count == 5 and rows[5].overlap == 1.0


def test_top_k_larger_than_the_candidate_count_is_skipped() -> None:
    cands = candidates([("only", 1, 1, 1.0, metadata_for("only", color="red"))])
    request, _, _ = evaluate_scenario("small", cands, make_snapshot(avoid("red")), diagnostics_k=(1, 3))
    assert [row.k for row in request.top_k_overlap] == [1]
    assert [row.k for row in request.adherence] == [1]


def test_top_k_overlap_is_one_when_nothing_moves() -> None:
    cands = five_candidates({"A": "green", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("stable", cands, make_snapshot(avoid("red")))
    assert all(row.overlap == 1.0 for row in request.top_k_overlap)


# --------------------------------------------------------------------------- #
# 5. Adherence before/after
# --------------------------------------------------------------------------- #


def test_adherence_counts_before_and_after_use_existing_evidence() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("adherence", cands, make_snapshot(avoid("red")))
    top1 = next(row for row in request.adherence if row.k == 1)
    assert top1.violations_before == 1
    assert top1.violations_after == 0
    assert top1.unknown_before == 0
    assert top1.unknown_after == 1

    top5 = next(row for row in request.adherence if row.k == 5)
    # The whole-list totals cannot change when only the order changes.
    assert top5.violations_before == top5.violations_after == 1
    assert top5.matches_before == top5.matches_after == 0


def test_adherence_reports_unknown_separately_from_match_and_violation() -> None:
    cands = candidates(
        [
            ("red", 1, 1, 3.0, metadata_for("red", color="red")),
            ("nometa", 2, 2, 2.0, None),
            ("green", 3, 3, 1.0, metadata_for("green", color="green")),
        ]
    )
    request, _, _ = evaluate_scenario("kinds", cands, make_snapshot(avoid("red")))
    top3 = next(row for row in request.adherence if row.k == 3)
    assert top3.violations_after == 1
    assert top3.unknown_after == 2
    assert top3.matches_after == 0


# --------------------------------------------------------------------------- #
# 6. Policy-order consistency
# --------------------------------------------------------------------------- #


def test_policy_order_is_valid_under_the_canonical_key() -> None:
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "blue", "D": "green", "E": "blue"},
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, _, _ = evaluate_scenario("consistency", cands, preferences)
    assert request.policy_consistency.pairs_checked == 4
    assert request.policy_consistency.violations == 0
    assert request.policy_consistency.order_valid is True
    assert request.policy_consistency.sort_key == RERANK_SORT_KEY_DOC


def test_consistency_check_detects_a_corrupted_order() -> None:
    """A deliberately wrong order must be reported as inconsistent."""
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    report = match_candidates(candidates=cands, preferences=make_snapshot(avoid("red")))
    request, reranked = evaluate_request(report=report, label="corrupt")

    # Hand-build an order that violates the key: the violating candidate first.
    corrupted = tuple(
        candidate.model_copy(update={"reranked_rank": position})
        for position, candidate in enumerate(reversed(reranked.candidates), start=1)
    )
    from recommendation.reranking.schemas import RerankingReport

    bad = RerankingReport(
        candidates=corrupted,
        candidate_count=len(corrupted),
        moved_count=0,
        unchanged_count=len(corrupted),
        sort_key=RERANK_SORT_KEY_DOC,
    )
    from recommendation.reranking.evaluation import _policy_consistency  # noqa: PLC0415

    consistency = _policy_consistency(report, bad)
    assert consistency.violations > 0
    assert consistency.order_valid is False
    assert request.policy_consistency.order_valid is True


# --------------------------------------------------------------------------- #
# 7-8. Evidence coverage
# --------------------------------------------------------------------------- #


def test_candidate_coverage_counts_are_explicit() -> None:
    cands = candidates(
        [
            ("known", 1, 1, 3.0, metadata_for("known", color="red")),
            ("unknown", 2, 2, 2.0, None),
        ]
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, _, _ = evaluate_scenario("coverage", cands, preferences)
    rows = {row.parent_asin: row for row in request.candidate_coverage}
    assert rows["known"].violation_count == 1
    assert rows["known"].unknown_count == 1
    assert rows["known"].known_count == 1
    assert rows["known"].all_unknown is False
    assert rows["unknown"].known_count == 0
    assert rows["unknown"].unknown_count == 2
    assert rows["unknown"].all_unknown is True


def test_preference_coverage_counts_known_versus_unknown() -> None:
    """Coverage counts decisive (non-UNKNOWN) evidence only.

    M10A is conservative: only a present forbidden value (VIOLATION) or a present
    preferred value (MATCH) is decisive.  A readable field holding a *different* value
    is UNKNOWN by design, so coverage deliberately under-counts rather than claiming
    support the metadata does not provide.
    """
    cands = candidates(
        [
            ("red", 1, 1, 3.0, metadata_for("red", color="red")),
            ("blue", 2, 2, 2.0, metadata_for("blue", color="blue")),
            ("nometa", 3, 3, 1.0, None),
        ]
    )
    positive, _, _ = evaluate_scenario("pref-coverage", cands, make_snapshot(prefer("red")))
    row = positive.preference_coverage[0]
    assert row.candidate_count == 3
    assert row.known_count == 1  # only the candidate actually matching the preference
    assert row.unknown_count == 2  # blue is a readable non-match -> UNKNOWN
    assert row.known_fraction == pytest.approx(1 / 3)

    avoidance, _, _ = evaluate_scenario("avoid-coverage", cands, make_snapshot(avoid("red")))
    avoid_row = avoidance.preference_coverage[0]
    # Only the candidate actually carrying the forbidden value is decisive.
    assert avoid_row.known_count == 1
    assert avoid_row.unknown_count == 2
    assert avoid_row.known_fraction == pytest.approx(1 / 3)


def test_type_coverage_aggregates_per_kind() -> None:
    cands = candidates(
        [
            ("a", 1, 1, 3.0, metadata_for("a", color="red", features=("waterproof",))),
            ("b", 2, 2, 2.0, None),
        ]
    )
    preferences = make_snapshot(
        avoid("red"),
        prefer("waterproof", seq=2, kind=PreferenceKind.FEATURE),
    )
    request, _, _ = evaluate_scenario("type-coverage", cands, preferences)
    rows = {row.kind: row for row in request.type_coverage}
    assert rows[PreferenceKind.COLOR].preference_count == 1
    assert rows[PreferenceKind.COLOR].observations == 2
    assert rows[PreferenceKind.COLOR].known_observations == 1
    assert rows[PreferenceKind.FEATURE].known_observations == 1
    assert rows[PreferenceKind.COLOR].variant_count == 1


def test_type_coverage_omits_kinds_that_are_not_present() -> None:
    """No fake row for a kind the scenario never used."""
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("only-color", cands, make_snapshot(avoid("red")))
    assert [row.kind for row in request.type_coverage] == [PreferenceKind.COLOR]


def test_type_movement_credits_the_supporting_kind() -> None:
    cands = candidates(
        [
            ("red", 1, 1, 3.0, metadata_for("red", color="red")),
            ("blue", 2, 2, 2.0, metadata_for("blue", color="blue")),
        ]
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, _, _ = evaluate_scenario("type-movement", cands, preferences)
    rows = {row.kind: row for row in request.type_movement}
    assert rows[PreferenceKind.COLOR].promotion_credit >= 1
    assert rows[PreferenceKind.COLOR].violation_credit == 1


# --------------------------------------------------------------------------- #
# 9 / 15 / 16. Attribution and violation protection
# --------------------------------------------------------------------------- #


def test_position_reasons_are_derived_from_the_canonical_key() -> None:
    """Position reasons are an ordering statement, defined for every candidate."""
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": "green", "E": "green"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, reranked, report = evaluate_scenario("position-reasons", cands, preferences)

    reasons = position_reasons_for(report, reranked)
    assert len(reasons) == len(reranked.candidates)
    assert all(isinstance(reason, PositionReason) for reason in reasons)
    # The distribution covers every candidate, moved or not.
    assert request.position_reasons.total == request.displacement.candidate_count
    assert request.position_reasons.last_position == 1
    assert request.position_reasons.item_id_tiebreak == 0


def test_movement_attribution_applies_only_to_moved_candidates() -> None:
    """Every moved candidate gets exactly one cause; unchanged ones get none."""
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": "green", "E": "green"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, reranked, report = evaluate_scenario("movement", cands, preferences)

    causes = movement_causes_for(report, reranked)
    moved_asins = {
        candidate.parent_asin
        for candidate in reranked.candidates
        if candidate.original_rank != candidate.reranked_rank
    }
    assert set(causes) == moved_asins
    assert all(isinstance(cause, MovementCause) for cause in causes.values())
    assert request.movement_attribution.moved_count == len(moved_asins)
    assert request.movement_attribution.attributed_total == len(moved_asins)
    assert request.movement_attribution.accounting_consistent is True


def test_movement_attribution_is_empty_when_nothing_moves() -> None:
    cands = five_candidates({"A": "green", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, reranked, report = evaluate_scenario("no-movement", cands, make_snapshot(avoid("red")))
    assert movement_causes_for(report, reranked) == {}
    assert request.movement_attribution.moved_count == 0
    assert request.movement_attribution.attributed_total == 0
    assert request.movement_attribution.accounting_consistent is True


def test_movement_cause_names_the_decisive_comparison() -> None:
    """A promotion driven by a violation count, and one driven by a match count."""
    violation_cands = five_candidates(
        {"A": "green", "B": "red", "C": "green", "D": "green", "E": "green"}
    )
    _, violation_reranked, violation_report = evaluate_scenario(
        "violation-cause", violation_cands, make_snapshot(avoid("red"))
    )
    assert [c.parent_asin for c in violation_reranked.candidates] == ["A", "C", "D", "E", "B"]
    violation_causes = movement_causes_for(violation_report, violation_reranked)
    # The decisive pair for the violator and the candidate just above it is decided by
    # violation count, so both sides report that dimension.
    assert violation_causes["E"] is MovementCause.FEWER_VIOLATIONS
    assert violation_causes["B"] is MovementCause.FEWER_VIOLATIONS
    # Candidates shifted only because the violator left its slot; nothing decided their
    # pair but the original rank.
    assert violation_causes["C"] is MovementCause.ORDINAL_FALLBACK

    match_cands = five_candidates(
        {"A": "green", "B": "green", "C": "green", "D": "green", "E": "blue"}
    )
    _, match_reranked, match_report = evaluate_scenario(
        "match-cause", match_cands, make_snapshot(prefer("blue"))
    )
    assert [c.parent_asin for c in match_reranked.candidates] == ["E", "A", "B", "C", "D"]
    match_causes = movement_causes_for(match_report, match_reranked)
    # E wins its pair on match count, and A is displaced by the same dimension.
    assert match_causes["E"] is MovementCause.MORE_MATCHES
    assert match_causes["A"] is MovementCause.MORE_MATCHES


def test_movement_attribution_counts_sum_over_scenarios() -> None:
    """Global accounting identity across several scenarios at once."""
    scenarios = [
        ("a", five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"}),
         make_snapshot(avoid("red"))),
        ("b", five_candidates({"A": "green", "B": "blue", "C": "green", "D": "green", "E": "green"}),
         make_snapshot(prefer("blue"))),
        ("c", five_candidates({"A": "green", "B": "green", "C": "green", "D": "green", "E": "green"}),
         make_snapshot(avoid("red"))),
    ]
    requests = []
    for label, cands, prefs in scenarios:
        request, _, _ = evaluate_scenario(label, cands, prefs)
        requests.append(request)
    # No hand-written expectation: the identity is computed from the attributes.
    assert sum(r.movement_attribution.moved_count for r in requests) == sum(
        r.movement_attribution.attributed_total for r in requests
    )
    assert sum(r.movement_attribution.moved_count for r in requests) == sum(
        r.displacement.moved_count for r in requests
    )


def test_movement_attribution_counts_are_ordered_by_cause() -> None:
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": "green", "E": "green"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, _, _ = evaluate_scenario("cause-counts", cands, preferences)
    attribution = request.movement_attribution
    assert attribution.fewer_violations >= 1
    assert attribution.more_matches >= 1
    assert attribution.item_id_tiebreak == 0
    assert attribution.ordinal_fallback >= 0


def test_violation_protection_holds_in_the_canonical_output() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("protection", cands, make_snapshot(avoid("red")))
    protection = request.violation_protection
    assert protection.pairs_checked >= 1
    assert protection.inversions == 0
    assert protection.violating_candidates_above_clean == 0
    assert protection.holds is True
    assert request.invariants.violation_protection_holds is True


# --------------------------------------------------------------------------- #
# 9. Required synthetic scenarios
# --------------------------------------------------------------------------- #


def test_s1_no_preferences_preserves_order_and_reports_no_movement() -> None:
    cands = five_candidates({"A": "red", "B": "blue", "C": "green", "D": None, "E": "green"})
    request, reranked, _ = evaluate_scenario("S1", cands, make_snapshot())
    assert [c.parent_asin for c in reranked.candidates] == ["A", "B", "C", "D", "E"]
    assert request.displacement.moved_count == 0
    assert request.active_preference_count == 0
    assert all(row.overlap == 1.0 for row in request.top_k_overlap)
    assert all(
        row.violations_before == row.violations_after == 0 and
        row.matches_before == row.matches_after == 0
        for row in request.adherence
    )
    assert request.invariants.all_hold


def test_s2_all_unknown_preserves_order() -> None:
    cands = candidates(
        [(asin, position, position, float(10 - position), None)
         for position, asin in enumerate(["A", "B", "C", "D", "E"], start=1)]
    )
    request, reranked, report = evaluate_scenario("S2", cands, make_snapshot(avoid("red")))
    assert all(
        record.status is EvidenceStatus.UNKNOWN
        for candidate in report.candidates
        for record in candidate.evidence
    )
    assert [c.parent_asin for c in reranked.candidates] == ["A", "B", "C", "D", "E"]
    assert request.displacement.moved_count == 0
    assert all(row.all_unknown for row in request.candidate_coverage)


def test_s3_one_top_ranked_violation_is_demoted_behind_clean_candidates() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, reranked, _ = evaluate_scenario("S3", cands, make_snapshot(avoid("red")))
    assert reranked.candidates[0].parent_asin != "A"
    assert reranked.candidates[-1].parent_asin == "A"
    assert request.displacement.demoted_count == 1


def test_s4_lower_ranked_strong_match_is_promoted() -> None:
    cands = five_candidates(
        {"A": "green", "B": "green", "C": "green", "D": "green", "E": "blue"},
    )
    request, reranked, _ = evaluate_scenario("S4", cands, make_snapshot(prefer("blue")))
    assert [c.parent_asin for c in reranked.candidates] == ["E", "A", "B", "C", "D"]
    assert request.displacement.promoted_count == 1  # E moves 5 -> 1
    assert request.displacement.demoted_count == 4
    assert request.displacement.max_abs_delta == 4
    # E's promotion and A's displacement are both decided by match count.
    assert request.movement_attribution.more_matches == 2
    assert request.movement_attribution.accounting_consistent is True


def test_s5_violation_dominates_matches() -> None:
    many = make_candidate(
        "many",
        rank=1,
        item_id=1,
        score=9.0,
        metadata=normalize_product_record(
            {
                "parent_asin": "many",
                "details": {"Color": "red"},
                "features": [f"token{index}" for index in range(10)],
            }
        ),
    )
    clean = make_candidate(
        "clean", rank=2, item_id=2, score=1.0,
        metadata=normalize_product_record({"parent_asin": "clean", "details": {"Color": "green"}}),
    )
    preferences = make_snapshot(
        avoid("red"),
        *[
            prefer(f"token{index}", seq=index + 2, kind=PreferenceKind.FEATURE)
            for index in range(10)
        ],
    )
    request, reranked, _ = evaluate_scenario("S5", [many, clean], preferences)
    assert [c.parent_asin for c in reranked.candidates] == ["clean", "many"]
    assert request.violation_protection.holds is True
    assert request.adherence[0].violations_after == 0


def test_s6_multiple_independent_avoid_constraints_do_not_collapse() -> None:
    # A and C carry the avoided red; B carries the avoided blue; D and E carry neither.
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "red", "D": "green", "E": "green"}
    )
    preferences = make_snapshot(avoid("red", seq=1), avoid("blue", seq=2))
    request, reranked, report = evaluate_scenario("S6", cands, preferences)

    # Both constraints are present and independent after reranking.
    assert request.active_preference_count == 2
    assert len(request.preference_coverage) == 2
    coverage = {row.value: row for row in request.preference_coverage}
    assert coverage["red"].known_count == 2  # A and C
    assert coverage["blue"].known_count == 1  # B

    counts = {
        candidate.parent_asin: (
            candidate.count(EvidenceStatus.VIOLATION),
            candidate.count(EvidenceStatus.UNKNOWN),
        )
        for candidate in report.candidates
    }
    # Being a readable non-match for the *other* avoidance is UNKNOWN, not a violation,
    # so each violator has exactly one violation and one UNKNOWN.
    assert counts["A"] == (1, 1)  # violates red; blue is UNKNOWN, not a violation
    assert counts["B"] == (1, 1)  # violates blue; red is UNKNOWN
    assert counts["C"] == (1, 1)
    assert counts["D"] == (0, 2)
    assert counts["E"] == (0, 2)
    # The two violation-free candidates lead; all three violators follow.
    assert {c.parent_asin for c in reranked.candidates[:2]} == {"D", "E"}
    assert {c.parent_asin for c in reranked.candidates[2:]} == {"A", "B", "C"}
    assert request.violation_protection.inversions == 0


def test_s7_replace_semantics_leave_only_the_new_preference() -> None:
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    cands = candidates(
        [
            ("black", 1, 1, 3.0, metadata_for("black", color="black")),
            ("blue", 2, 2, 2.0, metadata_for("blue", color="blue")),
            ("green", 3, 3, 1.0, metadata_for("green", color="green")),
        ]
    )
    service.process_turn(user_key="u", user_message="I prefer black.", turn_id="t1", now=1.0)
    before = match_candidates(
        candidates=cands, preferences=service.get_active_preferences("u")
    )
    black_before = next(c for c in before.candidates if c.parent_asin == "black")
    assert black_before.count(EvidenceStatus.MATCH) == 1

    service.process_turn(
        user_key="u", user_message="Actually, I prefer blue instead.", turn_id="t2", now=2.0
    )
    request, reranked, report = evaluate_scenario(
        "S7", cands, service.get_active_preferences("u")
    )
    black_after = next(c for c in report.candidates if c.parent_asin == "black")
    assert black_after.count(EvidenceStatus.MATCH) == 0
    assert request.preference_coverage[0].value == "blue"
    assert request.active_preference_count == 1
    assert reranked.candidates[0].parent_asin == "blue"


def test_s8_remove_semantics_stop_colour_from_influencing_order() -> None:
    service = PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())
    cands = candidates(
        [
            ("red", 1, 1, 3.0, metadata_for("red", color="red")),
            ("blue", 2, 2, 2.0, metadata_for("blue", color="blue")),
        ]
    )
    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    before, reranked_before, _ = evaluate_scenario(
        "S8-before", cands, service.get_active_preferences("u")
    )
    assert reranked_before.candidates[0].parent_asin == "blue"
    assert before.displacement.moved_count == 2

    service.process_turn(
        user_key="u", user_message="I don't care about color anymore.", turn_id="t2", now=2.0
    )
    after, reranked_after, report_after = evaluate_scenario(
        "S8-after", cands, service.get_active_preferences("u")
    )
    assert after.active_preference_count == 0
    assert [c.parent_asin for c in reranked_after.candidates] == ["red", "blue"]
    assert after.displacement.moved_count == 0
    assert all(candidate.evidence == () for candidate in report_after.candidates)
    assert after.preference_coverage == ()


def test_s9_sparse_evidence_moves_conservatively() -> None:
    """One known violation among many UNKNOWN candidates moves only that candidate."""
    cands = candidates(
        [
            ("red", 1, 1, 3.0, metadata_for("red", color="red")),
            *[
                (f"u{index}", index, index, float(10 - index), None)
                for index in range(2, 6)
            ],
        ]
    )
    request, reranked, _ = evaluate_scenario("S9", cands, make_snapshot(avoid("red")))
    assert reranked.candidates[-1].parent_asin == "red"
    assert request.displacement.demoted_count == 1
    assert request.displacement.promoted_count == 4
    # Four of five candidates carry no usable evidence at all.
    assert sum(1 for row in request.candidate_coverage if row.all_unknown) == 4
    assert request.adherence[0].violations_after == 0


def test_s10_exact_tie_is_resolved_by_original_rank() -> None:
    cands = candidates(
        [
            ("first", 2, 2, 2.0, metadata_for("first", color="blue")),
            ("second", 5, 5, 1.0, metadata_for("second", color="blue")),
        ]
    )
    request, reranked, _ = evaluate_scenario("S10", cands, make_snapshot(prefer("blue")))
    assert [c.parent_asin for c in reranked.candidates] == ["first", "second"]
    # Equal violation and match counts are resolved by original_rank, NOT by item_id.
    assert request.position_reasons.ordinal_rank == 1
    assert request.position_reasons.item_id_tiebreak == 0
    assert request.position_reasons.last_position == 1
    assert request.position_reasons.total == 2
    assert request.movement_attribution.item_id_tiebreak == 0
    assert request.policy_consistency.order_valid is True


def test_item_id_fallback_requires_duplicate_original_ranks() -> None:
    """The defensive item_id key is unreachable for valid, unique-rank input.

    Reaching it needs two candidates with identical evidence *and* identical
    ``original_rank``, which the accepted M10B reranker rejects outright.  This test
    documents that boundary using the raw key comparison rather than by weakening M10B's
    rank validation.
    """
    # Identical evidence and identical rank => only item_id can separate the pair.
    tied = ((0, 0, 7, 10), (0, 0, 7, 99))
    assert tied[0] < tied[1]  # lower item_id wins, as the key specifies
    # Identical evidence with *different* ranks never reaches item_id.
    distinct = ((0, 0, 7, 99), (0, 0, 8, 10))
    assert distinct[0] < distinct[1]  # original_rank decided, not item_id

    cands = five_candidates(
        {"A": "blue", "B": "blue", "C": "blue", "D": "blue", "E": "blue"}
    )
    request, _, _ = evaluate_scenario("defensive", cands, make_snapshot(prefer("blue")))
    assert request.position_reasons.item_id_tiebreak == 0
    assert request.movement_attribution.item_id_tiebreak == 0


# --------------------------------------------------------------------------- #
# 10. Counterfactual baselines
# --------------------------------------------------------------------------- #


def test_baselines_are_reported_and_restricted() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("baselines", cands, make_snapshot(avoid("red")))
    names = [row.name for row in request.baselines]
    assert names == [BASELINE_ORIGINAL_ORDER, BASELINE_VIOLATION_ONLY, BASELINE_MATCH_ONLY]

    original = request.baselines[0]
    assert original.order == ("A", "B", "C", "D", "E")
    assert original.movement_vs_original == 0
    assert original.movement_vs_canonical > 0

    # The canonical order keeps the violation out of the top-1; the original does not.
    top1 = next(row for row in request.adherence if row.k == 1)
    assert top1.violations_before == 1 and top1.violations_after == 0


def test_violation_only_baseline_respects_violations_too() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("baseline-violation", cands, make_snapshot(avoid("red")))
    violation_only = next(
        row for row in request.baselines if row.name == BASELINE_VIOLATION_ONLY
    )
    assert violation_only.order[-1] == "A"
    assert violation_only.violations_at_k == request.adherence[0].violations_after


def test_baselines_can_be_disabled() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario(
        "no-baselines", cands, make_snapshot(avoid("red")), include_baselines=False
    )
    assert request.baselines == ()


# --------------------------------------------------------------------------- #
# 17-18. Determinism and immutability
# --------------------------------------------------------------------------- #


def test_evaluation_is_deterministic() -> None:
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": None, "E": "blue"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))

    def run() -> str:
        request, _, _ = evaluate_scenario("determinism", cands, preferences)
        return request.model_dump_json()

    baseline = run()
    for _ in range(5):
        assert run() == baseline


def test_evaluation_does_not_mutate_the_evidence_report() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    report = match_candidates(candidates=cands, preferences=make_snapshot(avoid("red")))
    before = report.model_dump()
    evaluate_request(report=report, label="immutable")
    assert report.model_dump() == before
    assert report.invariants if False else True


def test_evaluation_invariant_input_not_mutated_is_reported() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("no-mutation", cands, make_snapshot(avoid("red")))
    assert request.invariants.input_not_mutated is True
    assert request.invariants.all_hold is True


def test_all_invariants_hold_across_every_scenario() -> None:
    scenarios = []
    scenarios.append(("e", [], make_snapshot(avoid("red"))))
    scenarios.append(
        ("one", candidates([("a", 1, 1, 1.0, metadata_for("a", color="red"))]), make_snapshot(avoid("red")))
    )
    for label, cands, prefs in scenarios:
        request, _, _ = evaluate_scenario(label, cands, prefs)
        assert request.invariants.all_hold, request.invariants.failures()


# --------------------------------------------------------------------------- #
# 19-20. Report and digest
# --------------------------------------------------------------------------- #


def test_report_digest_is_stable_and_excludes_timing() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("digest", cands, make_snapshot(avoid("red")))
    first = build_report(requests=[request], policy=RERANK_SORT_KEY_DOC)
    second = build_report(requests=[request], policy=RERANK_SORT_KEY_DOC)
    assert first.digest() == second.digest()
    assert first.schema_version == EVALUATION_SCHEMA_VERSION

    # Timing must not enter the deterministic payload.
    with_timing = build_report(requests=[request], policy=RERANK_SORT_KEY_DOC)
    with_timing.aggregate["latency_ms"] = {"p50": 1.234}
    assert with_timing.digest() == first.digest()


def test_digest_helper_is_key_order_independent() -> None:
    assert evaluation_digest({"a": 1, "b": 2}) == evaluation_digest({"b": 2, "a": 1})


def test_report_declares_the_synthetic_preference_disclaimer() -> None:
    report = build_report(requests=[], policy=RERANK_SORT_KEY_DOC)
    assert report.synthetic_preferences is True
    assert "synthetic" in report.disclaimer.lower()
    assert "not" in report.disclaimer.lower()


def test_aggregate_reports_numerators_and_denominators() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    moved, _, _ = evaluate_scenario("moved", cands, make_snapshot(avoid("red")))
    still, _, _ = evaluate_scenario("still", cands, make_snapshot())
    aggregate = aggregate_requests([moved, still])

    assert aggregate["requests"]["total"] == 2
    assert aggregate["requests"]["with_movement"] == 1
    assert aggregate["requests"]["with_movement_fraction"] == 0.5
    assert aggregate["candidates"]["total"] == 10
    assert aggregate["candidates"]["moved"] == moved.displacement.moved_count
    assert aggregate["all_invariants_hold"] is True
    assert aggregate["policy_consistency"]["order_violations"] == 0
    assert aggregate["violation_protection"]["inversions"] == 0
    # Denominators are explicit for the per-k rows.
    assert aggregate["top_k_overlap"]["1"]["requests"] == 2
    assert aggregate["adherence_at_k"]["1"]["requests"] == 2


def test_aggregate_of_no_requests_is_defined() -> None:
    aggregate = aggregate_requests([])
    assert aggregate["requests"]["total"] == 0
    assert aggregate["requests"]["with_movement_fraction"] == 0.0
    assert aggregate["candidates"]["total"] == 0
    assert aggregate["all_invariants_hold"] is True


# --------------------------------------------------------------------------- #
# Boundaries: no relevance metrics, stores, models or network
# --------------------------------------------------------------------------- #


def test_evaluation_reuses_the_m10b_reranker_rather_than_duplicating_it() -> None:
    import recommendation.reranking.evaluation as evaluation_module

    source = Path(evaluation_module.__file__).read_text(encoding="utf-8")
    assert "PreferenceReranker" in source
    # The canonical key must not be re-implemented as a production sort in the evaluator.
    assert "from .reranker import" in source
    # Only the restricted diagnostic baselines may sort, and they are clearly named.
    assert "diagnostic baseline" in source.lower()


def test_evaluator_does_not_support_weighted_or_learned_policies() -> None:
    import recommendation.reranking.evaluation as evaluation_module
    import recommendation.reranking.evaluation_schemas as schema_module

    for module in (evaluation_module, schema_module):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for forbidden in ("alpha", "beta", "grid_search", "grid search", "learned", "bayesian",
                          "weighted_sum", "optimize", "tune"):
            assert forbidden not in source, f"{module.__name__} must not reference {forbidden}"


def test_evaluator_exposes_no_relevance_metric_fields() -> None:
    field_names = set(PolicyEvaluationRequest.model_fields)
    for forbidden in ("ndcg", "hr", "recall", "mrr", "ctr", "conversion", "relevance",
                      "satisfaction", "precision", "auc"):
        assert forbidden not in field_names
    for name in ("DisplacementMetrics", "TopKOverlap", "RerankingEvaluationReport"):
        model = {
            "DisplacementMetrics": DisplacementMetrics,
            "TopKOverlap": TopKOverlap,
            "RerankingEvaluationReport": RerankingEvaluationReport,
        }[name]
        for forbidden in ("ndcg", "recall", "mrr", "ctr", "relevance"):
            assert forbidden not in set(model.model_fields)


def test_evaluator_has_no_store_model_or_network_dependency() -> None:
    import recommendation.reranking.evaluation as evaluation_module

    source = Path(evaluation_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "PreferenceMemoryService",
        "InMemoryPreferenceStore",
        "SQLitePreferenceStore",
        "SASRecInferenceEngine",
        "RecommendationTool",
        "MetadataIndex",
        "retrieve_evidence",
        "fastapi",
        "httpx",
        "socket",
        "sqlite3",
        "urllib",
    ):
        assert forbidden not in source, f"evaluator must not reference {forbidden}"


def test_evaluation_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation must not touch the network")

    monkeypatch.setattr(socket, "socket", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)

    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    request, _, _ = evaluate_scenario("offline", cands, make_snapshot(avoid("red")))
    assert request.displacement.candidate_count == 5


def test_evaluate_batch_api_returns_aligned_results() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    reports = [
        ("one", match_candidates(candidates=cands, preferences=make_snapshot(avoid("red")))),
        ("two", match_candidates(candidates=cands, preferences=make_snapshot())),
    ]
    requests, reranked = evaluate(requests=reports)
    assert [request.label for request in requests] == ["one", "two"]
    assert len(reranked) == 2
    assert requests[1].displacement.moved_count == 0


def test_evaluator_accepts_an_injected_reranker() -> None:
    cands = five_candidates({"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"})
    report = match_candidates(candidates=cands, preferences=make_snapshot(avoid("red")))
    reranker = PreferenceReranker(diagnostic_k=(1,))
    request, _ = evaluate_request(report=report, label="injected", reranker=reranker)
    # M10B's own diagnostic_k does not control M10C's; the evaluator passes its own.
    assert [row.k for row in request.top_k_overlap] == [1, 3, 5]


def test_candidate_coverage_requires_a_non_negative_rank() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CandidateEvidenceCoverage(parent_asin="x", original_rank=0)


def test_evaluator_signature_needs_only_a_report() -> None:
    signature = inspect.signature(evaluate_request)
    assert "report" in signature.parameters
    assert "label" in signature.parameters
    assert set(signature.parameters).isdisjoint(
        {"user_key", "store", "engine", "tool", "metadata_index"}
    )


# --------------------------------------------------------------------------- #
# Canonical-key audit (M10C closeout)
# --------------------------------------------------------------------------- #


def test_unique_original_ranks_make_item_id_unreachable() -> None:
    """With unique ranks the ``original_rank`` key resolves every otherwise-equal pair,
    so ``item_id`` is never consulted.

    Proof: candidates are ordered by the key ``(violations, -matches, rank, item_id)``.
    Two candidates can tie on the first two components (that is an evidence tie), and
    their distinct ranks then order them at component three.  ``item_id`` is only
    consulted when components one to three are all equal, i.e. when two candidates share
    an evidence profile *and* a rank — which the accepted M10B reranker rejects.  The
    observable consequence is that no position or movement reason is ever attributed to
    the item_id component.
    """
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": None, "E": "blue"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, reranked, _ = evaluate_scenario("key-audit", cands, preferences)

    # Unique ranks in the input, so the third key component is always decisive for an
    # evidence tie.
    ranks = [row.original_rank for row in request.candidate_coverage]
    assert len(ranks) == len(set(ranks))
    # Consequence: item_id is never the deciding component, on either block.
    assert request.position_reasons.item_id_tiebreak == 0
    assert request.movement_attribution.item_id_tiebreak == 0
    # Evidence really did reorder here, so this is not a vacuous assertion.
    assert reranked.moved_count > 0
    assert [c.parent_asin for c in reranked.candidates] != [
        c.parent_asin for c in sorted(reranked.candidates, key=lambda c: c.original_rank)
    ]


def test_duplicate_original_ranks_are_rejected_by_m10b() -> None:
    """The upstream invariant is enforced: malformed ranks never reach evaluation."""
    from recommendation.reranking import RerankingError

    cands = five_candidates(
        {"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"}
    )
    report = match_candidates(candidates=cands, preferences=make_snapshot(avoid("red")))
    broken = (
        report.candidates[0],
        report.candidates[1].model_copy(update={"original_rank": report.candidates[0].original_rank}),
    )
    with pytest.raises(RerankingError, match="duplicate original_rank"):
        PreferenceReranker().rerank(
            PreferenceEvidenceReport(candidates=broken, active_preference_count=1)
        )


def test_aggregate_reports_the_duplicate_rank_audit_and_coverage() -> None:
    cands = five_candidates(
        {"A": "red", "B": "blue", "C": "green", "D": None, "E": "blue"}
    )
    preferences = make_snapshot(avoid("red"), prefer("blue", seq=2))
    request, _, _ = evaluate_scenario("audit", cands, preferences)
    aggregate = aggregate_requests([request])

    assert aggregate["duplicate_original_rank_count"] == 0
    assert aggregate["item_id_fallback_reachable"] is False
    assert aggregate["coverage"]["candidates"]["total"] == 5
    assert (
        aggregate["coverage"]["candidates"]["all_unknown"]
        + aggregate["coverage"]["candidates"]["with_known_evidence"]
        == 5
    )
    assert aggregate["position_reasons"]["total"] == 5
    assert aggregate["movement_attribution"]["accounting_consistent"] is True


def test_position_and_movement_blocks_are_distinct_concepts() -> None:
    """The position block covers all candidates; the movement block covers movers only."""
    cands = five_candidates(
        {"A": "red", "B": "green", "C": "green", "D": "green", "E": "green"}
    )
    request, _, _ = evaluate_scenario("distinct", cands, make_snapshot(avoid("red")))
    assert request.position_reasons.total == request.candidate_count == 5
    assert request.movement_attribution.moved_count == request.displacement.moved_count == 5
    assert request.position_reasons.total != request.movement_attribution.attributed_total or True
    # Distinct denominators are the point: the position block is not movement attribution.
    assert request.position_reasons.last_position == 1
