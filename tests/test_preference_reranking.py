"""Milestone 10B tests: deterministic preference-aware reranking.

Fully offline and deterministic.  Fixtures reuse the accepted M10A evidence pipeline, so
the reranker is exercised on real ``PreferenceEvidenceReport`` objects rather than a
bespoke DTO.

Coverage follows the milestone's required list: the canonical policy and its
precedence rules, the preservation invariants (candidates, identity, count, scores,
evidence), the no-preference / all-UNKNOWN / equal-profile guarantees, rank and movement
diagnostics, input validation, M9 ADD/REPLACE/REMOVE integration, determinism and
immutability, and the absence of filtering, weighting, models and stores.
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
from recommendation.reranking import (  # noqa: E402
    RERANK_SORT_KEY_DOC,
    CandidateEvaluation,
    PreferenceReranker,
    RerankReason,
    RerankedCandidate,
    RerankingError,
    rerank_candidates,
    sort_key_for,
)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def metadata_with_color(color: str | None, *, price: float | None = None):
    """Normalized metadata carrying a colour (and optionally a price)."""
    payload: dict[str, object] = {"parent_asin": f"m-{color}"}
    if color is not None:
        payload["details"] = {"Color": color}
    if price is not None:
        payload["price"] = price
    return normalize_product_record(payload)


def candidate_list(specs):
    """Build candidates from ``(asin, rank, item_id, score, color)`` tuples."""
    candidates = []
    for asin, rank, item_id, score, color in specs:
        candidates.append(
            make_candidate(
                asin,
                rank=rank,
                item_id=item_id,
                score=score,
                metadata=None if color is None else metadata_with_color(color),
            )
        )
    return candidates


def avoid_red():
    """An active avoidance of red."""
    return make_snapshot(
        make_entry(memory_id="v-red", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1)
    )


def prefer_blue():
    """An active preference for blue."""
    return make_snapshot(
        make_entry(memory_id="m-blue", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=1)
    )


def rerank(candidates, preferences):
    """Match then rerank, returning the reranking report."""
    return rerank_candidates(
        report=match_candidates(candidates=candidates, preferences=preferences)
    )


# --------------------------------------------------------------------------- #
# 1-5. Degenerate and preservation cases
# --------------------------------------------------------------------------- #


def test_empty_candidate_report_yields_a_valid_empty_reranking() -> None:
    report = PreferenceEvidenceReport(candidates=(), active_preference_count=0)
    result = rerank_candidates(report=report)
    assert result.candidates == ()
    assert result.candidate_count == 0
    assert result.moved_count == 0
    assert result.diagnostics.candidate_count == 0
    assert result.diagnostics.top_k == ()


def test_single_candidate_stays_first_regardless_of_evidence() -> None:
    candidates = candidate_list([("only", 1, 11, 5.0, "red")])
    result = rerank(candidates, avoid_red())
    assert result.candidate_count == 1
    assert result.candidates[0].reranked_rank == 1
    assert result.candidates[0].original_rank == 1
    assert result.candidates[0].violation_count == 1
    assert result.moved_count == 0
    assert result.candidates[0].rank_delta == 0


def test_no_preferences_preserves_original_order_exactly() -> None:
    candidates = candidate_list(
        [("a", 1, 1, 5.0, "red"), ("b", 2, 2, 4.0, "blue"), ("c", 3, 3, 3.0, None), ("d", 4, 4, 2.0, "green")]
    )
    result = rerank(candidates, make_snapshot())
    assert result.original_ranks == (1, 2, 3, 4)
    assert result.reranked_ranks == (1, 2, 3, 4)
    assert result.moved_count == 0
    assert result.diagnostics.active_preference_count == 0
    # With no preferences every candidate has an identical (empty) evidence profile, so
    # each position is explained by the original rank, or by the final item_id fallback.
    assert all(
        c.rerank_reason
        in (
            RerankReason.PRESERVED_ORIGINAL_ORDER,
            RerankReason.RANKED_LAST,
            RerankReason.DETERMINISTIC_TIE_BREAK,
        )
        for c in result.candidates
    )


def test_all_unknown_preserves_original_order() -> None:
    """Metadata sparsity must not change ranking."""
    candidates = candidate_list(
        [("a", 1, 1, 5.0, None), ("b", 2, 2, 4.0, None), ("c", 3, 3, 3.0, None)]
    )
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    assert all(
        record.status is EvidenceStatus.UNKNOWN
        for c in report.candidates
        for record in c.evidence
    )
    result = rerank_candidates(report=report)
    assert result.reranked_ranks == (1, 2, 3)
    assert result.moved_count == 0
    assert result.diagnostics.top_k[0].violations_after == 0
    assert result.diagnostics.top_k[0].matches_after == 0
    assert all(c.unknown_count == 1 for c in result.candidates)


def test_identical_evidence_profile_preserves_original_order() -> None:
    """All three match blue equally; nothing should move."""
    candidates = candidate_list(
        [("a", 1, 1, 5.0, "blue"), ("b", 2, 2, 4.0, "blue"), ("c", 3, 3, 3.0, "blue")]
    )
    result = rerank(candidates, prefer_blue())
    assert [c.match_count for c in result.candidates] == [1, 1, 1]
    assert result.reranked_ranks == (1, 2, 3)
    assert result.moved_count == 0


# --------------------------------------------------------------------------- #
# 6-10. Canonical policy precedence
# --------------------------------------------------------------------------- #


def test_fewer_violations_wins() -> None:
    candidates = candidate_list([("bad", 1, 1, 9.0, "red"), ("good", 2, 2, 1.0, "green")])
    result = rerank(candidates, avoid_red())
    assert result.parent_asins == ("good", "bad")
    assert result.candidates[0].rerank_reason is RerankReason.FEWER_VIOLATIONS


def test_violations_dominate_any_number_of_matches() -> None:
    """Required example: A has 1 violation and 10 matches; B has neither. B wins."""
    candidate_a = make_candidate(
        "a",
        rank=1,
        item_id=1,
        score=9.0,
        metadata=normalize_product_record(
            {"parent_asin": "a", "details": {"Color": "red"}, "features": ["f"] * 1}
        ),
    )
    candidate_b = make_candidate(
        "b", rank=2, item_id=2, score=1.0, metadata=normalize_product_record(
            {"parent_asin": "b", "details": {"Color": "green"}}
        )
    )
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        *[
            make_entry(
                memory_id=f"m{index}",
                kind=PreferenceKind.FEATURE,
                value=f"token{index}",
                polarity=PreferencePolarity.PREFER,
                logical_seq=index + 2,
            )
            for index in range(10)
        ],
    )
    # Give A ten feature matches so the match count is genuinely large.
    candidate_a = make_candidate(
        "a",
        rank=1,
        item_id=1,
        score=9.0,
        metadata=normalize_product_record(
            {
                "parent_asin": "a",
                "details": {"Color": "red"},
                "features": [f"token{index}" for index in range(10)],
            }
        ),
    )
    report = match_candidates(candidates=[candidate_a, candidate_b], preferences=preferences)
    counts_a = report.candidates[0]
    assert counts_a.count(EvidenceStatus.VIOLATION) == 1
    assert counts_a.count(EvidenceStatus.MATCH) == 10

    result = rerank_candidates(report=report)
    assert result.parent_asins == ("b", "a")
    assert result.candidates[0].violation_count == 0
    assert result.candidates[1].match_count == 10


def test_more_matches_wins_when_violations_tie() -> None:
    candidates = candidate_list(
        [("low", 1, 1, 9.0, "green"), ("high", 9, 9, 1.0, "blue")]
    )
    result = rerank(candidates, prefer_blue())
    assert result.parent_asins == ("high", "low")
    assert result.candidates[0].rerank_reason is RerankReason.MORE_MATCHES


def test_original_rank_breaks_evidence_ties() -> None:
    candidates = candidate_list(
        [("first", 2, 20, 5.0, "blue"), ("second", 5, 50, 4.0, "blue")]
    )
    result = rerank(candidates, prefer_blue())
    assert result.parent_asins == ("first", "second")
    assert result.reranked_ranks == (1, 2)
    assert result.candidates[0].original_rank == 2
    assert result.moved_count == 2  # both moved up because rank 1 is absent here


def test_item_id_is_the_final_deterministic_key() -> None:
    """The policy key is total even when every richer component ties."""
    key_low = (0, 0, 7, 10)
    key_high = (0, 0, 7, 99)
    assert sorted([key_high, key_low]) == [key_low, key_high]
    assert RERANK_SORT_KEY_DOC.endswith("item_id ASC")


def test_sort_key_uses_counts_not_evidence_order() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "blue")])
    report = match_candidates(candidates=candidates, preferences=prefer_blue())
    evidence = report.candidates[0].evidence
    evaluation = CandidateEvaluation.from_evidence(evidence)
    key = sort_key_for(report.candidates[0], evaluation)
    assert key == (0, -1, 1, 1)


# --------------------------------------------------------------------------- #
# 11-15. Preservation invariants
# --------------------------------------------------------------------------- #


def test_candidate_count_ids_and_asins_are_preserved() -> None:
    candidates = candidate_list(
        [("a", 1, 1, 5.0, "red"), ("b", 2, 2, 4.0, "blue"), ("c", 3, 3, 3.0, None)]
    )
    report = match_candidates(
        candidates=candidates,
        preferences=make_snapshot(
            make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                       polarity=PreferencePolarity.AVOID, logical_seq=1),
            make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                       polarity=PreferencePolarity.PREFER, logical_seq=2),
        ),
    )
    result = rerank_candidates(report=report)

    assert len(result.candidates) == len(report.candidates)
    assert result.candidate_count == len(candidates)
    assert sorted(result.item_ids) == sorted(c.item_id for c in report.candidates)
    assert sorted(result.parent_asins) == sorted(c.parent_asin for c in report.candidates)
    # multiplicity is exactly one
    assert len(set(result.parent_asins)) == len(result.parent_asins)
    assert len(set(result.item_ids)) == len(result.item_ids)


def test_sasrec_scores_are_copied_exactly() -> None:
    candidates = candidate_list(
        [("a", 1, 1, 5.5555555555, "red"), ("b", 2, 2, 0.1 + 0.2, "blue")]
    )
    result = rerank(candidates, avoid_red())
    by_asin = {c.parent_asin: c.sasrec_score for c in result.candidates}
    assert by_asin["a"] == 5.5555555555
    assert by_asin["b"] == 0.1 + 0.2  # exact float equality, no rounding


def test_evidence_records_are_preserved_structurally() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "red"), ("b", 2, 2, 4.0, None)])
    report = match_candidates(
        candidates=candidates,
        preferences=make_snapshot(
            make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                       polarity=PreferencePolarity.AVOID, logical_seq=1),
            make_entry(memory_id="u", kind=PreferenceKind.FREE_FORM_CONSTRAINT, value="light",
                       polarity=PreferencePolarity.PREFER, logical_seq=2),
        ),
    )
    before = {c.parent_asin: c.evidence for c in report.candidates}
    result = rerank_candidates(report=report)
    for candidate in result.candidates:
        assert candidate.evidence == before[candidate.parent_asin]
    # UNKNOWN evidence was neither removed nor collapsed.
    assert [len(c.evidence) for c in result.candidates] == [2, 2]


def test_no_candidate_is_filtered_even_when_it_violates() -> None:
    candidates = candidate_list([("bad", 1, 1, 9.0, "red"), ("ok", 2, 2, 1.0, "green")])
    result = rerank(candidates, avoid_red())
    assert "bad" in result.parent_asins
    assert result.candidate_count == 2
    assert result.candidates[-1].parent_asin == "bad"
    assert result.candidates[-1].violation_count == 1


# --------------------------------------------------------------------------- #
# 16-19. Ranks, reasons and movement
# --------------------------------------------------------------------------- #


def test_original_rank_is_retained_separately_from_reranked_rank() -> None:
    candidates = candidate_list([("bad", 1, 1, 9.0, "red"), ("good", 2, 2, 1.0, "green")])
    result = rerank(candidates, avoid_red())
    assert result.original_ranks == (2, 1)
    assert result.reranked_ranks == (1, 2)


def test_reranked_ranks_are_contiguous_and_unique() -> None:
    candidates = candidate_list(
        [(f"c{i}", i, i, float(10 - i), "red" if i % 2 else "green") for i in range(1, 8)]
    )
    result = rerank(candidates, avoid_red())
    assert result.reranked_ranks == tuple(range(1, 8))
    assert len(set(result.reranked_ranks)) == 7


def test_rank_delta_sign_semantics() -> None:
    """positive = promoted, 0 = unchanged, negative = demoted."""
    promoted = RerankedCandidate(original_rank=4, reranked_rank=1, item_id=1,
                                 parent_asin="p", sasrec_score=1.0)
    unchanged = RerankedCandidate(original_rank=2, reranked_rank=2, item_id=2,
                                  parent_asin="u", sasrec_score=1.0)
    demoted = RerankedCandidate(original_rank=1, reranked_rank=5, item_id=3,
                                parent_asin="d", sasrec_score=1.0)
    assert promoted.rank_delta == 3 and promoted.moved is True
    assert unchanged.rank_delta == 0 and unchanged.moved is False
    assert demoted.rank_delta == -4 and demoted.moved is True


def test_promoted_and_demoted_counts_are_correct() -> None:
    candidates = candidate_list(
        [("bad", 1, 1, 9.0, "red"), ("mid", 2, 2, 5.0, "green"), ("good", 3, 3, 1.0, "blue")]
    )
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    result = rerank(candidates, preferences)
    diagnostics = result.diagnostics
    assert diagnostics.promoted_count == 1  # good moves 3 -> 1
    assert diagnostics.demoted_count == 1  # bad moves 1 -> 3
    assert diagnostics.unchanged_count == 1  # mid stays at 2
    assert diagnostics.moved_count == 2
    assert result.moved_count == 2
    assert result.unchanged_count == 1
    assert (
        diagnostics.promoted_count + diagnostics.demoted_count + diagnostics.unchanged_count
        == diagnostics.candidate_count
    )


def test_every_candidate_carries_a_machine_readable_reason() -> None:
    candidates = candidate_list([("bad", 1, 1, 9.0, "red"), ("good", 2, 2, 1.0, "blue")])
    result = rerank(candidates, avoid_red())
    for candidate in result.candidates:
        assert isinstance(candidate.rerank_reason, RerankReason)
    assert all(c.reason_detail for c in result.candidates)


# --------------------------------------------------------------------------- #
# 21. Input validation
# --------------------------------------------------------------------------- #


def test_duplicate_original_rank_is_rejected() -> None:
    report = PreferenceEvidenceReport(
        candidates=(
            match_candidates(
                candidates=[make_candidate("a", rank=1, item_id=1, score=1.0)],
                preferences=make_snapshot(),
            ).candidates[0],
            match_candidates(
                candidates=[make_candidate("b", rank=1, item_id=2, score=2.0)],
                preferences=make_snapshot(),
            ).candidates[0],
        )
    )
    with pytest.raises(RerankingError, match="duplicate original_rank"):
        rerank_candidates(report=report)


def test_duplicate_item_id_is_rejected() -> None:
    first = match_candidates(
        candidates=[make_candidate("a", rank=1, item_id=7, score=1.0)],
        preferences=make_snapshot(),
    ).candidates[0]
    second = match_candidates(
        candidates=[make_candidate("b", rank=2, item_id=7, score=2.0)],
        preferences=make_snapshot(),
    ).candidates[0]
    with pytest.raises(RerankingError, match="duplicate item_id"):
        rerank_candidates(report=PreferenceEvidenceReport(candidates=(first, second)))


def test_duplicate_parent_asin_is_rejected() -> None:
    first = match_candidates(
        candidates=[make_candidate("same", rank=1, item_id=1, score=1.0)],
        preferences=make_snapshot(),
    ).candidates[0]
    second = match_candidates(
        candidates=[make_candidate("same", rank=2, item_id=2, score=2.0)],
        preferences=make_snapshot(),
    ).candidates[0]
    with pytest.raises(RerankingError, match="duplicate parent_asin"):
        rerank_candidates(report=PreferenceEvidenceReport(candidates=(first, second)))


def test_non_positive_original_rank_is_rejected() -> None:
    report = PreferenceEvidenceReport(
        candidates=match_candidates(
            candidates=[make_candidate("a", rank=1, item_id=1, score=1.0)],
            preferences=make_snapshot(),
        ).candidates
    )
    broken = report.candidates[0].model_copy(update={"original_rank": 0})
    with pytest.raises(RerankingError, match="must be positive"):
        rerank_candidates(report=PreferenceEvidenceReport(candidates=(broken,)))


def test_malformed_input_is_not_silently_repaired() -> None:
    """A duplicate rank must raise rather than being renumbered."""
    report = match_candidates(
        candidates=[
            make_candidate("a", rank=1, item_id=1, score=1.0),
            make_candidate("b", rank=2, item_id=2, score=2.0),
        ],
        preferences=make_snapshot(),
    )
    broken = (
        report.candidates[0],
        report.candidates[1].model_copy(update={"original_rank": 1}),
    )
    with pytest.raises(RerankingError):
        rerank_candidates(report=PreferenceEvidenceReport(candidates=broken))


# --------------------------------------------------------------------------- #
# 25-30. Integration, neutrality and the absence of ranking artefacts
# --------------------------------------------------------------------------- #


def test_reranker_consumes_the_m10a_report_directly() -> None:
    """The accepted M10A output is the input type; no second DTO, no schema drift."""
    candidates = candidate_list([("a", 1, 1, 5.0, "red")])
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    assert isinstance(report, PreferenceEvidenceReport)
    result = PreferenceReranker().rerank(report)
    assert result.candidates[0].evidence == report.candidates[0].evidence


def test_unknown_is_neutral_between_violation_and_match() -> None:
    """0 violations beats 1 violation, and UNKNOWN neither helps nor hurts."""
    candidates = candidate_list([("violates", 1, 1, 9.0, "red"), ("unknown", 2, 2, 1.0, None)])
    result = rerank(candidates, avoid_red())
    assert result.parent_asins == ("unknown", "violates")
    assert result.candidates[0].violation_count == 0
    assert result.candidates[0].unknown_count == 1
    assert result.candidates[0].match_count == 0


def test_match_count_only_counts_match() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "blue")])
    report = match_candidates(
        candidates=candidates,
        preferences=make_snapshot(
            make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                       polarity=PreferencePolarity.PREFER, logical_seq=1),
            make_entry(memory_id="u", kind=PreferenceKind.FREE_FORM_CONSTRAINT, value="x",
                       polarity=PreferencePolarity.PREFER, logical_seq=2),
        ),
    )
    result = rerank_candidates(report=report)
    assert result.candidates[0].match_count == 1
    assert result.candidates[0].unknown_count == 1
    assert result.candidates[0].violation_count == 0


def test_violation_count_only_counts_violation() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "red")])
    report = match_candidates(
        candidates=candidates,
        preferences=make_snapshot(
            make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                       polarity=PreferencePolarity.AVOID, logical_seq=1),
            make_entry(memory_id="u", kind=PreferenceKind.FREE_FORM_CONSTRAINT, value="x",
                       polarity=PreferencePolarity.PREFER, logical_seq=2),
        ),
    )
    result = rerank_candidates(report=report)
    assert result.candidates[0].violation_count == 1
    assert result.candidates[0].match_count == 0
    assert result.candidates[0].unknown_count == 1


def test_output_contains_no_weighted_or_final_score() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "red")])
    result = rerank(candidates, avoid_red())
    dumped = result.as_dict()
    assert set(dumped) == {
        "candidates",
        "candidate_count",
        "moved_count",
        "unchanged_count",
        "sort_key",
        "diagnostics",
    }
    for candidate in dumped["candidates"]:
        for forbidden in (
            "final_score",
            "weighted_score",
            "preference_score",
            "adjusted_score",
            "penalty",
            "boost",
            "alpha",
            "beta",
        ):
            assert forbidden not in candidate


def test_sort_key_is_declared_in_the_report_for_audit() -> None:
    result = rerank(candidate_list([("a", 1, 1, 5.0, "red")]), avoid_red())
    assert result.sort_key == RERANK_SORT_KEY_DOC
    assert result.sort_key == "violation_count ASC, match_count DESC, original_rank ASC, item_id ASC"


def _executable_source(module: object) -> str:
    """Return executable lines only, so docstrings cannot trip the guards."""
    import ast

    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_lines.update(
                    range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1)
                )
    return "\n".join(
        line
        for number, line in enumerate(source.splitlines(), start=1)
        if number not in doc_lines and line.strip() and not line.lstrip().startswith("#")
    )


def test_reranker_has_no_model_store_or_retriever_dependency() -> None:
    import recommendation.reranking.reranker as reranker_module

    source = _executable_source(reranker_module)
    for forbidden in (
        "PreferenceMemoryService",
        "InMemoryPreferenceStore",
        "SQLitePreferenceStore",
        "SASRecInferenceEngine",
        "RecommendationTool",
        "MetadataIndex",
        "retrieve_evidence",
        "match_candidates",
        "fastapi",
        "httpx",
        "socket",
        "sqlite3",
    ):
        assert forbidden not in source, f"reranker must not reference {forbidden}"

    signature = inspect.signature(PreferenceReranker.rerank)
    assert set(signature.parameters) == {"self", "report"}


# --------------------------------------------------------------------------- #
# 31-35. M9 ADD / REPLACE / REMOVE integration
# --------------------------------------------------------------------------- #


def build_service():
    """A memory service over a fresh in-memory store with the rule extractor."""
    return PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())


def rerank_from_memory(candidates, service, user_key="u"):
    """Match against the service's current ACTIVE state, then rerank."""
    report = match_candidates(
        candidates=candidates, preferences=service.get_active_preferences(user_key)
    )
    return rerank_candidates(report=report)


def test_add_operation_affects_ranking_independently() -> None:
    """Each added avoidance is an independent constraint with its own effect."""
    candidates = candidate_list(
        [("red", 1, 1, 9.0, "red"), ("blue", 2, 2, 5.0, "blue"), ("green", 3, 3, 1.0, "green")]
    )
    service = build_service()
    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    after_first = rerank_from_memory(candidates, service)
    assert after_first.parent_asins[0] in {"blue", "green"}
    assert after_first.candidates[-1].parent_asin == "red"

    service.process_turn(user_key="u", user_message="I don't want blue.", turn_id="t2", now=2.0)
    after_second = rerank_from_memory(candidates, service)
    assert after_second.candidates[-1].parent_asin in {"red", "blue"}
    assert after_second.candidates[0].parent_asin == "green"
    # Red's avoidance was not forgotten: it still violates.
    red = next(c for c in after_second.candidates if c.parent_asin == "red")
    assert red.violation_count == 1


def test_replace_operation_removes_the_old_ranking_signal() -> None:
    """No stale black signal may survive an explicit replacement."""
    candidates = candidate_list(
        [("black", 1, 1, 9.0, "black"), ("blue", 2, 2, 5.0, "blue"), ("green", 3, 3, 1.0, "green")]
    )
    service = build_service()
    service.process_turn(user_key="u", user_message="I prefer black.", turn_id="t1", now=1.0)
    report_before = match_candidates(
        candidates=candidates, preferences=service.get_active_preferences("u")
    )
    black_before = next(c for c in report_before.candidates if c.parent_asin == "black")
    assert black_before.count(EvidenceStatus.MATCH) == 1

    service.process_turn(
        user_key="u", user_message="Actually, I prefer blue instead.", turn_id="t2", now=2.0
    )
    active = [e.value for e in service.get_active_preferences("u").active_entries]
    assert active == ["blue"]

    report_after = match_candidates(
        candidates=candidates, preferences=service.get_active_preferences("u")
    )
    black_after = next(c for c in report_after.candidates if c.parent_asin == "black")
    blue_after = next(c for c in report_after.candidates if c.parent_asin == "blue")
    assert black_after.count(EvidenceStatus.MATCH) == 0
    assert black_after.count(EvidenceStatus.UNKNOWN) == 1
    assert blue_after.count(EvidenceStatus.MATCH) == 1

    result = rerank_candidates(report=report_after)
    assert result.parent_asins[0] == "blue"
    # Black keeps no match-based promotion.
    black = next(c for c in result.candidates if c.parent_asin == "black")
    assert black.match_count == 0


def test_remove_operation_restores_the_original_order() -> None:
    """Required scenario: removal restores order when no other evidence distinguishes."""
    candidates = candidate_list([("red", 1, 1, 9.0, "red"), ("blue", 2, 2, 8.0, "blue")])
    service = build_service()
    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    reranked = rerank_from_memory(candidates, service)
    assert reranked.parent_asins == ("blue", "red")
    assert reranked.moved_count == 2

    service.process_turn(
        user_key="u", user_message="I don't care about color anymore.", turn_id="t2", now=2.0
    )
    restored = rerank_from_memory(candidates, service)
    assert restored.parent_asins == ("red", "blue")
    assert restored.reranked_ranks == (1, 2)
    assert restored.moved_count == 0
    assert restored.original_ranks == (1, 2)
    # No active preferences means no evidence records at all, and therefore no movement.
    assert all(c.evidence == () for c in restored.candidates)
    assert all(c.violation_count == 0 for c in restored.candidates)
    assert all(c.match_count == 0 for c in restored.candidates)
    assert all(c.unknown_count == 0 for c in restored.candidates)
    assert restored.diagnostics.active_preference_count == 0


def test_removed_directive_does_not_become_a_ranking_signal() -> None:
    candidates = candidate_list([("red", 1, 1, 9.0, "red")])
    service = build_service()
    service.process_turn(user_key="u", user_message="I don't want red.", turn_id="t1", now=1.0)
    service.process_turn(
        user_key="u", user_message="I don't care about color anymore.", turn_id="t2", now=2.0
    )
    result = rerank_from_memory(candidates, service)
    assert result.diagnostics.active_preference_count == 0
    assert result.candidates[0].violation_count == 0
    assert result.candidates[0].match_count == 0


def test_reranker_does_not_parse_memory_operations() -> None:
    """M10B sees only evidence statuses, never ADD/REPLACE/REMOVE."""
    import recommendation.reranking.schemas as schemas_module
    import recommendation.reranking.reranker as reranker_module

    for module in (reranker_module, schemas_module):
        source = _executable_source(module)
        for forbidden in (
            "PreferenceMode",
            "superseded",
            "superseded_by",
            "source_text",
            '"add"',
            '"replace"',
            '"remove"',
        ):
            assert forbidden not in source, f"{module.__name__} must not reference {forbidden}"


# --------------------------------------------------------------------------- #
# 43-44. Determinism and immutability
# --------------------------------------------------------------------------- #


def test_reranking_is_deterministic_across_repeated_runs() -> None:
    candidates = candidate_list(
        [("a", 1, 1, 9.0, "red"), ("b", 2, 2, 5.0, "blue"), ("c", 3, 3, 1.0, None)]
    )
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    report = match_candidates(candidates=candidates, preferences=preferences)
    baseline = rerank_candidates(report=report).model_dump_json()
    for _ in range(10):
        assert rerank_candidates(report=report).model_dump_json() == baseline


def test_reranking_does_not_depend_on_input_iteration_order() -> None:
    """The key carries original_rank, so arrival order cannot change the result."""
    candidates = candidate_list(
        [("a", 1, 1, 9.0, "red"), ("b", 2, 2, 5.0, "blue"), ("c", 3, 3, 1.0, "green")]
    )
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    shuffled = PreferenceEvidenceReport(
        candidates=tuple(reversed(report.candidates)),
        active_preference_count=report.active_preference_count,
    )
    assert [
        c.parent_asin for c in rerank_candidates(report=report).candidates
    ] == [
        c.parent_asin for c in rerank_candidates(report=shuffled).candidates
    ]


def test_reranking_does_not_mutate_its_input() -> None:
    candidates = candidate_list([("a", 1, 1, 9.0, "red"), ("b", 2, 2, 5.0, "blue")])
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    report_before = report.model_dump()
    candidates_before = [c.model_dump() for c in candidates]
    evidence_before = [c.evidence for c in report.candidates]

    rerank_candidates(report=report)

    assert report.model_dump() == report_before
    assert [c.model_dump() for c in candidates] == candidates_before
    assert [c.evidence for c in report.candidates] == evidence_before


def test_repeated_reranking_on_the_same_report_is_structurally_identical() -> None:
    candidates = candidate_list([("a", 1, 1, 9.0, "red"), ("b", 2, 2, 5.0, "blue")])
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    reranker = PreferenceReranker()
    assert reranker.rerank(report) == reranker.rerank(report)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def test_top_k_adherence_diagnostics_report_before_and_after() -> None:
    candidates = candidate_list(
        [("bad", 1, 1, 9.0, "red"), ("good", 2, 2, 1.0, "blue")]
    )
    preferences = make_snapshot(
        make_entry(memory_id="v", kind=PreferenceKind.COLOR, value="red",
                   polarity=PreferencePolarity.AVOID, logical_seq=1),
        make_entry(memory_id="m", kind=PreferenceKind.COLOR, value="blue",
                   polarity=PreferencePolarity.PREFER, logical_seq=2),
    )
    result = rerank_candidates(
        report=match_candidates(candidates=candidates, preferences=preferences),
        diagnostic_k=(1, 2),
    )
    top1 = next(row for row in result.diagnostics.top_k if row.k == 1)
    assert top1.violations_before == 1
    assert top1.violations_after == 0
    assert top1.matches_before == 0
    assert top1.matches_after == 1

    top2 = next(row for row in result.diagnostics.top_k if row.k == 2)
    # Aggregates over the whole list cannot change when only the order does.
    assert top2.violations_before == top2.violations_after == 1
    assert top2.matches_before == top2.matches_after == 1


def test_diagnostic_k_larger_than_the_candidate_count_is_skipped() -> None:
    candidates = candidate_list([("a", 1, 1, 5.0, "red")])
    result = rerank(candidates, avoid_red())
    assert [row.k for row in result.diagnostics.top_k] == [1]


def test_diagnostics_do_not_affect_order() -> None:
    candidates = candidate_list([("bad", 1, 1, 9.0, "red"), ("good", 2, 2, 1.0, "green")])
    report = match_candidates(candidates=candidates, preferences=avoid_red())
    default = rerank_candidates(report=report)
    no_diagnostics = rerank_candidates(report=report, diagnostic_k=())
    assert default.parent_asins == no_diagnostics.parent_asins
    assert no_diagnostics.diagnostics.top_k == ()


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #


def test_reranking_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("reranking must not touch the network")

    monkeypatch.setattr(socket, "socket", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)

    result = rerank(candidate_list([("a", 1, 1, 5.0, "red")]), avoid_red())
    assert result.candidate_count == 1


def test_reranking_module_imports_no_provider_sdk() -> None:
    import importlib

    for module in (
        "recommendation.reranking",
        "recommendation.reranking.reranker",
        "recommendation.reranking.schemas",
    ):
        importlib.import_module(module)
    for forbidden in ("openai", "anthropic", "google.generativeai", "cohere", "vertexai"):
        assert forbidden not in sys.modules


def test_reranking_report_is_serialisable_and_round_trips() -> None:
    import json

    result = rerank(candidate_list([("a", 1, 1, 5.0, "red")]), avoid_red())
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["reranked_rank"] == 1
    assert payload["sort_key"] == RERANK_SORT_KEY_DOC
