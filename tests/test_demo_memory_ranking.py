"""E-02A: memory-aware ranking of an agent run's presented candidates.

What is under test, in the three layers the round distinguishes:

1. **the adapter** (:mod:`recommendation.demo.memory_ranking`) - it must hand M10A exactly the
   run's presented candidates with their own scores, never a fusion score in the model-score slot,
   never a dropped or invented candidate, and it must report missing metadata as missing;
2. **M10A/M10B themselves** - over the real M8 metadata of the fixture catalogue, a PREFER must
   produce a real ``match``, an AVOID a real ``violation``, and neither may be invented for a
   candidate whose attribute is simply different or absent;
3. **the whole vertical slice over the real Chat API** - the same legal candidate pool, a different
   ACTIVE memory, a different final order, with identity, score, score kind and provenance
   unchanged, and with the pre-E-02A baseline reproducible both by removing the preference and by
   switching the capability off.

The only double is the model provider (the repository's ``ScriptedModelClient``) and the accepted
engine/metadata doubles of the offline demo fixture; the controller, the plane, the ledger, the
grounding verifier, the enricher, M10A, M10B and M9 are the real components.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.catalog import MetadataIndex
from recommendation.catalog.metadata import normalize_product_record
from recommendation.control.candidate_ledger import reciprocal_rank_fusion
from recommendation.control.model_client import ScriptedModelClient
from recommendation.demo.agent_service import memory_ranking_enabled
from recommendation.demo.memory_ranking import MemoryRankingError, rank_with_memory
from recommendation.memory.schemas import PreferenceMemoryEntry, PreferenceMemorySnapshot
from recommendation.preference_matching import PreferenceCandidateMatcher
from recommendation.rag import ProductEnricher
from recommendation.reranking import PreferenceReranker
from recommendation.tools.schemas import RecommendationToolResult, ToolRecommendation
from tests.agent_reranking_fixture import CANDIDATE_ROWS, build_index
from tests.demo_fixture import build_harness, close_harness

PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}

#: The fixture engine's own order, used by the adapter tests and by a history-only run.
ADAPTER_POOL = ("cand-red", "cand-blue", "cand-black", "cand-green")

#: The run's fused baseline for :data:`POOL_QUERY`, over the two sources this plan executes.
#:
#: E-02A-V corrected what "the fused baseline" is: both executed sources contribute, and the order
#: is the accepted reciprocal-rank fusion of their own rankings - the sequential source's
#: ``(red, blue, black, green)`` and the lexical source's ``(blue, green, red, black)`` - not the
#: most recent source's list.  Computed by hand: blue 1/61+1/62, red 1/61+1/63, green 1/62+1/64,
#: black 1/63+1/64.
FUSED_BASELINE = ("cand-blue", "cand-red", "cand-green", "cand-black")

#: ``prefer black`` + ``avoid red`` over :data:`FUSED_BASELINE`, by the accepted M10B key.
RERANKED_POOL = ("cand-black", "cand-blue", "cand-green", "cand-red")

#: A query whose tokens are exactly the four catalogue titles, so one lexical action retrieves the
#: whole pool.  The titles are single camel-case tokens, and the source matches tokens, not
#: substrings - which is why a partial word like "Widget" retrieves nothing at all.
POOL_QUERY = "RedWidget BlueWidget BlackWidget GreenWidget"

PREFER_BLACK = "I prefer black."
PREFER_GREEN = "I prefer green."
AVOID_RED = "I don't want red."
RETRACT_COLOUR = "I don't care about color anymore."


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _entry(index: int, kind: str, polarity: str, value: str) -> PreferenceMemoryEntry:
    return PreferenceMemoryEntry(
        memory_id=f"memory-{index}",
        user_key="session-key",
        kind=kind,
        polarity=polarity,
        value=value,
        source_text=f"I said {value}",
        source_turn_id=f"turn-{index}",
        logical_seq=index,
        created_at=1000.0 + index,
    )


def _snapshot(*values: tuple[str, str, str]) -> PreferenceMemorySnapshot:
    """A real M9 snapshot holding exactly the given ACTIVE (kind, polarity, value) triples."""
    return PreferenceMemorySnapshot(
        user_key="session-key",
        entries=tuple(
            _entry(index, kind, polarity, value)
            for index, (kind, polarity, value) in enumerate(values, start=1)
        ),
    )


def _candidates(identities: Any = ADAPTER_POOL, *, score: float = 0.0) -> tuple[Any, ...]:
    return tuple(
        ToolRecommendation(rank=position, item_id=900 + position, parent_asin=parent_asin,
                           score=score)
        for position, parent_asin in enumerate(identities, start=1)
    )


def _enricher(index: MetadataIndex | None = None) -> ProductEnricher:
    return ProductEnricher(index if index is not None else build_index())


def _rank(
    preferences: Any,
    *,
    candidates: Any = None,
    enricher: Any = None,
    tool_result: Any = None,
    query: str = "",
) -> Any:
    return rank_with_memory(
        candidates=_candidates() if candidates is None else candidates,
        tool_result=tool_result,
        preferences=preferences,
        enricher=_enricher() if enricher is None else enricher,
        matcher=PreferenceCandidateMatcher(),
        reranker=PreferenceReranker(),
        query=query,
    )


class _RecordingMatcher:
    """The real matcher, with the candidates it was handed recorded for inspection."""

    def __init__(self) -> None:
        self._real = PreferenceCandidateMatcher()
        self.seen: list[Any] = []

    def match(self, *, candidates: Any, preferences: Any) -> Any:
        self.seen = list(candidates)
        return self._real.match(candidates=candidates, preferences=preferences)


# --------------------------------------------------------------------------- #
# 1. the adapter: what M10A is handed, and what it refuses to do
# --------------------------------------------------------------------------- #


def test_no_active_preference_means_no_ranking_and_no_enrichment() -> None:
    calls: list[Any] = []

    class _SpyEnricher:
        def enrich(self, result: Any, query: str = "") -> Any:
            calls.append(result)
            raise AssertionError("an unranked run must not pay for enrichment")

    assert _rank(None, enricher=_SpyEnricher()) is None
    assert _rank(_snapshot(), enricher=_SpyEnricher()) is None, (
        "a snapshot with no ACTIVE entry is no memory for ranking purposes"
    )
    assert calls == []


def test_the_model_score_slot_never_carries_a_fusion_score() -> None:
    """A plane candidate's 0.0 placeholder stays 0.0: the BM25 value is the ledger's, not M10A's."""
    matcher = _RecordingMatcher()
    candidates = _candidates(("cand-red",), score=0.0)
    ranking = rank_with_memory(
        candidates=candidates,
        preferences=_snapshot(("color", "avoid", "red")),
        enricher=_enricher(),
        matcher=matcher,
        reranker=PreferenceReranker(),
        query="Widget",
    )
    assert ranking is not None
    assert [item.score for item in matcher.seen] == [0.0]
    assert [item.parent_asin for item in matcher.seen] == ["cand-red"]
    # The violation is real (the fixture's cand-red really is red), and it came from the metadata.
    assert ranking.candidates[0].violation_count == 1
    assert ranking.candidates[0].evidence[0].metadata_field == "details.Color"


def test_a_reranker_that_changes_the_candidate_set_is_an_error() -> None:
    """A drop or an addition must fail the turn, never silently shrink the answer."""

    class _Dropping:
        def rerank(self, report: Any) -> Any:
            return report.model_copy(update={"candidates": report.candidates[:-1]})

    with pytest.raises(MemoryRankingError):
        rank_with_memory(
            candidates=_candidates(),
            preferences=_snapshot(("color", "prefer", "black")),
            enricher=_enricher(),
            matcher=PreferenceCandidateMatcher(),
            reranker=_Dropping(),
        )


def test_missing_metadata_is_unknown_and_moves_nothing() -> None:
    """An identity the catalogue does not know is 'unknown', never a match and never a violation."""
    partial = MetadataIndex.from_records([
        normalize_product_record({"parent_asin": "cand-black", "title": "BlackWidget",
                                  "details": {"Color": "black"}})
    ])
    ranking = _rank(_snapshot(("color", "prefer", "black")), enricher=_enricher(partial))

    assert ranking is not None
    evidence = {candidate.parent_asin: candidate for candidate in ranking.candidates}
    known = evidence["cand-black"]
    assert (known.match_count, known.violation_count, known.unknown_count) == (1, 0, 0)
    for identity in ("cand-red", "cand-blue", "cand-green"):
        record = evidence[identity]
        assert (record.match_count, record.violation_count, record.unknown_count) == (0, 0, 1), (
            f"{identity} has no catalogue record, so its colour is unknown - not a violation"
        )
    # With one match and no violations anywhere, only the known candidate may move.
    assert ranking.ranked_order == ("cand-black", "cand-red", "cand-blue", "cand-green")
    assert ranking.baseline_order == ADAPTER_POOL


def test_evidence_that_decides_nothing_preserves_the_baseline_order() -> None:
    ranking = _rank(_snapshot(("color", "prefer", "chartreuse")))
    assert ranking is not None
    assert ranking.ranked_order == ranking.baseline_order == ADAPTER_POOL
    assert ranking.changed_order is False
    assert ranking.moved_count == 0


# --------------------------------------------------------------------------- #
# 2. M10A/M10B behaviour over the real fixture catalogue
# --------------------------------------------------------------------------- #


def test_a_preference_promotes_the_candidate_whose_metadata_matches() -> None:
    ranking = _rank(_snapshot(("color", "prefer", "black")))
    assert ranking is not None
    by_identity = ranking.by_identity()

    assert by_identity["cand-black"].match_count == 1
    assert by_identity["cand-black"].reranked_rank == 1
    assert by_identity["cand-black"].rank == 3, "the baseline position is kept, not rewritten"
    # The other colours are *different*, not violating: no negative inference is invented.
    for identity in ("cand-red", "cand-blue", "cand-green"):
        assert by_identity[identity].violation_count == 0
        assert by_identity[identity].unknown_count == 1
    assert ranking.ranked_order == ("cand-black", "cand-red", "cand-blue", "cand-green")


def test_an_avoidance_demotes_the_candidate_that_violates_it() -> None:
    ranking = _rank(_snapshot(("color", "avoid", "red")))
    assert ranking is not None
    by_identity = ranking.by_identity()

    assert by_identity["cand-red"].violation_count == 1
    assert by_identity["cand-red"].reranked_rank == 4
    for identity in ("cand-blue", "cand-black", "cand-green"):
        assert by_identity[identity].violation_count == 0
    # Ties are broken by the baseline (fused) order, which is what keeps this deterministic.
    assert ranking.ranked_order == ("cand-blue", "cand-black", "cand-green", "cand-red")


def test_two_preferences_order_the_pool_by_the_canonical_key() -> None:
    ranking = _rank(_snapshot(("color", "prefer", "black"), ("color", "avoid", "red")))
    assert ranking is not None
    assert ranking.active_preference_count == 2
    assert ranking.ranked_order == ("cand-black", "cand-blue", "cand-green", "cand-red")
    assert ranking.changed_order is True
    assert {candidate.parent_asin for candidate in ranking.candidates} == set(ADAPTER_POOL), (
        "reranking is a reordering: no candidate is added or dropped"
    )
    assert sorted(candidate.reranked_rank for candidate in ranking.candidates) == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# 3. the real controller, plane and ledger, over the Chat API
# --------------------------------------------------------------------------- #


def _client(*answers: Any) -> Any:
    client = ScriptedModelClient(list(answers), repeat_last=True)
    client.profile = SimpleNamespace(name="deepseek")
    client.model = "deepseek-chat"
    return client


DIRECT = {"route": "direct", "selected_sources": [], "reason": "statement"}
#: The sequential source first, then the lexical one: both really execute (the action counts below
#: assert it) and the lexical plane's grounded set is the one that gets presented, which is the
#: accepted adoption rule - the run's presented pool is the last candidate-producing step's set,
#: while the ledger keeps every source's records.
TWO_SOURCES = {"route": "recommend", "selected_sources": ["history", "catalog_search"],
               "reason": "two sources"}


def _turn(harness: Any, session_id: str, message: str, *answers: Any, k: int = 4) -> dict[str, Any]:
    harness.runtime.use_llm_client(_client(*answers))
    response = harness.client.post(
        f"/v1/demo/sessions/{session_id}/chat",
        json={"message": message, "k": k, "decision_mode": "llm_agent"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _identities(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"] for card in body["recommendations"]]


def _baseline_identities(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"]
            for card in sorted(body["recommendations"], key=lambda card: card["original_rank"])]


def _spine(body: dict[str, Any]) -> list[tuple[str, float, str, tuple[str, ...]]]:
    """The facts memory must never touch: identity, score, score kind, provenance."""
    return sorted(
        (card["parent_asin"], card["sasrec_score"], card["score_kind"],
         tuple(card["fusion_sources"]))
        for card in body["recommendations"]
    )


def _provenance(body: dict[str, Any]) -> list[tuple[str, str, int, float, str, str]]:
    return sorted(
        (candidate["parent_asin"], record["source"], record["source_rank"],
         record["source_score"], record["score_kind"], record["source_query"])
        for candidate in body["agent"]["grounded"]["candidates"]
        for record in candidate["provenance"]
    )


def test_two_real_sources_run_and_memory_reorders_their_pool(
    tmp_path: Path, configured: None
) -> None:
    """Two sources execute for real, and the ranking reorders that pool without touching it."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, PREFER_BLACK, DIRECT)
        _turn(harness, session, AVOID_RED, DIRECT)

        ranked = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        agent = ranked["agent"]

        # -- the run really consulted both planned sources ------------------------------- #
        assert agent["planned_sources"] == ["history", "catalog_search"]
        assert agent["action_counts"] == {"search_catalog": 1, "recommend_from_history": 1,
                                         "finish": 1}
        # Both executed sources are reported, and every identity both of them proposed is a real
        # multi-source candidate.  (Before E-02A-V this read ["catalog_search"] / 0, because the
        # response only ever contained the most recent source's candidates: the assertion recorded
        # the bug rather than the contract, and is replaced by this stronger one.)
        assert agent["grounded"]["sources_present"] == ["history", "catalog_search"]
        assert agent["grounded"]["multi_source_count"] == 0, (
            "the ledger's multi-source count counts *plane* sightings of one identity: the "
            "sequential source runs through the accepted capability, which is not a plane source, "
            "so it has no ledger record even though its candidates are in the pool and fused.  "
            "sources_present below names both sources, which is where the run's real source set is "
            "reported (see docs/V2_E02A_V_REPORT.md, boundary 2)."
        )
        assert agent["grounded"]["grounded_count"] == 4
        provenance = _provenance(ranked)
        assert len(provenance) == 4, "every presented candidate carries its ledger sighting"
        assert {record[1] for record in provenance} == {"catalog_search"}, (
            "only the plane records ledger provenance; the accepted capability is not a plane "
            "source, which is why the sequential source's own verified list is reported separately"
        )
        assert {record[4] for record in provenance} == {"bm25"}
        assert {record[5] for record in provenance} == {POOL_QUERY}, (
            "the source was asked for this turn's request, and nothing else"
        )

        # -- the same pool, reordered by real evidence ----------------------------------- #
        assert _identities(ranked) == list(RERANKED_POOL)
        assert _baseline_identities(ranked) == list(FUSED_BASELINE), (
            "the baseline (fused) order is preserved on every card, so the two orders differ"
        )
        cards = {card["parent_asin"]: card for card in ranked["recommendations"]}
        assert cards["cand-black"]["match_count"] == 1
        assert cards["cand-black"]["violation_count"] == 0
        assert cards["cand-black"]["unknown_count"] == 1
        assert [
            (record["status"], record["metadata_field"], record["metadata_value"])
            for record in cards["cand-black"]["evidence"]
        ] == [("match", "details.Color", "black"), ("unknown", "details.Color", "black")]
        assert cards["cand-red"]["violation_count"] == 1
        assert cards["cand-blue"]["unknown_count"] == 2
        assert ranked["audit"]["reranking_applied"] is True
        assert ranked["audit"]["moved_count"] == 3
        assert ranked["audit"]["ranked_with_preference_count"] == 2
        assert ranked["audit"]["original_order"] == list(FUSED_BASELINE)
        assert ranked["audit"]["reranked_order"] == _identities(ranked)
    finally:
        close_harness(harness)


def _fusion(body: dict[str, Any]) -> list[tuple[str, float]]:
    """The ledger's own RRF over the response's provenance, per source, recomputed publicly."""
    per_source: dict[str, list[tuple[int, str]]] = {}
    for candidate in body["agent"]["grounded"]["candidates"]:
        for record in candidate["provenance"]:
            per_source.setdefault(record["source"], []).append(
                (int(record["source_rank"]), candidate["parent_asin"])
            )
    rankings = [
        [identity for _, identity in sorted(records)] for records in per_source.values()
    ]
    return reciprocal_rank_fusion(rankings)


def test_memory_never_changes_identity_score_kind_or_provenance(
    tmp_path: Path, configured: None
) -> None:
    """The controlled comparison: same plan and message, ranking on vs off."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, PREFER_BLACK, DIRECT)
        _turn(harness, session, AVOID_RED, DIRECT)

        on = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        harness.runtime.agent_service()._memory_ranking_enabled = False
        off = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        harness.runtime.agent_service()._memory_ranking_enabled = True

        assert _identities(on) != _identities(off), "memory must be able to change the order"
        assert _identities(off) == list(FUSED_BASELINE), "with the capability off, the fused order"
        assert _spine(on) == _spine(off), "identity, score, score kind and provenance are untouched"
        assert _provenance(on) == _provenance(off), "the ledger records are untouched"
        assert _baseline_identities(on) == _identities(off), (
            "the ranked run's baseline order is the unranked run's order: same pool, same fusion"
        )
        # The fusion is a pure function of the provenance, so unchanged provenance means unchanged
        # RRF values - checked directly with the ledger's own public fusion function.
        assert _fusion(on) == _fusion(off), "the RRF values must be identical in both runs"
        assert [card["reranked_rank"] for card in off["recommendations"]] == [None] * 4
        assert off["audit"]["reranking_applied"] is False
        assert off["audit"]["moved_count"] == 0
        assert off["audit"]["ranked_with_preference_count"] == 0
    finally:
        close_harness(harness)


def test_without_memory_the_order_is_the_fused_order_byte_for_byte(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert _identities(body) == list(FUSED_BASELINE)
        assert [card["reranked_rank"] for card in body["recommendations"]] == [None] * 4
        assert all(card["evidence"] == [] for card in body["recommendations"])
        assert all(card["movement_summary"] is None for card in body["recommendations"]), (
            "a card that was not reranked carries no movement summary"
        )
        # The optional blocks added by later milestones (E-03A replanning, E-03C product feedback)
        # are asserted separately, so every pre-existing field is still compared byte for byte.
        audit = {
            key: value
            for key, value in body["audit"].items()
            if key not in ("replan", "feedback")
        }
        assert audit == {
            "reranking_applied": False,
            "candidate_count": 4,
            "moved_count": 0,
            "original_order": list(FUSED_BASELINE),
            "reranked_order": list(FUSED_BASELINE),
            "ranked_with_preference_count": 0,
            "ranked_with_preferences": [],
            # The optional E-02B block: null here, because the explanation capability is off in
            # this deployment.  Every field above is byte-for-byte what it was before E-02B.
            "rag": None,
        }
        # The optional E-03A decision block, on the same turn: a decision, and a correct one - the
        # fused answer already presents the requested k, so nothing further is proposed.
        replan = body["audit"]["replan"]
        assert (replan["continue_recommended"], replan["reason_code"]) == (
            False, "candidates_sufficient")
        assert replan["decision_source"] == "deterministic_rules"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 4. the Chat session: ADD, supersede, REMOVE, and the switch
# --------------------------------------------------------------------------- #


def test_add_supersede_and_remove_change_only_the_ranking(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline = _turn(harness, session, POOL_QUERY, TWO_SOURCES)

        # ADD
        added = _turn(harness, session, PREFER_BLACK, DIRECT)
        assert added["memory_update"]["changed"] is True
        promoted = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert _identities(promoted) == ["cand-black", "cand-blue", "cand-red", "cand-green"]
        assert _spine(promoted) == _spine(baseline)

        # SUPERSEDE: the new value is the ranking input, and the old one is gone.
        replaced = _turn(harness, session, "Actually, I prefer green instead.", DIRECT)
        assert [item["value"] for item in replaced["memory_update"]["superseded"]] == ["black"]
        superseded = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert _identities(superseded) == ["cand-green", "cand-blue", "cand-red", "cand-black"]
        green = {card["parent_asin"]: card for card in superseded["recommendations"]}["cand-green"]
        assert green["match_count"] == 1
        assert all(
            record["value"] == "green"
            for card in superseded["recommendations"]
            for record in card["evidence"]
        ), "a superseded value must not appear in any evidence record"
        assert _spine(superseded) == _spine(baseline)

        # REMOVE: back to the fused order, with the ranking fields cleared.
        removed = _turn(harness, session, RETRACT_COLOUR, DIRECT)
        assert removed["memory_update"]["removed"], removed["memory_update"]
        restored = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert _identities(restored) == _identities(baseline)
        assert [card["reranked_rank"] for card in restored["recommendations"]] == [None] * 4
        assert restored["audit"]["reranking_applied"] is False
        assert restored["active_preferences"] == []
        assert _spine(restored) == _spine(baseline)
    finally:
        close_harness(harness)


def test_a_turn_ranks_with_the_memory_it_read_before_it_wrote(
    tmp_path: Path, configured: None
) -> None:
    """A recommendation turn that also states a preference ranks with the pre-write set."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, PREFER_BLACK, DIRECT)

        # This turn states *green* and asks for products in the same message.
        body = _turn(
            harness, session, f"{POOL_QUERY}, and I prefer green.", TWO_SOURCES
        )

        assert body["memory_update"]["changed"] is True, "the statement is still persisted"
        after = {item["value"] for item in body["active_preferences"]}
        assert after != {"black"}, "the panel shows the post-write state, which changed"
        assert "green" in after
        cards = {card["parent_asin"]: card for card in body["recommendations"]}
        assert set(cards) == set(FUSED_BASELINE), "the pool is the same four candidates"
        assert cards["cand-black"]["reranked_rank"] < cards["cand-black"]["original_rank"], (
            "black was promoted: the ranking ran on the pre-write memory, because green cannot "
            "affect the turn that stated it"
        )
        assert cards["cand-green"]["reranked_rank"] == cards["cand-green"]["original_rank"] or (
            cards["cand-green"]["match_count"] == 0
        ), "green was not ranked as a match on the turn that introduced it"
        assert cards["cand-green"]["match_count"] == 0
        assert [item["value"] for item in body["audit"]["ranked_with_preferences"]] == ["black"], (
            "the audit names the set the ordering actually used, not the new one"
        )
        assert body["audit"]["ranked_with_preference_count"] == 1

        # The next turn ranks with both.
        following = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert {item["value"] for item in following["audit"]["ranked_with_preferences"]} == {
            "black", "green"}
    finally:
        close_harness(harness)


def test_the_deployment_switch_pins_the_pre_e02a_baseline(
    tmp_path: Path, configured: None
) -> None:
    # The switch is a plain reader: on unless explicitly disabled, and a typo keeps the default.
    assert memory_ranking_enabled() is True
    assert memory_ranking_enabled("0") is False
    assert memory_ranking_enabled("off") is False
    assert memory_ranking_enabled("FALSE") is False
    assert memory_ranking_enabled("yes") is True
    assert memory_ranking_enabled("") is True

    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, PREFER_BLACK, DIRECT)

        ranked = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        # Only ``prefer black`` is active here, so black leads and the rest keep the fused order.
        assert _identities(ranked) == ["cand-black", "cand-blue", "cand-red", "cand-green"]

        # The same flag the environment switch sets, so the turn below is the pinned baseline.
        harness.runtime.agent_service()._memory_ranking_enabled = False
        pinned = _turn(harness, session, POOL_QUERY, TWO_SOURCES)
        assert _identities(pinned) == list(FUSED_BASELINE)
        assert pinned["audit"]["reranking_applied"] is False
        assert pinned["active_preferences"], (
            "the preference is still remembered - it is simply not ranked with"
        )
    finally:
        close_harness(harness)


def test_the_agent_response_carries_the_evidence_and_the_ranking_fields(
    tmp_path: Path, configured: None
) -> None:
    """The service seam: the new fields are on the agent view itself, not only on the chat card."""
    from recommendation.demo.agent_schemas import AgentRecommendRequest
    from recommendation.demo.tool_selection import SourceSelection

    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        user_key = harness.manager.get(session).user_key
        snapshot = harness.runtime.memory_service.process_turn(
            user_key=user_key, user_message=PREFER_BLACK, turn_id=f"{session}:000001"
        ).active
        assert snapshot.active_entries

        selection = SourceSelection(route="recommend", selected_sources=("history",))
        payload = AgentRecommendRequest(
            message="anything", k=4,
            history=list(harness.manager.get(session).trusted_user_history),
        )
        response = harness.runtime.agent_service().run_selection(
            selection, payload, preference_snapshot=snapshot
        )
        first = response.recommendations[0]
        assert first.parent_asin == "cand-black"
        assert first.rank == 3, "the baseline position is preserved"
        assert first.reranked_rank == 1, "the memory-aware position is reported separately"
        assert (first.match_count, first.violation_count, first.unknown_count) == (1, 0, 0)
        assert [record.preference_value for record in first.evidence] == ["black"]
        assert first.rerank_reason, "M10B's own reason travels with the candidate"
        assert first.score_kind == "sasrec_logit" and first.score == pytest.approx(0.7), (
            "the model score is the real one and is untouched by the ranking"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 5. the capability is wired to llm_agent only
# --------------------------------------------------------------------------- #


def test_the_legacy_agent_endpoint_is_not_reranked(tmp_path: Path, configured: None) -> None:
    """The request-declared path keeps its response order: E-02A is wired to llm_agent only.

    A preference really is stored for this session's namespace, and the legacy request carries that
    same ``user_key`` - but it supplies no snapshot to the ranking step, so no card can claim a
    memory-aware position.  Whatever the accepted capability does internally is untouched and, as
    delivered, is not projected into this response.
    """
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, PREFER_BLACK, DIRECT)
        user_key = harness.manager.get(session).user_key
        assert harness.active_values(session) == {("color", "prefer", "black")}

        response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": "Widget", "k": 4, "user_key": user_key, "sources": ["history"],
                  "history": list(harness.manager.get(session).trusted_user_history)},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [card["parent_asin"] for card in body["recommendations"]] == list(ADAPTER_POOL)
        assert [card.get("reranked_rank") for card in body["recommendations"]] == [None] * 4
        assert all(card.get("evidence", []) == [] for card in body["recommendations"])
    finally:
        close_harness(harness)
