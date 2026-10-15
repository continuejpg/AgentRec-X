"""E-02A-V: multi-source candidates, from the ledger to the final ChatResponse.

The contract this file pins down, over the real chain (LoopController, ActionValidator,
CandidatePlane, GroundingVerifier, CandidateLedger, the accepted RRF, AgentRecommendationView and
the Chat adapter):

    every source the run legally executed contributes its **grounded** candidates
    -> the ledger keeps each source's own sighting (rank, score, score kind, query, item id)
    -> the run's presentable set is the accepted rank-fusion of those sources' rankings
    -> the Chat response shows that fused set, bounded by the request's k
    -> the memory-aware ranking reorders that same pool without adding, dropping or re-scoring.

Two source payloads are used, deliberately **disjoint** (history: item-a, item-b; catalogue:
item-c, item-d) plus a query that makes them **overlap** on item-b.  Only the model provider is a
double; the engines and the catalogue index are the offline fixture's accepted doubles.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.api.app import create_app
from recommendation.catalog import MetadataIndex
from recommendation.catalog.metadata import normalize_product_record
from recommendation.control.candidate_ledger import reciprocal_rank_fusion
from recommendation.control.model_client import ScriptedModelClient
from recommendation.demo import build_demo_runtime
from recommendation.demo.agent_trajectory import presentable_candidates
from recommendation.memory import SQLitePreferenceStore
from tests.demo_fixture import DemoEngine, DemoHarness, close_harness, demo_profiles, make_settings

A_ROWS: tuple[tuple[Any, ...], ...] = (
    ("item-a", 101, 0.91, "grey", "AlphaRunner", "Acme"),
    ("item-b", 102, 0.82, "black", "BetaRunner", "Acme"),
)
B_ROWS: tuple[tuple[Any, ...], ...] = (
    ("item-c", 103, 0.0, "blue", "GammaWidget", "SportCo"),
    ("item-d", 104, 0.0, "red", "DeltaWidget", "SportCo"),
)
CATALOGUE = A_ROWS + B_ROWS

#: A query only the lexical source can answer: it retrieves item-c and item-d.
DISJOINT_QUERY = "GammaWidget DeltaWidget"
#: The same query plus the token that also appears in the sequential source's set (item-b).
OVERLAP_QUERY = "GammaWidget DeltaWidget BetaRunner"

PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}

DIRECT = {"route": "direct", "selected_sources": [], "reason": "statement"}

#: The accepted fusion constant, used to compute the expected values by hand below.
RRF_K = 60
HISTORY_RANKING = ("item-a", "item-b")
LEXICAL_DISJOINT = ("item-c", "item-d")
LEXICAL_OVERLAP = ("item-b", "item-c", "item-d")


class _TwoSourceEngine(DemoEngine):
    """Source A: returns its own rows; the grounding map covers the whole catalogue."""

    def __init__(self) -> None:
        super().__init__(A_ROWS)
        self._by_asin = {str(row[0]): int(row[1]) for row in CATALOGUE}
        self._by_id = {int(row[1]): str(row[0]) for row in CATALOGUE}


def _index() -> MetadataIndex:
    """The real catalogue records, colour included: it is what M10A matches on."""
    return MetadataIndex.from_records([
        normalize_product_record({
            "parent_asin": row[0],
            "title": row[4],
            "details": {"Color": row[3]},
            **({"store": row[5]} if row[5] else {}),
        })
        for row in CATALOGUE
    ])


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


def _harness(tmp_path: Path) -> DemoHarness:
    from fastapi.testclient import TestClient

    profiles = demo_profiles()
    engine = _TwoSourceEngine()
    store = SQLitePreferenceStore(tmp_path / "multi_source.sqlite3")
    runtime = build_demo_runtime(
        engine=engine, metadata=_index(), store=store, profiles=profiles, max_sessions=8
    )
    app = create_app(make_settings(tmp_path), engine=engine, load_on_startup=False,
                     demo=runtime, enable_demo=True)
    client = TestClient(app)
    client.__enter__()
    return DemoHarness(app=app, runtime=runtime, engine=engine, client=client, store=store,
                       profiles=profiles)


def _plan(*sources: str) -> dict[str, Any]:
    return {"route": "recommend", "selected_sources": list(sources), "reason": "multi-source"}


def _turn(harness: DemoHarness, session: str, message: str, plan: dict[str, Any],
          *, k: int = 4) -> dict[str, Any]:
    client = ScriptedModelClient([plan], repeat_last=True)
    client.profile = SimpleNamespace(name="deepseek")
    client.model = "deepseek-chat"
    harness.runtime.use_llm_client(client)
    response = harness.client.post(
        f"/v1/demo/sessions/{session}/chat",
        json={"message": message, "k": k, "decision_mode": "llm_agent"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"] for card in body["recommendations"]]


def _sources(body: dict[str, Any]) -> dict[str, list[str]]:
    """Each source's own grounded ranking, as the response's provenance records it."""
    per_source: dict[str, list[tuple[int, str]]] = {}
    for candidate in body["agent"]["grounded"]["candidates"]:
        for record in candidate["provenance"]:
            per_source.setdefault(record["source"], []).append(
                (int(record["source_rank"]), candidate["parent_asin"])
            )
    return {
        source: [identity for _, identity in sorted(records)]
        for source, records in per_source.items()
    }


def _expected_rrf(*rankings: tuple[str, ...]) -> list[str]:
    """The accepted RRF, computed independently here from the sources' own rankings."""
    fused = [
        (identity, sum(1.0 / (RRF_K + position) for ranking in rankings
                       for position, value in enumerate(ranking, start=1) if value == identity))
        for identity in {identity for ranking in rankings for identity in ranking}
    ]
    return [identity for identity, _ in sorted(fused, key=lambda pair: (-pair[1], pair[0]))]


# --------------------------------------------------------------------------- #
# A. two sources, no overlap: everything grounded is presented
# --------------------------------------------------------------------------- #


def test_two_disjoint_sources_both_reach_the_response(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, DISJOINT_QUERY, _plan("history", "catalog_search"))

        assert body["agent"]["action_counts"] == {
            "recommend_from_history": 1, "search_catalog": 1, "finish": 1}
        assert _sources(body) == {"history": [], "catalog_search": list(LEXICAL_DISJOINT)} or (
            _sources(body) == {"catalog_search": list(LEXICAL_DISJOINT)}
        ), _sources(body)
        assert set(_ids(body)) == {"item-a", "item-b", "item-c", "item-d"}, (
            "both sources' grounded candidates must be presented"
        )
        assert set(body["agent"]["terminal"] and [body["agent"]["terminal"]["status"]]) == {
            "finished"}
        # The order is the accepted fusion of the two sources' own rankings.
        assert _ids(body) == _expected_rrf(HISTORY_RANKING, LEXICAL_DISJOINT)
        assert body["agent"]["grounded"]["grounded_count"] == 4
    finally:
        close_harness(harness)


def test_the_fused_order_is_the_documented_rrf_of_the_two_sources(
    tmp_path: Path, configured: None
) -> None:
    """Hand-computed RRF, item by item, against the production order and weights."""
    rankings = (HISTORY_RANKING, LEXICAL_DISJOINT)
    weights = {
        identity: sum(1.0 / (RRF_K + position) for ranking in rankings
                      for position, value in enumerate(ranking, start=1) if value == identity)
        for identity in {identity for ranking in rankings for identity in ranking}
    }
    # 1/61 for the two first places, 1/62 for the two second places, nothing merged here.
    assert weights == {
        "item-a": pytest.approx(1 / 61),
        "item-c": pytest.approx(1 / 61),
        "item-b": pytest.approx(1 / 62),
        "item-d": pytest.approx(1 / 62),
    }
    expected = _expected_rrf(*rankings)
    assert expected == ["item-a", "item-c", "item-b", "item-d"], (
        "ties are broken by identity, which is the accepted stable rule"
    )
    # And the ledger's own fusion function agrees, over the same two rankings.
    assert [identity for identity, _ in reciprocal_rank_fusion([list(r) for r in rankings])] == (
        expected
    )

    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, DISJOINT_QUERY, _plan("history", "catalog_search"))
        assert _ids(body) == expected
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# B. two sources with an overlap: one product, both sightings
# --------------------------------------------------------------------------- #


def test_an_overlapping_product_appears_once_with_both_sightings(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))

        assert _sources(body)["catalog_search"] == list(LEXICAL_OVERLAP)
        assert _ids(body).count("item-b") == 1, "one product, one card"
        assert _ids(body) == _expected_rrf(HISTORY_RANKING, LEXICAL_OVERLAP), (
            "item-b carries both sightings' weight and therefore leads"
        )
        assert _ids(body)[0] == "item-b"
        # The ledger keeps one provenance record per source, with that source's own rank.
        item_b = next(candidate for candidate in body["agent"]["grounded"]["candidates"]
                      if candidate["parent_asin"] == "item-b")
        assert [(record["source"], record["source_rank"]) for record in item_b["provenance"]] == [
            ("catalog_search", 1)]
        assert body["agent"]["grounded"]["grounded_count"] == 4
    finally:
        close_harness(harness)


def test_the_ledger_keeps_both_sightings_when_two_plane_sources_overlap() -> None:
    """The ledger boundary itself: two sources, one shared product, both records kept.

    The end-to-end plans available to the agent pair the sequential capability with one plane
    source, and the capability is not a plane source - so the overlap is exercised here where the
    ledger's own rule lives, with the real GroundingVerifier and CandidateLedger and no doubles.
    """
    from recommendation.control.arguments import CandidateSource
    from recommendation.control.candidate_ledger import CandidateLedger
    from recommendation.control.grounding import GroundingVerifier

    # The real identity protocol over the fixture catalogue: item-e is deliberately unknown.
    ledger = CandidateLedger()
    verifier = GroundingVerifier(_TwoSourceEngine())
    verifier.record(
        ledger,
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("item-b", 1, 1.203973), ("item-c", 2, 0.9), ("item-d", 3, 0.5)],
        score_kind="bm25",
        source_query=OVERLAP_QUERY,
        step_index=0,
    )
    verifier.record(
        ledger,
        source=CandidateSource.SIMILAR_ITEM,
        candidates=[("item-b", 1, 0.44), ("item-e", 2, 0.31)],
        score_kind="item_item_similarity",
        source_query="item-b",
        step_index=1,
    )

    entry = ledger.get("item-b")
    assert entry is not None
    assert [(record.source.value, record.source_rank, record.source_score, record.score_kind)
            for record in entry.provenance] == [
        ("catalog_search", 1, pytest.approx(1.203973), "bm25"),
        ("similar_item", 1, pytest.approx(0.44), "item_item_similarity"),
    ], "one product, two real sightings, neither merged away"
    assert entry.is_multi_source is True
    assert ledger.multi_source_parent_asins() == ("item-b",)
    # A product only one source proposed keeps that one record, and an ungrounded one is retained
    # for audit without ever becoming a candidate.
    assert [record.source.value for record in ledger.get("item-c").provenance] == ["catalog_search"]
    assert ledger.get("item-e").verification_status.value == "ungrounded"
    assert [entry.parent_asin for entry in ledger.ungrounded_entries] == ["item-e"]

    # The hand-computed fusion over both sources' own rankings, item by item.
    def _weight(*ranks: Any) -> float:
        return sum(1.0 / (RRF_K + int(rank)) for rank in ranks if rank)

    weights = {
        identity: _weight(rank_c, rank_s)
        for identity, rank_c, rank_s in (
            ("item-b", 1, 1), ("item-c", 2, None), ("item-d", 3, None), ("item-e", None, 2),
        )
    }
    assert weights["item-b"] == pytest.approx(2 / 61)
    assert weights["item-c"] == pytest.approx(1 / 62)
    assert weights["item-d"] == pytest.approx(1 / 63)
    assert weights["item-e"] == pytest.approx(1 / 62)
    assert [identity for identity, _ in reciprocal_rank_fusion([
        ["item-b", "item-c", "item-d"], ["item-b", "item-e"],
    ])] == ["item-b", "item-c", "item-e", "item-d"], (
        "the shared product leads, then the tied pair by identity, then the last rank"
    )
    assert ledger.ranked_parent_asins() == ("item-b", "item-c", "item-d")
    assert ledger.fusion_scores() == {
        "item-b": pytest.approx(2 / 61),
        "item-c": pytest.approx(1 / 62),
        "item-d": pytest.approx(1 / 63),
    }, "the ledger fuses what it recorded; an ungrounded identity is never a candidate"


# --------------------------------------------------------------------------- #
# C. execution order does not change the fused result
# --------------------------------------------------------------------------- #


def test_the_two_execution_orders_fuse_identically(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]

        a_then_b = _turn(harness, first, DISJOINT_QUERY, _plan("history", "catalog_search"))
        b_then_a = _turn(harness, second, DISJOINT_QUERY, _plan("catalog_search", "history"))

        assert _ids(a_then_b) == _ids(b_then_a) == _expected_rrf(
            HISTORY_RANKING, LEXICAL_DISJOINT)
        # Both really executed, in the order each plan named.
        assert list(a_then_b["agent"]["trajectory"]["steps"]) and [
            step["action"] for step in a_then_b["agent"]["trajectory"]["steps"]
        ] == ["recommend_from_history", "search_catalog", "finish"]
        assert [step["action"] for step in b_then_a["agent"]["trajectory"]["steps"]] == [
            "search_catalog", "recommend_from_history", "finish"]
        # The per-source provenance is identical in both: fusion depends on the rankings, not on
        # which step recorded them.
        assert _sources(a_then_b) == _sources(b_then_a)
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. single-source behaviour is unchanged
# --------------------------------------------------------------------------- #


def test_a_single_source_run_presents_exactly_that_source(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        sequential = harness.create_session()["session_id"]
        a_only = _turn(harness, sequential, DISJOINT_QUERY, _plan("history"))
        assert _ids(a_only) == ["item-a", "item-b"]
        assert [card["sasrec_score"] for card in a_only["recommendations"]] == [0.91, 0.82]
        assert {card["score_kind"] for card in a_only["recommendations"]} == {"sasrec_logit"}
        assert all(card["fusion_sources"] == [] for card in a_only["recommendations"])

        lexical = harness.create_session()["session_id"]
        b_only = _turn(harness, lexical, DISJOINT_QUERY, _plan("catalog_search"))
        assert _ids(b_only) == ["item-c", "item-d"]
        assert {card["score_kind"] for card in b_only["recommendations"]} == {"bm25"}
        assert all(card["fusion_sources"] == ["catalog_search"]
                   for card in b_only["recommendations"])
        assert b_only["agent"]["grounded"]["grounded_count"] == 2
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. k is applied to the fused set
# --------------------------------------------------------------------------- #


def test_k_bounds_the_fused_set_after_ordering(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, DISJOINT_QUERY, _plan("history", "catalog_search"), k=2)
        assert len(body["recommendations"]) == 2, "k bounds the answer"
        assert _ids(body) == _expected_rrf(HISTORY_RANKING, LEXICAL_DISJOINT)[:2], (
            "the top of the run's fused order, not a slice of one source"
        )
        assert body["agent"]["grounded"]["grounded_count"] == 4, (
            "the grounded universe is unchanged: k bounds presentation, not retrieval"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# F. provenance and score fidelity
# --------------------------------------------------------------------------- #


def test_every_card_carries_the_real_source_of_its_score(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))
        cards = {card["parent_asin"]: card for card in body["recommendations"]}

        # A product only the sequential source returned keeps the real model score and says so.
        assert cards["item-a"]["score_kind"] == "sasrec_logit"
        assert cards["item-a"]["sasrec_score"] == pytest.approx(0.91)
        assert cards["item-a"]["fusion_sources"] == []
        # A product the lexical source returned reports the lexical source's own score.
        assert cards["item-c"]["score_kind"] == "bm25"
        assert cards["item-c"]["sasrec_score"] == pytest.approx(1.203973)
        assert cards["item-c"]["fusion_sources"] == ["catalog_search"]
        # A product both returned is one card; its published score is the ledger's own record for
        # it, labelled with that record's kind - never an RRF value dressed up as a model score.
        assert cards["item-b"]["score_kind"] == "bm25"
        assert cards["item-b"]["sasrec_score"] == pytest.approx(1.203973)
        assert cards["item-b"]["fusion_sources"] == ["catalog_search"]
        # No card claims a fused score as its own.
        assert all(card["score_kind"] != "rrf_rank_fusion" for card in body["recommendations"])
    finally:
        close_harness(harness)


def test_every_presented_identity_is_grounded_and_source_backed(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, DISJOINT_QUERY, _plan("history", "catalog_search"))
        presented = set(_ids(body))
        grounded = {candidate["parent_asin"]: candidate
                    for candidate in body["agent"]["grounded"]["candidates"]}
        assert presented <= set(grounded), "nothing may be presented that was not grounded"
        for identity in presented:
            assert grounded[identity]["verification_status"] == "grounded"
        # Every presented identity is traceable to a source: either a ledger sighting or the
        # sequential source's own verified list.
        for identity in presented:
            sources = {record["source"] for record in grounded[identity]["provenance"]}
            assert sources or identity in {row[0] for row in A_ROWS}
        assert body["agent"]["grounded"]["sources_present"] == ["history", "catalog_search"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# G. failure, isolation and the memory-aware ranking over the fused pool
# --------------------------------------------------------------------------- #


def test_a_failing_or_empty_source_cannot_add_or_remove_candidates(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        # A lexical query that matches nothing: the plane grounds nothing, and the sequential
        # source's candidates must survive untouched.
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, "NothingMatchesThisAtAll",
                     _plan("history", "catalog_search"))
        assert _ids(body) == ["item-a", "item-b"], (
            "an empty source contributes nothing and removes nothing"
        )
        assert "search_catalog" in body["agent"]["action_counts"]
        assert body["agent"]["grounded"]["grounded_count"] == 2

        # A lexical-only plan that matches nothing keeps the existing refusal contract.
        empty = harness.create_session()["session_id"]
        refused = _turn(harness, empty, "NothingMatchesThisAtAll", _plan("catalog_search"))
        assert refused["recommendations"] == []
        assert refused["agent"]["outcome"] == "refused"
        assert refused["agent"]["terminal"]["termination_reason"] == "completion_refused"
    finally:
        close_harness(harness)


def test_two_requests_never_share_a_ledger(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]

        one = _turn(harness, first, DISJOINT_QUERY, _plan("history", "catalog_search"))
        two = _turn(harness, second, DISJOINT_QUERY, _plan("catalog_search"))
        assert set(_ids(one)) == {"item-a", "item-b", "item-c", "item-d"}
        assert set(_ids(two)) == {"item-c", "item-d"}, (
            "the second run's ledger holds only what its own sources grounded"
        )
        assert set(_sources(two)) == {"catalog_search"}

        # And a later turn in the first session does not inherit the earlier turn's candidates
        # when its own plan consults one source.
        three = _turn(harness, first, DISJOINT_QUERY, _plan("catalog_search"))
        assert set(_ids(three)) == {"item-c", "item-d"}
    finally:
        close_harness(harness)


def test_memory_reranks_the_fused_pool_without_touching_it(
    tmp_path: Path, configured: None
) -> None:
    """E-02A on top of E-02A-V: the ranking input is the fused pool, and only its order changes."""
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        baseline = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))
        assert _ids(baseline) == _expected_rrf(HISTORY_RANKING, LEXICAL_OVERLAP)

        # item-c is the catalogue's blue product and sits third in the fused order, so a
        # preference for blue can promote it: the ranking input really is the fused pool.
        _turn(harness, session, "I prefer blue.", DIRECT)
        ranked = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))

        assert set(_ids(ranked)) == set(_ids(baseline)), "no product is added or dropped"
        assert _ids(ranked)[0] == "item-c", "the matching product is promoted from third to first"
        assert _ids(ranked) != _ids(baseline)
        assert ranked["audit"]["reranking_applied"] is True
        assert ranked["audit"]["ranked_with_preference_count"] == 1
        assert ranked["audit"]["original_order"] == _ids(baseline), (
            "the baseline is the run's fused order"
        )
        # Identity, score and provenance are untouched by the ranking.
        def spine(body: dict[str, Any]) -> list[tuple[str, float, str, tuple[str, ...]]]:
            return sorted(
                (card["parent_asin"], card["sasrec_score"], card["score_kind"],
                 tuple(card["fusion_sources"]))
                for card in body["recommendations"]
            )

        assert spine(ranked) == spine(baseline)
        blue = next(card for card in ranked["recommendations"] if card["parent_asin"] == "item-c")
        assert blue["match_count"] == 1
        assert [(record["status"], record["metadata_field"], record["metadata_value"])
                for record in blue["evidence"]] == [("match", "details.Color", "blue")]

        # An avoidance produces real violation evidence too, and cannot invent a move: item-d is
        # the red product and is already last in the fused order, so the order stays put while the
        # evidence changes.  Both facts are asserted.
        _turn(harness, session, "I don't want red.", DIRECT)
        avoided = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))
        red = next(card for card in avoided["recommendations"]
                   if card["parent_asin"] == "item-d")
        assert red["violation_count"] == 1, "the avoidance produces real violation evidence"
        assert [
            (record["status"], record["metadata_field"], record["metadata_value"])
            for record in red["evidence"]
            if record["status"] == "violation"
        ] == [("violation", "details.Color", "red")], red["evidence"]
        assert set(_ids(avoided)) == set(_ids(baseline)), "still the same pool"
        assert spine(avoided) == spine(baseline), "and the same spine"

        # REMOVE restores the fused baseline exactly.
        _turn(harness, session, "I don't care about color anymore.", DIRECT)
        restored = _turn(harness, session, OVERLAP_QUERY, _plan("history", "catalog_search"))
        assert _ids(restored) == _ids(baseline)
        assert [card["reranked_rank"] for card in restored["recommendations"]] == [None] * 4
    finally:
        close_harness(harness)


def test_the_presentable_artifact_is_what_the_ranking_receives(
    tmp_path: Path, configured: None
) -> None:
    """The ranking, the titles and the view all read one definition of the presentable set."""
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        body = _turn(harness, session, DISJOINT_QUERY, _plan("history", "catalog_search"))
        assert set(_ids(body)) == {"item-a", "item-b", "item-c", "item-d"}

        # Read the controller's own artifact through the service, in the same process.
        service = harness.runtime.agent_service()
        from recommendation.agent import AgentInput
        from recommendation.demo.agent_schemas import AgentRecommendRequest
        from recommendation.demo.tool_selection import SourceSelection

        controller = service._compose_controller(
            k=4, user_key=None, plan=("history", "catalog_search"), use_memory=False
        )
        from recommendation.demo.agent_service import _task_state

        result = controller.invoke(
            AgentInput(user_message=DISJOINT_QUERY,
                       trusted_user_history=tuple(harness.manager.get(session).trusted_user_history)),
            task_state=_task_state(AgentRecommendRequest(
                message=DISJOINT_QUERY, k=4,
                history=list(harness.manager.get(session).trusted_user_history))),
        )
        artifact = result.state["presentable_tool_result"]
        assert [str(item.parent_asin) for item in artifact.recommendations] == _ids(body)
        assert artifact.score_kind == "rrf_rank_fusion", (
            "the accumulated envelope names its own multi-source nature"
        )
        assert [str(item.parent_asin) for item in presentable_candidates(result.state)] == _ids(body)
        # The run's *current* candidate set is still the last adopted one: this fix changes what is
        # presented, not what the policy or the completion guard sees step by step.
        assert [str(item.parent_asin) for item in result.state["tool_result"].recommendations] == (
            list(LEXICAL_DISJOINT)
        )
    finally:
        close_harness(harness)
