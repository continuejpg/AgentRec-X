"""Milestone 10D tests: preference reranking integrated into the agent route.

Fully offline.  A fixed engine, a synthetic metadata index, a synthetic preference
snapshot and the **real** M10A matcher / M10B reranker, so the integration contract is
tested without a checkpoint, the catalogue artifact or any provider API.

What must hold, and is asserted here:

* the accepted M7B/M7C, M8 and M9 graph configurations keep their exact topology,
  behaviour and rendered order;
* a full M10D graph builds only when matcher and reranker are supplied **together** over
  an enricher, and every other combination fails construction explicitly;
* the recommendation route adds evidence and order, and changes nothing else: candidate
  count, identity, metadata, evidence linkage and raw SASRec score are all preserved and
  ``original_rank`` is never overwritten;
* the final rendering follows the reranked order while keeping original ranks auditable,
  and a candidate is always printed with **its own** catalogue facts;
* M10C stays out of the serving path (and is used here, test-side, to validate output);
* the direct route performs no recommendation, enrichment, matching or reranking work;
* failures propagate instead of producing a partially reranked answer.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_fakes import HISTORY, ScriptedDecisionModel  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    INITIAL_ORDER,
    QUERY,
    CountingMatcher,
    CountingReranker,
    FixedEngine,
    StaticMemory,
    build_index,
    build_tool,
    make_service,
    rendered_order,
)
from tests.preference_matching_fixture import make_entry, make_snapshot  # noqa: E402
from recommendation.agent import (  # noqa: E402
    AGENT_GRAPH_VERSION,
    NODE_DECIDE,
    NODE_ENRICH,
    NODE_FINALIZE,
    NODE_LOAD_MEMORY,
    NODE_MATCH_PREFERENCES,
    NODE_PERSIST_MEMORY,
    NODE_RECOMMEND,
    NODE_RERANK,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentConfigurationError,
    AgentGraph,
    AgentGraphError,
    PreferenceMatcherLike,
    PreferenceRerankerLike,
)
from recommendation.catalog import MetadataIndex  # noqa: E402
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
    PreferenceCandidateMatcher,
    PreferenceEvidenceReport,
)
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.reranking import (  # noqa: E402
    CandidateEvaluation,
    PreferenceReranker,
    RerankingReport,
    sort_key_for,
)
from recommendation.tools import RecommendationToolError  # noqa: E402

RECOMMEND = {"action": "recommend", "k": 4}
DIRECT = {"action": "direct_response", "direct_response": "Hello there."}

AGENT_DIR = REPO_ROOT / "recommendation" / "agent"

#: Phrases that would turn policy adherence into a quality or relevance claim.
FORBIDDEN_CLAIMS = (
    "best for you",
    "most relevant",
    "better recommendation",
    "more personalized",
    "more personalised",
    "optimal choice",
    "perfectly matches",
    "match score",
    "preference score",
    "item_id",
    "broke the tie",
)

#: Words that may only ever appear inside the explicit score disclaimer, never as the
#: label of the score itself.
SCORE_MISLABELS = ("confidence", "probability", "rating", "preference score")

#: The exact sentence that keeps the raw score from being read as product evidence.
SCORE_DISCLAIMER_FRAGMENT = "not probabilities or confidence values"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


class CountingEnricher:
    """The real M8 enricher plus a call counter."""

    def __init__(self, delegate: ProductEnricher | None = None) -> None:
        self._delegate = delegate or ProductEnricher(build_index())
        self.calls: list[Any] = []

    @property
    def call_count(self) -> int:
        """How many times ``enrich`` was invoked."""
        return len(self.calls)

    def enrich(self, result: Any, query: str = "") -> Any:
        """Record the call and delegate."""
        self.calls.append(result)
        return self._delegate.enrich(result, query)


class CountingLookup:
    """A metadata index wrapper recording every identifier the enricher asked about."""

    def __init__(self, delegate: MetadataIndex) -> None:
        self._delegate = delegate
        self.requested: list[str] = []

    @property
    def requested_asins(self) -> tuple[str, ...]:
        """Every identifier requested, in request order."""
        return tuple(self.requested)

    def lookup(self, parent_asin: str) -> Any:
        """Record and delegate."""
        self.requested.append(parent_asin)
        return self._delegate.lookup(parent_asin)

    def lookup_many(self, parent_asins: Any) -> Any:
        """Record and delegate, preserving alignment."""
        self.requested.extend(parent_asins)
        return self._delegate.lookup_many(parent_asins)

    def __contains__(self, parent_asin: object) -> bool:
        """Delegate membership."""
        return parent_asin in self._delegate


def colour_entry(
    value: str,
    *,
    polarity: PreferencePolarity = PreferencePolarity.AVOID,
    logical_seq: int = 1,
    kind: PreferenceKind = PreferenceKind.COLOR,
) -> Any:
    """Build one active preference entry."""
    return make_entry(
        memory_id=f"mem-{value}-{logical_seq}",
        kind=kind,
        value=value,
        polarity=polarity,
        logical_seq=logical_seq,
    )


def snapshot(*entries: Any, user_key: str = "alice") -> Any:
    """Build a preference snapshot from entries."""
    return make_snapshot(*entries, user_key=user_key)


class CountingService:
    """The real M9 memory service plus read/write counters and a user-key log."""

    def __init__(self, delegate: PreferenceMemoryService) -> None:
        self._delegate = delegate
        self.loads: list[str] = []
        self.turns: list[dict[str, Any]] = []

    @property
    def load_count(self) -> int:
        """How many snapshot reads happened."""
        return len(self.loads)

    @property
    def turn_count(self) -> int:
        """How many memory writes happened."""
        return len(self.turns)

    def get_active_preferences(self, user_key: str) -> Any:
        """Record the read and delegate."""
        self.loads.append(user_key)
        return self._delegate.get_active_preferences(user_key)

    def get_memory_history(self, user_key: str) -> Any:
        """Delegate an audit-trail read."""
        return self._delegate.get_memory_history(user_key)

    def process_turn(self, **kwargs: Any) -> Any:
        """Record the write and delegate."""
        self.turns.append(kwargs)
        return self._delegate.process_turn(**kwargs)


def _candidate_blocks(text: str) -> dict[str, str]:
    """Split a rendered response into one text block per candidate identity.

    Blocks are keyed by ``parent_asin`` so an assertion can ask "what did the response
    print next to *this* product?" without depending on the position it appears in.
    """
    blocks: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\d+\. ((?:B[0-9A-Za-z]+)|(?:cand-[a-z]+)) \(", line)
        if match:
            current = match.group(1)
            blocks[current] = ""
            continue
        if current is not None:
            blocks[current] += line + "\n"
    return blocks


#: ``avoid red`` + ``prefer blue``: rank 1 has a supported violation, rank 2 a match.
VIOLATION_PREFS = (colour_entry("red"), colour_entry("blue", polarity=PreferencePolarity.PREFER, logical_seq=2))

#: ``prefer black`` + ``prefer Acme``: only the rank 3 candidate has matches.
MATCH_PREFS = (
    colour_entry("black", polarity=PreferencePolarity.PREFER),
    colour_entry("Acme", polarity=PreferencePolarity.PREFER, logical_seq=2, kind=PreferenceKind.BRAND),
)


def make_graph(
    payload: Any = RECOMMEND,
    *,
    engine: FixedEngine | None = None,
    enricher: Any = None,
    memory: Any = None,
    user_key: str | None = "alice",
    matcher: Any = None,
    reranker: Any = None,
) -> tuple[AgentGraph, ScriptedDecisionModel, FixedEngine]:
    """Build a graph over the offline doubles with an explicit dependency set."""
    engine = engine or FixedEngine()
    decision_model = ScriptedDecisionModel(payload)
    graph = AgentGraph(
        decision_model,
        build_tool(engine),
        product_enricher=enricher,
        memory_service=memory,
        user_key=user_key if memory is not None else None,
        preference_matcher=matcher,
        reranker=reranker,
    )
    return graph, decision_model, engine


def full_graph(
    prefs: Any = None,
    *,
    payload: Any = RECOMMEND,
    rows: Any = CANDIDATE_ROWS,
    memory: Any = None,
    matcher: Any = None,
    reranker: Any = None,
) -> tuple[AgentGraph, CountingMatcher, CountingReranker, CountingEnricher, CountingLookup, Any]:
    """Build a full M10D graph with counting collaborators.

    ``prefs`` may be a ready snapshot, a sequence of preference entries (wrapped into a
    snapshot here), or ``None`` for an empty snapshot.  When ``memory`` is not supplied a
    :class:`StaticMemory` over that snapshot is used, so every call is observable.
    """
    if prefs is None:
        resolved = snapshot()
    elif hasattr(prefs, "active_entries"):
        resolved = prefs
    else:
        resolved = snapshot(*prefs)
    lookup = CountingLookup(build_index(row[0] for row in rows))
    enricher = CountingEnricher(ProductEnricher(lookup))
    memory = memory if memory is not None else StaticMemory(resolved)
    matcher = matcher if matcher is not None else CountingMatcher()
    reranker = reranker if reranker is not None else CountingReranker()
    graph, _, engine = make_graph(
        payload,
        engine=FixedEngine(rows),
        enricher=enricher,
        memory=memory,
        matcher=matcher,
        reranker=reranker,
    )
    return graph, matcher, reranker, enricher, lookup, engine


def run_full(prefs: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Run a full M10D recommendation route and return the final state."""
    graph, *_ = full_graph(prefs, **kwargs)
    return graph.run(QUERY, HISTORY)


# --------------------------------------------------------------------------- #
# 1-5. Topology and the dependency matrix
# --------------------------------------------------------------------------- #


def declared_nodes(graph: AgentGraph) -> set[str]:
    """Declared node names, excluding LangGraph's synthetic start/end."""
    return set(graph.node_names()) - {"__start__", "__end__"}


def test_legacy_graph_topology_is_unchanged() -> None:
    """Tool only: exactly the accepted M7B topology, and version 1 shape."""
    graph, _, _ = make_graph()
    assert declared_nodes(graph) == {NODE_DECIDE, NODE_RECOMMEND, NODE_FINALIZE}
    assert graph.version == AGENT_GRAPH_VERSION == 2
    assert not graph.enriches_products
    assert not graph.matches_preferences
    assert not graph.reranks_preferences


def test_enricher_only_graph_topology_is_unchanged() -> None:
    """M8: the enrich node is inserted and nothing else appears."""
    graph, _, _ = make_graph(enricher=CountingEnricher())
    assert declared_nodes(graph) == {
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_ENRICH,
        NODE_FINALIZE,
    }
    assert graph.enriches_products
    assert not graph.matches_preferences
    assert not graph.reranks_preferences


def test_memory_and_enricher_graph_topology_is_unchanged() -> None:
    """M9: load/persist appear, and no preference-evidence node does."""
    graph, _, _ = make_graph(enricher=CountingEnricher(), memory=StaticMemory(snapshot()))
    assert declared_nodes(graph) == {
        NODE_LOAD_MEMORY,
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_ENRICH,
        NODE_FINALIZE,
        NODE_PERSIST_MEMORY,
    }
    assert graph.uses_memory
    assert not graph.matches_preferences


def test_full_m10d_graph_adds_exactly_two_nodes() -> None:
    """Full M10D: match_preferences and rerank are appended after enrich."""
    graph, *_ = full_graph(VIOLATION_PREFS)
    assert declared_nodes(graph) == {
        NODE_LOAD_MEMORY,
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_ENRICH,
        NODE_MATCH_PREFERENCES,
        NODE_RERANK,
        NODE_FINALIZE,
        NODE_PERSIST_MEMORY,
    }
    assert graph.matches_preferences
    assert graph.reranks_preferences


def test_rerank_follows_match_preferences_in_the_mermaid_contract() -> None:
    """The declared edges put evidence before policy, and policy before finalize."""
    graph, *_ = full_graph(VIOLATION_PREFS)
    mermaid = graph.mermaid()
    assert f"{NODE_MATCH_PREFERENCES} --> {NODE_RERANK}" in mermaid
    assert f"{NODE_ENRICH} --> {NODE_MATCH_PREFERENCES}" in mermaid
    assert f"{NODE_RERANK} --> {NODE_FINALIZE}" in mermaid


@pytest.mark.parametrize(
    ("matcher", "reranker"),
    [
        (CountingMatcher(), None),
        (None, CountingReranker()),
    ],
)
def test_half_configured_reranking_is_rejected(matcher: Any, reranker: Any) -> None:
    """A matcher without a policy (or the reverse) is a hard configuration error."""
    with pytest.raises(AgentConfigurationError, match="configured together"):
        make_graph(enricher=CountingEnricher(), matcher=matcher, reranker=reranker)


def test_matcher_requires_an_enricher() -> None:
    """M10A reads the metadata M8 attached, so matching without enrichment is invalid."""
    with pytest.raises(AgentConfigurationError, match="requires product_enricher"):
        make_graph(matcher=CountingMatcher(), reranker=CountingReranker())


def test_matcher_without_a_callable_match_is_rejected() -> None:
    """A collaborator missing its one method is not silently accepted."""
    with pytest.raises(AgentConfigurationError, match="preference_matcher must provide"):
        make_graph(
            enricher=CountingEnricher(),
            matcher=object(),
            reranker=CountingReranker(),
        )


def test_reranker_without_a_callable_rerank_is_rejected() -> None:
    """Same for the policy half of the pair."""
    with pytest.raises(AgentConfigurationError, match="reranker must provide"):
        make_graph(
            enricher=CountingEnricher(),
            matcher=CountingMatcher(),
            reranker=object(),
        )


def test_reranking_pair_is_valid_without_memory() -> None:
    """Documented contract: no memory means an empty snapshot, not a configuration error.

    The matcher then runs with zero active preferences, produces an evidence report with
    no records, and the accepted M10B policy is order-preserving.
    """
    graph, _, _ = make_graph(
        enricher=CountingEnricher(),
        matcher=CountingMatcher(),
        reranker=CountingReranker(),
    )
    assert graph.matches_preferences and graph.reranks_preferences
    assert not graph.uses_memory

    state = graph.run(QUERY, HISTORY)
    assert "preference_snapshot" not in state
    assert state["preference_evidence"].active_preference_count == 0
    assert state["preference_evidence"].candidates[0].evidence == ()
    assert rendered_order(state["final_response"]) == INITIAL_ORDER

def test_structural_protocols_are_satisfied_by_the_real_collaborators() -> None:
    """The real accepted matcher and reranker conform to the graph's seams."""
    assert isinstance(PreferenceCandidateMatcher(), PreferenceMatcherLike)
    assert isinstance(PreferenceReranker(), PreferenceRerankerLike)


def test_graph_never_constructs_a_default_matcher_or_reranker() -> None:
    """With nothing injected the stored collaborators are ``None``."""
    graph, _, _ = make_graph()
    assert graph.preference_matcher is None
    assert graph.reranker is None


# --------------------------------------------------------------------------- #
# 6-8. Call counts: once on the recommendation route, never on the direct route
# --------------------------------------------------------------------------- #


def test_recommendation_route_calls_matcher_and_reranker_once() -> None:
    """The route runs each optional stage exactly once."""
    graph, matcher, reranker, enricher, _, engine = full_graph(VIOLATION_PREFS)
    graph.run(QUERY, HISTORY)

    assert engine.call_count == 1
    assert enricher.call_count == 1
    assert matcher.call_count == 1
    assert reranker.call_count == 1


def test_direct_route_calls_neither_matcher_nor_reranker() -> None:
    """DIRECT does no recommendation, enrichment, matching, reranking or metadata work."""
    graph, matcher, reranker, enricher, lookup, engine = full_graph(VIOLATION_PREFS, payload=DIRECT)
    state = graph.run("hello", HISTORY)

    assert state["route"] == ROUTE_DIRECT
    assert state["final_response"] == "Hello there."
    assert engine.call_count == 0
    assert enricher.call_count == 0
    assert matcher.call_count == 0
    assert reranker.call_count == 0
    assert lookup.requested_asins == ()
    assert "tool_result" not in state
    assert "enrichment" not in state
    assert "preference_evidence" not in state
    assert "reranking" not in state


def test_direct_route_still_reads_and_persists_memory() -> None:
    """Accepted M9 behaviour is preserved on the direct route."""
    service = make_service()
    graph, matcher, reranker, _, _, engine = full_graph(
        None, payload=DIRECT, memory=service
    )
    graph.run("I don't want red.", HISTORY)

    assert service.get_active_preferences("alice").active_count == 1
    assert engine.call_count == 0
    assert matcher.call_count == 0
    assert reranker.call_count == 0


# --------------------------------------------------------------------------- #
# 9-10. No preferences / all-UNKNOWN preserve the original order
# --------------------------------------------------------------------------- #


def test_no_active_preferences_preserves_the_original_order() -> None:
    """Zero active preferences must reproduce the M9/M8 order exactly."""
    state = run_full(snapshot())
    assert rendered_order(state["final_response"]) == INITIAL_ORDER
    assert state["reranking"].moved_count == 0
    assert [c.reranked_rank for c in state["reranking"].candidates] == [1, 2, 3, 4]
    assert state["preference_evidence"].active_preference_count == 0


def test_all_unknown_evidence_preserves_the_original_order() -> None:
    """UNKNOWN is neutral: no promotion and no penalty."""
    state = run_full(snapshot(colour_entry("purple", polarity=PreferencePolarity.PREFER)))

    candidates = state["reranking"].candidates
    assert rendered_order(state["final_response"]) == INITIAL_ORDER
    assert [c.match_count for c in candidates] == [0, 0, 0, 0]
    assert [c.violation_count for c in candidates] == [0, 0, 0, 0]
    assert [c.unknown_count for c in candidates] == [1, 1, 1, 1]
    assert all(
        record.status is EvidenceStatus.UNKNOWN
        for candidate in state["preference_evidence"].candidates
        for record in candidate.evidence
    )
    assert state["reranking"].moved_count == 0


def test_no_memory_configured_preserves_the_original_order() -> None:
    """Without memory the matcher receives an empty sequence and M10B is a no-op."""
    graph, _, _ = make_graph(
        enricher=CountingEnricher(),
        matcher=CountingMatcher(),
        reranker=CountingReranker(),
    )
    state = graph.run(QUERY, HISTORY)
    assert rendered_order(state["final_response"]) == INITIAL_ORDER
    assert state["preference_evidence"].candidates[0].evidence == ()


# --------------------------------------------------------------------------- #
# 11-12. Violation demotes; match promotes at equal violations
# --------------------------------------------------------------------------- #


def test_violation_demotes_but_never_removes() -> None:
    """The violating rank 1 candidate is demoted, still present, and score-identical."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    candidates = {c.parent_asin: c for c in state["reranking"].candidates}

    red = candidates["cand-red"]
    blue = candidates["cand-blue"]

    assert red.original_rank == 1
    assert red.reranked_rank == 4
    assert red.violation_count == 1
    assert red.match_count == 0
    assert blue.reranked_rank == 1
    assert blue.match_count == 1
    assert blue.violation_count == 0

    # Not filtered: same count, same identities, and the raw score is untouched.
    assert len(state["reranking"].candidates) == len(CANDIDATE_ROWS)
    assert set(candidates) == set(INITIAL_ORDER)
    assert red.sasrec_score == 0.9
    assert candidates["cand-blue"].sasrec_score == 0.8


def test_violation_does_not_change_the_tool_result() -> None:
    """The Tool result keeps SASRec order and rank even when a candidate is demoted."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    result = state["tool_result"]
    assert [r.parent_asin for r in result.recommendations] == list(INITIAL_ORDER)
    assert [r.rank for r in result.recommendations] == [1, 2, 3, 4]
    assert result.recommendations[0].parent_asin == "cand-red"
    assert result.recommendations[0].rank == 1


def test_match_promotes_at_equal_violation_count() -> None:
    """Equal violations (all zero) and more matches moves the lower candidate up."""
    state = run_full(snapshot(*MATCH_PREFS))
    candidates = {c.parent_asin: c for c in state["reranking"].candidates}

    black = candidates["cand-black"]
    assert black.original_rank == 3
    assert black.reranked_rank == 1
    assert black.match_count == 2
    assert black.violation_count == 0

    # Every other candidate has zero violations too, so the promotion is match-driven.
    for asin in ("cand-red", "cand-blue", "cand-green"):
        assert candidates[asin].violation_count == 0
    assert [c.violation_count for c in state["reranking"].candidates] == [0, 0, 0, 0]


def test_policy_order_matches_the_frozen_m10b_key() -> None:
    """The integrated order is exactly what the accepted key produces, candidate for candidate."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    report = state["preference_evidence"]
    by_asin = {c.parent_asin: c for c in report.candidates}
    expected = sorted(
        (c.parent_asin for c in report.candidates),
        key=lambda asin: sort_key_for(
            by_asin[asin], CandidateEvaluation.from_evidence(by_asin[asin].evidence)
        ),
    )
    assert list(state["reranking"].parent_asins) == expected


# --------------------------------------------------------------------------- #
# 13-15. Candidate invariants: count, identity, score
# --------------------------------------------------------------------------- #


def test_candidate_multiset_is_unchanged_by_the_full_route() -> None:
    """Same count, same identities, same multiplicities, same raw scores, no additions."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    before = state["tool_result"].recommendations
    after = state["reranking"].candidates

    assert len(before) == len(after) == len(CANDIDATE_ROWS)
    assert [r.parent_asin for r in before] == list(INITIAL_ORDER)
    assert sorted(c.parent_asin for c in after) == sorted(INITIAL_ORDER)
    assert sorted(c.item_id for c in after) == sorted(r.item_id for r in before)
    assert sorted(c.sasrec_score for c in after) == sorted(r.score for r in before)
    for candidate in after:
        assert len([c for c in after if c.parent_asin == candidate.parent_asin]) == 1


def test_reranked_ranks_are_contiguous_and_unique() -> None:
    """Reranked ranks form exactly ``1..N``."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    ranks = [c.reranked_rank for c in state["reranking"].candidates]
    assert ranks == list(range(1, len(CANDIDATE_ROWS) + 1))


def test_original_rank_is_never_overwritten() -> None:
    """Every candidate keeps the SASRec rank it arrived with, and the set is unchanged."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    original = {c.parent_asin: c.original_rank for c in state["preference_evidence"].candidates}
    assert {c.parent_asin: c.original_rank for c in state["reranking"].candidates} == original
    assert sorted(original.values()) == [1, 2, 3, 4]
    assert state["preference_evidence"].ranks == (1, 2, 3, 4)


def test_evidence_report_keeps_upstream_order() -> None:
    """M10A evidence is ordered by SASRec rank, not by the policy."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    report = state["preference_evidence"]
    assert report.parent_asins == INITIAL_ORDER
    assert report.ranks == (1, 2, 3, 4)
    assert [c.rank for c in state["enrichment"].items] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# 16-18. Metadata and evidence stay attached to the right product
# --------------------------------------------------------------------------- #


def test_metadata_lookup_is_candidate_scoped() -> None:
    """The metadata layer is asked about exactly the Tool's candidates and no one else."""
    graph, _, _, _, lookup, engine = full_graph(VIOLATION_PREFS)
    graph.run(QUERY, HISTORY)
    assert engine.call_count == 1
    assert set(lookup.requested_asins) == set(INITIAL_ORDER)
    assert len(lookup.requested_asins) == len(INITIAL_ORDER)


def test_metadata_stays_attached_to_the_same_product_after_reordering() -> None:
    """A reranked candidate prints its own catalogue facts, never its neighbour's.

    Adversarial: the original order is ``red, blue, black`` and the policy order is
    ``black, blue, red``.  A positional ``zip`` would give the first printed block the
    *red* title; the identity-based alignment must give it the *black* title.
    """
    rows = CANDIDATE_ROWS[:3]
    state = run_full(snapshot(colour_entry("red"), colour_entry("black", polarity=PreferencePolarity.PREFER, logical_seq=2)), rows=rows)

    assert rendered_order(state["final_response"]) == ("cand-black", "cand-blue", "cand-red")
    assert state["enrichment"].items[0].parent_asin == "cand-red"

    text = state["final_response"]
    blocks = text.split("\n1. ")[1].split("\n2. ")[0]
    assert "BlackWidget" in blocks
    assert "RedWidget" not in blocks
    assert "BlueWidget" not in blocks

    second = text.split("\n2. ")[1].split("\n3. ")[0]
    assert "BlueWidget" in second
    assert "RedWidget" not in second
    assert "BlackWidget" not in second

    third = text.split("\n3. ")[1]
    assert "RedWidget" in third
    assert "BlackWidget" not in third


def test_evidence_stays_attached_to_the_same_product() -> None:
    """Each candidate's counts come from its own evidence records."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    report = state["preference_evidence"]
    by_asin = {c.parent_asin: c for c in report.candidates}
    for candidate in state["reranking"].candidates:
        assert candidate.evidence == by_asin[candidate.parent_asin].evidence
        assert candidate.item_id == by_asin[candidate.parent_asin].item_id
        assert candidate.sasrec_score == by_asin[candidate.parent_asin].sasrec_score


def test_matcher_receives_the_enriched_candidates_and_the_turn_snapshot() -> None:
    """The matcher's inputs are exactly the enriched candidates and the loaded snapshot."""
    memory = StaticMemory(snapshot(*VIOLATION_PREFS))
    graph, matcher, _, _, _, _ = full_graph(None, memory=memory)
    state = graph.run(QUERY, HISTORY)

    assert matcher.last_candidates == tuple(state["enrichment"].items)
    assert matcher.last_preferences is memory.snapshot
    assert memory.load_calls == ["alice"]


def test_reranker_receives_exactly_the_matcher_report() -> None:
    """The report handed to M10B is the very object M10A produced."""
    graph, _, reranker, _, _, _ = full_graph(VIOLATION_PREFS)
    state = graph.run(QUERY, HISTORY)
    assert reranker.last_report is state["preference_evidence"]
    assert isinstance(reranker.last_report, PreferenceEvidenceReport)
    assert isinstance(state["reranking"], RerankingReport)


# --------------------------------------------------------------------------- #
# 19-20. Presentation follows the reranked order
# --------------------------------------------------------------------------- #


def test_final_response_follows_the_reranked_order() -> None:
    """The rendered sequence equals the M10B order, not the SASRec order."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    assert rendered_order(state["final_response"]) == state["reranking"].parent_asins
    assert rendered_order(state["final_response"]) != INITIAL_ORDER


def test_final_response_keeps_original_ranks_visible() -> None:
    """Both ranks are printed, so the movement is auditable from the text alone."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    text = state["final_response"]
    assert "original SASRec rank 1" in text
    assert "moved from rank 1 to rank 4 under the configured explicit-preference policy" in text
    assert "preference evidence: 1 supported match(es), 0 supported violation(s), 1 unknown" in text


def test_enrichment_only_response_keeps_the_raw_order() -> None:
    """M8 behaviour is untouched: without a reranker the rendered order is the Tool's."""
    graph, _, _ = make_graph(enricher=CountingEnricher())
    state = graph.run(QUERY, HISTORY)
    assert "reranking" not in state
    assert rendered_order(state["final_response"]) == INITIAL_ORDER


def test_plain_response_keeps_the_raw_order() -> None:
    """M7B/M7C behaviour is untouched."""
    graph, _, _ = make_graph()
    state = graph.run(QUERY, HISTORY)
    assert rendered_order(state["final_response"]) == INITIAL_ORDER
    assert state["route"] == ROUTE_RECOMMEND


def test_rendered_response_contains_no_unsupported_claims() -> None:
    """No quality, relevance or tie-break claim may be rendered."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    lowered = state["final_response"].lower()
    for forbidden in FORBIDDEN_CLAIMS:
        assert forbidden not in lowered


def test_score_is_never_labelled_as_confidence_probability_or_rating() -> None:
    """The raw model score is presented as a ranking score and disclaimed explicitly."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    text = state["final_response"]

    assert SCORE_DISCLAIMER_FRAGMENT in text
    # Candidate lines only: the disclaimer itself names the words it forbids.
    score_lines = [
        line
        for line in text.splitlines()
        if re.match(r"^\d+\. ", line) and "ranking score" in line.lower()
    ]
    assert score_lines, "the score must be labelled as a ranking score"
    for line in score_lines:
        for mislabel in SCORE_MISLABELS:
            assert mislabel not in line.lower()

    # The words may appear only inside the disclaimer that forbids reading them that way.
    body = text.replace(SCORE_DISCLAIMER_FRAGMENT, "")
    for mislabel in ("confidence", "probability"):
        assert mislabel not in body.lower()


def test_rendered_response_quotes_only_this_candidates_metadata() -> None:
    """Grounded: each printed title is that candidate's own metadata title, and no other."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    blocks = _candidate_blocks(state["final_response"])
    titles = {
        item.parent_asin: (item.metadata.title if item.metadata is not None else None)
        for item in state["enrichment"].items
    }

    assert set(blocks) == set(INITIAL_ORDER)
    for asin, block in blocks.items():
        printed = [
            line.split("Title: ", 1)[1]
            for line in block.splitlines()
            if line.strip().startswith("Title: ")
        ]
        if titles[asin] is None:
            assert printed == []
        else:
            assert printed == [titles[asin]]
        for other_asin, other_title in titles.items():
            if other_asin != asin and other_title is not None:
                assert other_title not in block


def test_empty_candidate_list_still_renders_the_exhaustion_message() -> None:
    """Candidate exhaustion stays a normal outcome, not an error or padding."""
    graph, *_ = full_graph(VIOLATION_PREFS, rows=())
    state = graph.run(QUERY, HISTORY)
    assert state["reranking"].candidates == ()
    assert "No unseen product is left" in state["final_response"]


# --------------------------------------------------------------------------- #
# 21-24. M9 lifecycle through the graph: ADD / REPLACE / REMOVE, turn semantics
# --------------------------------------------------------------------------- #


def lifecycle_graph() -> tuple[AgentGraph, PreferenceMemoryService, CountingMatcher, CountingReranker]:
    """A full M10D graph over the real M9 memory service."""
    service = make_service()
    graph, matcher, reranker, _, _, _ = full_graph(None, memory=service)
    return graph, service, matcher, reranker


def test_add_keeps_both_preferences_and_both_reach_matching() -> None:
    """ADD: two avoidances coexist, neither is dropped, both produce evidence."""
    graph, service, _, _ = lifecycle_graph()
    for turn, message in (("t1", "avoid red"), ("t2", "avoid blue")):
        graph.invoke(
            __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
                user_message=message, trusted_user_history=HISTORY, turn_id=turn
            )
        )

    active = service.get_active_preferences("alice")
    assert active.active_count == 2
    assert sorted(entry.value for entry in active.active_entries) == ["blue", "red"]

    state = graph.run(QUERY, HISTORY)
    report = state["preference_evidence"]
    assert report.active_preference_count == 2
    assert len(report.candidates[0].evidence) == 2
    assert {record.preference_value for record in report.candidates[0].evidence} == {"red", "blue"}
    assert all(len(candidate.evidence) == 2 for candidate in report.candidates)


def test_replace_drops_the_superseded_value() -> None:
    """REPLACE: the earlier value no longer reaches evidence; the new one does."""
    graph, service, _, _ = lifecycle_graph()
    for turn, message in (("t1", "I prefer black."), ("t2", "Actually, I prefer blue instead.")):
        graph.invoke(
            __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
                user_message=message, trusted_user_history=HISTORY, turn_id=turn
            )
        )

    active = service.get_active_preferences("alice")
    assert [entry.value for entry in active.active_entries] == ["blue"]

    state = graph.run(QUERY, HISTORY)
    report = state["preference_evidence"]
    assert report.active_preference_count == 1
    values = {
        record.preference_value
        for candidate in report.candidates
        for record in candidate.evidence
    }
    assert values == {"blue"}
    assert "black" not in values

    # And the surviving preference really drives the policy: blue is promoted.
    assert state["reranking"].candidates[0].parent_asin == "cand-blue"


def test_remove_leaves_no_active_colour_evidence() -> None:
    """REMOVE: a retraction removes the colour constraint from evidence entirely."""
    graph, service, _, _ = lifecycle_graph()
    for turn, message in (
        ("t1", "I don't want red."),
        ("t2", "I don't care about color anymore."),
    ):
        graph.invoke(
            __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
                user_message=message, trusted_user_history=HISTORY, turn_id=turn
            )
        )

    assert service.get_active_preferences("alice").active_count == 0

    state = graph.run(QUERY, HISTORY)
    report = state["preference_evidence"]
    assert report.active_preference_count == 0
    assert all(candidate.evidence == () for candidate in report.candidates)
    assert rendered_order(state["final_response"]) == INITIAL_ORDER
    assert state["reranking"].moved_count == 0


def test_superseded_and_removed_entries_never_reach_matching() -> None:
    """The audit trail keeps history, but only ACTIVE entries are ever evaluated."""
    graph, service, _, _ = lifecycle_graph()
    messages = (
        ("t1", "I don't want red."),
        ("t2", "I don't care about color anymore."),
        ("t3", "I prefer black."),
    )
    for turn, message in messages:
        graph.invoke(
            __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
                user_message=message, trusted_user_history=HISTORY, turn_id=turn
            )
        )

    history = service.get_memory_history("alice")
    assert len(history.entries) >= 2  # provenance survives
    active_values = {entry.value for entry in service.get_active_preferences("alice").active_entries}
    assert active_values == {"black"}

    state = graph.run(QUERY, HISTORY)
    values = {
        record.preference_value
        for candidate in state["preference_evidence"].candidates
        for record in candidate.evidence
    }
    assert values == {"black"}


def test_current_turn_preference_does_not_affect_its_own_recommendation() -> None:
    """Mandatory regression: the write lands, but turn N is ranked by turn N-1's memory."""
    graph, service, _, _ = lifecycle_graph()

    first = graph.invoke(
        __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
            user_message="I don't want red.", trusted_user_history=HISTORY, turn_id="t1"
        )
    )

    # The write happened...
    assert service.get_active_preferences("alice").active_count == 1
    assert first["memory_update"] is not None
    assert [entry.value for entry in first["memory_update"].update.added] == ["red"]
    # ...but this turn's own ranking used the snapshot loaded before it: no evidence.
    assert first["preference_evidence"].active_preference_count == 0
    assert first["reranking"].moved_count == 0
    assert rendered_order(first["final_response"]) == INITIAL_ORDER

    # The next turn does see it, and the violating candidate is demoted.
    second = graph.run(QUERY, HISTORY)
    assert second["preference_evidence"].active_preference_count == 1
    assert second["reranking"].candidates[-1].parent_asin == "cand-red"


def test_preferences_are_never_inferred_from_candidate_data() -> None:
    """Candidates full of colourful metadata yield no evidence without stored preferences.

    This is the direct guard against inferring preferences from the recommendation
    result, the SASRec scores, the candidate identities or the retrieved evidence: an
    empty snapshot must produce an empty evidence report even though every candidate has
    a readable colour and a brand.
    """
    graph, matcher, _, _, _, _ = full_graph(())
    state = graph.run(QUERY, HISTORY)

    assert state["preference_evidence"].active_preference_count == 0
    assert all(candidate.evidence == () for candidate in state["preference_evidence"].candidates)
    assert state["preference_evidence"].counts.match_count == 0
    assert state["preference_evidence"].counts.violation_count == 0
    # Every candidate really does carry metadata; the emptiness is about preferences.
    assert all(item.metadata_status == "found" for item in state["enrichment"].items)
    assert matcher.last_preferences is not None
    assert matcher.last_preferences.active_count == 0


def test_preference_snapshot_is_read_once_per_turn() -> None:
    """One turn uses one coherent snapshot: exactly one read, one write, one matcher call."""
    service = CountingService(make_service())
    graph, _, _, _, _, _ = full_graph(None, memory=service)

    graph.invoke(
        __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
            user_message="avoid red", trusted_user_history=HISTORY, turn_id="t1"
        )
    )
    assert service.load_count == 1
    assert service.turn_count == 1
    assert service.loads == ["alice"]

    state = graph.run(QUERY, HISTORY)
    assert service.load_count == 2
    assert service.turn_count == 2

    # The snapshot the matcher evaluated is the snapshot the load node produced.
    assert state["preference_snapshot"] is not None
    assert state["preference_snapshot"].active_count == 1
    assert [entry.value for entry in state["preference_snapshot"].active_entries] == ["red"]


def test_extractor_receives_only_the_user_message() -> None:
    """No conversation-to-interaction conversion and no history-to-preference inference."""
    seen: list[str] = []

    class SpyExtractor:
        def extract(self, user_message: str) -> Any:
            seen.append(user_message)
            return RuleBasedPreferenceExtractor().extract(user_message)

    service = PreferenceMemoryService(InMemoryPreferenceStore(), SpyExtractor())
    graph, *_ = full_graph(None, memory=service)
    graph.run(QUERY, HISTORY)

    assert seen == [QUERY]
    for item in HISTORY:
        assert item not in seen[0]


def test_process_turn_is_called_without_any_history_argument() -> None:
    """The memory write seam carries the user-authored turn only."""
    memory = StaticMemory(snapshot())
    graph, _, _, _, _, _ = full_graph(None, memory=memory)
    graph.invoke(
        __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
            user_message="avoid red", trusted_user_history=HISTORY, turn_id="t1"
        )
    )
    assert memory.turn_calls
    for call in memory.turn_calls:
        assert set(call) == {"user_key", "user_message", "turn_id"}
        assert call["user_message"] == "avoid red"
        assert call["user_key"] == "alice"


# --------------------------------------------------------------------------- #
# 25-30. Immutability, failure paths, reuse, determinism
# --------------------------------------------------------------------------- #


def test_trusted_history_is_not_mutated() -> None:
    """The caller's own history object is untouched by a full M10D run."""
    supplied = list(HISTORY)
    graph, *_ = full_graph(VIOLATION_PREFS)
    graph.invoke(
        __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
            user_message=QUERY, trusted_user_history=supplied
        )
    )
    assert supplied == list(HISTORY)
    assert isinstance(supplied, list)


def test_upstream_objects_are_not_rewritten() -> None:
    """Derived state is added; the Tool result, enrichment and evidence are preserved."""
    graph, _, reranker, enricher, _, engine = full_graph(VIOLATION_PREFS)
    state = graph.run(QUERY, HISTORY)

    result_dump = state["tool_result"].model_dump()
    enrichment_dump = state["enrichment"].model_dump()
    evidence_dump = state["preference_evidence"].model_dump()

    assert engine.last_history == list(HISTORY)
    # Re-deriving from the same upstream objects gives byte-identical payloads.
    assert state["tool_result"].model_dump() == result_dump
    assert state["enrichment"].model_dump() == enrichment_dump
    assert state["preference_evidence"].model_dump() == evidence_dump
    # The report M10B consumed is the report M10A produced, unmodified.
    assert reranker.last_report.model_dump() == evidence_dump
    assert enricher.call_count == 1


def test_candidate_count_drift_is_a_hard_error_not_a_silent_filter() -> None:
    """A reranker that drops a candidate fails loudly instead of hiding the loss."""

    class DroppingReranker:
        def rerank(self, report: Any) -> Any:
            full = PreferenceReranker().rerank(report)
            return full.model_copy(update={"candidates": full.candidates[:-1]})

    graph, _, _, _, _, _ = full_graph(
        VIOLATION_PREFS, reranker=DroppingReranker()
    )
    with pytest.raises(AgentGraphError, match="candidate count"):
        graph.run(QUERY, HISTORY)


def test_unknown_candidate_identity_is_a_hard_error() -> None:
    """A reranker that invents a candidate fails instead of printing orphaned facts."""

    class InventingReranker:
        def rerank(self, report: Any) -> Any:
            full = PreferenceReranker().rerank(report)
            forged = full.candidates[0].model_copy(
                update={"parent_asin": "cand-forged", "item_id": 999}
            )
            return full.model_copy(update={"candidates": (forged, *full.candidates[1:])})

    graph, _, _, _, _, _ = full_graph(VIOLATION_PREFS, reranker=InventingReranker())
    with pytest.raises(AgentGraphError, match="not in the enriched candidate set"):
        graph.run(QUERY, HISTORY)


def test_recommendation_failure_propagates_and_reranking_is_not_attempted() -> None:
    """A Tool failure stops the route: no enrichment, no evidence, no reranking."""
    engine = FixedEngine(error=RuntimeError("engine exploded"))
    matcher = CountingMatcher()
    reranker = CountingReranker()
    enricher = CountingEnricher()
    graph = AgentGraph(
        ScriptedDecisionModel(RECOMMEND),
        build_tool(engine),
        product_enricher=enricher,
        memory_service=StaticMemory(snapshot(*VIOLATION_PREFS)),
        user_key="alice",
        preference_matcher=matcher,
        reranker=reranker,
    )
    with pytest.raises(RecommendationToolError):
        graph.run(QUERY, HISTORY)

    assert engine.call_count == 1
    assert enricher.call_count == 0
    assert matcher.call_count == 0
    assert reranker.call_count == 0


def test_enrichment_failure_produces_no_preference_evidence() -> None:
    """M10A never runs on a failed enrichment, so no fabricated evidence is created."""

    class FailingEnricher:
        def enrich(self, result: Any, query: str = "") -> Any:
            raise RuntimeError("metadata layer unavailable")

    matcher = CountingMatcher()
    reranker = CountingReranker()
    graph, _, _ = make_graph(
        enricher=FailingEnricher(),
        memory=StaticMemory(snapshot(*VIOLATION_PREFS)),
        matcher=matcher,
        reranker=reranker,
    )
    with pytest.raises(RuntimeError):
        graph.run(QUERY, HISTORY)
    assert matcher.call_count == 0
    assert reranker.call_count == 0


def test_matcher_failure_does_not_produce_a_partially_reranked_list() -> None:
    """A matcher error propagates; the reranker is never reached."""

    class FailingMatcher:
        def match(self, *, candidates: Any, preferences: Any) -> Any:
            raise RuntimeError("evidence layer failed")

    reranker = CountingReranker()
    graph, _, _ = make_graph(
        enricher=CountingEnricher(),
        memory=StaticMemory(snapshot(*VIOLATION_PREFS)),
        matcher=FailingMatcher(),
        reranker=reranker,
    )
    with pytest.raises(RuntimeError):
        graph.run(QUERY, HISTORY)
    assert reranker.call_count == 0


def test_reranker_failure_does_not_pretend_the_policy_was_applied() -> None:
    """An M10B error propagates instead of falling back to a raw-order answer."""

    class FailingReranker:
        def rerank(self, report: Any) -> Any:
            raise RuntimeError("policy failed")

    graph, _, _, _, _, _ = full_graph(
        VIOLATION_PREFS, reranker=FailingReranker()
    )
    with pytest.raises(RuntimeError):
        graph.run(QUERY, HISTORY)


def test_direct_route_is_unaffected_by_a_failing_reranking_stage() -> None:
    """A broken preference stage cannot break the direct route."""

    class FailingMatcher:
        def match(self, *, candidates: Any, preferences: Any) -> Any:
            raise RuntimeError("evidence layer failed")

    memory = StaticMemory(snapshot(*VIOLATION_PREFS))
    graph, _, _, _, _, engine = full_graph(
        None, payload=DIRECT, memory=memory, matcher=FailingMatcher()
    )
    state = graph.run("hello", HISTORY)
    assert state["route"] == ROUTE_DIRECT
    assert engine.call_count == 0


def test_collaborators_are_reused_across_turns() -> None:
    """No reload per turn: the same engine, Tool, enricher, matcher and reranker are reused."""
    lookup = CountingLookup(build_index())
    enricher = CountingEnricher(ProductEnricher(lookup))
    matcher = CountingMatcher()
    reranker = CountingReranker()
    memory = StaticMemory(snapshot(*VIOLATION_PREFS))
    engine = FixedEngine()
    tool = build_tool(engine)
    graph = AgentGraph(
        ScriptedDecisionModel(RECOMMEND),
        tool,
        product_enricher=enricher,
        memory_service=memory,
        user_key="alice",
        preference_matcher=matcher,
        reranker=reranker,
    )

    assert graph.tool is tool
    assert graph.product_enricher is enricher
    assert graph.memory_service is memory
    assert graph.preference_matcher is matcher
    assert graph.reranker is reranker

    first = graph.run(QUERY, HISTORY)
    second = graph.run(QUERY, HISTORY)
    third = graph.run(QUERY, HISTORY)

    assert engine.call_count == 3
    assert matcher.call_count == 3
    assert reranker.call_count == 3
    assert first["final_response"] == second["final_response"] == third["final_response"]
    assert first["reranking"].model_dump() == third["reranking"].model_dump()


def test_repeated_execution_is_deterministic() -> None:
    """Same inputs, same evidence, same order, same text."""
    graph, *_ = full_graph(VIOLATION_PREFS)
    dumps = set()
    for _ in range(4):
        state = graph.run(QUERY, HISTORY)
        dumps.add(state["reranking"].model_dump_json())
        texts = state["final_response"]
    assert len(dumps) == 1
    assert isinstance(texts, str)


def test_evidence_report_is_not_mutated_by_reranking() -> None:
    """M10A's report still describes the upstream order after M10B has run."""
    state = run_full(snapshot(*VIOLATION_PREFS))
    assert state["preference_evidence"].parent_asins == INITIAL_ORDER
    assert state["preference_evidence"].ranks == (1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# 31-33. Purity: no duplicated policy, no offline evaluator in the serving path
# --------------------------------------------------------------------------- #


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _sorted_with_key_calls(tree: ast.Module) -> list[str]:
    """Describe every ``sorted(...)``/``.sort(...)`` call that passes a ``key=``."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_sorted = isinstance(node.func, ast.Name) and node.func.id == "sorted"
        is_sort = isinstance(node.func, ast.Attribute) and node.func.attr == "sort"
        if not (is_sorted or is_sort):
            continue
        if any(keyword.arg == "key" for keyword in node.keywords):
            found.append(f"line {node.lineno}")
    return found


@pytest.mark.parametrize("filename", ["graph.py", "state.py", "decision.py", "__init__.py"])
def test_agent_module_does_not_import_the_matching_or_reranking_packages(filename: str) -> None:
    """The graph talks to M10A/M10B only through injected protocols, never imports."""
    imported = _imported_modules(_module_ast(AGENT_DIR / filename))
    offenders = {
        name
        for name in imported
        if name.startswith("recommendation.preference_matching")
        or name.startswith("recommendation.reranking")
    }
    assert offenders == set()


@pytest.mark.parametrize("filename", ["graph.py", "state.py", "decision.py", "__init__.py"])
def test_agent_module_does_not_import_the_offline_evaluator(filename: str) -> None:
    """M10C is diagnostics, not a serving dependency."""
    imported = _imported_modules(_module_ast(AGENT_DIR / filename))
    assert not any("evaluation" in name for name in imported)


def _module_identifiers(tree: ast.Module) -> set[str]:
    """Every identifier that appears in executable code (docstrings excluded)."""
    names: set[str] = set()

    class _Collector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
            names.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast API
            names.add(node.attr)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:  # noqa: N802 - ast API
            names.add(node.arg)

        def visit_keyword(self, node: ast.keyword) -> None:  # noqa: N802 - ast API
            if node.arg:
                names.add(node.arg)
            self.generic_visit(node)

    _Collector().visit(tree)
    return names


def test_agent_graph_does_not_reimplement_the_sort_policy() -> None:
    """No ordering key is written in the agent package: ordering belongs to M10B."""
    tree = _module_ast(AGENT_DIR / "graph.py")
    assert _sorted_with_key_calls(tree) == []

    source = (AGENT_DIR / "graph.py").read_text(encoding="utf-8")
    for policy_fragment in ("violation_count ASC", "match_count DESC", "sort_key_for"):
        assert policy_fragment not in source


def test_graph_source_does_not_use_the_m10b_reason_label_for_explanations() -> None:
    """M10C found the M10B tail reason label imprecise, so it is not user-facing here.

    Checked over executable identifiers (docstrings excluded, so the module may *say* in
    prose that it avoids the label) and over the rendered text itself.
    """
    tree = _module_ast(AGENT_DIR / "graph.py")
    identifiers = _module_identifiers(tree)
    assert "rerank_reason" not in identifiers
    assert "reason_detail" not in identifiers

    state = run_full(snapshot(*VIOLATION_PREFS))
    text = state["final_response"].lower()
    for label in ("deterministic_tie_break", "tie_break", "rerank_reason",
                  "preserved_original_order", "ranked_last"):
        assert label not in text

    # The rule label is still available on the report, where it is structured audit data
    # rather than an explanation shown to a user.
    assert state["reranking"].candidates[0].rerank_reason is not None


def test_no_network_or_provider_sdk_in_the_agent_package() -> None:
    """The integration stays offline."""
    for filename in ("graph.py", "state.py", "decision.py"):
        imported = _imported_modules(_module_ast(AGENT_DIR / filename))
        for banned in ("requests", "httpx", "openai", "anthropic", "urllib3", "aiohttp"):
            assert banned not in imported


# --------------------------------------------------------------------------- #
# 34-36. M10C validates the graph output, test-side only
# --------------------------------------------------------------------------- #


def test_m10c_evaluator_confirms_the_graph_output_invariants() -> None:
    """The offline evaluator accepts the report the graph produced, with all invariants."""
    from recommendation.reranking.evaluation import aggregate_requests, evaluate_request

    state = run_full(snapshot(*VIOLATION_PREFS))
    request, reranked = evaluate_request(
        report=state["preference_evidence"], label="m10d-graph", diagnostics_k=(1, 3)
    )

    assert reranked.model_dump() == state["reranking"].model_dump()
    assert request.invariants.all_hold
    assert request.invariants.failures() == ()
    assert request.movement_attribution.accounting_consistent
    assert request.movement_attribution.item_id_tiebreak == 0

    aggregate = aggregate_requests([request])
    assert aggregate["all_invariants_hold"]
    assert aggregate["duplicate_original_rank_count"] == 0
    assert aggregate["item_id_fallback_reachable"] is False


def test_m10c_evaluator_and_graph_agree_on_no_movement() -> None:
    """With no preferences the evaluator sees a zero-movement, zero-attribution run."""
    from recommendation.reranking.evaluation import evaluate_request

    state = run_full(snapshot())
    request, _ = evaluate_request(
        report=state["preference_evidence"], label="m10d-no-prefs", diagnostics_k=(1, 3)
    )
    assert request.displacement.moved_count == 0
    assert request.movement_attribution.moved_count == 0
    assert request.invariants.all_hold


def test_graph_module_is_not_imported_by_the_offline_evaluator() -> None:
    """The dependency direction is one-way: the evaluator knows nothing of the agent."""
    import recommendation.reranking.evaluation as evaluation

    source = inspect.getsource(evaluation)
    assert "recommendation.agent" not in source


# --------------------------------------------------------------------------- #
# 37. Audit information is available in the returned state
# --------------------------------------------------------------------------- #


def test_returned_state_answers_the_audit_questions() -> None:
    """The final state is enough to explain every rank and every evidence record."""
    state = run_full(snapshot(*VIOLATION_PREFS))

    # What was the original rank / what is the reranked rank?
    pairs = [(c.parent_asin, c.original_rank, c.reranked_rank) for c in state["reranking"].candidates]
    assert pairs == [
        ("cand-blue", 2, 1),
        ("cand-black", 3, 2),
        ("cand-green", 4, 3),
        ("cand-red", 1, 4),
    ]

    # What preferences were active?
    assert [entry.value for entry in state["preference_snapshot"].active_entries] == ["red", "blue"]

    # What evidence was MATCH / VIOLATION / UNKNOWN?
    statuses = {
        (candidate.parent_asin, record.preference_value): record.status
        for candidate in state["preference_evidence"].candidates
        for record in candidate.evidence
    }
    assert statuses[("cand-red", "red")] is EvidenceStatus.VIOLATION
    assert statuses[("cand-blue", "blue")] is EvidenceStatus.MATCH
    assert statuses[("cand-green", "red")] is EvidenceStatus.UNKNOWN

    # Why did the policy move this candidate?  Directly supported facts, no reason label.
    red = next(c for c in state["reranking"].candidates if c.parent_asin == "cand-red")
    assert (red.violation_count, red.match_count) == (1, 0)
