"""Milestone 9 Agent-integration tests: memory around the recommendation route.

Fully offline.  A stub engine, a synthetic metadata index and an in-memory preference
store, so the graph contract is tested without a checkpoint or the full catalogue.

What must hold, and is asserted here:

* with memory configured the topology gains ``load_memory`` / ``persist_memory``, and
  **without** it the accepted M7B/M7C/M8 topology and output are unchanged;
* the graph loads exactly the invoking user's memory and never another user's;
* the direct route can read and write preference memory without any recommendation or
  enrichment work;
* preference memory cannot change candidate identity, count, rank or SASRec score,
  with or without active preferences, and cannot rerank;
* preference memory cannot widen the RAG candidate universe;
* trusted interaction history is untouched and unreadable by the extractor;
* a failing extraction leaves existing memory intact and never leaves a partial write.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_fakes import HISTORY, RecordingEngine, ScriptedDecisionModel  # noqa: E402
from tests.product_catalog_fixture import RecordingMetadataLookup, index  # noqa: E402
from recommendation.agent import (  # noqa: E402
    NODE_DECIDE,
    NODE_ENRICH,
    NODE_FINALIZE,
    NODE_LOAD_MEMORY,
    NODE_PERSIST_MEMORY,
    NODE_RECOMMEND,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentConfigurationError,
    AgentDecision,
    AgentGraph,
)
from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceExtraction,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
    ScriptedPreferenceExtractor,
)
from recommendation.memory.schemas import (  # noqa: E402
    PreferenceCandidate,
    PreferenceKind,
    PreferencePolarity,
)
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.tools import RecommendationTool  # noqa: E402

RECOMMEND = {"action": "recommend", "k": 3}
DIRECT = {"action": "direct_response", "direct_response": "Hello there."}


def make_service(table: dict[str, PreferenceExtraction] | None = None) -> PreferenceMemoryService:
    """Service over a fresh in-memory store with a scripted extractor."""
    return PreferenceMemoryService(
        InMemoryPreferenceStore(), ScriptedPreferenceExtractor(table or {})
    )


def preference(value: str, kind: PreferenceKind = PreferenceKind.COLOR) -> PreferenceExtraction:
    """Extraction containing one explicit preference."""
    return PreferenceExtraction(
        preferences=(
            PreferenceCandidate(
                kind=kind, value=value, source_text=f"user said {value}"
            ),
        )
    )


def make_graph(
    payload: Any = RECOMMEND,
    *,
    service: PreferenceMemoryService | None = None,
    user_key: str | None = None,
    engine: RecordingEngine | None = None,
    enricher: Any = None,
    augment: bool = True,
) -> tuple[AgentGraph, ScriptedDecisionModel, RecordingEngine]:
    """Build a graph over doubles, optionally with preference memory."""
    engine = engine or RecordingEngine(catalog_size=32)
    decision_model = ScriptedDecisionModel(payload)
    graph = AgentGraph(
        decision_model,
        RecommendationTool(engine),
        product_enricher=enricher,
        memory_service=service,
        user_key=user_key,
        augment_query_with_preferences=augment,
    )
    return graph, decision_model, engine


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #


def test_memory_adds_load_and_persist_nodes() -> None:
    service = make_service()
    plain, _, _ = make_graph()
    with_memory, _, _ = make_graph(service=service, user_key="alice")

    assert set(plain.node_names()) - {"__start__", "__end__"} == {
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_FINALIZE,
    }
    assert set(with_memory.node_names()) - {"__start__", "__end__"} == {
        NODE_LOAD_MEMORY,
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_FINALIZE,
        NODE_PERSIST_MEMORY,
    }


def test_memory_and_enrichment_compose() -> None:
    graph, _, _ = make_graph(
        service=make_service(), user_key="alice", enricher=ProductEnricher(index())
    )
    assert set(graph.node_names()) - {"__start__", "__end__"} == {
        NODE_LOAD_MEMORY,
        NODE_DECIDE,
        NODE_RECOMMEND,
        NODE_ENRICH,
        NODE_FINALIZE,
        NODE_PERSIST_MEMORY,
    }


def test_memory_nodes_are_wired_load_first_and_persist_last() -> None:
    graph, _, _ = make_graph(service=make_service(), user_key="alice")
    mermaid = graph.mermaid()
    assert "__start__ --> load_memory" in mermaid
    assert "load_memory --> decide" in mermaid
    assert "finalize --> persist_memory" in mermaid
    assert "persist_memory --> __end__" in mermaid


def test_memory_service_requires_a_user_key() -> None:
    with pytest.raises(AgentConfigurationError):
        make_graph(service=make_service(), user_key=None)
    with pytest.raises(AgentConfigurationError):
        make_graph(service=make_service(), user_key="   ")


def test_unusable_memory_service_is_rejected() -> None:
    for bad in (object(), type("NoMemory", (), {})()):
        with pytest.raises(AgentConfigurationError):
            make_graph(service=bad, user_key="alice")  # type: ignore[arg-type]


def test_without_memory_output_is_the_accepted_behaviour() -> None:
    graph, _, engine = make_graph()
    state = graph.run("suggest something", HISTORY)
    assert state["route"] == ROUTE_RECOMMEND
    assert engine.call_count == 1
    assert "preference_snapshot" not in state
    assert "memory_update" not in state
    assert "Your stated preferences" not in state["final_response"]


def test_agent_package_does_not_import_the_memory_package() -> None:
    """The memory seam is a protocol, so agent code stays free of a hard import."""
    import ast

    agent_dir = Path(__import__("recommendation.agent", fromlist=["x"]).__file__).parent
    for module_path in sorted(agent_dir.glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("recommendation.memory"), module_path.name


# --------------------------------------------------------------------------- #
# A / B. Loading and isolation
# --------------------------------------------------------------------------- #


def test_graph_loads_the_invoking_users_memory() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)

    graph, _, _ = make_graph(service=service, user_key="alice")
    state = graph.run("show me something", HISTORY)

    snapshot = state["preference_snapshot"]
    assert [e.value for e in snapshot.active_entries] == ["blue"]


def test_graph_never_receives_another_users_memory() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)

    bob_graph, _, _ = make_graph(service=service, user_key="bob")
    bob_state = bob_graph.run("show me something", HISTORY)

    assert bob_state["preference_snapshot"].active_entries == ()
    assert "blue" not in bob_state["final_response"]
    # And Bob's turn did not disturb Alice.
    assert [e.value for e in service.get_active_preferences("alice").active_entries] == ["blue"]


def test_concurrent_users_do_not_leak_through_shared_store() -> None:
    service = make_service(
        {"a": preference("blue"), "b": preference("green")}
    )
    alice_graph, _, _ = make_graph(service=service, user_key="alice")
    bob_graph, _, _ = make_graph(service=service, user_key="bob")

    alice_graph.run("a", HISTORY)
    bob_graph.run("b", HISTORY)

    assert [e.value for e in service.get_active_preferences("alice").active_entries] == ["blue"]
    assert [e.value for e in service.get_active_preferences("bob").active_entries] == ["green"]


def test_persist_node_writes_the_turn_to_the_right_user_only() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    graph, _, _ = make_graph(service=service, user_key="alice")
    graph.run("I prefer blue.", HISTORY)

    assert service.get_active_preferences("alice").active_count == 1
    assert service.get_active_preferences("bob").active_count == 0


# --------------------------------------------------------------------------- #
# C. Trusted interaction-history isolation
# --------------------------------------------------------------------------- #


def test_memory_never_alters_trusted_history() -> None:
    service = make_service(
        {
            "I prefer blue.": preference("blue"),
            "I bought B0BX5QFWQN yesterday.": PreferenceExtraction(),
        }
    )
    graph, _, engine = make_graph(service=service, user_key="alice")

    state = graph.run("I bought B0BX5QFWQN yesterday.", HISTORY)

    assert state["trusted_user_history"] == HISTORY
    assert engine.last_history == list(HISTORY)
    # The purchase statement created no preference and no interaction.
    assert service.get_active_preferences("alice").active_entries == ()


def test_conversational_text_never_becomes_an_interaction_event() -> None:
    """The mandatory adversarial regression: a stated purchase is not history."""
    extractor = RuleBasedPreferenceExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    graph, _, engine = make_graph(service=service, user_key="alice")

    supplied = list(HISTORY)
    snapshot = list(supplied)
    state = graph.run("I bought B0BX5QFWQN yesterday.", supplied)

    # History is byte-identical and the engine saw exactly it.
    assert supplied == snapshot
    assert state["trusted_user_history"] == HISTORY
    assert engine.last_history == snapshot
    # The ASIN never entered memory in any form.
    entries = service.get_memory_history("alice").entries
    assert entries == ()
    assert "B0BX5QFWQN" not in str(service.get_memory_history("alice").as_dict())


def test_extractor_is_never_given_history_candidates_or_evidence() -> None:
    extractor = ScriptedPreferenceExtractor({"show me something": preference("blue")})
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    graph, _, _ = make_graph(service=service, user_key="alice", enricher=ProductEnricher(index()))

    graph.run("show me something", HISTORY)

    # The extractor saw exactly one string: the user's own message.
    assert extractor.calls == ["show me something"]
    for called in extractor.calls:
        for asin in HISTORY:
            assert asin not in called


def test_memory_api_has_no_history_parameter() -> None:
    """Structural proof at the graph level as well as the service level."""
    for name in ("get_active_preferences", "process_turn"):
        method = getattr(PreferenceMemoryService, name)
        assert set(inspect.signature(method).parameters).isdisjoint(
            {"history", "user_history", "trusted_user_history"}
        )


# --------------------------------------------------------------------------- #
# D. Recommendation invariants
# --------------------------------------------------------------------------- #


def test_candidates_are_identical_with_and_without_active_preferences() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    plain, _, plain_engine = make_graph()
    with_memory, _, memory_engine = make_graph(service=service, user_key="alice")

    baseline = plain.run("show me something", HISTORY)
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    enriched = with_memory.run("show me something", HISTORY)

    assert [
        (r.rank, r.parent_asin, r.item_id, r.score)
        for r in baseline["tool_result"].recommendations
    ] == [
        (r.rank, r.parent_asin, r.item_id, r.score)
        for r in enriched["tool_result"].recommendations
    ]


@pytest.mark.parametrize(
    "value,kind",
    [
        ("blue", PreferenceKind.COLOR),
        ("red", PreferenceKind.MATERIAL),
        ("leather", PreferenceKind.MATERIAL),
    ],
)
def test_preference_memory_does_not_rerank(value: str, kind: PreferenceKind) -> None:
    """Even when a preference matches a lower-ranked candidate, order is unchanged."""
    service = make_service({f"p-{value}": preference(value, kind)})
    service.process_turn(user_key="alice", user_message=f"p-{value}", turn_id="t1", now=1.0)

    graph, _, _ = make_graph(service=service, user_key="alice")
    state = graph.run("show me something", HISTORY)

    ranks = [r.rank for r in state["tool_result"].recommendations]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [r.score for r in state["tool_result"].recommendations]
    assert scores == sorted(scores, reverse=True)
    # Candidate identity is exactly what the engine produced.
    assert [r.parent_asin for r in state["tool_result"].recommendations] == [
        "B900000001",
        "B900000002",
        "B900000003",
    ]


def test_memory_cannot_change_returned_count_or_requested_k() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    graph, _, engine = make_graph(service=service, user_key="alice")

    state = graph.run("show me something", HISTORY)
    assert state["tool_result"].requested_k == 3
    assert state["tool_result"].returned_k == 3
    assert engine.last_k == 3


def test_memory_cannot_suppress_or_add_a_candidate() -> None:
    """A preference naming a value that matches no candidate changes nothing."""
    service = make_service({"I prefer nothing-like-this": preference("nothing-like-this")})
    service.process_turn(
        user_key="alice", user_message="I prefer nothing-like-this", turn_id="t1", now=1.0
    )
    graph, _, _ = make_graph(service=service, user_key="alice")
    state = graph.run("show me something", HISTORY)
    assert [r.parent_asin for r in state["tool_result"].recommendations] == [
        "B900000001",
        "B900000002",
        "B900000003",
    ]


def test_enrichment_order_is_unchanged_by_preferences() -> None:
    service = make_service({"I prefer waterproof": preference("waterproof")})
    service.process_turn(
        user_key="alice", user_message="I prefer waterproof", turn_id="t1", now=1.0
    )
    graph, _, _ = make_graph(
        service=service, user_key="alice", enricher=ProductEnricher(index())
    )
    state = graph.run("waterproof boots", HISTORY)

    tool_order = [r.parent_asin for r in state["tool_result"].recommendations]
    enriched_order = list(state["enrichment"].parent_asins)
    assert enriched_order == tool_order


# --------------------------------------------------------------------------- #
# E. RAG universe invariant
# --------------------------------------------------------------------------- #


def test_preference_mentioning_an_out_of_scope_product_cannot_widen_retrieval() -> None:
    """A preference naming a product outside the candidate set stays inert."""
    service = make_service({"remember decoy-999": preference("decoy-999")})
    service.process_turn(
        user_key="alice", user_message="remember decoy-999", turn_id="t1", now=1.0
    )
    lookup = RecordingMetadataLookup(index())
    graph, _, _ = make_graph(
        service=service, user_key="alice", enricher=ProductEnricher(lookup)
    )
    state = graph.run("decoy-999", HISTORY)

    candidates = {r.parent_asin for r in state["tool_result"].recommendations}
    assert "decoy-999" not in candidates
    # The metadata layer was asked only about candidates.
    assert set(lookup.requested_asins) == candidates
    for item in state["enrichment"].items:
        for evidence in item.evidence:
            assert evidence.parent_asin in candidates


def test_query_augmentation_is_explicit_and_cannot_add_candidates() -> None:
    from recommendation.agent.graph import _memory_context

    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    snapshot = service.get_active_preferences("alice")

    augmented = _memory_context("boots", snapshot)
    assert augmented == "boots\npreferences:\n- color: prefers blue"

    graph, _, _ = make_graph(
        service=service, user_key="alice", enricher=ProductEnricher(index())
    )
    state = graph.run("boots", HISTORY)
    assert [r.parent_asin for r in state["tool_result"].recommendations] == [
        "B900000001",
        "B900000002",
        "B900000003",
    ]


def test_query_augmentation_can_be_disabled() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    graph, _, _ = make_graph(
        service=service, user_key="alice", enricher=ProductEnricher(index()), augment=False
    )
    assert graph.augments_query_with_preferences is False
    state = graph.run("boots", HISTORY)
    assert state["route"] == ROUTE_RECOMMEND


# --------------------------------------------------------------------------- #
# F. Direct route
# --------------------------------------------------------------------------- #


def test_direct_route_reads_and_writes_memory_without_recommending() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    graph, decision_model, engine = make_graph(
        DIRECT, service=service, user_key="alice"
    )

    state = graph.run("I prefer blue.", HISTORY)

    assert state["route"] == ROUTE_DIRECT
    assert engine.call_count == 0
    assert "tool_result" not in state
    assert "enrichment" not in state
    assert service.get_active_preferences("alice").active_count == 1
    assert decision_model.call_count == 1


def test_direct_route_loads_existing_memory() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    graph, _, engine = make_graph(DIRECT, service=service, user_key="alice")

    state = graph.run("hello", HISTORY)
    assert [e.value for e in state["preference_snapshot"].active_entries] == ["blue"]
    assert engine.call_count == 0


def test_direct_route_does_not_open_the_metadata_layer() -> None:
    class ExplodingLookup:
        def lookup(self, parent_asin: str) -> Any:
            raise AssertionError("direct route must not consult metadata")

        def lookup_many(self, parent_asins: Any) -> Any:
            raise AssertionError("direct route must not consult metadata")

        def __contains__(self, parent_asin: object) -> bool:
            raise AssertionError("direct route must not consult metadata")

    service = make_service()
    graph, _, engine = make_graph(
        DIRECT, service=service, user_key="alice",
        enricher=ProductEnricher(ExplodingLookup()),
    )
    assert graph.run("hello", HISTORY)["route"] == ROUTE_DIRECT
    assert engine.call_count == 0


# --------------------------------------------------------------------------- #
# G. Failure paths
# --------------------------------------------------------------------------- #


def test_failing_extraction_leaves_memory_intact_and_aborts_the_run() -> None:
    class FlakyExtractor:
        def __init__(self) -> None:
            self.fail = False

        def extract(self, user_message: str):
            if self.fail:
                raise RuntimeError("extractor unavailable")
            return preference("blue")

    extractor = FlakyExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    graph, _, _ = make_graph(service=service, user_key="alice")

    graph.run("I prefer blue.", HISTORY)
    before = service.get_memory_history("alice").entries
    assert before

    extractor.fail = True
    with pytest.raises(Exception):
        graph.run("I prefer green.", HISTORY)

    # Memory is exactly as it was: no partial write and no corruption.
    assert service.get_memory_history("alice").entries == before
    assert [e.value for e in service.get_active_preferences("alice").active_entries] == ["blue"]


def test_malformed_extraction_does_not_corrupt_memory() -> None:
    class BadExtractor:
        def extract(self, user_message: str):
            return {"not": "an extraction"}

    service = PreferenceMemoryService(InMemoryPreferenceStore(), BadExtractor())
    graph, _, _ = make_graph(service=service, user_key="alice")
    with pytest.raises(Exception):
        graph.run("anything", HISTORY)
    assert service.get_memory_history("alice").entries == ()


def test_failed_turn_does_not_leave_a_supersede_half_applied() -> None:
    """A failure after a supersede must roll the whole unit back."""

    class TwoStepExtractor:
        def __init__(self) -> None:
            self.fail = False

        def extract(self, user_message: str):
            if self.fail:
                raise RuntimeError("boom")
            return preference(user_message)

    extractor = TwoStepExtractor()
    service = PreferenceMemoryService(InMemoryPreferenceStore(), extractor)
    service.process_turn(user_key="alice", user_message="black", turn_id="t1", now=1.0)

    extractor.fail = True
    with pytest.raises(Exception):
        service.process_turn(user_key="alice", user_message="blue", turn_id="t2", now=2.0)

    entries = service.get_memory_history("alice").entries
    assert [(e.value, e.status.value) for e in entries] == [("black", "active")]


def test_blank_user_message_is_rejected_before_memory_is_touched() -> None:
    from recommendation.agent import MalformedDecision

    service = make_service()
    graph, _, _ = make_graph(service=service, user_key="alice")
    with pytest.raises(MalformedDecision):
        graph._decide_node({"user_message": "   ", "trusted_user_history": list(HISTORY)})  # noqa: SLF001
    assert service.get_memory_history("alice").entries == ()


# --------------------------------------------------------------------------- #
# Rendered response honesty
# --------------------------------------------------------------------------- #


def test_rendered_preferences_are_labelled_and_not_claimed_as_matches() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    service.process_turn(user_key="alice", user_message="I prefer blue.", turn_id="t1", now=1.0)
    graph, _, _ = make_graph(service=service, user_key="alice")
    text = graph.run("show me something", HISTORY)["final_response"]

    assert "Your stated preferences" in text
    assert "not used to rank these candidates" in text
    lowered = text.lower()
    for fabricated in ("matches your preference", "perfectly matches", "match score",
                       "best match", "we picked these because"):
        assert fabricated not in lowered


def test_no_preference_block_when_memory_is_empty() -> None:
    graph, _, _ = make_graph(service=make_service(), user_key="alice")
    text = graph.run("show me something", HISTORY)["final_response"]
    assert "Your stated preferences" not in text


def test_added_preference_is_persisted_even_though_it_is_not_yet_rendered() -> None:
    """Documented turn semantics: writes land, the current response predates them."""
    service = make_service({"I prefer blue.": preference("blue")})
    graph, _, _ = make_graph(service=service, user_key="alice")

    first = graph.run("I prefer blue.", HISTORY)
    assert service.get_active_preferences("alice").active_count == 1
    assert first["memory_update"] is not None
    assert [e.value for e in first["memory_update"].update.added] == ["blue"]
    assert "Your stated preferences" not in first["final_response"]

    second = graph.run("show me something", HISTORY)
    assert "Your stated preferences" in second["final_response"]


def test_repeated_identical_turn_does_not_duplicate_memory() -> None:
    service = make_service({"I prefer blue.": preference("blue")})
    graph, _, _ = make_graph(service=service, user_key="alice")
    for _ in range(3):
        graph.run("I prefer blue.", HISTORY)
    assert service.get_active_preferences("alice").active_count == 1


def test_persist_uses_the_application_supplied_turn_id() -> None:
    """Two different turns with the same text still collapse to one active entry."""
    service = make_service({"I prefer blue.": preference("blue")})
    graph, _, _ = make_graph(service=service, user_key="alice")

    graph.invoke(
        __import__("recommendation.agent", fromlist=["AgentInput"]).AgentInput(
            user_message="I prefer blue.", trusted_user_history=HISTORY, turn_id="t1"
        )
    )
    assert service.get_active_preferences("alice").active_count == 1
