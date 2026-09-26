"""Real-catalogue integration tests for the M3 harness factory.

These run against the **canonical 156,746-item catalogue** — the real ``MetadataIndex``, the real
``CatalogSearchSource``, the real frozen sequential head — with only the model answers scripted.

Scope limits, stated honestly:

* ``ScriptedModelClient`` is the test double, as the rule requires. It never stands in for the
  policy in an evaluation, and no test here reads, asserts or persists a recommendation metric.
* Cohort alignment is *not* verified here. The factory maps ``user_int_id`` to a head row through
  an explicit ``user_order``; these tests pass a small fabricated order so they exercise wiring
  without loading the 20 000-user cohort. Proving the real cohort order matches the frozen head
  rows is an execution-time check, not a test-time one.
"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.m3_agent_arm import M3AgentRunner, summarize_behavior
from experiments.m3_agent_harness import RealCatalogueHarnessFactory
from recommendation.control.arguments import CandidateSource
from recommendation.control.model_client import ScriptedModelClient
from recommendation.control.model_policy import LLMAgentPolicy
from recommendation.evaluation.split import EvaluationCase

CATALOGUE_SIZE = 156746
USER_ORDER = (1, 2, 3)


def _case(user_int_id: int, history_length: int, target: int = 5) -> EvaluationCase:
    history = tuple(range(1, history_length + 1))
    return EvaluationCase(
        user_id=str(user_int_id),
        user_int_id=user_int_id,
        train_history=history,
        validation_target=target,
        test_target=target,
        sequence_length=history_length + 1,
    )


def _factory(script: list[Any], *, history_length: int = 5):
    def policy_factory() -> Any:
        return LLMAgentPolicy(
            ScriptedModelClient(list(script), repeat_last=True), max_attempts=2
        )

    factory = RealCatalogueHarnessFactory(
        policy_factory=policy_factory,
        user_order=USER_ORDER,
        driver="direct",
    )
    case = _case(USER_ORDER[0], history_length)
    return factory, M3AgentRunner(harness_factory=factory, item2id=factory.shared().item2id), case


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_factory_builds_the_canonical_catalogue_once() -> None:
    factory, _runner, _case_ = _factory([{"action": "finish"}])

    first = factory.shared()
    second = factory.shared()

    assert first is second, "the catalogue is shared, not rebuilt per user"
    assert factory.shared_builds == 1
    assert first.records == CATALOGUE_SIZE, "canonical 156,746-item catalogue"
    assert first.catalog_search is not None and first.grounding is not None
    assert len(first.item2id) == CATALOGUE_SIZE


def test_user_harness_reuses_the_shared_components() -> None:
    factory, _runner, case = _factory([{"action": "finish"}])
    a = factory(case)
    b = factory(case)

    assert a.controller is not b.controller, "each user gets its own control plane"
    assert a.ledger is not b.ledger, "each user gets its own ledger"
    assert factory.shared_builds == 1, "and the catalogue was still built only once"


def test_head_rows_come_from_the_frozen_artifact() -> None:
    factory, _runner, case = _factory([{"action": "finish"}])
    rows = factory.head_rows(case)

    assert rows, "the frozen sequential head is non-empty for this user"
    assert all(isinstance(r[0], str) and r[1] > 0 for r in rows)
    scores = [r[2] for r in rows]
    assert scores == sorted(scores, reverse=True), "head order preserved as descending score"


# --------------------------------------------------------------------------- #
# RECOMMEND_FROM_HISTORY
# --------------------------------------------------------------------------- #


def test_recommend_from_history_path_runs_on_the_real_catalogue() -> None:
    factory, runner, case = _factory(
        [{"action": "recommend_from_history", "k": 5}, {"action": "finish"}]
    )
    outcome = runner.run(case)

    assert outcome.behavior["action_sequence"][:2] == ["recommend_from_history", "finish"]
    assert outcome.behavior["succeeded"] is True, (
        f"termination={outcome.behavior['termination_reason']} "
        f"detail_failure={outcome.behavior['failure']}"
    )
    assert outcome.behavior["termination_reason"] == "completed"
    assert outcome.ranking, "the history path produced a ranking"
    assert len(outcome.ranking) <= 5


# --------------------------------------------------------------------------- #
# SEARCH_CATALOG
# --------------------------------------------------------------------------- #


def test_search_catalog_path_runs_on_the_real_catalogue() -> None:
    factory, runner, case = _factory(
        [
            {"action": "search_catalog", "arguments": {"terms": ["waterproof", "jacket"], "limit": 5}},
            {"action": "finish"},
        ]
    )
    outcome = runner.run(case)

    assert "search_catalog" in outcome.behavior["action_sequence"]
    assert outcome.behavior["succeeded"] is True, (
        f"termination={outcome.behavior['termination_reason']}"
    )
    assert outcome.behavior["sources_present"], "the lexical source contributed candidates"
    assert "catalog_search" in outcome.behavior["sources_present"]
    assert outcome.behavior["candidates"] > 0
    assert outcome.ranking, "the catalogue path produced a ranking"


# --------------------------------------------------------------------------- #
# the history_length contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("history_length", [4, 5, 7, 12])
def test_history_length_contract_holds_for_non_three_lengths(history_length: int) -> None:
    """The regression guard: the repository's test engine hardcodes history_length=3."""
    factory, runner, case = _factory(
        [{"action": "recommend_from_history", "k": 3}, {"action": "finish"}],
        history_length=history_length,
    )
    outcome = runner.run(case)

    assert outcome.behavior["succeeded"] is True, (
        "a history of length "
        f"{history_length} must verify; got termination={outcome.behavior['termination_reason']}"
    )
    harness = factory(case)
    # the engine reports the true length rather than a constant
    result = harness.engine.recommend(["x"] * history_length, k=2)
    assert result.history_length == history_length
    assert result.effective_history_length == history_length
    assert harness.engine.last_history is not None


def test_engine_rows_are_the_runs_own_frozen_head() -> None:
    factory, _runner, case = _factory([{"action": "finish"}])
    harness = factory(case)
    result = harness.engine.recommend(["a"] * 6, k=3)
    expected = factory.head_rows(case)[:3]
    assert [r.parent_asin for r in result.recommendations] == [r[0] for r in expected]


# --------------------------------------------------------------------------- #
# grounding / ledger / ranking / logging
# --------------------------------------------------------------------------- #


def test_grounding_ledger_ranking_and_behavior_logging() -> None:
    factory, runner, case = _factory(
        [
            {"action": "search_catalog", "arguments": {"terms": ["camping", "stove"], "limit": 6}},
            {"action": "recommend_from_history", "k": 4},
            {"action": "finish"},
        ]
    )
    outcome = runner.run(case)

    # grounding: every recorded candidate was confirmed
    assert outcome.behavior["candidates"] > 0
    assert outcome.behavior["grounded_candidates"] == outcome.behavior["candidates"]

    # ledger -> ranking
    assert outcome.ranking
    assert len(set(outcome.ranking)) == len(outcome.ranking), "deduplicated"
    shared = factory.shared()
    assert set(outcome.ranking) <= set(range(1, shared.records + 1))
    assert outcome.behavior["ranking_size"] == len(outcome.ranking)

    # logging
    for key in (
        "status",
        "termination_reason",
        "steps",
        "tool_calls",
        "retries",
        "action_sequence",
        "source_selection",
        "sources_present",
        "budget_exhausted",
        "failure",
        "empty_ranking",
    ):
        assert key in outcome.behavior, key

    summary = summarize_behavior([outcome.behavior])
    assert summary["users"] == 1
    assert 0.0 <= summary["failure_rate"] <= 1.0


def test_select_source_remains_unreachable() -> None:
    """The recorded architecture fact: no similar-item tool, so the frozen rule never offers it."""
    factory, runner, case = _factory([{"action": "finish"}])
    harness = factory(case)

    assert harness.controller.candidate_plane.has_source(CandidateSource.SIMILAR_ITEM) is False
    assert harness.controller.candidate_plane.has_source(CandidateSource.CATALOG_SEARCH) is True
    assert harness.controller.candidate_plane.has_source(CandidateSource.HISTORY) is True

    outcome = runner.run(case)
    assert "select_source" not in outcome.behavior["action_sequence"]
    assert outcome.behavior["source_selection"] == []
