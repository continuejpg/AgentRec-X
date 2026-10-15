"""E-03A: the run-level observation and the deterministic replan gate.

Everything asserted here comes from real runs of the accepted chain (LoopController, CandidatePlane,
GroundingVerifier, ledger, trajectory) through the Demo API: the observation is read from the
response the run produced, and the gate's decision is read from the same response.  Nothing is
inferred from prose, and nothing in this milestone executes a second plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recommendation.api.app import create_app
from recommendation.catalog import MetadataIndex
from recommendation.catalog.metadata import normalize_product_record
from recommendation.control import LoopLimits
from recommendation.demo import build_demo_runtime
from recommendation.demo.agent_schemas import AgentObservationView, ReplanDecisionView
from recommendation.demo.observation import (
    BLOCKING_TERMINATIONS,
    DECISION_SOURCE_RULES,
    REPLAN_REASON_CODES,
    build_observation,
    decide_replan,
    observation_context,
)
from recommendation.memory import SQLitePreferenceStore
from tests.demo_fixture import (
    DemoHarness,
    close_harness,
    demo_profiles,
    make_settings,
)
from tests.test_v2_phase4_joint_acceptance import (  # the E-02C joint fixture
    DIRECT,
    FUSED_BASELINE,
    REQUEST,
    ROWS,
    _Engine,
    _cards,
    _harness,
    _ids,
    _turn,
)

PROVIDER_ENV_KEYS = (
    "AGENTRECX_LLM_BASE_URL",
    "AGENTRECX_LLM_MODEL",
    "AGENTRECX_LLM_API_KEY",
    "AGENTRECX_LLM_PROFILE",
)


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_ENV_KEYS:
        monkeypatch.setenv(name, "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")


def _agent_run(
    harness: DemoHarness, *, message: str = REQUEST, sources: Any = ("history", "catalog_search"),
    k: int = 4,
) -> dict[str, Any]:
    """One request-declared run through the real chain, returning the agent response."""
    session = harness.create_session()["session_id"]
    response = harness.client.post(
        "/v1/demo/agent/recommend",
        json={
            "message": message,
            "k": k,
            "sources": list(sources),
            "history": list(harness.manager.get(session).trusted_user_history),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _observation(observation: AgentObservationView, **overrides: Any) -> AgentObservationView:
    """A copy of a real observation with named fields replaced (rule-level cases only)."""
    return observation.model_copy(update=overrides)


# --------------------------------------------------------------------------- #
# 7.1 enough candidates: nothing to add
# --------------------------------------------------------------------------- #


def test_enough_candidates_do_not_suggest_another_retrieval(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        body = _agent_run(harness)
        observation, decision = body["observation"], body["replan"]

        assert [card["parent_asin"] for card in body["recommendations"]] == FUSED_BASELINE
        assert observation["executed_sources"] == ["history", "catalog_search"]
        assert observation["source_calls"] == {"history": 1, "catalog_search": 1}
        assert observation["source_outcomes"] == {
            "history": "candidates", "catalog_search": "candidates"}
        assert observation["grounded_count"] == 4 and observation["presented_count"] == 4
        assert observation["candidates_sufficient"] is True
        assert observation["termination_reason"] == "completed"
        assert decision["continue_recommended"] is False
        assert decision["reason_code"] == "candidates_sufficient"
        assert decision["decision_source"] == DECISION_SOURCE_RULES
        # The request's k was met exactly, so no source is proposed.
        assert decision["eligible_sources"] == []
    finally:
        close_harness(harness)


def test_exactly_k_presented_counts_as_sufficient(tmp_path: Path, configured: None) -> None:
    """Three products were grounded; the answer asked for two and delivered two."""
    harness = _harness(tmp_path)
    try:
        body = _agent_run(harness, k=2)
        observation, decision = body["observation"], body["replan"]
        assert observation["grounded_count"] == 3 and observation["presented_count"] == 2
        assert observation["requested_k"] == 2
        assert observation["candidates_sufficient"] is True
        assert decision["continue_recommended"] is False
        assert decision["reason_code"] == "candidates_sufficient"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.3 fewer than k: insufficient, and it says only that
# --------------------------------------------------------------------------- #


def test_fewer_than_k_is_reported_as_a_count_not_a_quality_claim(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        body = _agent_run(harness, sources=("history",), k=9)
        observation, decision = body["observation"], body["replan"]
        assert observation["presented_count"] == 4 and observation["requested_k"] == 9
        assert observation["candidates_sufficient"] is False
        assert decision["continue_recommended"] is True
        assert decision["reason_code"] == "candidates_below_k"
        assert "relevance" in (decision["explanation"] or ""), (
            "the gate must say that it counted products and did not judge them"
        )
        assert decision["eligible_sources"] == ["catalog_search"]
        assert decision["remaining_tool_budget"] > 0 and decision["remaining_step_budget"] > 0
        # The suggestion is a decision only: the answer is what the one consulted source produced,
        # in that source's own order (no fusion ran), and the observation agrees with it.
        assert [card["parent_asin"] for card in body["recommendations"]] == [
            "j-red", "j-blue", "j-black", "j-plain"]
        assert observation["presented_count"] == len(body["recommendations"])
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.4 a failed tool, an empty tool and a refused identity stay distinct
# --------------------------------------------------------------------------- #


def test_a_failed_tool_and_an_empty_tool_are_reported_differently(
    tmp_path: Path, configured: None
) -> None:
    empty_harness = _harness(tmp_path / "empty")
    failing_harness = _harness(tmp_path / "failing")
    try:
        # A source that ran and returned nothing.
        empty = _agent_run(
            empty_harness, message="NothingMatchesThisRequest",
            sources=("catalog_search",),
        )
        assert empty["observation"]["source_outcomes"]["catalog_search"] == "empty"
        assert empty["observation"]["empty_sources"] == ["catalog_search"]
        assert empty["observation"]["failed_sources"] == []
        assert empty["observation"]["grounded_count"] == 0

        # A source that actually failed.
        failing_harness.engine.error = RuntimeError("engine exploded")
        failing = _agent_run(failing_harness, sources=("history",))
        assert failing["observation"]["source_outcomes"]["history"] == "failed"
        assert failing["observation"]["failed_sources"] == ["history"]
        assert failing["observation"]["empty_sources"] == []
        assert failing["observation"]["grounded_count"] == 0

        # Both runs really stopped, and the gate refuses to route around the refusal (§4G).
        for body in (empty, failing):
            assert body["recommendations"] == []
            assert body["observation"]["termination_reason"] == "completion_refused"
            assert body["replan"]["continue_recommended"] is False
            assert body["replan"]["reason_code"] == "blocked_by_termination"
    finally:
        close_harness(empty_harness)
        close_harness(failing_harness)


def test_an_ungrounded_identity_is_not_a_candidate(tmp_path: Path, configured: None) -> None:
    """A source can return an identity the trusted mapping does not know; it is never a candidate."""
    from fastapi.testclient import TestClient

    records = [
        normalize_product_record({
            "parent_asin": row[0], "title": row[4],
            **({"details": {"Color": row[3]}} if row[3] else {}),
        })
        for row in ROWS
    ]
    records.append(normalize_product_record(
        {"parent_asin": "j-ghost", "title": "GhostWidget"}))
    engine = _Engine()  # the trusted mapping knows only ROWS
    store = SQLitePreferenceStore(tmp_path / "ungrounded.sqlite3")
    runtime = build_demo_runtime(
        engine=engine, metadata=MetadataIndex.from_records(records), store=store,
        profiles=demo_profiles(), max_sessions=4,
    )
    app = create_app(make_settings(tmp_path), engine=engine, load_on_startup=False, demo=runtime,
                     enable_demo=True)
    harness = DemoHarness(app=app, runtime=runtime, engine=engine, client=TestClient(app),
                          store=store, profiles=runtime.profiles)
    harness.client.__enter__()
    try:
        body = _agent_run(harness, message="GhostWidget RedWidget")
        observation = body["observation"]
        assert body["grounded"]["ungrounded_count"] == 1
        assert observation["ungrounded_count"] == 1
        assert observation["grounded_count"] == 4, (
            "the refused identity is not added to the verified count"
        )
        assert observation["presented_count"] == 4
        assert "j-ghost" not in _ids(body), "and it is never presented"
        # The source itself is still reported as having returned candidates: the finding is that a
        # returned identity did not verify, not that retrieval found nothing.
        assert observation["source_outcomes"]["catalog_search"] == "candidates"
        assert observation["empty_sources"] == []
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.6 / 7.7 budget and sources: the two hard limits
# --------------------------------------------------------------------------- #


def test_an_exhausted_budget_forbids_continuing(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        # One tool call is allowed: the sequential source uses it, the lexical source cannot run.
        harness.runtime.agent_service()._limits = LoopLimits(
            max_steps=3, max_tool_calls=1, max_retries=0
        )
        body = _agent_run(harness, k=9)
        observation, decision = body["observation"], body["replan"]
        assert observation["remaining_tool_calls"] == 0
        assert observation["has_further_budget"] is False
        assert observation["presented_count"] == 4 < observation["requested_k"] == 9
        assert observation["unexecuted_sources"] == ["catalog_search"], (
            "a source really was left untried - the budget is what forbids it"
        )
        assert decision["continue_recommended"] is False
        assert decision["reason_code"] == "budget_exhausted"
    finally:
        close_harness(harness)


def test_exhausted_sources_forbid_continuing(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        body = _agent_run(harness, k=9)  # both sources ran, and k can never be met
        observation, decision = body["observation"], body["replan"]
        assert observation["unexecuted_sources"] == []
        assert observation["has_further_budget"] is True
        assert decision["eligible_sources"] == []
        assert decision["continue_recommended"] is False
        assert decision["reason_code"] == "sources_exhausted"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.8 refusals and invalid plans are never routed around
# --------------------------------------------------------------------------- #


def test_a_safety_refusal_is_not_bypassed_and_an_invalid_plan_never_runs(
    tmp_path: Path, configured: None
) -> None:
    from recommendation.control.model_client import ScriptedModelClient
    from types import SimpleNamespace

    harness = _harness(tmp_path)
    try:
        # A refused run records the refusal, and the gate refuses to continue regardless of budget.
        refused = _agent_run(harness, message="NothingMatchesThisRequest",
                             sources=("catalog_search",))
        assert refused["observation"]["termination_reason"] == "completion_refused"
        assert refused["replan"]["continue_recommended"] is False
        assert refused["replan"]["reason_code"] == "blocked_by_termination"
        assert refused["replan"]["eligible_sources"] == ["history"], (
            "an eligible source exists; the refusal is what forbids using it"
        )

        # An unauthorised source never executes at all: the run is refused by validation, so there
        # is no observation to replan from.
        client = ScriptedModelClient(
            [{"route": "recommend", "selected_sources": ["tiger"], "reason": "x"}],
            repeat_last=True,
        )
        client.profile = SimpleNamespace(name="deepseek")
        client.model = "deepseek-chat"
        harness.runtime.use_llm_client(client)
        session = harness.create_session()["session_id"]
        rejected = harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": REQUEST, "k": 4, "decision_mode": "llm_agent"},
        )
        assert rejected.status_code == 502
        assert rejected.json()["error"] == "llm_invalid_plan"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.9 the observation describes the run's accumulated set
# --------------------------------------------------------------------------- #


def test_the_observation_describes_the_accumulated_candidates(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        body = _agent_run(harness)
        observation = body["observation"]
        assert observation["executed_sources"] == ["history", "catalog_search"]
        assert observation["grounded_count"] == body["grounded"]["grounded_count"] == 4
        assert observation["grounded_count"] == len(body["recommendations"])
        # The ledger itself only holds the plane's sightings, so this count cannot come from the
        # last tool snapshot: it is the run's accumulated verified set.
        ledger_sources = {
            record["source"]
            for candidate in body["grounded"]["candidates"]
            for record in candidate["provenance"]
        }
        assert ledger_sources == {"catalog_search"}
        assert body["grounded"]["multi_source_count"] == 0
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.10 two requests keep their own observation
# --------------------------------------------------------------------------- #


def test_two_requests_do_not_share_an_observation(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        first = _agent_run(harness, sources=("history",), k=4)
        second = _agent_run(harness, sources=("catalog_search",), k=2)

        assert first["observation"]["run_ref"] != second["observation"]["run_ref"]
        assert first["observation"]["executed_sources"] == ["history"]
        assert second["observation"]["executed_sources"] == ["catalog_search"]
        assert first["observation"]["unexecuted_sources"] == ["catalog_search"]
        assert second["observation"]["unexecuted_sources"] == ["history"]
        assert first["observation"]["source_calls"] == {"history": 1}
        assert second["observation"]["source_calls"] == {"catalog_search": 1}
        assert first["observation"]["requested_k"] == 4
        assert second["observation"]["requested_k"] == 2
        assert second["replan"]["eligible_sources"] == ["history"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.11 the observation changes nothing about the answer
# --------------------------------------------------------------------------- #


def test_observing_the_run_changes_nothing_about_the_run(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _turn(harness, session, "I prefer black.", plan=DIRECT)
        ranked, _client = _turn(harness, session, REQUEST)
        # The same session and the same request again: same inputs, so the answer must not move.
        again, _client2 = _turn(harness, session, REQUEST)

        # The observation is derived from the answer's own records, not computed beside them.
        observation = ranked["audit"]["replan"]
        assert observation["reason_code"] in REPLAN_REASON_CODES
        agent_response = _agent_run(harness, sources=("history",))
        derived = agent_response["observation"]
        assert derived["presented_count"] == len(agent_response["recommendations"])
        assert derived["grounded_count"] == agent_response["grounded"]["grounded_count"]
        assert derived["steps_used"] == agent_response["terminal"]["steps"]
        assert derived["tool_calls_used"] == agent_response["terminal"]["tool_calls"]
        # No tool ran because of the observation: candidate calls are a subset of the tool calls the
        # run itself consumed, and every count is a subset of the same run's record.
        assert sum(derived["source_calls"].values()) <= derived["tool_calls_used"]
        # The observation carries no product identity, no title and no score.
        rendered = json.dumps(derived)
        for identity in derived.get("plan", ()):
            assert identity in {"history", "catalog_search"}
        for card in agent_response["recommendations"]:
            assert card["parent_asin"] not in rendered
            assert str(card["item_id"]) not in rendered
        # Two independent identical turns produce identical answers (the memory-ranked one too).
        assert _ids(ranked) == _ids(again)
        assert [card["sasrec_score"] for card in ranked["recommendations"]] == [
            card["sasrec_score"] for card in again["recommendations"]]
        assert [card["fusion_sources"] for card in ranked["recommendations"]] == [
            card["fusion_sources"] for card in again["recommendations"]]
        assert [card["reason"] for card in ranked["recommendations"]] == [
            card["reason"] for card in again["recommendations"]]
        assert [card["evidence"] for card in ranked["recommendations"]] == [
            card["evidence"] for card in again["recommendations"]]
        # And the chat's own decision block is the same decision, with no execution attached.
        assert ranked["audit"]["replan"]["continue_recommended"] is False
        assert ranked["audit"]["replan"]["decision_source"] == DECISION_SOURCE_RULES
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7.12 the other paths are untouched
# --------------------------------------------------------------------------- #


def test_other_paths_carry_no_observation_or_decision(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        deterministic = harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend something.", "k": 4})
        assert deterministic.status_code == 200, deterministic.text
        body = deterministic.json()
        assert body["audit"]["replan"] is None, (
            "the deterministic chat path runs the graph, not the agent service's plan projection"
        )
        direct, _c = _turn(harness, harness.create_session()["session_id"], "hello there",
                           plan=DIRECT)
        assert direct["audit"]["replan"] is None, "a direct turn runs no controller"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# the gate's rule table (cases the accepted controller cannot reach, and the
# precedence between them), exercised on constructed observations
# --------------------------------------------------------------------------- #


def _rule_observation(**overrides: Any) -> AgentObservationView:
    base: dict[str, Any] = {
        "run_ref": "run-1",
        "requested_k": 5,
        "plan": ("history", "catalog_search"),
        "executed_sources": ("history",),
        "unexecuted_sources": ("catalog_search",),
        "source_calls": {"history": 1},
        "source_outcomes": {"history": "candidates"},
        "grounded_count": 1,
        "presented_count": 1,
        "candidates_sufficient": False,
        "termination_status": "finished",
        "termination_reason": "completed",
        "steps_used": 2,
        "steps_limit": 6,
        "tool_calls_used": 1,
        "tool_calls_limit": 4,
        "remaining_steps": 4,
        "remaining_tool_calls": 3,
        "has_further_budget": True,
    }
    base.update(overrides)
    return AgentObservationView(**base)


def test_the_rule_table_covers_the_cases_the_controller_cannot_reach() -> None:
    # §4B: no candidates, budget left, an untried legal source -> a bounded suggestion.
    no_candidates = decide_replan(_rule_observation(presented_count=0, grounded_count=0))
    assert (no_candidates.continue_recommended, no_candidates.reason_code) == (True, "no_candidates")
    assert no_candidates.eligible_sources == ("catalog_search",)

    # §4D: a consulted source failed while the run still completed: a different source may work.
    failed = decide_replan(
        _rule_observation(failed_sources=("history",), source_outcomes={"history": "failed"})
    )
    assert (failed.continue_recommended, failed.reason_code) == (True, "tool_failed")

    # §4A guards the failure rule: a sufficiency claim is skipped when a source failed.
    failed_but_enough = decide_replan(
        _rule_observation(presented_count=5, candidates_sufficient=True,
                          failed_sources=("history",))
    )
    assert (failed_but_enough.continue_recommended, failed_but_enough.reason_code) == (
        True, "tool_failed")

    # §4G: every blocking termination wins over budget, sources and sufficiency.
    for reason in sorted(BLOCKING_TERMINATIONS):
        blocked = decide_replan(
            _rule_observation(termination_reason=reason, presented_count=5,
                              candidates_sufficient=True)
        )
        assert (blocked.continue_recommended, blocked.reason_code) == (
            False, "blocked_by_termination"), reason

    # A run that consulted nothing has nothing to replan.
    idle = decide_replan(_rule_observation(executed_sources=(), source_calls={}))
    assert (idle.continue_recommended, idle.reason_code) == (False, "not_a_recommendation")

    # Every code the gate can return is in its published vocabulary.
    assert {code for code in REPLAN_REASON_CODES} >= {
        no_candidates.reason_code, failed.reason_code, blocked.reason_code, idle.reason_code,
    }


def test_the_gate_never_widens_what_the_run_was_allowed() -> None:
    observation = _rule_observation()
    decision = decide_replan(observation)
    # The decision carries the run's remaining budgets verbatim and can only name sources the
    # observation already listed as legal-but-untried.
    assert decision.remaining_tool_budget == observation.remaining_tool_calls
    assert decision.remaining_step_budget == observation.remaining_steps
    assert set(decision.eligible_sources) <= set(observation.unexecuted_sources)
    assert set(decision.eligible_sources) <= {"history", "catalog_search", "similar_item",
                                              "two_tower", "tiger"}
    # There is no field through which a decision could raise a budget or add a source.
    assert set(ReplanDecisionView.model_fields) == {
        "continue_recommended", "reason_code", "eligible_sources", "remaining_tool_budget",
        "remaining_step_budget", "decision_source", "explanation",
    }


def test_the_decision_projection_discloses_only_counts_and_names() -> None:
    observation = _rule_observation()
    context = observation_context(observation, decide_replan(observation))
    assert set(context) == {
        "requested_k", "executed_sources", "unexecuted_sources", "source_outcomes",
        "grounded_count", "ungrounded_count", "presented_count", "candidates_sufficient",
        "termination_reason", "remaining_tool_budget", "remaining_step_budget",
        "continue_recommended", "reason_code", "decision_source",
    }
    rendered = json.dumps(context)
    for forbidden in ("parent_asin", "item_id", "title", "score", "source_text", "memory_id",
                      "user_key", "trusted_user_history"):
        assert forbidden not in rendered, forbidden


def test_build_observation_needs_no_run_to_have_happened() -> None:
    """The composition reads only the views it is handed - no controller, no store, no network."""
    from types import SimpleNamespace

    terminal = SimpleNamespace(
        run_id="run-x", status="finished", termination_reason="completed", succeeded=True,
        steps=1, tool_calls=1, retries=0, limits={"max_steps": 6, "max_tool_calls": 4},
    )
    grounded = SimpleNamespace(grounded_count=2, ungrounded_count=1, multi_source_count=1)
    eligibility = SimpleNamespace(evaluated=False, verified_eligible_count=0)
    trajectory = SimpleNamespace(steps=[
        SimpleNamespace(action="search_catalog", observation_type="candidate_set",
                        candidate_count=2, refusal_codes=()),
    ])
    observation = build_observation(
        terminal=terminal, grounded=grounded, eligibility=eligibility, trajectory=trajectory,
        plan=("catalog_search",), available_sources=("history", "catalog_search"),
        requested_k=2, presented_count=2,
    )
    assert observation.executed_sources == ("catalog_search",)
    assert observation.unexecuted_sources == ("history",)
    assert observation.source_outcomes == {"catalog_search": "candidates"}
    assert observation.remaining_steps == 5 and observation.remaining_tool_calls == 3
    assert decide_replan(observation).reason_code == "candidates_sufficient"
