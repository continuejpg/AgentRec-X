"""Targeted P7-B regression: Planner query must not write M5/M9 preferences.

The planner may rewrite a RETRIEVE query.  That query is a retrieval instrument, not a
new user utterance: it must not add, re-activate, or retract preferences in the
session's preference memory.
"""

from __future__ import annotations

from pathlib import Path

from recommendation.demo.agent_schemas import AgentRecommendRequest
from recommendation.demo.planner_contract import ActionProposal, PlannerAction, PlannerObservation
from recommendation.demo.planner_service_executor import ServicePlannerExecutor
from tests.demo_fixture import PROFILE_A, build_harness, close_harness


def _active_light_preferences(service, user_key: str) -> list[str]:
    snapshot = service._memory_service.get_active_preferences(user_key)
    values: list[str] = []
    for entry in getattr(snapshot, "active_entries", ()) or ():
        text = f"{getattr(entry, 'value', '')} {getattr(entry, 'source_text', '')}".casefold()
        if "light" in text:
            values.append(str(getattr(entry, "value", "")))
    return values


def test_planner_retrieve_query_cannot_reactivate_retracted_preference(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.runtime.sessions.create(PROFILE_A)
        service = harness.runtime.agent_service()
        request = AgentRecommendRequest(
            message="I need a water filter. I prefer lighter options.",
            history=list(session.trusted_user_history),
            k=3,
            sources=("history", "catalog_search"),
            user_key=session.user_key,
        )
        service.run_shopping_turn(
            request, session_id=session.user_key, task_id=session.session_id, turn_id="seed-1"
        )
        assert _active_light_preferences(service, session.user_key), "seed preference was not stored"

        retract = request.model_copy(update={"message": "Forget about the lighter options."})
        service.run_shopping_turn(
            retract, session_id=session.user_key, task_id=session.session_id, turn_id="retract-1"
        )
        assert not _active_light_preferences(service, session.user_key), "retraction failed"

        # Simulate a planner rewrite that contains the retracted preference language.
        payload = request.model_copy(update={"message": "lighter options water filter"})
        executor = ServicePlannerExecutor(
            service=service,
            payload=payload,
            session_id=session.user_key,
            task_id=session.session_id,
            turn_prefix="p7b",
            effective_state={"price_max": None, "goal_type": "water_filter", "role": "main"},
            effective_role_target={"type": "water_filter", "role": "main"},
            state_message="I need a water filter.",
        )
        observation = PlannerObservation(step=0, query="lighter options water filter")
        executor.execute(
            ActionProposal(action=PlannerAction.RETRIEVE, arguments={"query": "lighter options water filter"}),
            observation,
        )
        assert not _active_light_preferences(
            service, session.user_key
        ), "planner RETRIEVE query re-activated a retracted preference"

        # A second retrieval in the same user turn must be equally inert.
        executor.execute(
            ActionProposal(action=PlannerAction.RETRIEVE, arguments={"query": "lighter options water filter"}),
            observation,
        )
        assert not _active_light_preferences(
            service, session.user_key
        ), "repeated planner RETRIEVE re-activated a retracted preference"
    finally:
        close_harness(harness)


def test_planner_retrieve_persists_user_turn_but_not_rewritten_query(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.runtime.sessions.create(PROFILE_A)
        service = harness.runtime.agent_service()
        payload = AgentRecommendRequest(
            message="lighter options water filter",
            history=list(session.trusted_user_history),
            k=3,
            sources=("history", "catalog_search"),
            user_key=session.user_key,
        )
        executor = ServicePlannerExecutor(
            service=service,
            payload=payload,
            session_id=session.user_key,
            task_id=session.session_id,
            turn_prefix="p7b-m9",
            effective_state={"price_max": None, "goal_type": "water_filter", "role": "main"},
            effective_role_target={"type": "water_filter", "role": "main"},
            state_message="I need a water filter. I prefer lighter options.",
        )
        observation = PlannerObservation(step=0, query="lighter options water filter")
        executor.execute(
            ActionProposal(action=PlannerAction.RETRIEVE, arguments={"query": "lighter options water filter"}),
            observation,
        )
        assert _active_light_preferences(
            service, session.user_key
        ), "first RETRIEVE did not persist the real user turn's preference"

        before = _active_light_preferences(service, session.user_key)
        executor.execute(
            ActionProposal(
                action=PlannerAction.RETRIEVE,
                arguments={"query": "forget the lighter options and show water filters"},
            ),
            observation,
        )
        assert _active_light_preferences(service, session.user_key) == before, (
            "a later planner query changed M9 without a new user turn"
        )
    finally:
        close_harness(harness)
