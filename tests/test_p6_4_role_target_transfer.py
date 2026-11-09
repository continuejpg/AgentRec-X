"""Targeted regression for P6.4 role-target propagation.

The HTTP shopping route passes two identities to ``run_shopping_turn``:
``session_id=session.user_key`` (the memory namespace) and
``task_id=session.session_id`` (the task owner).  Role state is written under the task
id but consumed through ``payload.user_key``.  This test drives the real service over
the offline demo harness and pins the bridge: after the first turn establishes an
explicit product target, a follow-up turn without type words must still present that
target to the role-filtering layer, and another session must never see it.
"""

from __future__ import annotations

from pathlib import Path

from recommendation.demo.agent_schemas import AgentRecommendRequest
from tests.demo_fixture import PROFILE_A, build_harness, close_harness


def _turn(harness, session, *, message: str, turn_number: int):
    service = harness.runtime.agent_service()
    payload = AgentRecommendRequest(
        message=message,
        history=list(session.trusted_user_history),
        k=3,
        sources=("history", "catalog_search"),
        user_key=session.user_key,
    )
    return service.run_shopping_turn(
        payload,
        session_id=session.user_key,
        task_id=session.session_id,
        turn_id=session.turn_id_for(turn_number),
    )


def test_role_target_reaches_rule_path_and_stays_session_scoped(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        first = harness.runtime.sessions.create(PROFILE_A)
        _turn(
            harness,
            first,
            message="I need a water filter under $30, please show me options.",
            turn_number=1,
        )
        service = harness.runtime.agent_service()
        first_target = service.role_diagnostics(first.user_key)
        assert first_target is not None, "the first turn must establish a role target"
        assert first_target["target"]["type"] == "water_filter"

        # A follow-up turn has no product-type words.  Without the session bridge the
        # role stage would read no target and stop filtering; with it the same target is
        # still the effective one.
        _turn(harness, first, message="Show me more options.", turn_number=2)
        follow_up_target = service.role_diagnostics(first.user_key)
        assert follow_up_target is not None, "the follow-up turn lost the session role target"
        assert follow_up_target["target"]["type"] == "water_filter"
        assert follow_up_target["target"]["role"] == "main"

        # A second session has its own namespace and must not inherit the first session's
        # role target before it has ever run a turn.
        second = harness.runtime.sessions.create(PROFILE_A)
        assert second.user_key != first.user_key
        assert service.role_diagnostics(second.user_key) is None
        assert first_target["target"]["type"] == "water_filter"
    finally:
        close_harness(harness)
