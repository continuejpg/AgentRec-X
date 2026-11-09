"""P1B-0: verify the real M5 -> R1 input channel.

These tests seed the M5 event store directly, then exercise the real multi-turn
shopping route.  Shadow must leave the legacy planner input unchanged; apply_constraints
must expose the trusted M5 hard set to R1 and must keep soft preferences soft.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state import shopping_adapter  # noqa: E402
from recommendation.state.schemas import PreferenceKind  # noqa: E402
from tests.test_m7_grounded_actions import (  # noqa: E402
    _create_session,
    _harness,
    _shopping,
    _task_id,
)


def _seed_constraint(harness: Any, session_id: str, *, kind: PreferenceKind, value: str) -> tuple[str, str]:
    user_key, task_id = _task_id(harness, session_id)
    integration = harness.runtime.agent_service()._state_integration
    assert integration is not None
    integration.integrate_turn(
        user_key=user_key,
        session_id=task_id,
        task_id=task_id,
        turn_key="seed:hard",
        message="seed an explicit remembered requirement",
        intent_label="recommend",
        preferences=(),
        constraints=(
            shopping_adapter.ExplicitConstraint(
                kind=kind, value=value, source_text="seed requirement"
            ),
        ),
        trusted_constraints=(),
    )
    return user_key, task_id


def _seed_soft(harness: Any, session_id: str, *, kind: PreferenceKind, value: str) -> tuple[str, str]:
    user_key, task_id = _task_id(harness, session_id)
    integration = harness.runtime.agent_service()._state_integration
    assert integration is not None
    integration.integrate_turn(
        user_key=user_key,
        session_id=task_id,
        task_id=task_id,
        turn_key="seed:soft",
        message="seed an explicit soft preference",
        intent_label="recommend",
        preferences=(shopping_adapter.ExplicitPreference(kind, value),),
        constraints=(),
        trusted_constraints=(),
    )
    return user_key, task_id


def test_applied_m5_hard_constraint_enters_the_r1_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    with _harness(
        tmp_path,
        monkeypatch,
        name="p1b-hard",
        state_mode="apply_constraints",
    ) as (harness, _):
        session_id = _create_session(harness)
        _user_key, task_id = _seed_constraint(
            harness, session_id, kind=PreferenceKind.COLOR, value="red"
        )
        response = _shopping(
            harness,
            session_id,
            "show me options",
            sources=["history", "catalog_search"],
            preference_discovery=False,
        )
        assert response.status_code == 200, response.text
        diagnostic = harness.runtime.agent_service().r1_diagnostics(task_id)
        assert diagnostic is not None
        assert diagnostic["constraints"] == [{"kind": "color", "value": "red"}]
        assert [item["parent_asin"] for item in response.json()["recommendations"]] == [
            "cand-red"
        ]


def test_shadow_does_not_consume_the_seeded_m5_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    with _harness(
        tmp_path,
        monkeypatch,
        name="p1b-shadow",
        state_mode="shadow",
    ) as (harness, _):
        session_id = _create_session(harness)
        _user_key, task_id = _seed_constraint(
            harness, session_id, kind=PreferenceKind.COLOR, value="red"
        )
        response = _shopping(
            harness,
            session_id,
            "show me options",
            sources=["history", "catalog_search"],
            preference_discovery=False,
        )
        assert response.status_code == 200, response.text
        diagnostic = harness.runtime.agent_service().r1_diagnostics(task_id)
        assert diagnostic is not None
        assert diagnostic["constraints"] == []
        assert len(response.json()["recommendations"]) > 1


def test_applied_m5_soft_preference_is_not_promoted_to_a_hard_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    with _harness(
        tmp_path,
        monkeypatch,
        name="p1b-soft",
        state_mode="apply_constraints",
    ) as (harness, _):
        session_id = _create_session(harness)
        _user_key, task_id = _seed_soft(
            harness, session_id, kind=PreferenceKind.FEATURE, value="red"
        )
        response = _shopping(
            harness,
            session_id,
            "show me options",
            sources=["history", "catalog_search"],
            preference_discovery=False,
        )
        assert response.status_code == 200, response.text
        diagnostic = harness.runtime.agent_service().r1_diagnostics(task_id)
        assert diagnostic is not None
        assert diagnostic["constraints"] == []
        assert any(
            item["value"] == "red"
            for item in diagnostic.get("m5_soft_preferences", [])
        )
        assert len(response.json()["recommendations"]) > 1


def test_retracted_m5_constraint_is_not_consumed_by_r1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    with _harness(
        tmp_path,
        monkeypatch,
        name="p1b-retract",
        state_mode="apply_constraints",
    ) as (harness, _):
        session_id = _create_session(harness)
        user_key, task_id = _seed_constraint(
            harness, session_id, kind=PreferenceKind.COLOR, value="red"
        )
        first = _shopping(
            harness,
            session_id,
            "show me options",
            sources=["history", "catalog_search"],
            preference_discovery=False,
        )
        assert first.status_code == 200, first.text
        assert harness.runtime.agent_service().r1_diagnostics(task_id)["constraints"] == [
            {"kind": "color", "value": "red"}
        ]
        integration = harness.runtime.agent_service()._state_integration
        assert integration is not None
        integration.integrate_turn(
            user_key=user_key,
            session_id=task_id,
            task_id=task_id,
            turn_key="retract:hard",
            message="forget the red requirement",
            intent_label="recommend",
            preferences=(),
            constraints=(),
            retractions=(shopping_adapter.ExplicitRetraction(kind=PreferenceKind.COLOR, value="red"),),
            trusted_constraints=(),
        )
        second = _shopping(
            harness,
            session_id,
            "show me options",
            sources=["history", "catalog_search"],
            preference_discovery=False,
        )
        assert second.status_code == 200, second.text
        diagnostic = harness.runtime.agent_service().r1_diagnostics(task_id)
        assert diagnostic is not None
        assert diagnostic["constraints"] == []
        assert diagnostic["action"] == "RETRIEVE"
        assert diagnostic["reason"] == "m5_effective_constraints_changed"
        assert len(second.json()["recommendations"]) > 1


def test_unmatched_m5_retraction_is_not_invented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    with _harness(
        tmp_path,
        monkeypatch,
        name="p1b-unmatched",
        state_mode="apply_constraints",
    ) as (harness, database):
        session_id = _create_session(harness)
        user_key, task_id = _task_id(harness, session_id)
        integration = harness.runtime.agent_service()._state_integration
        assert integration is not None
        outcome = integration.integrate_turn(
            user_key=user_key,
            session_id=task_id,
            task_id=task_id,
            turn_key="retract:missing",
            message="forget a preference that was never established",
            intent_label="recommend",
            preferences=(),
            constraints=(),
            retractions=(shopping_adapter.ExplicitRetraction(value="weight"),),
            trusted_constraints=(),
        )
        assert outcome is not None
        assert outcome.diagnostics["retraction_match"]["resolved"] == 0
        diagnostic = harness.runtime.agent_service().r1_diagnostics(task_id)
        assert diagnostic is None or diagnostic.get("constraints") == []
        from recommendation.state import SQLiteEventStore

        store = SQLiteEventStore(database)
        try:
            event_types = [event.event_type.value for event in store.get_events(task_id, user_key)]
        finally:
            store.close()
        assert "preference_retracted" not in event_types


