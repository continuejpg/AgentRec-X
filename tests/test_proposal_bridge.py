from __future__ import annotations

import json

from recommendation.demo.proposal_bridge import map_proposal
from recommendation.state import SQLiteEventStore, StateManager, shopping_adapter
from recommendation.state.shopping_adapter import TurnIdentity
from tests.state_fixtures import temporary_database

USER = "proposal-user"
SESSION = "proposal-session"
TASK = "proposal-task"


def _identity(turn: str) -> TurnIdentity:
    return TurnIdentity(user_key=USER, session_id=SESSION, task_id=TASK, turn_key=turn)


def _proposal(ops):
    return {"schema_version": "agentrecx.preference_operation.v1", "operations": ops, "clarification": None}


def test_valid_price_proposal_changes_m5_effective_state_and_replays() -> None:
    message = "Actually make the budget under $50."
    payload = _proposal([{"op": "update_constraint", "attribute": "price_max", "value": 50, "direction": None, "scope": "task", "evidence": "under $50"}])
    actions = map_proposal(payload, message)
    assert actions.valid
    assert len(actions.constraints) == 1
    events = shopping_adapter.build_turn_events(
        identity=_identity("t1"),
        message=message,
        intent_label="recommend",
        constraints=actions.constraints,
        preferences=actions.preferences,
        retractions=actions.retractions,
        timestamp=1.0,
    )
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        manager = StateManager(store)
        manager.load_or_create(USER, SESSION, TASK)
        for event in events:
            manager.apply_event(event)
        state = manager.replay_state(USER, TASK)
        assert [c.value for c in state.authorized_hard_constraints] == ["50"]
        # Re-applying the same identities is idempotent.
        count = store.count(USER)
        for event in events:
            manager.apply_event(event)
        assert store.count(USER) == count
        assert manager.replay_state(USER, TASK).state_digest() == state.state_digest()
        store.close()


def test_invalid_evidence_proposal_is_rejected_without_actions() -> None:
    actions = map_proposal(
        _proposal([{"op": "set_constraint", "attribute": "price_max", "value": 50, "direction": None, "scope": "task", "evidence": "not spoken"}]),
        "Actually make the budget under $50.",
    )
    assert not actions.valid
    assert actions.constraints == ()
    assert actions.errors


def test_retraction_does_not_revive_superseded_budget() -> None:
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        manager = StateManager(store)
        manager.load_or_create(USER, SESSION, TASK)
        for turn, value, evidence in [("t1", 30, "under 30"), ("t2", 10, "under 10"), ("t3", 50, "under 50")]:
            actions = map_proposal(_proposal([{"op": "update_constraint", "attribute": "price_max", "value": value, "direction": None, "scope": "task", "evidence": evidence}]), evidence)
            for event in shopping_adapter.build_turn_events(identity=_identity(turn), message=evidence, intent_label="recommend", constraints=actions.constraints, preferences=actions.preferences, retractions=actions.retractions, timestamp=float(len(turn))):
                manager.apply_event(event)
        assert [c.value for c in manager.replay_state(USER, TASK).authorized_hard_constraints] == ["50"]
        retraction = map_proposal(_proposal([{"op": "retract_constraint", "attribute": "price_max", "value": None, "direction": None, "scope": "task", "evidence": "Forget the budget."}]), "Forget the budget.")
        for event in shopping_adapter.build_turn_events(identity=_identity("t4"), message="Forget the budget.", intent_label="recommend", constraints=retraction.constraints, preferences=retraction.preferences, retractions=retraction.retractions, timestamp=5.0):
            manager.apply_event(event)
        assert manager.replay_state(USER, TASK).authorized_hard_constraints == ()
        store.close()


def test_noop_and_clarify_do_not_create_state_updates() -> None:
    for text, op in [("Carry on with the current plan.", "noop"), ("Help me choose a category.", "clarify")]:
        payload = _proposal([{"op": op, "attribute": "product_type" if op == "clarify" else None, "value": None, "direction": None, "scope": "task" if op == "clarify" else None, "evidence": text, "reason": "declined_preference" if op == "noop" else None}])
        if op == "clarify":
            payload["clarification"] = {"attribute": "product_type", "question": "Which category?", "choices": ["filter", "cookware"]}
        actions = map_proposal(payload, text)
        assert actions.valid
        assert actions.preferences == () and actions.constraints == () and actions.retractions == ()


def test_same_message_different_valid_proposals_change_effective_state() -> None:
    message = "Use a budget of $30 or $50."
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        manager = StateManager(store)
        manager.load_or_create(USER, SESSION, TASK)
        for turn, value in [("t1", 30), ("t2", 50)]:
            actions = map_proposal(
                _proposal([{"op": "update_constraint", "attribute": "price_max", "value": value, "direction": None, "scope": "task", "evidence": message}]),
                message,
            )
            for event in shopping_adapter.build_turn_events(
                identity=_identity(turn), message=message, intent_label="recommend",
                constraints=actions.constraints, preferences=actions.preferences,
                retractions=actions.retractions, timestamp=float(len(turn)),
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER, TASK)
            assert [c.value for c in state.authorized_hard_constraints] == [str(value)]
        store.close()
