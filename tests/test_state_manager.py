"""Tests for the M5-MVP state manager (replay-based orchestration).

The manager is verified at its public boundary: it replays stored events, stamps the
caller's session, appends through the store's transaction, is idempotent for duplicate
submissions, rejects invalid transitions *before* they reach the append-only store, and
never mixes users.  The reducer itself is covered by ``tests/test_state_beliefs.py``; these
tests assert the orchestration contract, not the belief rules.

Every test uses ``tests.state_fixtures.temporary_database`` (never ``tmp_path``: 0700
temporary directories are unusable on the confined Windows host this project is developed
on) and fixed timestamps, so replay determinism is asserted against content, not clocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.decision_state import empty_state  # noqa: E402
from recommendation.state.errors import (  # noqa: E402
    EventConflictError,
    UnsupportedTransitionError,
)
from recommendation.state.event_store import SQLiteEventStore  # noqa: E402
from recommendation.state.state_manager import StateManager  # noqa: E402
from tests.state_fixtures import (  # noqa: E402
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_TS,
    DEFAULT_USER,
    comparison_event,
    confirmed_event,
    constraint_event,
    context_event,
    explanation_requested_event,
    hypothesis_event,
    preference_stated_event,
    retracted_event,
    temporary_database,
)

OTHER_USER = "user-b"
OTHER_TASK = "task-2"


@pytest.fixture()
def manager_env():
    """Yield ``(StateManager, SQLiteEventStore)`` over one temporary database."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            yield StateManager(store), store
        finally:
            store.close()


def _belief_values(state) -> set[str]:
    """Values of the preference beliefs in a state (test readability helper)."""
    return {belief.value for belief in state.preference_beliefs}


# --------------------------------------------------------------------------- #
# load_or_create and empty state
# --------------------------------------------------------------------------- #


def test_load_or_create_returns_empty_state_for_a_new_task(manager_env) -> None:
    manager, store = manager_env
    state = manager.load_or_create(DEFAULT_USER, "session-current", "task-new")

    expected = empty_state(user_key=DEFAULT_USER, task_id="task-new")
    assert state.user_key == DEFAULT_USER
    assert state.task_id == "task-new"
    assert state.session_id == "session-current"
    assert state.preference_beliefs == ()
    assert state.interaction.event_count == 0
    # Apart from the caller's current session, a new task is the canonical empty state.
    assert state.model_copy(update={"session_id": None}) == expected
    # Loading never writes.
    assert store.count(DEFAULT_USER) == 0

    no_session = manager.load_or_create(DEFAULT_USER, None, "task-new")
    assert no_session.session_id is None
    assert no_session == expected


def test_load_or_create_uses_the_caller_session_over_the_replayed_one(
    manager_env,
) -> None:
    manager, _store = manager_env
    manager.apply_event(
        preference_stated_event(event_id="evt-1", session_id=DEFAULT_SESSION)
    )
    replayed = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert replayed.session_id == DEFAULT_SESSION

    loaded = manager.load_or_create(DEFAULT_USER, "session-current", DEFAULT_TASK)
    assert loaded.session_id == "session-current"
    assert loaded.preference_beliefs == replayed.preference_beliefs
    assert loaded.interaction == replayed.interaction
    assert loaded.state_digest() != replayed.state_digest()

    # Passing None keeps the session recorded by the last task-scoped event.
    kept = manager.load_or_create(DEFAULT_USER, None, DEFAULT_TASK)
    assert kept.session_id == DEFAULT_SESSION
    assert kept == replayed


# --------------------------------------------------------------------------- #
# apply_event, get_state and replay_state agreement
# --------------------------------------------------------------------------- #


def test_apply_event_returns_the_updated_state_and_reads_agree(manager_env) -> None:
    manager, store = manager_env
    event = preference_stated_event(
        event_id="evt-1", user_key=DEFAULT_USER, task_id=DEFAULT_TASK
    )

    state = manager.apply_event(event)

    assert state.user_key == DEFAULT_USER
    assert state.task_id == DEFAULT_TASK
    assert state.session_id == DEFAULT_SESSION
    assert _belief_values(state) == {"lightweight"}
    assert state.preference_beliefs[0].lifecycle.value == "explicit"
    assert state.interaction.event_count == 1
    assert state.interaction.user_turn_count == 1
    assert state.interaction.last_event_id == "evt-1"
    assert state.interaction.last_event_sequence == 1

    assert manager.get_state(DEFAULT_USER, DEFAULT_TASK) == state
    assert manager.replay_state(DEFAULT_USER, DEFAULT_TASK) == state
    assert (
        manager.get_state(DEFAULT_USER, DEFAULT_TASK).state_digest()
        == state.state_digest()
    )
    assert store.count(DEFAULT_USER) == 1


def test_replay_state_reconstructs_the_incrementally_built_state(manager_env) -> None:
    manager, store = manager_env
    events = (
        preference_stated_event(event_id="evt-1", value="lightweight"),
        context_event(event_id="evt-2", intent="recommend"),
        constraint_event(event_id="evt-3", kind="price_max", value="150"),
        comparison_event(event_id="evt-4"),
        explanation_requested_event(event_id="evt-5", attribute="waterproofing"),
    )
    incremental = None
    for event in events:
        incremental = manager.apply_event(event)
    assert incremental is not None

    replayed = manager.replay_state(DEFAULT_USER, DEFAULT_TASK)
    assert replayed == incremental
    assert replayed.state_digest() == incremental.state_digest()
    # Replay is idempotent: same store, same empty-state seed, same state.
    assert manager.replay_state(DEFAULT_USER, DEFAULT_TASK) == replayed
    # A second manager over the same store reconstructs the identical state.
    second = StateManager(store)
    assert second.replay_state(DEFAULT_USER, DEFAULT_TASK) == replayed
    assert (
        second.replay_state(DEFAULT_USER, DEFAULT_TASK).state_digest()
        == replayed.state_digest()
    )


def test_duplicate_apply_event_is_idempotent(manager_env) -> None:
    manager, store = manager_env
    event = preference_stated_event(event_id="evt-dup")

    first = manager.apply_event(event)
    second = manager.apply_event(event)
    assert second == first
    assert second.state_digest() == first.state_digest()

    # A retry with a fresh timestamp is the same semantic event: no new row, no new
    # sequence, identical derived state.
    retried = event.model_copy(update={"timestamp": DEFAULT_TS + 250.0})
    third = manager.apply_event(retried)
    assert third == first
    assert third.state_digest() == first.state_digest()
    assert store.count(DEFAULT_USER) == 1
    assert len(store.get_all_events(DEFAULT_USER)) == 1
    assert store.get_all_events(DEFAULT_USER)[0].sequence == 1


def test_caller_placeholder_sequence_does_not_block_validation_or_ordering(
    manager_env,
) -> None:
    """The store assigns ordering; a caller placeholder must not trip the order guard."""
    manager, store = manager_env
    manager.apply_event(preference_stated_event(event_id="evt-1", value="lightweight"))

    placeholder = preference_stated_event(
        event_id="evt-2", value="blue"
    ).model_copy(update={"sequence": 1})
    state = manager.apply_event(placeholder)

    stored = store.get_event("evt-2")
    assert stored is not None
    assert stored.sequence == 2
    assert state.interaction.last_event_sequence == 2
    assert _belief_values(state) == {"lightweight", "blue"}


# --------------------------------------------------------------------------- #
# Rejected transitions and rejected appends store nothing
# --------------------------------------------------------------------------- #


def test_rejected_transition_stores_nothing_on_an_empty_store(manager_env) -> None:
    manager, store = manager_env
    with pytest.raises(UnsupportedTransitionError):
        manager.apply_event(confirmed_event(event_id="evt-confirm-missing"))

    assert store.count(DEFAULT_USER) == 0
    state = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert state == empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    assert state.state_digest() == empty_state(
        user_key=DEFAULT_USER, task_id=DEFAULT_TASK
    ).state_digest()

    # The rejection did not poison the store: a valid transition still works afterwards,
    # and it takes the first sequence because a rejected event never consumed one.
    recovered = manager.apply_event(
        preference_stated_event(event_id="evt-valid", value="lightweight")
    )
    assert store.count(DEFAULT_USER) == 1
    assert _belief_values(recovered) == {"lightweight"}
    assert store.get_all_events(DEFAULT_USER)[0].sequence == 1


def test_rejected_transition_leaves_the_previous_state_unchanged(manager_env) -> None:
    manager, store = manager_env
    # A real hypothesis exists, but the rejected event confirms a different identity.
    manager.apply_event(
        hypothesis_event(event_id="evt-hyp", value="lightweight", kind="feature")
    )
    before = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert _belief_values(before) == {"lightweight"}

    with pytest.raises(UnsupportedTransitionError):
        manager.apply_event(
            confirmed_event(event_id="evt-confirm-missing", value="heavy")
        )

    # The append-only store was not touched and replay is byte-identical.
    assert store.count(DEFAULT_USER) == 1
    after = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert after == before
    assert after.state_digest() == before.state_digest()


def test_rejected_retraction_stores_nothing(manager_env) -> None:
    """A retraction that matches no belief is a rejected transition, not an event."""
    manager, store = manager_env
    manager.apply_event(preference_stated_event(event_id="evt-1", value="lightweight"))
    before = manager.get_state(DEFAULT_USER, DEFAULT_TASK)

    with pytest.raises(UnsupportedTransitionError):
        manager.apply_event(
            retracted_event(event_id="evt-retract-nothing", value="nonexistent")
        )

    assert store.count(DEFAULT_USER) == 1
    after = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert after == before
    assert after.state_digest() == before.state_digest()


def test_rejected_append_leaves_the_previous_state_unchanged(manager_env) -> None:
    manager, store = manager_env
    manager.apply_event(preference_stated_event(event_id="evt-1", value="lightweight"))
    before = manager.get_state(DEFAULT_USER, DEFAULT_TASK)

    with pytest.raises(EventConflictError):
        manager.apply_event(
            preference_stated_event(event_id="evt-1", value="heavy")
        )

    assert store.count(DEFAULT_USER) == 1
    after = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
    assert after == before
    assert after.state_digest() == before.state_digest()
    assert _belief_values(after) == {"lightweight"}


# --------------------------------------------------------------------------- #
# Cross-user isolation
# --------------------------------------------------------------------------- #


def test_states_are_isolated_per_user(manager_env) -> None:
    manager, store = manager_env
    state_a = manager.apply_event(
        preference_stated_event(event_id="evt-a", user_key="user-a", value="lightweight")
    )
    state_b = manager.apply_event(
        preference_stated_event(event_id="evt-b", user_key=OTHER_USER, value="blue")
    )

    assert state_a.user_key == "user-a"
    assert state_b.user_key == OTHER_USER
    assert _belief_values(state_a) == {"lightweight"}
    assert _belief_values(state_b) == {"blue"}
    assert state_a.state_digest() != state_b.state_digest()
    assert manager.get_state("user-a", DEFAULT_TASK) == state_a
    assert manager.get_state(OTHER_USER, DEFAULT_TASK) == state_b
    assert store.count("user-a") == 1
    assert store.count(OTHER_USER) == 1

    # A duplicate event id under another user is a conflict; the other user's state and
    # row count stay unchanged.
    before = manager.get_state(OTHER_USER, DEFAULT_TASK)
    with pytest.raises(EventConflictError):
        manager.apply_event(
            preference_stated_event(event_id="evt-a", user_key=OTHER_USER)
        )
    assert store.count(OTHER_USER) == 1
    assert manager.get_state(OTHER_USER, DEFAULT_TASK) == before


# --------------------------------------------------------------------------- #
# Long-term events cross tasks
# --------------------------------------------------------------------------- #


def test_long_term_event_from_another_task_is_visible_when_replaying(manager_env) -> None:
    manager, store = manager_env
    manager.apply_event(
        preference_stated_event(
            event_id="evt-a-task", task_id="task-a", value="lightweight"
        )
    )
    manager.apply_event(
        preference_stated_event(
            event_id="evt-a-long-term",
            task_id="task-a",
            scope="long_term",
            value="linen only",
        )
    )

    state_b = manager.replay_state(DEFAULT_USER, OTHER_TASK)
    # The long-term belief crosses the task boundary; the task-a-scoped belief does not.
    assert _belief_values(state_b) == {"linen only"}
    assert {(belief.value, belief.scope.value) for belief in state_b.preference_beliefs} == {
        ("linen only", "long_term")
    }

    loaded = manager.load_or_create(DEFAULT_USER, "session-b", OTHER_TASK)
    assert loaded.session_id == "session-b"
    assert _belief_values(loaded) == {"linen only"}

    # A task-b event is applied on top of the replayed long-term state.
    state_b = manager.apply_event(
        preference_stated_event(
            event_id="evt-b-task", task_id=OTHER_TASK, value="blue"
        )
    )
    assert _belief_values(state_b) == {"linen only", "blue"}
    # task-a still replays its own task-scoped event plus the long-term one.
    state_a = manager.replay_state(DEFAULT_USER, "task-a")
    assert _belief_values(state_a) == {"lightweight", "linen only"}
    assert store.count(DEFAULT_USER) == 3


# --------------------------------------------------------------------------- #
# Replaying a task with no events
# --------------------------------------------------------------------------- #


def test_replaying_a_task_with_no_events_returns_an_empty_state(manager_env) -> None:
    manager, store = manager_env
    manager.apply_event(
        preference_stated_event(event_id="evt-1", task_id=DEFAULT_TASK)
    )

    state = manager.replay_state(DEFAULT_USER, "task-empty")
    expected = empty_state(user_key=DEFAULT_USER, task_id="task-empty")
    assert state == expected
    assert state.state_digest() == expected.state_digest()
    assert manager.get_state(DEFAULT_USER, "task-empty") == expected
    assert store.get_events("task-empty", DEFAULT_USER) == ()
