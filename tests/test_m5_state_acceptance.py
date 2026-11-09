"""Independent black-box acceptance tests for M5-MVP (Unified Decision State).

This file is deliberately written against the *documented contract*, not against the
implementation: ``docs/M5_UNIFIED_DECISION_STATE.md`` sections 2-9 plus the frozen
modules ``recommendation.state.{schemas,events,decision_state,errors}``.  It imports the
public modules directly (never the package root) and only uses public APIs:

* ``recommendation.state.event_store.SQLiteEventStore``
* ``recommendation.state.state_manager.StateManager``
* ``recommendation.state.belief_updater.BeliefUpdater``
* ``recommendation.state.state_projector.for_agent/for_ranker/for_constraints``
* ``recommendation.state.legacy_adapter.project_legacy_memory``

One test per required acceptance category, in the milestone's numbering:

01 new state creation                       09 duplicate event submission
02 explicit preference insertion            10 deterministic replay
03 inferred preference creation             11 cross-user isolation
04 inferred -> confirmed                    12 invalid/unauthorized constraints
05 user correction of an inference          13 knowledge-state transitions
06 preference retraction                    14 serialization/deserialization
07 task vs long-term isolation              15 observations never imply preferences
08 conflicting preferences                  16 legacy adapter read-only parity

Every test is independent and deterministic: its own temporary SQLite database (via
``tests.state_fixtures.temporary_database`` -- never pytest's ``tmp_path``), fixed event
timestamps, and explicit event ids.  No model checkpoint, dataset, network or LLM is
used.

A note on the failing-path assertions: the reducer is *pure*, so a rejected transition
must raise and leave the caller's state untouched, and the manager must store nothing.
Those are asserted with the state digest / event count, not just the exception type.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Iterator

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceMemoryEntry,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
)
from recommendation.state import belief_updater, legacy_adapter, state_projector  # noqa: E402
from recommendation.state.decision_state import (  # noqa: E402
    UnifiedDecisionState,
    comparison_observation_id,
    decision_constraint_id,
    empty_state,
    preference_belief_id,
)
from recommendation.state.errors import (  # noqa: E402
    CrossUserStateError,
    EventConflictError,
    UnsupportedTransitionError,
)
from recommendation.state.event_store import SQLiteEventStore  # noqa: E402
from recommendation.state.events import (  # noqa: E402
    PreferenceHypothesisProposed,
    ProductComparisonFeedback,
    StateEvent,
)
from recommendation.state.schemas import (  # noqa: E402
    BeliefOrigin,
    BeliefStrength,
    ConstraintAuthority,
    KnowledgeStatus,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
    STATE_SCHEMA_VERSION,
)
from recommendation.state.state_manager import StateManager  # noqa: E402
from tests.state_fixtures import (  # noqa: E402
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_TS,
    DEFAULT_USER,
    candidate_pool_event,
    comparison_event,
    confirmed_event,
    constraint_event,
    context_event,
    explanation_requested_event,
    explanation_shown_event,
    hypothesis_event,
    preference_stated_event,
    priority_event,
    ranking_snapshot_event,
    retracted_event,
    temporary_database,
    understanding_event,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def open_state_engine(database: Path) -> Iterator[tuple[SQLiteEventStore, StateManager]]:
    """Yield a store and its manager for one temporary database, closing both."""
    store = SQLiteEventStore(database)
    try:
        yield store, StateManager(store)
    finally:
        store.close()


def make_reducer():
    """Return the documented pure reducer (``BeliefUpdater.apply(state, event)``)."""
    updater = belief_updater.BeliefUpdater()
    assert callable(getattr(updater, "apply", None)), (
        "BeliefUpdater must expose apply(state, event) per the contract (section 4)"
    )
    return updater


def find_belief(
    state: UnifiedDecisionState,
    *,
    kind: PreferenceKind | str = PreferenceKind.FEATURE,
    value: str,
    polarity: PreferencePolarity | str = PreferencePolarity.PREFER,
):
    """Return the single belief with this identity (kind/value/polarity)."""
    kind = PreferenceKind(kind)
    polarity = PreferencePolarity(polarity)
    matches = [
        belief
        for belief in state.preference_beliefs
        if belief.kind is kind and belief.value == value and belief.polarity is polarity
    ]
    assert len(matches) == 1, (
        f"expected exactly one belief for ({kind.value}, {value!r}, {polarity.value}); "
        f"got {[(b.kind.value, b.value, b.polarity.value, b.lifecycle.value, b.status.value) for b in state.preference_beliefs]}"
    )
    return matches[0]


def ranker_feature(projection, belief_id: str):
    """Return the single ranker feature carrying ``belief_id``."""
    matches = [feature for feature in projection.features if feature.preference_id == belief_id]
    assert len(matches) == 1, (
        f"expected one ranker feature for {belief_id}; "
        f"got {[f.preference_id for f in projection.features]}"
    )
    return matches[0]


def find_constraint(state: UnifiedDecisionState, constraint_id: str):
    """Return the single constraint in ``state`` with ``constraint_id``."""
    matches = [
        constraint
        for constraint in state.hard_constraints
        if constraint.constraint_id == constraint_id
    ]
    assert len(matches) == 1, (
        f"expected one constraint record for {constraint_id}; "
        f"got {[c.constraint_id for c in state.hard_constraints]}"
    )
    return matches[0]


def agent_preference(projection, belief_id: str):
    """Return the single for_agent preference summary carrying ``belief_id``."""
    matches = [
        summary for summary in projection.preferences if summary.preference_id == belief_id
    ]
    assert len(matches) == 1, (
        f"expected one for_agent preference summary for {belief_id}; "
        f"got {[s.preference_id for s in projection.preferences]}"
    )
    return matches[0]


# --------------------------------------------------------------------------- #
# 01 -- new state creation
# --------------------------------------------------------------------------- #


def test_01_new_state_creation() -> None:
    """A missing task yields a typed, versioned, fully empty state."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        state = manager.load_or_create(DEFAULT_USER, DEFAULT_SESSION, DEFAULT_TASK)

        assert isinstance(state, UnifiedDecisionState)
        assert state.state_version == STATE_SCHEMA_VERSION == 1
        assert state.user_key == DEFAULT_USER
        assert state.task_id == DEFAULT_TASK
        # load_or_create installs the caller's current session on the returned state.
        assert state.session_id == DEFAULT_SESSION

        # Every collection is empty; no context, no references, no counters.
        assert state.preference_beliefs == ()
        assert state.preference_priorities == ()
        assert state.knowledge_beliefs == ()
        assert state.hard_constraints == ()
        assert state.observations == ()
        assert state.candidate_pool_ref is None
        assert state.ranking_snapshot_ref is None
        assert state.task_context.intent == "unknown"
        assert state.task_context.request_text == ""
        assert state.task_context.updated_by_event_id is None
        assert state.interaction.event_count == 0
        assert state.interaction.user_turn_count == 0
        assert state.interaction.last_event_id is None
        assert state.interaction.last_event_sequence == 0

        # The state is exactly the contract's empty state (same canonical digest).
        assert state == empty_state(
            user_key=DEFAULT_USER, task_id=DEFAULT_TASK, session_id=DEFAULT_SESSION
        )
        assert state.state_digest() == empty_state(
            user_key=DEFAULT_USER, task_id=DEFAULT_TASK, session_id=DEFAULT_SESSION
        ).state_digest()

        # No event was written by creating a state.
        assert store.count(DEFAULT_USER) == 0
        assert store.get_events(DEFAULT_TASK, DEFAULT_USER) == ()
        assert store.replay(DEFAULT_TASK, DEFAULT_USER) == ()

        # Replay of a task with no events yields the same empty (session-less) state.
        replayed = manager.get_state(DEFAULT_USER, DEFAULT_TASK)
        assert replayed == empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
        assert replayed.state_digest() == empty_state(
            user_key=DEFAULT_USER, task_id=DEFAULT_TASK
        ).state_digest()

        # load_or_create is repeatable and does not accumulate events.
        assert manager.load_or_create(DEFAULT_USER, DEFAULT_SESSION, DEFAULT_TASK) == state


# --------------------------------------------------------------------------- #
# 02 -- explicit preference insertion
# --------------------------------------------------------------------------- #


def test_02_explicit_preference_insertion() -> None:
    """A user statement creates EXPlICIT/ACTIVE/STRONG belief with provenance."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        event = preference_stated_event(
            event_id="evt-explicit-1",
            kind="feature",
            value="lightweight",
            polarity="prefer",
            scope="task",
            source_text="I want a lightweight backpack.",
        )
        state = manager.apply_event(event)

        belief = find_belief(state, value="lightweight")
        assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
        assert belief.status is PreferenceStatus.ACTIVE
        assert belief.origin is BeliefOrigin.USER_STATED
        assert belief.strength is BeliefStrength.STRONG
        assert belief.scope is PreferenceScope.TASK
        assert belief.is_active is True
        assert belief.is_soft_signal is False
        assert belief.kind is PreferenceKind.FEATURE
        assert belief.value == "lightweight"
        assert belief.polarity is PreferencePolarity.PREFER
        assert belief.supporting_event_ids == ("evt-explicit-1",)
        assert belief.last_event_id == "evt-explicit-1"
        assert belief.legacy_memory_id is None
        # Deterministic identity: (user, task, scope, kind, value, polarity).
        assert belief.belief_id == preference_belief_id(
            user_key=DEFAULT_USER,
            task_id=DEFAULT_TASK,
            scope=PreferenceScope.TASK,
            kind=PreferenceKind.FEATURE,
            value="lightweight",
            polarity=PreferencePolarity.PREFER,
        )

        # Interaction bookkeeping: one event, one user turn, last-applied marker.
        assert state.interaction.event_count == 1
        assert state.interaction.user_turn_count == 1
        assert state.interaction.last_event_id == "evt-explicit-1"
        assert state.interaction.last_event_sequence == 1
        assert state.session_id == DEFAULT_SESSION

        # for_ranker carries it as a hard (non-soft) ordinal signal.
        ranker = state_projector.for_ranker(state)
        assert ranker.ranker_signal_version == 1
        feature = ranker_feature(ranker, belief.belief_id)
        assert feature.kind is PreferenceKind.FEATURE
        assert feature.value == "lightweight"
        assert feature.polarity is PreferencePolarity.PREFER
        assert feature.scope is PreferenceScope.TASK
        assert feature.lifecycle is PreferenceLifecycle.EXPLICIT
        assert feature.is_soft is False
        assert feature.strength_rank == BeliefStrength.STRONG.rank == 3

        # for_agent counts it as an explicit preference; for_constraints ignores it.
        agent = state_projector.for_agent(state)
        assert agent.event_count == 1
        assert agent.user_turn_count == 1
        assert agent.explicit_preference_count == 1
        assert agent.inferred_preference_count == 0
        assert agent_preference(agent, belief.belief_id).is_soft is False
        constraints = state_projector.for_constraints(state)
        assert constraints.constraints == ()
        assert constraints.constraints_version == 1

        # One row, with the store-assigned ordering key; timestamp is metadata only.
        stored = store.get_event("evt-explicit-1")
        assert isinstance(stored, StateEvent)
        assert stored.sequence == 1
        assert stored.payload_hash() == event.payload_hash()
        assert stored.timestamp == DEFAULT_TS
        assert store.count(DEFAULT_USER) == 1


# --------------------------------------------------------------------------- #
# 03 -- inferred preference creation (soft signal, never hard)
# --------------------------------------------------------------------------- #


def test_03_inferred_preference_creation() -> None:
    """A hypothesis becomes INFERRED_HYPOTHESIS soft evidence and never a filter."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        event = hypothesis_event(
            event_id="evt-hyp-1",
            kind="feature",
            value="lightweight",
            polarity="prefer",
            scope="task",
            strength="weak",
            basis="browsing_history",
        )
        state = manager.apply_event(event)

        belief = find_belief(state, value="lightweight")
        assert belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
        assert belief.status is PreferenceStatus.ACTIVE
        assert belief.origin is BeliefOrigin.SYSTEM_INFERRED
        assert belief.strength is BeliefStrength.WEAK
        assert belief.scope is PreferenceScope.TASK
        assert belief.is_soft_signal is True
        assert belief.may_be_committed is False  # soft evidence cannot become memory
        assert belief.supporting_event_ids == ("evt-hyp-1",)

        # A named trusted basis may raise the strength to MODERATE, but never STRONG.
        state = manager.apply_event(
            hypothesis_event(
                event_id="evt-hyp-2",
                kind="feature",
                value="waterproof",
                polarity="prefer",
                strength="moderate",
                basis="catalogue_evidence",
            )
        )
        moderate = find_belief(state, value="waterproof")
        assert moderate.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
        assert moderate.strength is BeliefStrength.MODERATE
        assert moderate.is_soft_signal is True

        # The projection is explicitly soft: the Agent may use it for a clarification.
        ranker = state_projector.for_ranker(state)
        soft_feature = ranker_feature(ranker, belief.belief_id)
        assert soft_feature.is_soft is True
        assert soft_feature.strength_rank == BeliefStrength.WEAK.rank == 1
        assert soft_feature.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
        agent = state_projector.for_agent(state)
        assert agent.inferred_preference_count == 2
        assert agent.explicit_preference_count == 0
        soft_summary = agent_preference(agent, belief.belief_id)
        assert soft_summary.is_soft is True
        assert soft_summary.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS

        # Inference can never become a hard constraint: no field, no filter, no belief.
        assert "hard_constraints" not in PreferenceHypothesisProposed.model_fields
        assert "constraint" not in PreferenceHypothesisProposed.model_fields
        constraints = state_projector.for_constraints(state)
        assert constraints.constraints == ()
        assert constraints.excluded_hypothesis_count == 2
        assert state.hard_constraints == ()

        # The payload itself refuses long-term inference and STRONG inference.
        with pytest.raises(ValidationError):
            hypothesis_event(scope="long_term")
        with pytest.raises(ValidationError):
            hypothesis_event(strength="strong")
        # ... and an inference has no field through which a constraint could be smuggled
        # (the payload model is closed: extra="forbid").
        with pytest.raises(ValidationError):
            PreferenceHypothesisProposed(
                kind="feature",
                value="lightweight",
                basis="browsing_history",
                constraint="price_max<=150",
            )


# --------------------------------------------------------------------------- #
# 04 -- inferred -> confirmed transition
# --------------------------------------------------------------------------- #


def test_04_inferred_to_confirmed_transition() -> None:
    """PREFERENCE_CONFIRMED promotes exactly the inferred hypothesis, and nothing else."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        state = manager.apply_event(
            hypothesis_event(event_id="evt-hyp-1", value="lightweight")
        )
        inferred = find_belief(state, value="lightweight")
        assert inferred.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS

        state = manager.apply_event(
            confirmed_event(
                event_id="evt-confirm-1", value="lightweight", source_text="Yes, lightweight."
            )
        )
        confirmed = find_belief(state, value="lightweight")
        # The same record is promoted; nothing is deleted or duplicated.
        assert len(state.preference_beliefs) == 1
        assert confirmed.belief_id == inferred.belief_id
        assert confirmed.lifecycle is PreferenceLifecycle.CONFIRMED
        assert confirmed.status is PreferenceStatus.ACTIVE
        assert confirmed.origin is BeliefOrigin.USER_CONFIRMED
        assert confirmed.strength is BeliefStrength.STRONG
        # Confirmation preserves the promoted hypothesis's scope: it confirms the same
        # scoped identity rather than inventing a new one.
        assert confirmed.scope is inferred.scope is PreferenceScope.TASK
        assert confirmed.is_soft_signal is False
        assert confirmed.supporting_event_ids == ("evt-hyp-1", "evt-confirm-1")
        assert confirmed.last_event_id == "evt-confirm-1"

        # The ranker sees CONFIRMED (no longer soft, ordinal strength STRONG).
        ranker = state_projector.for_ranker(state)
        feature = ranker_feature(ranker, confirmed.belief_id)
        assert feature.lifecycle is PreferenceLifecycle.CONFIRMED
        assert feature.is_soft is False
        assert feature.strength_rank == 3
        agent = state_projector.for_agent(state)
        assert agent_preference(agent, confirmed.belief_id).lifecycle is PreferenceLifecycle.CONFIRMED

        # Rejections store nothing: confirming with a mismatched scope is not the same
        # identity (a hypothesis is task/session scoped), and confirming an absent
        # hypothesis is not a legal transition either.
        before_digest = state.state_digest()
        before_count = store.count(DEFAULT_USER)
        with pytest.raises(UnsupportedTransitionError):
            manager.apply_event(
                confirmed_event(
                    event_id="evt-confirm-bad-scope",
                    value="lightweight",
                    scope="session",
                )
            )
        with pytest.raises(UnsupportedTransitionError):
            manager.apply_event(
                confirmed_event(event_id="evt-confirm-bad", value="rugged")
            )
        assert store.count(DEFAULT_USER) == before_count
        assert store.get_event("evt-confirm-bad-scope") is None
        assert store.get_event("evt-confirm-bad") is None
        assert manager.get_state(DEFAULT_USER, DEFAULT_TASK).state_digest() == before_digest


# --------------------------------------------------------------------------- #
# 05 -- user correction of an inferred preference
# --------------------------------------------------------------------------- #


def test_05_user_correction_of_inferred_preference() -> None:
    """An explicit opposite-polarity statement retracts the inference, keeping the record."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        state = manager.apply_event(
            hypothesis_event(event_id="evt-hyp-1", value="lightweight", polarity="prefer")
        )
        inferred = find_belief(state, value="lightweight")
        assert inferred.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS

        # The user explicitly states the opposite: "actually, avoid lightweight".
        state = manager.apply_event(
            preference_stated_event(
                event_id="evt-correct-1",
                kind="feature",
                value="lightweight",
                polarity="avoid",
                scope="task",
                source_text="Actually I want to avoid lightweight packs.",
            )
        )

        # The inference is retracted/removed but the record is kept for provenance.
        retracted = find_belief(state, value="lightweight", polarity="prefer")
        assert retracted.belief_id == inferred.belief_id
        assert retracted.lifecycle is PreferenceLifecycle.RETRACTED
        assert retracted.status is PreferenceStatus.REMOVED
        assert retracted.is_active is False
        assert retracted.supporting_event_ids  # provenance is not deleted

        # The explicit correction is active, STRONG and user-stated.
        explicit = find_belief(state, value="lightweight", polarity="avoid")
        assert explicit.lifecycle is PreferenceLifecycle.EXPLICIT
        assert explicit.status is PreferenceStatus.ACTIVE
        assert explicit.origin is BeliefOrigin.USER_STATED
        assert explicit.strength is BeliefStrength.STRONG
        assert state.inferred_hypotheses == ()
        assert len(state.preference_beliefs) == 2  # nothing silently deleted

        # The ranker sees only the active explicit correction, never the retracted guess.
        ranker = state_projector.for_ranker(state)
        assert [feature.preference_id for feature in ranker.features] == [explicit.belief_id]
        feature = ranker_feature(ranker, explicit.belief_id)
        assert feature.polarity is PreferencePolarity.AVOID
        assert feature.lifecycle is PreferenceLifecycle.EXPLICIT
        assert feature.is_soft is False


# --------------------------------------------------------------------------- #
# 06 -- preference retraction
# --------------------------------------------------------------------------- #


def test_06_preference_retraction() -> None:
    """A retraction marks the belief RETRACTED/REMOVED, keeps it, and stops projection."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        state = manager.apply_event(
            preference_stated_event(event_id="evt-explicit-1", value="lightweight")
        )
        belief = find_belief(state, value="lightweight")
        assert belief.is_active is True

        state = manager.apply_event(
            retracted_event(
                event_id="evt-retract-1",
                kind="feature",
                value="lightweight",
                scope="task",
                source_text="Forget the lightweight requirement.",
            )
        )

        retracted = find_belief(state, value="lightweight")
        assert retracted.belief_id == belief.belief_id  # same record, not a new one
        assert retracted.lifecycle is PreferenceLifecycle.RETRACTED
        assert retracted.status is PreferenceStatus.REMOVED
        assert retracted.is_active is False
        assert retracted.supporting_event_ids == ("evt-explicit-1", "evt-retract-1")
        assert retracted.last_event_id == "evt-retract-1"
        assert state.active_preference_beliefs == ()
        assert state.preference_beliefs == (retracted,)  # record kept, not deleted

        # No longer projected: not a ranking signal, not a constraint.
        assert state_projector.for_ranker(state).features == ()
        assert state_projector.for_constraints(state).constraints == ()
        # The event count still reflects both events (and the retraction is a user turn).
        assert state.interaction.event_count == 2
        assert state.interaction.user_turn_count == 2
        assert store.count(DEFAULT_USER) == 2

        # Retracting nothing is a rejected transition that stores nothing.
        before = state.state_digest()
        before_count = store.count(DEFAULT_USER)
        with pytest.raises(UnsupportedTransitionError):
            manager.apply_event(
                retracted_event(event_id="evt-retract-none", kind="brand", value="acme")
            )
        assert store.count(DEFAULT_USER) == before_count
        assert manager.get_state(DEFAULT_USER, DEFAULT_TASK).state_digest() == before


# --------------------------------------------------------------------------- #
# 07 -- task-scope vs long-term isolation
# --------------------------------------------------------------------------- #


def test_07_task_scope_vs_long_term_isolation() -> None:
    """Long-term state crosses tasks; task-scope events can never modify it."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        # A long-term explicit statement made in task-1.
        state1 = manager.apply_event(
            preference_stated_event(
                event_id="evt-lt-1",
                task_id="task-1",
                scope="long_term",
                value="lightweight",
                source_text="I always prefer lightweight packs.",
            )
        )
        long_term = state1.preference_beliefs[0]
        assert long_term.scope is PreferenceScope.LONG_TERM
        assert long_term.lifecycle is PreferenceLifecycle.EXPLICIT
        assert long_term.is_active is True

        # The same event is replayed for a different task of the same user (section 5).
        task2_events = store.get_events("task-2", DEFAULT_USER)
        assert tuple(event.event_id for event in task2_events) == ("evt-lt-1",)
        state2 = manager.get_state(DEFAULT_USER, "task-2")
        task2_belief = state2.preference_beliefs[0]
        assert task2_belief.scope is PreferenceScope.LONG_TERM
        assert task2_belief.is_active is True
        # A long-term belief identity is task-independent (section 2.5).
        assert task2_belief.belief_id == long_term.belief_id

        reducer = make_reducer()

        # A task-scoped event emitted in another task is rejected by the reducer.
        other_task_event = preference_stated_event(
            event_id="evt-task-1", task_id="task-1", scope="task", value="rugged"
        )
        with pytest.raises(UnsupportedTransitionError):
            reducer.apply(state2, other_task_event)

        # A task-scope retraction can only touch task-scope records: with only a
        # long-term belief present it matches nothing and is rejected.
        with pytest.raises(UnsupportedTransitionError):
            reducer.apply(
                state2,
                retracted_event(
                    event_id="evt-retract-task2",
                    task_id="task-2",
                    scope="task",
                    value="lightweight",
                ),
            )
        # The reducer is pure and the stored long-term memory is untouched.
        assert state2.preference_beliefs[0].is_active is True
        assert state2.state_digest() == manager.get_state(DEFAULT_USER, "task-2").state_digest()
        assert (
            manager.get_state(DEFAULT_USER, "task-1").preference_beliefs[0].is_active is True
        )

        # A later long-term statement of the same identity (from another task) updates
        # the same record and keeps the task-independent identity.
        state2 = manager.apply_event(
            preference_stated_event(
                event_id="evt-lt-2",
                task_id="task-2",
                scope="long_term",
                value="lightweight",
            )
        )
        updated = state2.preference_beliefs[0]
        assert len(state2.preference_beliefs) == 1
        assert updated.belief_id == long_term.belief_id
        assert updated.supporting_event_ids == ("evt-lt-1", "evt-lt-2")


# --------------------------------------------------------------------------- #
# 08 -- conflicting preferences
# --------------------------------------------------------------------------- #


def test_08_conflicting_preferences() -> None:
    """Two explicit opposite-polarity statements supersede deterministically, newest wins."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        state = manager.apply_event(
            preference_stated_event(
                event_id="evt-prefer-1", value="lightweight", polarity="prefer"
            )
        )
        prefer = find_belief(state, value="lightweight", polarity="prefer")
        assert prefer.is_active is True

        state = manager.apply_event(
            preference_stated_event(
                event_id="evt-avoid-1",
                value="lightweight",
                polarity="avoid",
                source_text="On second thought, I don't want lightweight.",
            )
        )

        assert len(state.preference_beliefs) == 2  # history kept, nothing deleted
        prefer = find_belief(state, value="lightweight", polarity="prefer")
        avoid = find_belief(state, value="lightweight", polarity="avoid")
        assert prefer.status is PreferenceStatus.SUPERSEDED
        assert prefer.is_active is False
        assert avoid.status is PreferenceStatus.ACTIVE
        assert avoid.lifecycle is PreferenceLifecycle.EXPLICIT
        assert avoid.origin is BeliefOrigin.USER_STATED

        # Provenance is retained: the superseded record still carries the event that
        # created it, and its audit marker advances to the event that superseded it.
        assert "evt-prefer-1" in prefer.supporting_event_ids
        assert prefer.last_event_id == "evt-avoid-1"
        assert avoid.supporting_event_ids == ("evt-avoid-1",)
        assert avoid.last_event_id == "evt-avoid-1"

        # Deterministic "latest wins": only the newest active belief is projected.
        ranker = state_projector.for_ranker(state)
        assert [feature.preference_id for feature in ranker.features] == [avoid.belief_id]
        assert ranker_feature(ranker, avoid.belief_id).polarity is PreferencePolarity.AVOID
        assert state.active_preference_beliefs == (avoid,)
        # ... and the full audited order is first-appearance order.
        assert state.preference_beliefs == (prefer, avoid)


# --------------------------------------------------------------------------- #
# 09 -- duplicate event submission
# --------------------------------------------------------------------------- #


def test_09_duplicate_event_submission() -> None:
    """Same id + same semantics is idempotent; same id + different payload conflicts."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        event = preference_stated_event(event_id="evt-dup-1", value="lightweight")
        state1 = manager.apply_event(event)
        digest1 = state1.state_digest()
        stored1 = store.get_event("evt-dup-1")
        assert store.count(DEFAULT_USER) == 1
        assert stored1.sequence == 1

        # Re-submitting the same event object is idempotent: same state, same row,
        # same sequence -- no duplicate row and no second sequence.
        state2 = manager.apply_event(event)
        assert state2 == state1
        assert state2.state_digest() == digest1
        assert store.count(DEFAULT_USER) == 1
        assert store.get_event("evt-dup-1").sequence == 1

        # A retry with a fresh timestamp is the same semantic event (timestamp is
        # metadata), so it is also idempotent and the stored copy is unchanged.
        retry = preference_stated_event(
            event_id="evt-dup-1", value="lightweight", timestamp=DEFAULT_TS + 100.0
        )
        state3 = manager.apply_event(retry)
        assert state3.state_digest() == digest1
        assert store.count(DEFAULT_USER) == 1
        assert store.get_event("evt-dup-1").timestamp == DEFAULT_TS
        assert state3.interaction.event_count == 1

        # The store exposes the same duplicate semantics when appended directly.
        assert store.append(event).sequence == 1
        assert store.count(DEFAULT_USER) == 1

        # Same id with a different payload is a hard conflict; nothing is stored and
        # the effective state does not change.
        conflicting = preference_stated_event(event_id="evt-dup-1", value="heavy")
        with pytest.raises(EventConflictError):
            manager.apply_event(conflicting)
        assert store.count(DEFAULT_USER) == 1
        assert store.get_event("evt-dup-1").payload_hash() == event.payload_hash()
        assert manager.get_state(DEFAULT_USER, DEFAULT_TASK).state_digest() == digest1

        with pytest.raises(EventConflictError):
            store.append(conflicting)


# --------------------------------------------------------------------------- #
# 10 -- deterministic replay
# --------------------------------------------------------------------------- #


def test_10_deterministic_replay() -> None:
    """Incremental build and full replay from one or two loaders agree byte-for-byte."""
    events = (
        context_event(event_id="evt-10-01"),
        preference_stated_event(event_id="evt-10-02", value="lightweight"),
        hypothesis_event(event_id="evt-10-03", value="waterproof"),
        confirmed_event(event_id="evt-10-04", value="waterproof"),
        constraint_event(event_id="evt-10-05", kind="price_max", value="150"),
        priority_event(event_id="evt-10-06", higher="comfort", lower="weight"),
        comparison_event(event_id="evt-10-07"),
        explanation_requested_event(event_id="evt-10-08", attribute="waterproofing"),
        explanation_shown_event(event_id="evt-10-09", attribute="waterproofing"),
        understanding_event(event_id="evt-10-10", attribute="waterproofing"),
    )

    with temporary_database() as database:
        with open_state_engine(database) as (store, manager):
            incremental = manager.load_or_create(DEFAULT_USER, DEFAULT_SESSION, DEFAULT_TASK)
            for index, event in enumerate(events):
                event = event.model_copy(update={"timestamp": DEFAULT_TS + index})
                incremental = manager.apply_event(event)

            assert incremental.interaction.event_count == len(events)
            replay_1 = manager.replay_state(DEFAULT_USER, DEFAULT_TASK)
            replay_2 = manager.replay_state(DEFAULT_USER, DEFAULT_TASK)
            assert replay_1 == incremental
            assert replay_2 == incremental
            assert (
                replay_1.state_digest()
                == replay_2.state_digest()
                == incremental.state_digest()
            )
            assert replay_1.to_canonical_json() == incremental.to_canonical_json()

            # The ordered stream is sequence-ordered, never timestamp-ordered (a later
            # append may carry a smaller timestamp, but it must not reorder the fold).
            ordered = store.replay(DEFAULT_TASK, DEFAULT_USER)
            assert [event.sequence for event in ordered] == list(
                range(1, len(events) + 1)
            )

            first_digest = incremental.state_digest()
            first_json = incremental.to_canonical_json()

        # ---- second loader, after the first one is closed ------------------ #
        with open_state_engine(database) as (store2, manager2):
            again_a = manager2.replay_state(DEFAULT_USER, DEFAULT_TASK)
            again_b = manager2.replay_state(DEFAULT_USER, DEFAULT_TASK)
            assert again_a == incremental
            assert again_b == incremental
            assert again_a.state_digest() == again_b.state_digest() == first_digest
            assert again_a.to_canonical_json() == first_json

            # A pure manual fold of the store stream produces the identical state too.
            manual = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
            reducer = make_reducer()
            for event in store2.replay(DEFAULT_TASK, DEFAULT_USER):
                manual = reducer.apply(manual, event)
            assert manual == incremental
            assert manual.state_digest() == first_digest


# --------------------------------------------------------------------------- #
# 11 -- cross-user isolation
# --------------------------------------------------------------------------- #


def test_11_cross_user_isolation() -> None:
    """Two users on the same task id never observe each other's events or state."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        shared_task = "task-shared"
        state_a = manager.apply_event(
            preference_stated_event(
                event_id="evt-a-1",
                user_key="user-a",
                task_id=shared_task,
                value="lightweight",
            )
        )
        state_b = manager.apply_event(
            preference_stated_event(
                event_id="evt-b-1",
                user_key="user-b",
                task_id=shared_task,
                value="rugged",
            )
        )

        assert state_a.user_key == "user-a"
        assert state_b.user_key == "user-b"
        assert [event.event_id for event in store.get_events(shared_task, "user-a")] == ["evt-a-1"]
        assert [event.event_id for event in store.get_events(shared_task, "user-b")] == ["evt-b-1"]
        assert store.count("user-a") == 1
        assert store.count("user-b") == 1

        # Replay is user-scoped: each state contains only its owner's belief.
        replay_a = manager.get_state("user-a", shared_task)
        replay_b = manager.get_state("user-b", shared_task)
        assert [belief.value for belief in replay_a.preference_beliefs] == ["lightweight"]
        assert [belief.value for belief in replay_b.preference_beliefs] == ["rugged"]
        assert replay_a.state_digest() != replay_b.state_digest()
        assert replay_a.state_digest() == state_a.state_digest()
        assert replay_b.state_digest() == state_b.state_digest()

        # A third user sees nothing at all.
        state_c = manager.get_state("user-c", shared_task)
        assert state_c == empty_state(user_key="user-c", task_id=shared_task)
        assert store.get_all_events("user-c") == ()

        # The reducer itself refuses an event from another user.
        reducer = make_reducer()
        with pytest.raises(CrossUserStateError):
            reducer.apply(
                replay_a,
                preference_stated_event(
                    event_id="evt-x-1", user_key="user-b", task_id=shared_task, value="heavy"
                ),
            )

        # Reusing an event id across users is a conflict, never a shared row.
        with pytest.raises(EventConflictError):
            manager.apply_event(
                preference_stated_event(
                    event_id="evt-a-1",
                    user_key="user-b",
                    task_id=shared_task,
                    value="lightweight",
                )
            )
        assert store.count("user-a") == 1
        assert store.count("user-b") == 1


# --------------------------------------------------------------------------- #
# 12 -- invalid or unauthorized hard constraints
# --------------------------------------------------------------------------- #


def test_12_invalid_or_unauthorized_hard_constraints() -> None:
    """Only an explicit user constraint event can become a hard constraint."""
    # (a) A constraint is not even representable by the inference payload model.
    assert "scope" in PreferenceHypothesisProposed.model_fields
    assert "hard_constraints" not in PreferenceHypothesisProposed.model_fields
    assert "constraint" not in PreferenceHypothesisProposed.model_fields
    with pytest.raises(ValidationError):
        PreferenceHypothesisProposed(
            kind="feature",
            value="lightweight",
            basis="browsing_history",
            hard_constraint="price_max<=150",
        )

    # (b) Only a user actor may state a constraint: the event construction refuses
    # agent/trusted-component constraint statements (decisive boundary, not reducer).
    with pytest.raises(ValidationError):
        constraint_event(actor="agent")
    with pytest.raises(ValidationError):
        constraint_event(actor="trusted_component")
    with pytest.raises(ValidationError):
        constraint_event(actor="system")

    # (c) Malformed constraint payloads are refused by payload validation.
    with pytest.raises(ValidationError):
        constraint_event(value="x" * 121)  # M9 MAX_VALUE_LENGTH bound
    with pytest.raises(ValidationError):
        constraint_event(value="sk-abcdefghijklmnop")  # credential-like value

    with temporary_database() as database, open_state_engine(database) as (store, manager):
        # A confirmed preference remains a ranking signal, never a filter.
        state = manager.apply_event(hypothesis_event(event_id="evt-h-1", value="rugged"))
        state = manager.apply_event(confirmed_event(event_id="evt-c-1", value="rugged"))
        assert find_belief(state, value="rugged").lifecycle is PreferenceLifecycle.CONFIRMED
        assert state.hard_constraints == ()
        assert state_projector.for_constraints(state).constraints == ()

        # A task-scope constraint cannot overwrite the long-term memory record: both
        # coexist with distinct deterministic ids, and projection precedence reports
        # the suppressed long-term id without mutating it.
        state = manager.apply_event(
            constraint_event(
                event_id="evt-lt-constraint",
                task_id="task-1",
                scope="long_term",
                kind="price_max",
                value="300",
                source_text="My budget is normally up to $300.",
            )
        )
        long_term = state.hard_constraints[0]
        assert long_term.scope is PreferenceScope.LONG_TERM
        assert long_term.authority is ConstraintAuthority.USER_EXPLICIT
        assert long_term.is_authorized_hard is True
        assert long_term.constraint_id == decision_constraint_id(
            user_key=DEFAULT_USER,
            task_id="task-1",
            scope=PreferenceScope.LONG_TERM,
            kind=PreferenceKind.PRICE_MAX,
            value="300",
            polarity=PreferencePolarity.PREFER,
        )

        state2 = manager.apply_event(
            constraint_event(
                event_id="evt-task-constraint",
                task_id="task-2",
                scope="task",
                kind="price_max",
                value="150",
                source_text="For this trip, my maximum budget is $150.",
            )
        )
        records = {record.constraint_id: record for record in state2.hard_constraints}
        assert len(records) == 2  # nothing deleted or overwritten
        kept_long_term = find_constraint(state2, long_term.constraint_id)
        assert kept_long_term == long_term  # the long-term record is unchanged
        task_record = [
            record
            for record in state2.hard_constraints
            if record.scope is PreferenceScope.TASK
        ]
        assert len(task_record) == 1
        assert task_record[0].value == "150"
        assert task_record[0].constraint_id != long_term.constraint_id

        # A task-scope inference must never become a filter: it is left out of the
        # constraint projection (and counted), while the authorized constraints stay.
        state2 = manager.apply_event(
            hypothesis_event(
                event_id="evt-task2-hypothesis", task_id="task-2", value="lightweight"
            )
        )
        assert len(state2.hard_constraints) == 2
        assert {record.value for record in state2.hard_constraints} == {"300", "150"}

        # The projector applies deterministic task-over-long-term precedence for the
        # current task only and reports the suppressed long-term id.
        projection = state_projector.for_constraints(state2)
        assert projection.constraints_version == 1
        assert len(projection.constraints) == 1
        assert projection.constraints[0].value == "150"
        assert projection.constraints[0].scope is PreferenceScope.TASK
        assert long_term.constraint_id in projection.suppressed_long_term_ids
        # No inference ever appears in the constraint projection.
        assert projection.excluded_hypothesis_count == 1
        assert len(state2.inferred_hypotheses) == 1
        # The underlying long-term record remains in the state (already asserted).


# --------------------------------------------------------------------------- #
# 13 -- knowledge-state transitions
# --------------------------------------------------------------------------- #


def test_13_knowledge_state_transitions() -> None:
    """Explanation shown is not understanding, and understanding is never downgraded."""
    attribute = "waterproofing"
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        # UNKNOWN -> NEEDS_EXPLANATION when the user asks what the attribute means.
        state = manager.apply_event(
            explanation_requested_event(event_id="evt-k-1", attribute=attribute)
        )
        knowledge = state.knowledge_beliefs[0]
        assert knowledge.attribute == attribute
        assert knowledge.status is KnowledgeStatus.NEEDS_EXPLANATION
        agent = state_projector.for_agent(state)
        assert attribute in agent.clarification_attributes

        # EXPLANATION_PROVIDED when the agent displays an explanation -- display only.
        state = manager.apply_event(
            explanation_shown_event(
                event_id="evt-k-2", attribute=attribute, explanation_ref="exp-1"
            )
        )
        knowledge = state.knowledge_beliefs[0]
        assert knowledge.status is KnowledgeStatus.EXPLANATION_PROVIDED
        assert knowledge.is_understood is False  # showing does not prove understanding
        assert "evt-k-2" in knowledge.explanation_event_ids
        assert state_projector.for_agent(state).clarification_attributes == ()

        # A second display changes nothing: still not understanding.
        state = manager.apply_event(
            explanation_shown_event(
                event_id="evt-k-3", attribute=attribute, explanation_ref="exp-1"
            )
        )
        assert state.knowledge_beliefs[0].status is KnowledgeStatus.EXPLANATION_PROVIDED
        assert state.knowledge_beliefs[0].is_understood is False

        # Only the user's own demonstration moves it to UNDERSTANDING_DEMONSTRATED;
        # the evidence span is mandatory.
        with pytest.raises(ValidationError):
            understanding_event(event_id="evt-k-bad", attribute=attribute, evidence_text="   ")
        state = manager.apply_event(
            understanding_event(event_id="evt-k-4", attribute=attribute)
        )
        knowledge = state.knowledge_beliefs[0]
        assert knowledge.status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
        assert knowledge.is_understood is True
        assert knowledge.last_event_id == "evt-k-4"

        # A later explanation request or display never silently downgrades it.
        state = manager.apply_event(
            explanation_requested_event(event_id="evt-k-5", attribute=attribute)
        )
        assert state.knowledge_beliefs[0].status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
        state = manager.apply_event(
            explanation_shown_event(event_id="evt-k-6", attribute=attribute)
        )
        assert state.knowledge_beliefs[0].status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
        assert state.knowledge_beliefs[0].is_understood is True

        # Actor authority: only an agent/system may display an explanation and only
        # the user may demonstrate understanding.
        with pytest.raises(ValidationError):
            explanation_shown_event(event_id="evt-k-7", actor="user")
        with pytest.raises(ValidationError):
            understanding_event(event_id="evt-k-8", actor="agent")


# --------------------------------------------------------------------------- #
# 14 -- serialization / deserialization consistency
# --------------------------------------------------------------------------- #


def test_14_serialization_deserialization_consistency() -> None:
    """Canonical JSON round-trips state and events with identical digests/payloads."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        events = (
            context_event(event_id="evt-s-01"),
            preference_stated_event(event_id="evt-s-02", value="lightweight"),
            hypothesis_event(event_id="evt-s-03", value="waterproof"),
            constraint_event(event_id="evt-s-04", kind="price_max", value="150"),
            priority_event(event_id="evt-s-05", higher="comfort", lower="weight"),
            comparison_event(event_id="evt-s-06"),
            candidate_pool_event(event_id="evt-s-07"),
            ranking_snapshot_event(event_id="evt-s-08"),
            explanation_requested_event(event_id="evt-s-09", attribute="waterproofing"),
            explanation_shown_event(event_id="evt-s-10", attribute="waterproofing"),
        )
        state = None
        for event in events:
            state = manager.apply_event(event)
        assert state is not None

        # ---- state: canonical JSON -> model -> byte-identical digest ---------- #
        canonical = state.to_canonical_json()
        assert isinstance(canonical, str)
        assert json.loads(canonical) == state.model_dump(mode="json")
        restored = UnifiedDecisionState.model_validate_json(canonical)
        assert restored == state
        assert restored.state_digest() == state.state_digest()
        assert restored.to_canonical_json() == canonical
        # Canonical: sorted keys, no insignificant whitespace.
        assert json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")) == canonical

        # ---- events: round-trip through JSON with payload and sequence ------- #
        stored_events = store.get_events(DEFAULT_TASK, DEFAULT_USER)
        assert len(stored_events) == len(events)
        assert stored_events == store.replay(DEFAULT_TASK, DEFAULT_USER)
        for stored in stored_events:
            restored_event = StateEvent.model_validate_json(stored.model_dump_json())
            assert restored_event == stored  # includes sequence
            assert restored_event.payload == stored.payload
            assert restored_event.sequence == stored.sequence
            assert restored_event.payload_hash() == stored.payload_hash()
            assert restored_event.semantic_fields() == stored.semantic_fields()
            fetched = store.get_event(stored.event_id)
            assert fetched == stored
            assert fetched.payload_hash() == stored.payload_hash()

        # The replayed state reconstructed after a JSON round-trip of its events is
        # identical, so serialization is not a lossy side path.
        json_stream = [event.model_dump_json() for event in stored_events]
        manual = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
        reducer = make_reducer()
        for raw in json_stream:
            manual = reducer.apply(manual, StateEvent.model_validate_json(raw))
        assert manual == state
        assert manual.state_digest() == state.state_digest()


# --------------------------------------------------------------------------- #
# 15 -- observations never imply preferences
# --------------------------------------------------------------------------- #


def test_15_observations_never_imply_preferences() -> None:
    """Comparison feedback is evidence: it creates no belief, constraint or knowledge."""
    with temporary_database() as database, open_state_engine(database) as (store, manager):
        # Structural guarantee: the observation payload has no preference/feature field.
        assert "kind" not in ProductComparisonFeedback.model_fields
        assert "preference" not in ProductComparisonFeedback.model_fields
        assert "attribute" not in ProductComparisonFeedback.model_fields

        event = comparison_event(
            event_id="evt-obs-1",
            compared_refs=("pool:item-a", "pool:item-b"),
            selected_ref="pool:item-b",
            feedback_text="Item B feels more comfortable.",
        )
        state = manager.apply_event(event)

        assert len(state.observations) == 1
        observation = state.observations[0]
        assert observation.kind.value == "product_comparison"
        assert observation.compared_refs == ("pool:item-a", "pool:item-b")
        assert observation.selected_ref == "pool:item-b"
        assert observation.event_id == "evt-obs-1"
        assert observation.observation_id == comparison_observation_id(event_id="evt-obs-1")
        # The explicit empty tuple is the contract: a choice proves no feature preference.
        assert observation.implied_preferences == ()

        # No belief, priority, constraint or knowledge change of any kind.
        assert state.preference_beliefs == ()
        assert state.preference_priorities == ()
        assert state.hard_constraints == ()
        assert state.knowledge_beliefs == ()
        assert state.inferred_hypotheses == ()
        assert state_projector.for_ranker(state).features == ()
        assert state_projector.for_constraints(state).constraints == ()
        agent = state_projector.for_agent(state)
        assert agent.observation_count == 1
        assert agent.inferred_preference_count == 0
        assert agent.explicit_preference_count == 0

        # Only the user may report a comparison; the actor table is declarative.
        with pytest.raises(ValidationError):
            comparison_event(event_id="evt-obs-2", actor="agent")


# --------------------------------------------------------------------------- #
# 16 -- legacy adapter read-only parity
# --------------------------------------------------------------------------- #


def _m9_entry(
    *,
    memory_id: str,
    user_key: str = DEFAULT_USER,
    kind: PreferenceKind | str = PreferenceKind.FEATURE,
    value: str = "lightweight",
    polarity: PreferencePolarity | str = PreferencePolarity.PREFER,
    status: PreferenceStatus | str = PreferenceStatus.ACTIVE,
    logical_seq: int = 1,
) -> PreferenceMemoryEntry:
    """Build one valid accepted-M9 memory entry (no extractor involved)."""
    return PreferenceMemoryEntry(
        memory_id=memory_id,
        user_key=user_key,
        kind=kind,
        value=value,
        polarity=polarity,
        source_text=f"I stated {value}.",
        source_turn_id=f"turn-{logical_seq}",
        extractor="test",
        status=status,
        logical_seq=logical_seq,
        created_at=float(logical_seq),
    )


def test_16_legacy_adapter_read_only_parity() -> None:
    """M9 active entries project to LONG_TERM explicit beliefs; nothing is written back."""
    memory_store = InMemoryPreferenceStore()
    service = PreferenceMemoryService(memory_store, RuleBasedPreferenceExtractor())
    memory_store.add_entry(_m9_entry(memory_id="mem-1", kind="feature", value="lightweight"))
    memory_store.add_entry(_m9_entry(memory_id="mem-2", kind="brand", value="acme", logical_seq=2))
    # A removed old entry must not be projected at all.
    memory_store.add_entry(
        _m9_entry(
            memory_id="mem-3",
            kind="color",
            value="red",
            status=PreferenceStatus.REMOVED,
            logical_seq=3,
        )
    )
    before = service.get_memory_history(DEFAULT_USER)
    snapshot = service.get_active_preferences(DEFAULT_USER)
    assert len(snapshot.active_entries) == 2  # active-only view (mem-3 excluded)

    with temporary_database() as database, open_state_engine(database) as (store, manager):
        # Event-sourced state first: a long-term explicit belief that must survive the
        # read-only legacy projection ("event-sourced wins").
        state = manager.apply_event(
            preference_stated_event(
                event_id="evt-lt-1",
                scope="long_term",
                kind="feature",
                value="lightweight",
                source_text="I always prefer lightweight packs.",
            )
        )
        event_belief = find_belief(state, value="lightweight")
        assert event_belief.scope is PreferenceScope.LONG_TERM
        assert event_belief.origin is BeliefOrigin.USER_STATED
        event_digest = state.state_digest()

        # Step 1: pure read-only projection of the already-produced M9 snapshot.
        projection = legacy_adapter.project_legacy_memory(snapshot, task_id=state.task_id)
        assert isinstance(projection, legacy_adapter.LegacyMemoryProjection)
        assert projection.user_key == DEFAULT_USER
        assert projection.skipped_memory_ids == ()
        assert len(projection.beliefs) == 2  # active entries only (mem-3 excluded)
        for belief in projection.beliefs:
            assert belief.scope is PreferenceScope.LONG_TERM
            assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
            assert belief.status is PreferenceStatus.ACTIVE
            assert belief.origin is BeliefOrigin.LEGACY_MEMORY
            assert belief.strength is BeliefStrength.STRONG
            assert belief.supporting_event_ids == ()
            assert belief.legacy_memory_id in {"mem-1", "mem-2"}
        projected_by_kind = {belief.kind: belief for belief in projection.beliefs}
        assert set(projected_by_kind) == {PreferenceKind.FEATURE, PreferenceKind.BRAND}
        assert projected_by_kind[PreferenceKind.FEATURE].legacy_memory_id == "mem-1"
        assert projected_by_kind[PreferenceKind.BRAND].value == "acme"
        # The removed old entry is not resurrected by the adapter.
        assert all(belief.value != "red" for belief in projection.beliefs)
        # Projecting is a read: the accepted M9 store and its snapshot are untouched.
        assert service.get_memory_history(DEFAULT_USER).entries == before.entries
        assert service.get_active_preferences(DEFAULT_USER).entries == snapshot.entries

        # Step 2: merge into the event-sourced state; event-sourced beliefs win.
        projected = legacy_adapter.merge_legacy_memory(state, projection)
        assert isinstance(projected, UnifiedDecisionState)
        assert projected.user_key == DEFAULT_USER
        assert projected.task_id == state.task_id

        # Only active old entries become beliefs, all long-term and explicit.  The
        # legacy duplicate of the event-sourced lightweight belief is not re-added.
        added = [
            belief
            for belief in projected.preference_beliefs
            if belief.origin is BeliefOrigin.LEGACY_MEMORY
        ]
        assert [(belief.kind, belief.value) for belief in added] == [
            (PreferenceKind.BRAND, "acme")
        ]
        legacy = added[0]
        assert legacy.scope is PreferenceScope.LONG_TERM
        assert legacy.lifecycle is PreferenceLifecycle.EXPLICIT
        assert legacy.status is PreferenceStatus.ACTIVE
        assert legacy.strength is BeliefStrength.STRONG
        assert legacy.legacy_memory_id == "mem-2"

        # Event-sourced wins: the existing long-term explicit belief is not replaced by
        # the legacy duplicate, and its provenance survives unchanged.
        kept = [
            belief
            for belief in projected.preference_beliefs
            if belief.belief_id == event_belief.belief_id
        ]
        assert len(kept) == 1
        assert kept[0] == event_belief
        assert kept[0].origin is BeliefOrigin.USER_STATED
        assert kept[0].supporting_event_ids == ("evt-lt-1",)
        assert kept[0].legacy_memory_id is None

        # Legacy beliefs are strong, non-soft, long-term signals for the ranker.
        feature = ranker_feature(state_projector.for_ranker(projected), legacy.belief_id)
        assert feature.scope is PreferenceScope.LONG_TERM
        assert feature.lifecycle is PreferenceLifecycle.EXPLICIT
        assert feature.is_soft is False
        assert feature.strength_rank == 3

        # The merge touches only preference_beliefs: no counters, observations, context
        # or trusted references are fabricated, and no event is written.
        assert projected.interaction == state.interaction
        assert projected.observations == state.observations
        assert projected.task_context == state.task_context
        assert store.count(DEFAULT_USER) == 1

        # The adapter is a read-only bridge: the accepted M9 store is untouched, the
        # event-sourced state is unchanged, and the replay is unchanged.
        after = service.get_memory_history(DEFAULT_USER)
        assert after.entries == before.entries
        assert service.get_active_preferences(DEFAULT_USER).entries == snapshot.entries
        assert state.state_digest() == event_digest
        assert manager.get_state(DEFAULT_USER, DEFAULT_TASK).state_digest() == event_digest
