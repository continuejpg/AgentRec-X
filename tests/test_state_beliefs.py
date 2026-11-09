"""Transition-rule tests for the deterministic M5 belief updater.

These tests drive :class:`~recommendation.state.belief_updater.BeliefUpdater` directly
with validated events (no event store and no state manager), so each rule in section 4 of
``docs/M5_UNIFIED_DECISION_STATE.md`` is pinned to observable state.  The properties
under test are the milestone's hard guarantees:

* explicit statements create authoritative beliefs, inferences stay soft and scoped;
* the user always wins: an inference never contradicts or overwrites an explicit belief,
  and retraction/supersession keep the record for provenance instead of deleting it;
* task/session events never touch long-term records, while long-term events cross tasks;
* ordering and duplicate rules are enforced by the reducer itself;
* comparison feedback is evidence only -- it cannot create a preference or a constraint;
* knowledge display is not understanding;
* replay is deterministic (same stream twice -> identical state digest).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommendation.state.belief_updater import BeliefUpdater
from recommendation.state.decision_state import (
    empty_state,
    knowledge_belief_id,
    preference_belief_id,
)
from recommendation.state.errors import CrossUserStateError, UnsupportedTransitionError
from recommendation.state.schemas import (
    BeliefOrigin,
    BeliefStrength,
    ConstraintAuthority,
    KnowledgeStatus,
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)
from tests.state_fixtures import (
    DEFAULT_SESSION,
    DEFAULT_TASK,
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
    understanding_event,
)

UPDATER = BeliefUpdater()


def _empty_state(
    *, task_id: str = DEFAULT_TASK, user_key: str = DEFAULT_USER
):
    """Empty state for the default user/task (or an explicitly named scope)."""
    return empty_state(user_key=user_key, task_id=task_id)


def _apply(state, event):
    """Apply one event through the reducer under test."""
    return UPDATER.apply(state, event)


# --------------------------------------------------------------------------- #
# Explicit beliefs
# --------------------------------------------------------------------------- #


def test_explicit_statement_creates_strong_active_belief() -> None:
    event = preference_stated_event(event_id="evt-explicit", value="lightweight")
    state = _apply(_empty_state(), event)

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
    assert belief.status is PreferenceStatus.ACTIVE
    assert belief.strength is BeliefStrength.STRONG
    assert belief.origin is BeliefOrigin.USER_STATED
    assert belief.scope is PreferenceScope.TASK
    assert belief.kind is PreferenceKind.FEATURE
    assert belief.polarity is PreferencePolarity.PREFER
    assert belief.supporting_event_ids == ("evt-explicit",)
    assert belief.last_event_id == "evt-explicit"
    assert belief.belief_id == preference_belief_id(
        user_key=DEFAULT_USER,
        task_id=DEFAULT_TASK,
        scope=PreferenceScope.TASK,
        kind=PreferenceKind.FEATURE,
        value="lightweight",
        polarity=PreferencePolarity.PREFER,
    )


def test_explicit_statement_honours_direction() -> None:
    event = preference_stated_event(
        event_id="evt-direction", kind="feature", value="weight", direction="minimize"
    )
    state = _apply(_empty_state(), event)

    (belief,) = state.preference_beliefs
    assert belief.direction is PreferenceDirection.MINIMIZE


def test_repeated_explicit_statement_keeps_one_record_and_appends_provenance() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-repeat-1", value="lightweight"),
    )
    state = _apply(
        state, preference_stated_event(event_id="evt-repeat-2", value="lightweight")
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
    assert belief.supporting_event_ids == ("evt-repeat-1", "evt-repeat-2")
    assert belief.last_event_id == "evt-repeat-2"


# --------------------------------------------------------------------------- #
# Inferred hypotheses
# --------------------------------------------------------------------------- #


def test_hypothesis_is_soft_task_scoped_inference() -> None:
    event = hypothesis_event(event_id="evt-hypothesis", strength="moderate")
    state = _apply(_empty_state(), event)

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
    assert belief.status is PreferenceStatus.ACTIVE
    assert belief.strength is BeliefStrength.MODERATE
    assert belief.origin is BeliefOrigin.SYSTEM_INFERRED
    assert belief.scope is PreferenceScope.TASK
    assert belief.supporting_event_ids == ("evt-hypothesis",)
    assert belief.is_soft_signal


def test_hypothesis_payload_rejects_long_term_and_strong() -> None:
    with pytest.raises(ValidationError):
        hypothesis_event(scope="long_term")
    with pytest.raises(ValidationError):
        hypothesis_event(strength="strong")


def test_hypothesis_repeated_with_same_identity_updates_strength_in_place() -> None:
    state = _apply(
        _empty_state(), hypothesis_event(event_id="evt-h1", strength="weak")
    )
    state = _apply(
        state, hypothesis_event(event_id="evt-h2", strength="moderate")
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
    assert belief.strength is BeliefStrength.MODERATE
    assert belief.supporting_event_ids == ("evt-h1", "evt-h2")


def test_hypothesis_cannot_contradict_explicit_belief() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-user", value="lightweight"),
    )
    digest_before = state.state_digest()

    with pytest.raises(UnsupportedTransitionError):
        _apply(
            state,
            hypothesis_event(
                event_id="evt-contradiction",
                value="lightweight",
                polarity="avoid",
            ),
        )

    assert state.state_digest() == digest_before
    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
    assert belief.polarity is PreferencePolarity.PREFER
    assert belief.supporting_event_ids == ("evt-user",)


def test_hypothesis_of_same_polarity_keeps_explicit_belief() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-user", value="lightweight"),
    )
    state = _apply(
        state, hypothesis_event(event_id="evt-agree", value="lightweight")
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
    assert belief.origin is BeliefOrigin.USER_STATED
    assert belief.supporting_event_ids == ("evt-user", "evt-agree")


# --------------------------------------------------------------------------- #
# Confirmation
# --------------------------------------------------------------------------- #


def test_confirmation_promotes_active_hypothesis() -> None:
    state = _apply(
        _empty_state(), hypothesis_event(event_id="evt-hypothesis")
    )
    state = _apply(
        state, confirmed_event(event_id="evt-confirm", value="lightweight")
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.CONFIRMED
    assert belief.status is PreferenceStatus.ACTIVE
    assert belief.strength is BeliefStrength.STRONG
    assert belief.origin is BeliefOrigin.USER_CONFIRMED
    assert belief.supporting_event_ids == ("evt-hypothesis", "evt-confirm")


def test_confirmation_without_hypothesis_is_refused() -> None:
    with pytest.raises(UnsupportedTransitionError):
        _apply(
            _empty_state(), confirmed_event(event_id="evt-confirm", value="lightweight")
        )


def test_confirmation_of_explicit_belief_is_refused() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-user", value="lightweight"),
    )
    with pytest.raises(UnsupportedTransitionError):
        _apply(state, confirmed_event(event_id="evt-confirm", value="lightweight"))


# --------------------------------------------------------------------------- #
# Explicit statements versus existing inferences
# --------------------------------------------------------------------------- #


def test_explicit_statement_upgrades_same_identity_hypothesis_in_place() -> None:
    state = _apply(
        _empty_state(), hypothesis_event(event_id="evt-hypothesis")
    )
    state = _apply(
        state,
        preference_stated_event(event_id="evt-explicit", value="lightweight"),
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
    assert belief.strength is BeliefStrength.STRONG
    assert belief.origin is BeliefOrigin.USER_STATED
    assert belief.supporting_event_ids == ("evt-hypothesis", "evt-explicit")


def test_explicit_opposite_polarity_retracts_inferred_hypothesis() -> None:
    state = _apply(
        _empty_state(),
        hypothesis_event(event_id="evt-hypothesis", value="lightweight", polarity="prefer"),
    )
    state = _apply(
        state,
        preference_stated_event(
            event_id="evt-explicit", value="lightweight", polarity="avoid"
        ),
    )

    assert len(state.preference_beliefs) == 2
    retracted, explicit = state.preference_beliefs
    assert retracted.lifecycle is PreferenceLifecycle.RETRACTED
    assert retracted.status is PreferenceStatus.REMOVED
    assert retracted.supporting_event_ids == ("evt-hypothesis", "evt-explicit")
    assert explicit.lifecycle is PreferenceLifecycle.EXPLICIT
    assert explicit.status is PreferenceStatus.ACTIVE
    assert explicit.polarity is PreferencePolarity.AVOID


# --------------------------------------------------------------------------- #
# Retraction
# --------------------------------------------------------------------------- #


def test_retraction_keeps_record_with_removed_status() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-explicit", value="lightweight"),
    )
    state = _apply(
        state, retracted_event(event_id="evt-retract", value="lightweight")
    )

    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.RETRACTED
    assert belief.status is PreferenceStatus.REMOVED
    assert belief.supporting_event_ids == ("evt-explicit", "evt-retract")
    assert belief.last_event_id == "evt-retract"


def test_retraction_without_match_is_refused() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-explicit", value="lightweight"),
    )
    with pytest.raises(UnsupportedTransitionError):
        _apply(state, retracted_event(event_id="evt-retract", value="cotton"))


def test_kind_wide_retraction_removes_every_active_value() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-red", kind="color", value="red"),
    )
    state = _apply(
        state,
        preference_stated_event(event_id="evt-blue", kind="color", value="blue"),
    )
    state = _apply(
        state,
        retracted_event(event_id="evt-retract-kind", kind="color", value=None),
    )

    assert len(state.preference_beliefs) == 2
    for belief in state.preference_beliefs:
        assert belief.lifecycle is PreferenceLifecycle.RETRACTED
        assert belief.status is PreferenceStatus.REMOVED


def test_targeted_retraction_supersedes_the_matching_hard_constraint_only() -> None:
    """A requirement represented as both belief and constraint is withdrawn together."""
    state = _apply(
        _empty_state(),
        preference_stated_event(
            event_id="evt-mouthpiece-pref",
            value="replacement mouthpiece for a water filter",
        ),
    )
    state = _apply(
        state,
        preference_stated_event(
            event_id="evt-lightweight-pref",
            value="lightweight",
        ),
    )
    state = _apply(
        state,
        constraint_event(
            event_id="evt-mouthpiece-constraint",
            kind="feature",
            value="replacement mouthpiece for a water filter",
        ),
    )
    state = _apply(
        state,
        retracted_event(
            event_id="evt-mouthpiece-retract",
            value="replacement mouthpiece for a water filter",
        ),
    )

    assert [belief.value for belief in state.active_preference_beliefs] == ["lightweight"]
    assert state.authorized_hard_constraints == ()
    assert {
        constraint.value: constraint.status for constraint in state.hard_constraints
    }["replacement mouthpiece for a water filter"] is PreferenceStatus.SUPERSEDED


def test_retraction_can_clear_a_constraint_without_an_active_preference() -> None:
    """A constraint-only survivor can be withdrawn without inventing a preference."""
    state = _apply(
        _empty_state(),
        constraint_event(
            event_id="evt-only-constraint",
            kind="feature",
            value="replacement mouthpiece for a water filter",
        ),
    )
    state = _apply(
        state,
        retracted_event(
            event_id="evt-constraint-retract",
            value="replacement mouthpiece for a water filter",
        ),
    )

    assert state.active_preference_beliefs == ()
    assert state.authorized_hard_constraints == ()
    assert state.hard_constraints[0].status is PreferenceStatus.SUPERSEDED


def test_task_retraction_does_not_delete_a_long_term_constraint_of_the_same_kind() -> None:
    """Scope is part of the identity: task retraction may not clear user-scope memory."""
    state = _apply(
        _empty_state(),
        constraint_event(
            event_id="evt-long-term-constraint",
            kind="feature",
            value="lightweight",
            scope="long_term",
        ),
    )
    state = _apply(
        state,
        constraint_event(
            event_id="evt-task-constraint",
            kind="feature",
            value="lightweight",
            scope="task",
        ),
    )
    state = _apply(
        state,
        retracted_event(
            event_id="evt-task-retract",
            kind="feature",
            value="lightweight",
            scope="task",
        ),
    )

    assert [(c.scope.value, c.status.value) for c in state.hard_constraints] == [
        ("long_term", "active"),
        ("task", "superseded"),
    ]
    assert [c.scope.value for c in state.authorized_hard_constraints] == ["long_term"]


def test_price_correction_supersedes_old_budget_and_retraction_does_not_restore_it() -> None:
    """Same-scope budget updates have a real effective lifecycle, not an append-only OR."""
    state = _apply(
        _empty_state(),
        constraint_event(event_id="evt-price-30", kind="price_max", value="30"),
    )
    state = _apply(
        state,
        constraint_event(event_id="evt-price-10", kind="price_max", value="10"),
    )
    state = _apply(
        state,
        constraint_event(event_id="evt-price-50", kind="price_max", value="50"),
    )
    assert [(c.value, c.status.value) for c in state.hard_constraints] == [
        ("30", "superseded"),
        ("10", "superseded"),
        ("50", "active"),
    ]
    assert [c.value for c in state.authorized_hard_constraints] == ["50"]
    state = _apply(
        state,
        retracted_event(event_id="evt-price-retract", kind="price_max", value="50"),
    )
    assert state.authorized_hard_constraints == ()
    # A withdrawn current budget does not resurrect either old budget value.
    assert {c.value for c in state.hard_constraints if c.is_active} == set()


# --------------------------------------------------------------------------- #
# Conflict resolution between authoritative statements
# --------------------------------------------------------------------------- #


def test_conflicting_explicit_beliefs_older_is_superseded() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(
            event_id="evt-prefer", value="lightweight", polarity="prefer"
        ),
    )
    state = _apply(
        state,
        preference_stated_event(
            event_id="evt-avoid", value="lightweight", polarity="avoid"
        ),
    )

    assert len(state.preference_beliefs) == 2
    older, newer = state.preference_beliefs
    assert older.lifecycle is PreferenceLifecycle.EXPLICIT
    assert older.status is PreferenceStatus.SUPERSEDED
    assert newer.lifecycle is PreferenceLifecycle.EXPLICIT
    assert newer.status is PreferenceStatus.ACTIVE
    assert newer.polarity is PreferencePolarity.AVOID


def test_replaces_value_supersedes_only_the_named_value() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-red", kind="color", value="red"),
    )
    state = _apply(
        state,
        preference_stated_event(event_id="evt-black", kind="color", value="black"),
    )
    state = _apply(
        state,
        preference_stated_event(
            event_id="evt-blue",
            kind="color",
            value="blue",
            replaces_value="red",
        ),
    )

    by_value = {belief.value: belief for belief in state.preference_beliefs}
    assert by_value["red"].status is PreferenceStatus.SUPERSEDED
    assert by_value["black"].status is PreferenceStatus.ACTIVE
    assert by_value["blue"].status is PreferenceStatus.ACTIVE
    assert by_value["blue"].lifecycle is PreferenceLifecycle.EXPLICIT


# --------------------------------------------------------------------------- #
# Task scope versus long-term scope
# --------------------------------------------------------------------------- #


def test_task_scope_statement_does_not_touch_long_term_belief() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(
            event_id="evt-long-term", scope="long_term", value="lightweight"
        ),
    )
    (long_term_before,) = state.preference_beliefs
    state = _apply(
        state,
        preference_stated_event(
            event_id="evt-task", scope="task", value="lightweight"
        ),
    )

    assert len(state.preference_beliefs) == 2
    long_term = [
        belief
        for belief in state.preference_beliefs
        if belief.scope is PreferenceScope.LONG_TERM
    ]
    assert long_term == [long_term_before]
    assert long_term_before.scope is PreferenceScope.LONG_TERM


def test_task_scope_retraction_cannot_remove_long_term_memory() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(
            event_id="evt-long-term", scope="long_term", value="lightweight"
        ),
    )
    digest_before = state.state_digest()

    with pytest.raises(UnsupportedTransitionError):
        _apply(
            state,
            retracted_event(
                event_id="evt-task-retract", scope="task", value="lightweight"
            ),
        )
    assert state.state_digest() == digest_before

    # A long-term retraction of the same kind, even from another task, is the way to
    # withdraw the record -- and it still keeps it with provenance.
    state = _apply(
        state,
        retracted_event(
            event_id="evt-long-term-retract",
            task_id="task-2",
            scope="long_term",
            value=None,
            kind="feature",
        ),
    )
    (belief,) = state.preference_beliefs
    assert belief.lifecycle is PreferenceLifecycle.RETRACTED
    assert belief.status is PreferenceStatus.REMOVED


def test_long_term_event_from_foreign_task_is_accepted() -> None:
    state = _empty_state()
    event = preference_stated_event(
        event_id="evt-long-term-foreign",
        task_id="task-2",
        scope="long_term",
        value="lightweight",
    )
    state = _apply(state, event)

    (belief,) = state.preference_beliefs
    assert belief.scope is PreferenceScope.LONG_TERM
    # Long-term identity ignores the task bucket, so every task resolves the same id.
    assert belief.belief_id == preference_belief_id(
        user_key=DEFAULT_USER,
        task_id="some-other-task",
        scope=PreferenceScope.LONG_TERM,
        kind=PreferenceKind.FEATURE,
        value="lightweight",
        polarity=PreferencePolarity.PREFER,
    )
    assert state.interaction.event_count == 1
    # A foreign-task event may not refresh the state's current session.
    assert state.session_id is None


def test_task_event_for_foreign_task_is_refused() -> None:
    state = _empty_state()
    with pytest.raises(UnsupportedTransitionError):
        _apply(
            state,
            preference_stated_event(
                event_id="evt-foreign-task",
                task_id="task-2",
                scope="task",
                value="lightweight",
            ),
        )


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


def test_comparison_feedback_only_adds_observation() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-explicit", value="lightweight"),
    )
    state = _apply(state, constraint_event(event_id="evt-constraint", value="150"))
    beliefs_before = state.preference_beliefs
    constraints_before = state.hard_constraints

    state = _apply(state, comparison_event(event_id="evt-comparison"))

    assert state.preference_beliefs == beliefs_before
    assert state.hard_constraints == constraints_before
    (observation,) = state.observations
    assert observation.event_id == "evt-comparison"
    assert observation.selected_ref == "pool:item-b"
    assert observation.compared_refs == ("pool:item-a", "pool:item-b")
    assert observation.implied_preferences == ()
    assert observation.observation_id.startswith("ob_")


# --------------------------------------------------------------------------- #
# Knowledge state
# --------------------------------------------------------------------------- #


def test_knowledge_ladder_reaches_demonstrated_understanding() -> None:
    state = _apply(
        _empty_state(),
        explanation_requested_event(event_id="evt-request", attribute="waterproofing"),
    )
    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.NEEDS_EXPLANATION

    state = _apply(
        state,
        explanation_shown_event(event_id="evt-shown", attribute="waterproofing"),
    )
    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.EXPLANATION_PROVIDED
    assert belief.explanation_event_ids == ("evt-shown",)

    state = _apply(
        state,
        understanding_event(event_id="evt-understood", attribute="waterproofing"),
    )
    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
    assert belief.is_understood
    assert belief.supporting_event_ids == (
        "evt-request",
        "evt-shown",
        "evt-understood",
    )
    assert belief.knowledge_id == knowledge_belief_id(
        user_key=DEFAULT_USER, attribute="waterproofing"
    )
    assert len(state.knowledge_beliefs) == 1


def test_explanation_shown_never_marks_understanding() -> None:
    state = _apply(
        _empty_state(),
        explanation_requested_event(event_id="evt-request"),
    )
    state = _apply(state, explanation_shown_event(event_id="evt-shown"))

    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.EXPLANATION_PROVIDED
    assert belief.status is not KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
    assert not belief.is_understood


def test_explanation_shown_creates_provided_record_when_unknown() -> None:
    state = _apply(
        _empty_state(), explanation_shown_event(event_id="evt-shown")
    )
    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.EXPLANATION_PROVIDED
    assert belief.explanation_event_ids == ("evt-shown",)


def test_explanation_request_keeps_demonstrated_understanding() -> None:
    state = _apply(
        _empty_state(),
        understanding_event(event_id="evt-understood"),
    )
    state = _apply(
        state, explanation_requested_event(event_id="evt-request")
    )

    (belief,) = state.knowledge_beliefs
    assert belief.status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED
    assert belief.supporting_event_ids == ("evt-understood", "evt-request")


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #


def test_only_constraint_event_creates_hard_constraint() -> None:
    state = _apply(_empty_state(), hypothesis_event(event_id="evt-hypothesis"))
    assert state.hard_constraints == ()
    assert state.authorized_hard_constraints == ()

    state = _apply(
        state,
        constraint_event(event_id="evt-constraint", kind="price_max", value="150"),
    )
    (constraint,) = state.hard_constraints
    assert constraint.authority is ConstraintAuthority.USER_EXPLICIT
    assert constraint.status is PreferenceStatus.ACTIVE
    assert constraint.kind is PreferenceKind.PRICE_MAX
    assert constraint.is_authorized_hard
    assert constraint.supporting_event_ids == ("evt-constraint",)


def test_constraint_event_does_not_create_preference_belief() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-explicit", value="lightweight"),
    )
    beliefs_before = state.preference_beliefs
    state = _apply(
        state, constraint_event(event_id="evt-constraint", value="150")
    )

    assert state.preference_beliefs == beliefs_before
    assert len(state.hard_constraints) == 1


def test_repeated_constraint_upserts_one_record() -> None:
    state = _apply(
        _empty_state(), constraint_event(event_id="evt-constraint-1", value="150")
    )
    state = _apply(
        state, constraint_event(event_id="evt-constraint-2", value="150")
    )

    (constraint,) = state.hard_constraints
    assert constraint.status is PreferenceStatus.ACTIVE
    assert constraint.supporting_event_ids == (
        "evt-constraint-1",
        "evt-constraint-2",
    )


# --------------------------------------------------------------------------- #
# Priorities and trusted references
# --------------------------------------------------------------------------- #


def test_priority_reverse_pair_is_superseded() -> None:
    state = _apply(
        _empty_state(),
        priority_event(event_id="evt-comfort-first", higher="comfort", lower="weight"),
    )
    state = _apply(
        state,
        priority_event(event_id="evt-weight-first", higher="weight", lower="comfort"),
    )

    assert len(state.preference_priorities) == 2
    first, second = state.preference_priorities
    assert first.identity() == (
        PreferenceScope.TASK.value,
        PreferenceKind.FEATURE.value,
        "comfort",
        "weight",
    )
    assert first.status is PreferenceStatus.SUPERSEDED
    assert second.status is PreferenceStatus.ACTIVE
    assert second.higher == "weight"


def test_candidate_and_ranking_references_latest_wins() -> None:
    state = _apply(
        _empty_state(),
        candidate_pool_event(event_id="evt-pool-1", ref="pool-1", candidate_count=5),
    )
    state = _apply(
        state,
        candidate_pool_event(event_id="evt-pool-2", ref="pool-2", candidate_count=7),
    )
    state = _apply(
        state,
        ranking_snapshot_event(
            event_id="evt-rank-1", ref="rank-1", method="rrf_rank_fusion"
        ),
    )
    state = _apply(
        state,
        ranking_snapshot_event(
            event_id="evt-rank-2", ref="rank-2", method="weighted_score_fusion"
        ),
    )

    assert state.candidate_pool_ref is not None
    assert state.candidate_pool_ref.ref == "pool-2"
    assert state.candidate_pool_ref.candidate_count == 7
    assert state.candidate_pool_ref.referenced_by_event_id == "evt-pool-2"
    assert state.candidate_pool_ref.read_only is True
    assert state.ranking_snapshot_ref is not None
    assert state.ranking_snapshot_ref.ref == "rank-2"
    assert state.ranking_snapshot_ref.method == "weighted_score_fusion"
    assert state.ranking_snapshot_ref.referenced_by_event_id == "evt-rank-2"


def test_task_context_replaces_intent_and_request_text() -> None:
    state = _apply(
        _empty_state(),
        context_event(event_id="evt-context", intent="recommend", request_text="hiking"),
    )

    assert state.task_context.intent == "recommend"
    assert state.task_context.request_text == "hiking"
    assert state.task_context.updated_by_event_id == "evt-context"


# --------------------------------------------------------------------------- #
# Interaction metadata and ordering guards
# --------------------------------------------------------------------------- #


def test_interaction_metadata_counts_events_and_user_turns() -> None:
    state = _apply(
        _empty_state(), preference_stated_event(event_id="evt-user-1")
    )
    state = _apply(state, hypothesis_event(event_id="evt-agent"))
    state = _apply(state, comparison_event(event_id="evt-user-2"))

    assert state.interaction.event_count == 3
    assert state.interaction.user_turn_count == 2
    assert state.interaction.last_event_id == "evt-user-2"
    assert state.session_id == DEFAULT_SESSION


def test_sequence_must_increase_strictly() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-seq-1", value="lightweight").stored(5),
    )
    assert state.interaction.last_event_sequence == 5

    with pytest.raises(UnsupportedTransitionError):
        _apply(
            state,
            preference_stated_event(event_id="evt-seq-2", value="waterproof").stored(5),
        )
    with pytest.raises(UnsupportedTransitionError):
        _apply(
            state,
            preference_stated_event(event_id="evt-seq-3", value="waterproof").stored(4),
        )

    state = _apply(
        state,
        preference_stated_event(event_id="evt-seq-4", value="waterproof").stored(6),
    )
    assert state.interaction.last_event_sequence == 6


def test_duplicate_last_event_id_is_refused() -> None:
    event = preference_stated_event(event_id="evt-duplicate").stored(1)
    state = _apply(_empty_state(), event)
    digest_before = state.state_digest()

    with pytest.raises(UnsupportedTransitionError):
        _apply(state, event)
    with pytest.raises(UnsupportedTransitionError):
        _apply(state, event.stored(2))

    assert state.state_digest() == digest_before


def test_unsequenced_event_does_not_advance_last_event_sequence() -> None:
    state = _apply(
        _empty_state(),
        preference_stated_event(event_id="evt-seq-1", value="lightweight").stored(3),
    )
    state = _apply(
        state, preference_stated_event(event_id="evt-unsequenced", value="waterproof")
    )

    assert state.interaction.last_event_sequence == 3
    assert state.interaction.event_count == 2


def test_cross_user_event_is_refused() -> None:
    state = _empty_state()
    with pytest.raises(CrossUserStateError):
        _apply(
            state,
            preference_stated_event(event_id="evt-other-user", user_key="user-b"),
        )


# --------------------------------------------------------------------------- #
# Purity and deterministic replay
# --------------------------------------------------------------------------- #


def test_apply_does_not_mutate_input_state() -> None:
    empty = _empty_state()
    digest_before = empty.state_digest()

    state = _apply(
        empty, preference_stated_event(event_id="evt-explicit")
    )

    assert empty.state_digest() == digest_before
    assert empty.preference_beliefs == ()
    assert empty.interaction.event_count == 0
    assert len(state.preference_beliefs) == 1


def test_apply_all_is_deterministic() -> None:
    events = [
        context_event(event_id="evt-a1").stored(1),
        preference_stated_event(event_id="evt-a2", value="lightweight").stored(2),
        hypothesis_event(event_id="evt-a3", value="waterproof").stored(3),
        constraint_event(event_id="evt-a4", value="150").stored(4),
        comparison_event(event_id="evt-a5").stored(5),
        explanation_requested_event(event_id="evt-a6").stored(6),
        explanation_shown_event(event_id="evt-a7").stored(7),
        understanding_event(event_id="evt-a8").stored(8),
        priority_event(event_id="evt-a9").stored(9),
        candidate_pool_event(event_id="evt-a10").stored(10),
        ranking_snapshot_event(event_id="evt-a11").stored(11),
    ]

    first = UPDATER.apply_all(_empty_state(), events)
    second = UPDATER.apply_all(_empty_state(), events)

    assert first == second
    assert first.state_digest() == second.state_digest()
    assert first.interaction.event_count == len(events)


def test_apply_all_leaves_input_state_untouched() -> None:
    empty = _empty_state()
    UPDATER.apply_all(
        empty,
        [
            preference_stated_event(event_id="evt-all-1").stored(1),
            hypothesis_event(event_id="evt-all-2", value="waterproof").stored(2),
        ],
    )

    assert empty == _empty_state()
    assert empty.preference_beliefs == ()
