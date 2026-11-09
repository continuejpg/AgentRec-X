"""Projection tests for the M5 state projector.

The three projections are the only way a consumer is meant to read the unified decision
state, so these tests pin both their content and their safety properties:

* ``for_agent`` carries the compact decision summary (scoped preferences including soft
  hypotheses, knowledge, clarification attributes, priorities, authorized constraints,
  opaque references and counters);
* ``for_ranker`` carries typed active preference features with an ordinal
  ``strength_rank``, flags hypotheses ``is_soft``, excludes retracted/superseded records
  and never carries a hard constraint;
* ``for_constraints`` carries only active ``USER_EXPLICIT`` constraints, applies the
  documented task-over-long-term precedence at projection time, reports the suppressed
  long-term ids, and counts the hypotheses deliberately kept out of the filter path;
* no projection exposes ``user_key`` or a raw support span, and all projection models
  are frozen.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from recommendation.state.belief_updater import BeliefUpdater
from recommendation.state.decision_state import (
    UnifiedDecisionState,
    decision_constraint_id,
    empty_state,
)
from recommendation.state.schemas import (
    BeliefStrength,
    KnowledgeStatus,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)
from recommendation.state.state_projector import (
    AGENT_PROJECTION_VERSION,
    CONSTRAINT_PROJECTION_VERSION,
    RANKER_SIGNAL_VERSION,
    AgentStateProjection,
    ConstraintStateProjection,
    RankerStateProjection,
    StateProjector,
    for_agent,
    for_constraints,
    for_ranker,
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
)

UPDATER = BeliefUpdater()


def _rich_state() -> UnifiedDecisionState:
    """State exercising every projection field: explicit/confirmed/inferred/retracted."""
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    events = [
        context_event(
            event_id="evt-context",
            intent="recommend",
            request_text="I need a waterproof hiking backpack for a weekend trip.",
        ),
        preference_stated_event(
            event_id="evt-lightweight",
            kind="feature",
            value="lightweight",
            source_text="I want something lightweight.",
        ),
        hypothesis_event(
            event_id="evt-waterproof", kind="feature", value="waterproof", strength="weak"
        ),
        hypothesis_event(
            event_id="evt-acme-hypothesis", kind="brand", value="acme", strength="moderate"
        ),
        confirmed_event(
            event_id="evt-acme-confirm",
            kind="brand",
            value="acme",
            source_text="Yes, the Acme brand is what I want.",
        ),
        hypothesis_event(event_id="evt-compact", kind="feature", value="compact"),
        retracted_event(event_id="evt-compact-retract", kind="feature", value="compact"),
        preference_stated_event(event_id="evt-red", kind="color", value="red"),
        preference_stated_event(
            event_id="evt-red-avoid", kind="color", value="red", polarity="avoid"
        ),
        priority_event(event_id="evt-priority", higher="comfort", lower="weight"),
        constraint_event(
            event_id="evt-task-price", kind="price_max", value="150", scope="task"
        ),
        constraint_event(
            event_id="evt-long-term-price", kind="price_max", value="90", scope="long_term"
        ),
        constraint_event(
            event_id="evt-long-term-brand", kind="brand", value="acme", scope="long_term"
        ),
        explanation_requested_event(
            event_id="evt-needs-explanation", attribute="waterproofing"
        ),
        explanation_shown_event(
            event_id="evt-explanation-shown", attribute="breathability"
        ),
        comparison_event(event_id="evt-comparison-1"),
        comparison_event(
            event_id="evt-comparison-2",
            compared_refs=("pool:item-a", "pool:item-c"),
            selected_ref="pool:item-c",
            feedback_text="Item C looks more breathable.",
        ),
        candidate_pool_event(event_id="evt-candidate-pool"),
        ranking_snapshot_event(event_id="evt-ranking-snapshot"),
    ]
    for sequence, event in enumerate(events, start=1):
        state = UPDATER.apply(state, event.stored(sequence))
    return state


# --------------------------------------------------------------------------- #
# for_agent
# --------------------------------------------------------------------------- #


def test_agent_projection_scalar_summary_fields() -> None:
    state = _rich_state()
    projection = for_agent(state)

    assert isinstance(projection, AgentStateProjection)
    assert projection.projection_version == AGENT_PROJECTION_VERSION == 1
    assert projection.task_id == DEFAULT_TASK
    assert projection.session_id == DEFAULT_SESSION
    assert projection.state_version == state.state_version
    assert projection.event_count == 19
    assert projection.user_turn_count == 12
    assert projection.intent == "recommend"
    assert "hiking backpack" in projection.request_text
    assert projection.observation_count == 2
    assert projection.inferred_preference_count == 1
    assert projection.explicit_preference_count == 3
    assert projection.candidate_pool_ref == "candidate-pool-1"
    assert projection.candidate_pool_candidate_count == 10
    assert projection.ranking_snapshot_ref == "ranking-snapshot-1"
    assert projection.ranking_snapshot_method == "rrf_rank_fusion"


def test_agent_projection_preferences_active_only_and_soft_flagged() -> None:
    projection = for_agent(_rich_state())
    by_value = {summary.value: summary for summary in projection.preferences}

    assert set(by_value) == {"lightweight", "waterproof", "acme", "red"}
    # Retracted and superseded records are never projected.
    assert "compact" not in by_value
    assert len(projection.preferences) == 4

    waterproof = by_value["waterproof"]
    assert waterproof.is_soft is True
    assert waterproof.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
    assert waterproof.strength is BeliefStrength.WEAK

    lightweight = by_value["lightweight"]
    assert lightweight.is_soft is False
    assert lightweight.lifecycle is PreferenceLifecycle.EXPLICIT
    assert lightweight.strength is BeliefStrength.STRONG
    assert lightweight.status is PreferenceStatus.ACTIVE

    assert by_value["acme"].lifecycle is PreferenceLifecycle.CONFIRMED
    assert by_value["red"].polarity is PreferencePolarity.AVOID


def test_agent_projection_knowledge_and_clarification_attributes() -> None:
    projection = for_agent(_rich_state())

    statuses = {summary.attribute: summary.status for summary in projection.knowledge}
    assert statuses == {
        "waterproofing": KnowledgeStatus.NEEDS_EXPLANATION,
        "breathability": KnowledgeStatus.EXPLANATION_PROVIDED,
    }
    assert projection.clarification_attributes == ("waterproofing",)


def test_agent_projection_priorities_and_authorized_constraints() -> None:
    state = _rich_state()
    projection = for_agent(state)

    (priority,) = projection.priorities
    assert priority.higher == "comfort"
    assert priority.lower == "weight"
    assert priority.scope is PreferenceScope.TASK

    constraint_kinds = {summary.kind for summary in projection.constraints}
    assert constraint_kinds == {PreferenceKind.PRICE_MAX, PreferenceKind.BRAND}
    assert len(projection.constraints) == len(state.authorized_hard_constraints) == 3


# --------------------------------------------------------------------------- #
# for_ranker
# --------------------------------------------------------------------------- #


def test_ranker_projection_active_features_with_ordinal_strength() -> None:
    projection = for_ranker(_rich_state())

    assert isinstance(projection, RankerStateProjection)
    assert projection.ranker_signal_version == RANKER_SIGNAL_VERSION == 1
    assert projection.task_id == DEFAULT_TASK
    features = {feature.value: feature for feature in projection.features}
    assert set(features) == {"lightweight", "waterproof", "acme", "red"}
    assert "compact" not in features

    # Retracted / superseded records are excluded entirely: the older "red prefer"
    # record was superseded; only the active AVOID survives.
    red_values = [feature for feature in projection.features if feature.value == "red"]
    assert len(red_values) == 1
    assert red_values[0].polarity is PreferencePolarity.AVOID

    assert features["lightweight"].strength_rank == 3
    assert features["acme"].strength_rank == 3
    assert features["waterproof"].strength_rank == 1
    assert features["waterproof"].is_soft is True
    assert features["lightweight"].is_soft is False
    for feature in projection.features:
        assert 1 <= feature.strength_rank <= 3


def test_ranker_projection_excludes_hard_constraints() -> None:
    projection = for_ranker(_rich_state())

    assert "constraints" not in RankerStateProjection.model_fields
    assert not hasattr(projection, "constraints")
    assert "constraints" not in projection.model_dump()


def test_ranker_projection_priorities_and_references() -> None:
    projection = for_ranker(_rich_state())

    (priority,) = projection.priorities
    assert priority.higher == "comfort"
    assert priority.lower == "weight"
    assert projection.candidate_pool_ref == "candidate-pool-1"
    assert projection.ranking_snapshot_ref == "ranking-snapshot-1"


# --------------------------------------------------------------------------- #
# for_constraints
# --------------------------------------------------------------------------- #


def test_constraints_projection_contains_only_user_explicit_constraints() -> None:
    state = _rich_state()
    projection = for_constraints(state)

    assert isinstance(projection, ConstraintStateProjection)
    assert projection.constraints_version == CONSTRAINT_PROJECTION_VERSION == 1
    assert projection.task_id == DEFAULT_TASK
    # The active inferred hypothesis (feature=waterproof) is never a filter.
    for view in projection.constraints:
        assert view.kind in (PreferenceKind.PRICE_MAX, PreferenceKind.BRAND)
    assert projection.excluded_hypothesis_count == 1
    assert len(state.inferred_hypotheses) == 1


def test_task_constraint_suppresses_long_term_same_kind_in_projection_only() -> None:
    state = _rich_state()
    projection = for_constraints(state)

    long_term_price_id = decision_constraint_id(
        user_key=DEFAULT_USER,
        task_id=DEFAULT_TASK,
        scope=PreferenceScope.LONG_TERM,
        kind=PreferenceKind.PRICE_MAX,
        value="90",
        polarity=PreferencePolarity.PREFER,
    )
    assert projection.suppressed_long_term_ids == (long_term_price_id,)

    projected_values = {(view.kind, view.value) for view in projection.constraints}
    assert (PreferenceKind.PRICE_MAX, "150") in projected_values
    assert (PreferenceKind.BRAND, "acme") in projected_values
    assert (PreferenceKind.PRICE_MAX, "90") not in projected_values

    # The long-term record is suppressed for this task only; the state keeps it.
    assert len(state.hard_constraints) == 3
    long_term_price = [
        constraint
        for constraint in state.hard_constraints
        if constraint.constraint_id == long_term_price_id
    ]
    assert len(long_term_price) == 1
    assert long_term_price[0].scope is PreferenceScope.LONG_TERM
    assert long_term_price[0].status is PreferenceStatus.ACTIVE


def test_constraints_projection_excludes_inferred_hypotheses_entirely() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    state = UPDATER.apply(
        state, hypothesis_event(event_id="evt-hypothesis", kind="feature", value="light")
    )
    state = UPDATER.apply(
        state, constraint_event(event_id="evt-constraint", kind="price_max", value="150")
    )

    projection = for_constraints(state)
    assert [(view.kind, view.value) for view in projection.constraints] == [
        (PreferenceKind.PRICE_MAX, "150")
    ]
    assert projection.excluded_hypothesis_count == 1
    assert projection.suppressed_long_term_ids == ()


# --------------------------------------------------------------------------- #
# Safety properties shared by every projection
# --------------------------------------------------------------------------- #


def _state_with_support_marker() -> tuple[UnifiedDecisionState, str]:
    """State whose raw support span is a unique marker the projections must not leak."""
    marker = "support-span-marker-do-not-leak"
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    state = UPDATER.apply(
        state,
        preference_stated_event(
            event_id="evt-marker", value="lightweight", source_text=marker
        ),
    )
    state = UPDATER.apply(
        state,
        context_event(event_id="evt-context", request_text="a backpack request"),
    )
    return state, marker


def test_projections_omit_user_key_and_raw_support_text() -> None:
    state, marker = _state_with_support_marker()

    for projection in (for_agent(state), for_ranker(state), for_constraints(state)):
        payload = projection.model_dump(mode="json")
        serialized = json.dumps(payload)
        assert "user_key" not in payload
        assert DEFAULT_USER not in serialized
        assert marker not in serialized
        assert "supporting_event_ids" not in payload
        assert "source_text" not in payload


def test_projection_models_are_frozen() -> None:
    state, _ = _state_with_support_marker()
    agent_projection = for_agent(state)
    ranker_projection = for_ranker(state)
    constraint_projection = for_constraints(state)

    with pytest.raises(ValidationError):
        agent_projection.intent = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ranker_projection.task_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        constraint_projection.task_id = "other"  # type: ignore[misc]


def test_empty_state_projects_to_empty_views() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)

    agent = for_agent(state)
    assert agent.preferences == ()
    assert agent.knowledge == ()
    assert agent.clarification_attributes == ()
    assert agent.priorities == ()
    assert agent.constraints == ()
    assert agent.candidate_pool_ref is None
    assert agent.candidate_pool_candidate_count is None
    assert agent.ranking_snapshot_ref is None
    assert agent.ranking_snapshot_method is None
    assert agent.observation_count == 0
    assert agent.inferred_preference_count == 0
    assert agent.explicit_preference_count == 0

    ranker = for_ranker(state)
    assert ranker.features == ()
    assert ranker.priorities == ()
    assert ranker.candidate_pool_ref is None

    constraints = for_constraints(state)
    assert constraints.constraints == ()
    assert constraints.suppressed_long_term_ids == ()
    assert constraints.excluded_hypothesis_count == 0


def test_module_level_functions_delegate_to_shared_projector() -> None:
    state = _rich_state()
    projector = StateProjector()

    assert for_agent(state) == projector.for_agent(state)
    assert for_ranker(state) == projector.for_ranker(state)
    assert for_constraints(state) == projector.for_constraints(state)
