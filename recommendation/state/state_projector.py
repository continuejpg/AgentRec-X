"""Read-only projections of the M5-MVP unified decision state.

Three consumers get three deliberately narrow views of one state:

* :func:`for_agent` -- a compact decision summary for an Agent policy: task context,
  preference summaries (soft hypotheses included, so the policy can choose a
  clarification), knowledge states, active priorities, authorized constraints and
  opaque read-only references;
* :func:`for_ranker` -- typed preference features with an ordinal strength for a future
  fine ranker.  Hard constraints are deliberately absent: a ranker may order
  candidates, it may not filter them;
* :func:`for_constraints` -- only user-authorized hard constraints, with a deterministic
  task-over-long-term precedence and a count of the hypotheses that were deliberately
  *not* turned into filters.

Properties every projection preserves
-------------------------------------
* **frozen value objects**: every projection model is a frozen Pydantic model, so a
  consumer cannot patch the derived view (or the state behind it) in place;
* **no user identity**: ``user_key`` is never projected;
* **no raw provenance text**: support spans, reasons, evidence spans and source text are
  not copied into a projection -- only typed, bounded semantic fields are;
* **no product data and no scores**: candidate pools and ranking snapshots appear only
  as opaque references, counts and a method label;
* **soft evidence stays soft**: an inferred hypothesis never appears in
  ``for_constraints`` and is flagged ``is_soft`` where it appears at all.

The projector is stateless and pure; the module-level functions delegate to one shared
default instance so there is exactly one projection implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .decision_state import DecisionConstraint, UnifiedDecisionState
from .schemas import (
    HARD_CONSTRAINT_AUTHORITIES,
    BeliefStrength,
    KnowledgeStatus,
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)

__all__ = [
    "AGENT_PROJECTION_VERSION",
    "CONSTRAINT_PROJECTION_VERSION",
    "RANKER_SIGNAL_VERSION",
    "AgentConstraintSummary",
    "AgentKnowledgeSummary",
    "AgentPreferenceSummary",
    "AgentPrioritySummary",
    "AgentStateProjection",
    "ConstraintStateProjection",
    "ConstraintView",
    "RankerFeature",
    "RankerPriority",
    "RankerStateProjection",
    "StateProjector",
    "for_agent",
    "for_constraints",
    "for_ranker",
]

#: Contract version of the Agent decision projection.
AGENT_PROJECTION_VERSION = 1

#: Contract version of the soft-signal ranker projection.
RANKER_SIGNAL_VERSION = 1

#: Contract version of the hard-constraint projection.
CONSTRAINT_PROJECTION_VERSION = 1

_FROZEN = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Agent projection models
# --------------------------------------------------------------------------- #


class AgentPreferenceSummary(BaseModel):
    """One preference belief as the Agent policy sees it (identity and authority)."""

    model_config = _FROZEN

    preference_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity
    scope: PreferenceScope
    lifecycle: PreferenceLifecycle
    strength: BeliefStrength
    status: PreferenceStatus
    #: True for an unconfirmed system hypothesis: soft evidence, never a filter.
    is_soft: bool


class AgentKnowledgeSummary(BaseModel):
    """Whether the user knows one attribute; only non-``UNKNOWN`` records are projected."""

    model_config = _FROZEN

    attribute: str
    status: KnowledgeStatus


class AgentPrioritySummary(BaseModel):
    """A relative feature-importance statement (``higher`` matters more than ``lower``)."""

    model_config = _FROZEN

    kind: PreferenceKind
    higher: str
    lower: str
    scope: PreferenceScope


class AgentConstraintSummary(BaseModel):
    """One authorized hard constraint, without provenance or authority fields."""

    model_config = _FROZEN

    constraint_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity
    scope: PreferenceScope


class AgentStateProjection(BaseModel):
    """Compact read-only decision-state summary for the Agent policy.

    Active preferences include soft hypotheses (flagged ``is_soft``) because choosing a
    clarification question is a legitimate Agent action.  Retracted and superseded
    records are never projected.  ``inferred_preference_count`` /
    ``explicit_preference_count`` count the active soft and authoritative beliefs
    respectively, so the policy can reason about how much of the signal is a guess.
    """

    model_config = _FROZEN

    projection_version: int = AGENT_PROJECTION_VERSION
    task_id: str
    session_id: str | None = None
    state_version: int
    event_count: int
    user_turn_count: int
    intent: str
    request_text: str
    preferences: tuple[AgentPreferenceSummary, ...] = ()
    knowledge: tuple[AgentKnowledgeSummary, ...] = ()
    clarification_attributes: tuple[str, ...] = ()
    priorities: tuple[AgentPrioritySummary, ...] = ()
    constraints: tuple[AgentConstraintSummary, ...] = ()
    candidate_pool_ref: str | None = None
    candidate_pool_candidate_count: int | None = None
    ranking_snapshot_ref: str | None = None
    ranking_snapshot_method: str | None = None
    observation_count: int = 0
    inferred_preference_count: int = 0
    explicit_preference_count: int = 0


# --------------------------------------------------------------------------- #
# Ranker projection models
# --------------------------------------------------------------------------- #


class RankerFeature(BaseModel):
    """One active preference feature for the fine ranker.

    ``strength_rank`` is the ordinal 1-3 weight from
    :class:`~recommendation.state.schemas.BeliefStrength` -- an ordering weight, never a
    probability.  ``is_soft`` is true for an unconfirmed hypothesis, so a ranker can
    discount it without having to re-derive its lifecycle.
    """

    model_config = _FROZEN

    preference_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity
    direction: PreferenceDirection | None = None
    scope: PreferenceScope
    lifecycle: PreferenceLifecycle
    strength_rank: int = Field(ge=1, le=3)
    is_soft: bool


class RankerPriority(BaseModel):
    """A relative feature-importance signal for the ranker."""

    model_config = _FROZEN

    kind: PreferenceKind
    higher: str
    lower: str
    scope: PreferenceScope


class RankerStateProjection(BaseModel):
    """Read-only soft-signal projection consumed by a future fine ranker.

    There is no constraint field: the ranker reorders a trusted candidate pool, it never
    filters it.  ``ranker_signal_version`` versions this contract independently of the
    state schema so a model trained on it can pin the exact signal shape.
    """

    model_config = _FROZEN

    ranker_signal_version: int = RANKER_SIGNAL_VERSION
    task_id: str
    state_version: int
    features: tuple[RankerFeature, ...] = ()
    priorities: tuple[RankerPriority, ...] = ()
    candidate_pool_ref: str | None = None
    ranking_snapshot_ref: str | None = None


# --------------------------------------------------------------------------- #
# Constraint projection models
# --------------------------------------------------------------------------- #


class ConstraintView(BaseModel):
    """One authorized hard constraint as the filter path sees it."""

    model_config = _FROZEN

    constraint_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity
    scope: PreferenceScope


class ConstraintStateProjection(BaseModel):
    """Read-only hard-constraint projection.

    Only active constraints whose authority is ``USER_EXPLICIT`` are projected.  A task
    or session constraint of a kind overrides the long-term constraint of the same kind
    **for this projection only**; the overridden record is reported in
    ``suppressed_long_term_ids`` and remains untouched in the underlying state.

    ``excluded_hypothesis_count`` makes the "inference never becomes a filter" rule
    observable: those active hypotheses were deliberately left out of ``constraints``.
    """

    model_config = _FROZEN

    constraints_version: int = CONSTRAINT_PROJECTION_VERSION
    task_id: str
    constraints: tuple[ConstraintView, ...] = ()
    suppressed_long_term_ids: tuple[str, ...] = ()
    excluded_hypothesis_count: int = 0


# --------------------------------------------------------------------------- #
# The projector
# --------------------------------------------------------------------------- #


def _authorized_constraints(
    state: UnifiedDecisionState,
) -> tuple[DecisionConstraint, ...]:
    """Active constraints whose authority may define feasibility (in state order)."""
    return tuple(
        constraint
        for constraint in state.hard_constraints
        if constraint.status is PreferenceStatus.ACTIVE
        and constraint.authority in HARD_CONSTRAINT_AUTHORITIES
    )


def _candidate_pool_ref(state: UnifiedDecisionState) -> str | None:
    """The opaque candidate-pool reference, or ``None`` when no trusted reference exists."""
    ref = state.candidate_pool_ref
    return ref.ref if ref is not None else None


def _ranking_snapshot_ref(state: UnifiedDecisionState) -> str | None:
    """The opaque ranking-snapshot reference, or ``None`` when absent."""
    ref = state.ranking_snapshot_ref
    return ref.ref if ref is not None else None


class StateProjector:
    """Stateless builder of the three read-only decision-state projections."""

    def for_agent(self, state: UnifiedDecisionState) -> AgentStateProjection:
        """Project the state into the Agent decision summary."""
        candidate_pool = state.candidate_pool_ref
        ranking_snapshot = state.ranking_snapshot_ref
        return AgentStateProjection(
            projection_version=AGENT_PROJECTION_VERSION,
            task_id=state.task_id,
            session_id=state.session_id,
            state_version=state.state_version,
            event_count=state.interaction.event_count,
            user_turn_count=state.interaction.user_turn_count,
            intent=state.task_context.intent,
            request_text=state.task_context.request_text,
            preferences=tuple(
                AgentPreferenceSummary(
                    preference_id=belief.belief_id,
                    kind=belief.kind,
                    value=belief.value,
                    polarity=belief.polarity,
                    scope=belief.scope,
                    lifecycle=belief.lifecycle,
                    strength=belief.strength,
                    status=belief.status,
                    is_soft=belief.is_soft_signal,
                )
                for belief in state.active_preference_beliefs
            ),
            knowledge=tuple(
                AgentKnowledgeSummary(attribute=belief.attribute, status=belief.status)
                for belief in state.active_knowledge_beliefs
            ),
            clarification_attributes=state.attributes_needing_explanation,
            priorities=tuple(
                AgentPrioritySummary(
                    kind=priority.kind,
                    higher=priority.higher,
                    lower=priority.lower,
                    scope=priority.scope,
                )
                for priority in state.active_priorities
            ),
            constraints=tuple(
                AgentConstraintSummary(
                    constraint_id=constraint.constraint_id,
                    kind=constraint.kind,
                    value=constraint.value,
                    polarity=constraint.polarity,
                    scope=constraint.scope,
                )
                for constraint in _authorized_constraints(state)
            ),
            candidate_pool_ref=(
                candidate_pool.ref if candidate_pool is not None else None
            ),
            candidate_pool_candidate_count=(
                candidate_pool.candidate_count if candidate_pool is not None else None
            ),
            ranking_snapshot_ref=(
                ranking_snapshot.ref if ranking_snapshot is not None else None
            ),
            ranking_snapshot_method=(
                ranking_snapshot.method if ranking_snapshot is not None else None
            ),
            observation_count=len(state.observations),
            inferred_preference_count=len(state.inferred_hypotheses),
            explicit_preference_count=len(state.explicit_or_confirmed_beliefs),
        )

    def for_ranker(self, state: UnifiedDecisionState) -> RankerStateProjection:
        """Project the state into the typed soft-signal ranker input.

        Only ``ACTIVE`` beliefs are features; retracted and superseded records are
        excluded, and hypotheses are flagged ``is_soft`` instead of being dropped (the
        ranker may use them at reduced weight).  Hard constraints are not present.
        """
        return RankerStateProjection(
            ranker_signal_version=RANKER_SIGNAL_VERSION,
            task_id=state.task_id,
            state_version=state.state_version,
            features=tuple(
                RankerFeature(
                    preference_id=belief.belief_id,
                    kind=belief.kind,
                    value=belief.value,
                    polarity=belief.polarity,
                    direction=belief.direction,
                    scope=belief.scope,
                    lifecycle=belief.lifecycle,
                    strength_rank=belief.strength.rank,
                    is_soft=belief.is_soft_signal,
                )
                for belief in state.active_preference_beliefs
            ),
            priorities=tuple(
                RankerPriority(
                    kind=priority.kind,
                    higher=priority.higher,
                    lower=priority.lower,
                    scope=priority.scope,
                )
                for priority in state.active_priorities
            ),
            candidate_pool_ref=_candidate_pool_ref(state),
            ranking_snapshot_ref=_ranking_snapshot_ref(state),
        )

    def for_constraints(
        self, state: UnifiedDecisionState
    ) -> ConstraintStateProjection:
        """Project only authorized hard constraints, applying task-over-long-term precedence."""
        authorized = _authorized_constraints(state)
        #: A task/session constraint of a kind overrides the long-term constraint of that
        #: kind for this projection.  Only *kinds* matter: the long-term record itself is
        #: never modified.
        local_kinds = {
            constraint.kind
            for constraint in authorized
            if constraint.scope is not PreferenceScope.LONG_TERM
        }
        projected: list[ConstraintView] = []
        suppressed: list[str] = []
        for constraint in authorized:
            if (
                constraint.scope is PreferenceScope.LONG_TERM
                and constraint.kind in local_kinds
            ):
                suppressed.append(constraint.constraint_id)
                continue
            projected.append(
                ConstraintView(
                    constraint_id=constraint.constraint_id,
                    kind=constraint.kind,
                    value=constraint.value,
                    polarity=constraint.polarity,
                    scope=constraint.scope,
                )
            )
        return ConstraintStateProjection(
            constraints_version=CONSTRAINT_PROJECTION_VERSION,
            task_id=state.task_id,
            constraints=tuple(projected),
            suppressed_long_term_ids=tuple(suppressed),
            excluded_hypothesis_count=len(state.inferred_hypotheses),
        )


#: Shared stateless projector behind the module-level convenience functions.
_DEFAULT_PROJECTOR = StateProjector()


def for_agent(state: UnifiedDecisionState) -> AgentStateProjection:
    """Project ``state`` into the Agent decision summary."""
    return _DEFAULT_PROJECTOR.for_agent(state)


def for_ranker(state: UnifiedDecisionState) -> RankerStateProjection:
    """Project ``state`` into the typed soft-signal ranker input."""
    return _DEFAULT_PROJECTOR.for_ranker(state)


def for_constraints(state: UnifiedDecisionState) -> ConstraintStateProjection:
    """Project ``state`` into the authorized hard-constraint view."""
    return _DEFAULT_PROJECTOR.for_constraints(state)
