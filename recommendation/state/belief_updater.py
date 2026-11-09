"""Deterministic belief updater for the M5-MVP unified decision state.

The updater is the single transition function of the decision-state engine: it turns
one validated :class:`~recommendation.state.events.StateEvent` plus the current
:class:`~recommendation.state.decision_state.UnifiedDecisionState` into a **new** state.
It is a pure function in the strong sense:

* no I/O, no network, no SQLite, no filesystem;
* no wall clock and no randomness -- ``timestamp`` is metadata and never orders events;
* no model, no LLM and no process-global mutable register;
* no mutation of the caller's state: the state and every record type are frozen, and a
  transition returns a rebuilt state, so a rejected event leaves the input untouched.

The reducer imports only the frozen vocabulary modules (``schemas``, ``events``,
``decision_state``, ``errors``); it never imports the event store, the state manager or
a projector, which keeps the dependency direction one-way (store -> manager -> reducer,
projector -> state) and makes offline replay possible.

Semantic rules, in one place
----------------------------
The authoritative rule table is section 4 of ``docs/M5_UNIFIED_DECISION_STATE.md`` and
the implementation follows it event by event.  Three invariants are worth restating
because they are structural, not conventional:

* **inference is soft**: a hypothesis is only ever ``INFERRED_HYPOTHESIS`` /
  ``ACTIVE`` / ``SYSTEM_INFERRED`` in a task/session scope with ``WEAK``/``MODERATE``
  strength; there is no event through which an inference can create a hard constraint
  or contradict an explicit user statement;
* **the user wins**: an explicit statement retracts a conflicting inference (record
  kept, provenance preserved), and an existing inferred belief the user states again is
  upgraded in place to ``EXPLICIT`` instead of silently remaining a guess;
* **nothing is deleted**: supersession and retraction move a record to a non-active
  lifecycle/status and append the event id; tuple order is the record's first-appearance
  order, so replay and first-application produce byte-identical states.

Deterministic identity
----------------------
Every created record uses the deterministic id builders of ``decision_state.py``
(``preference_belief_id`` and friends), so the same event stream always rebuilds the
same ids and the same :meth:`UnifiedDecisionState.state_digest`.  ``last_event_sequence``
is only advanced for store-sequenced events; a bare event (``sequence is None``) leaves
it untouched so a test can fold unsequenced events while the store path stays strict.
"""

from __future__ import annotations

from typing import Any, Iterable

from .decision_state import (
    CandidatePoolRef,
    DecisionConstraint,
    DecisionObservation,
    FeaturePriority,
    InteractionMetadata,
    KnowledgeBelief,
    PreferenceBelief,
    RankingSnapshotRef,
    TaskContext,
    UnifiedDecisionState,
    comparison_observation_id,
    decision_constraint_id,
    feature_priority_id,
    knowledge_belief_id,
    preference_belief_id,
)
from .errors import CrossUserStateError, UnsupportedTransitionError
from .events import (
    EVENT_ACTORS,
    CandidatePoolReferenced,
    EventActor,
    EventType,
    FeatureExplanationRequested,
    FeatureExplanationShown,
    PreferenceConfirmed,
    PreferenceHypothesisProposed,
    PreferencePriorityStated,
    PreferenceRetracted,
    ProductComparisonFeedback,
    RankingSnapshotReferenced,
    StateEvent,
    StateEventPayload,
    TaskContextUpdated,
    UserConstraintStated,
    UserPreferenceStated,
    UserUnderstandingDemonstrated,
)
from .schemas import (
    BeliefOrigin,
    BeliefStrength,
    ConstraintAuthority,
    KnowledgeStatus,
    ObservationKind,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)

__all__ = ["BeliefUpdater"]


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #


def _normalised(value: str) -> str:
    """Whitespace-collapsed, case-folded form used for identity/value comparison."""
    return " ".join(value.split()).casefold()


def _append_unique(values: tuple[str, ...], event_id: str) -> tuple[str, ...]:
    """Append ``event_id`` as provenance unless it is already recorded."""
    if event_id in values:
        return values
    return values + (event_id,)


def _expect(event: StateEvent, model: type[StateEventPayload]) -> StateEventPayload:
    """Return the payload, defensively asserting it is the model its event type names.

    :class:`~recommendation.state.events.StateEvent` already enforces the
    ``event_type -> payload model`` registry, so this can only fire for an object built
    around the model validation (``model_construct``/duck typing).  Raising the typed
    rejection keeps that failure inside the engine vocabulary.
    """
    payload = event.payload
    if not isinstance(payload, model):
        raise UnsupportedTransitionError(
            f"event {event.event_id!r} of type {event.event_type.value!r} carries "
            f"{type(payload).__name__}, expected {model.__name__}"
        )
    return payload


def _find_belief_index(
    beliefs: list[PreferenceBelief],
    *,
    scope: PreferenceScope,
    kind: PreferenceKind,
    value: str,
    polarity: PreferencePolarity,
) -> int | None:
    """Index of the record with this belief identity, preferring an active record.

    The reducer maintains at most one record per ``(scope, kind, value, polarity)``
    identity; the active-first lookup is defensive against a hand-built state and keeps
    the in-place-update rule deterministic if such a duplicate ever appears.
    """
    target = (scope.value, kind.value, _normalised(value), polarity.value)
    fallback: int | None = None
    for index, belief in enumerate(beliefs):
        if belief.identity() != target:
            continue
        if belief.is_active:
            return index
        if fallback is None:
            fallback = index
    return fallback


def _updated_belief(
    belief: PreferenceBelief, *, event_id: str, **changes: Any
) -> PreferenceBelief:
    """Return a validated copy with provenance appended and ``changes`` applied."""
    data = belief.model_dump()
    data["supporting_event_ids"] = _append_unique(belief.supporting_event_ids, event_id)
    data["last_event_id"] = event_id
    data.update(changes)
    return PreferenceBelief.model_validate(data)


def _retracted_belief(belief: PreferenceBelief, *, event_id: str) -> PreferenceBelief:
    """Move a belief to ``RETRACTED`` / ``REMOVED`` (record kept, provenance appended)."""
    return _updated_belief(
        belief,
        event_id=event_id,
        lifecycle=PreferenceLifecycle.RETRACTED,
        status=PreferenceStatus.REMOVED,
    )


def _superseded_belief(belief: PreferenceBelief, *, event_id: str) -> PreferenceBelief:
    """Mark an explicit/confirmed belief ``SUPERSEDED`` (lifecycle preserved)."""
    return _updated_belief(
        belief, event_id=event_id, status=PreferenceStatus.SUPERSEDED
    )


def _authority_removal(belief: PreferenceBelief, *, event_id: str) -> PreferenceBelief:
    """Withdraw a belief due to an explicit user statement.

    An inference is retracted (it was never user truth); an explicit/confirmed record is
    only superseded, because the user's own statement is still part of the audit trail.
    """
    if belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS:
        return _retracted_belief(belief, event_id=event_id)
    return _superseded_belief(belief, event_id=event_id)


def _find_constraint_index(
    constraints: list[DecisionConstraint],
    *,
    scope: PreferenceScope,
    kind: PreferenceKind,
    value: str,
    polarity: PreferencePolarity,
) -> int | None:
    """Index of the constraint with this identity, preferring an active record."""
    target = (scope.value, kind.value, _normalised(value), polarity.value)
    fallback: int | None = None
    for index, constraint in enumerate(constraints):
        if constraint.identity() != target:
            continue
        if constraint.is_active:
            return index
        if fallback is None:
            fallback = index
    return fallback


def _updated_constraint(
    constraint: DecisionConstraint, *, event_id: str, **changes: Any
) -> DecisionConstraint:
    """Return a validated constraint copy with provenance appended."""
    data = constraint.model_dump()
    data["supporting_event_ids"] = _append_unique(
        constraint.supporting_event_ids, event_id
    )
    data["last_event_id"] = event_id
    data.update(changes)
    return DecisionConstraint.model_validate(data)


def _find_priority_index(
    priorities: list[FeaturePriority], target: tuple[str, str, str, str]
) -> int | None:
    """Index of the priority with this identity, preferring an active record."""
    fallback: int | None = None
    for index, priority in enumerate(priorities):
        if priority.identity() != target:
            continue
        if priority.is_active:
            return index
        if fallback is None:
            fallback = index
    return fallback


def _updated_priority(
    priority: FeaturePriority, *, event_id: str, **changes: Any
) -> FeaturePriority:
    """Return a validated priority copy with provenance appended."""
    data = priority.model_dump()
    data["supporting_event_ids"] = _append_unique(priority.supporting_event_ids, event_id)
    data["last_event_id"] = event_id
    data.update(changes)
    return FeaturePriority.model_validate(data)


def _find_knowledge_index(
    knowledge: list[KnowledgeBelief], attribute: str
) -> int | None:
    """Index of the knowledge record for ``attribute`` (normalised attribute identity)."""
    target = _normalised(attribute)
    for index, belief in enumerate(knowledge):
        if belief.identity() == target:
            return index
    return None


def _updated_knowledge(
    belief: KnowledgeBelief, *, event_id: str, **changes: Any
) -> KnowledgeBelief:
    """Return a validated knowledge copy with provenance appended."""
    data = belief.model_dump()
    data["supporting_event_ids"] = _append_unique(belief.supporting_event_ids, event_id)
    data["last_event_id"] = event_id
    data.update(changes)
    return KnowledgeBelief.model_validate(data)


# --------------------------------------------------------------------------- #
# The reducer
# --------------------------------------------------------------------------- #


class BeliefUpdater:
    """Pure deterministic reducer: ``apply(state, event) -> new state``.

    Instances are stateless and therefore shareable and reusable; ``apply_all`` folds a
    store-ordered stream in the order it is given (the reducer never sorts or re-times
    events, because the store's per-user ``sequence`` is the only ordering authority).
    """

    # -- public API ------------------------------------------------------- #

    def apply(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> UnifiedDecisionState:
        """Return the state produced by applying ``event``; raise if it is not legal."""
        self._validate_envelope(state, event)
        updates = self._dispatch(state, event)
        return self._seal(state, event, updates)

    def apply_all(
        self,
        state: UnifiedDecisionState,
        events: Iterable[StateEvent],
    ) -> UnifiedDecisionState:
        """Fold ``events`` over ``state`` in the given order and return the new state."""
        current = state
        for event in events:
            current = self.apply(current, event)
        return current

    # -- common envelope rules -------------------------------------------- #

    @staticmethod
    def _validate_envelope(state: UnifiedDecisionState, event: StateEvent) -> None:
        """Apply the rules that hold for every event, in the documented order."""
        if event.user_key != state.user_key:
            raise CrossUserStateError(
                f"event {event.event_id!r} belongs to user {event.user_key!r}, "
                f"not {state.user_key!r}"
            )
        if event.task_id != state.task_id and not event.applies_long_term:
            raise UnsupportedTransitionError(
                f"event {event.event_id!r} is scoped to task {event.task_id!r} and "
                f"has no long-term scope, so it cannot be applied to task "
                f"{state.task_id!r}"
            )
        if (
            event.sequence is not None
            and event.sequence <= state.interaction.last_event_sequence
        ):
            raise UnsupportedTransitionError(
                f"event {event.event_id!r} has sequence {event.sequence}, which is "
                f"not after the last applied sequence "
                f"{state.interaction.last_event_sequence}"
            )
        if (
            state.interaction.last_event_id is not None
            and event.event_id == state.interaction.last_event_id
        ):
            raise UnsupportedTransitionError(
                f"event {event.event_id!r} was already applied to this state "
                f"(duplicate last event id)"
            )
        if event.actor not in EVENT_ACTORS[event.event_type]:
            raise UnsupportedTransitionError(
                f"actor {event.actor.value!r} is not allowed to emit "
                f"{event.event_type.value!r}"
            )

    @staticmethod
    def _seal(
        state: UnifiedDecisionState,
        event: StateEvent,
        updates: dict[str, Any],
    ) -> UnifiedDecisionState:
        """Apply the accepted event's bookkeeping and return the new frozen state."""
        interaction = InteractionMetadata(
            event_count=state.interaction.event_count + 1,
            user_turn_count=state.interaction.user_turn_count
            + (1 if event.actor is EventActor.USER else 0),
            last_event_id=event.event_id,
            last_event_sequence=(
                event.sequence
                if event.sequence is not None
                else state.interaction.last_event_sequence
            ),
        )
        # Only an event from the state's own task may refresh its current session.
        session_id = (
            event.session_id if event.task_id == state.task_id else state.session_id
        )
        changes: dict[str, Any] = dict(updates)
        changes["interaction"] = interaction
        changes["session_id"] = session_id
        return state.model_copy(update=changes)

    def _dispatch(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Route a validated event to its documented transition rule."""
        event_type = event.event_type
        if event_type is EventType.USER_PREFERENCE_STATED:
            return self._apply_user_preference_stated(state, event)
        if event_type is EventType.PREFERENCE_HYPOTHESIS_PROPOSED:
            return self._apply_preference_hypothesis_proposed(state, event)
        if event_type is EventType.PREFERENCE_CONFIRMED:
            return self._apply_preference_confirmed(state, event)
        if event_type is EventType.PREFERENCE_RETRACTED:
            return self._apply_preference_retracted(state, event)
        if event_type is EventType.USER_CONSTRAINT_STATED:
            return self._apply_user_constraint_stated(state, event)
        if event_type is EventType.PREFERENCE_PRIORITY_STATED:
            return self._apply_preference_priority_stated(state, event)
        if event_type is EventType.PRODUCT_COMPARISON_FEEDBACK:
            return self._apply_product_comparison_feedback(state, event)
        if event_type is EventType.FEATURE_EXPLANATION_REQUESTED:
            return self._apply_feature_explanation_requested(state, event)
        if event_type is EventType.FEATURE_EXPLANATION_SHOWN:
            return self._apply_feature_explanation_shown(state, event)
        if event_type is EventType.USER_UNDERSTANDING_DEMONSTRATED:
            return self._apply_user_understanding_demonstrated(state, event)
        if event_type is EventType.TASK_CONTEXT_UPDATED:
            return self._apply_task_context_updated(state, event)
        if event_type is EventType.CANDIDATE_POOL_REFERENCED:
            return self._apply_candidate_pool_referenced(state, event)
        if event_type is EventType.RANKING_SNAPSHOT_REFERENCED:
            return self._apply_ranking_snapshot_referenced(state, event)
        raise UnsupportedTransitionError(
            f"event type {event_type.value!r} has no transition rule"
        )

    # -- preference beliefs ----------------------------------------------- #

    def _apply_user_preference_stated(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Explicit user statement: upsert an explicit belief and resolve conflicts.

        Conflict order is deliberate: first the named ``replaces_value`` correction,
        then a same-kind/value opposite-polarity conflict (inference retracted, explicit
        superseded), finally the upsert itself (which upgrades an existing inference of
        the same identity to ``EXPLICIT`` because the user just said it explicitly).
        """
        payload = _expect(event, UserPreferenceStated)
        scope = payload.scope
        target_value = _normalised(payload.value)
        beliefs = list(state.preference_beliefs)

        if payload.replaces_value is not None:
            replaced_value = _normalised(payload.replaces_value)
            for index, belief in enumerate(beliefs):
                if (
                    belief.is_active
                    and belief.scope is scope
                    and belief.kind is payload.kind
                    and _normalised(belief.value) == replaced_value
                ):
                    beliefs[index] = _authority_removal(
                        belief, event_id=event.event_id
                    )

        if payload.replace_same_kind:
            # M5.2: an accepted M9 REPLACE without a named target corrects the current
            # same-kind/same-polarity value.  Other values of the same kind and polarity
            # are superseded (or retracted when they were only an inference); different
            # polarities are handled by the contradiction rule below.
            for index, belief in enumerate(beliefs):
                if (
                    belief.is_active
                    and belief.scope is scope
                    and belief.kind is payload.kind
                    and belief.polarity is payload.polarity
                    and _normalised(belief.value) != target_value
                ):
                    if belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS:
                        beliefs[index] = _retracted_belief(
                            belief, event_id=event.event_id
                        )
                    elif belief.lifecycle in (
                        PreferenceLifecycle.EXPLICIT,
                        PreferenceLifecycle.CONFIRMED,
                    ):
                        beliefs[index] = _superseded_belief(
                            belief, event_id=event.event_id
                        )

        for index, belief in enumerate(beliefs):
            if (
                belief.is_active
                and belief.scope is scope
                and belief.kind is payload.kind
                and _normalised(belief.value) == target_value
                and belief.polarity != payload.polarity
            ):
                if belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS:
                    beliefs[index] = _retracted_belief(belief, event_id=event.event_id)
                elif belief.lifecycle in (
                    PreferenceLifecycle.EXPLICIT,
                    PreferenceLifecycle.CONFIRMED,
                ):
                    beliefs[index] = _superseded_belief(belief, event_id=event.event_id)

        index = _find_belief_index(
            beliefs,
            scope=scope,
            kind=payload.kind,
            value=payload.value,
            polarity=payload.polarity,
        )
        if index is None:
            beliefs.append(
                PreferenceBelief(
                    belief_id=preference_belief_id(
                        user_key=state.user_key,
                        task_id=state.task_id,
                        scope=scope,
                        kind=payload.kind,
                        value=payload.value,
                        polarity=payload.polarity,
                    ),
                    kind=payload.kind,
                    value=payload.value,
                    polarity=payload.polarity,
                    direction=payload.direction,
                    lifecycle=PreferenceLifecycle.EXPLICIT,
                    status=PreferenceStatus.ACTIVE,
                    scope=scope,
                    strength=BeliefStrength.STRONG,
                    origin=BeliefOrigin.USER_STATED,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            existing = beliefs[index]
            # An omitted direction means "not restated", so a known direction is kept.
            direction = (
                payload.direction
                if payload.direction is not None
                else existing.direction
            )
            beliefs[index] = _updated_belief(
                existing,
                event_id=event.event_id,
                direction=direction,
                lifecycle=PreferenceLifecycle.EXPLICIT,
                status=PreferenceStatus.ACTIVE,
                strength=BeliefStrength.STRONG,
                origin=BeliefOrigin.USER_STATED,
            )
        return {"preference_beliefs": tuple(beliefs)}

    def _apply_preference_hypothesis_proposed(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Soft system inference: upsert a hypothesis without ever overriding the user."""
        payload = _expect(event, PreferenceHypothesisProposed)
        scope = payload.scope
        # The payload model already refuses both; re-checked here so the reducer alone
        # guarantees "inference is never durable and never STRONG".
        if scope is PreferenceScope.LONG_TERM:
            raise UnsupportedTransitionError(
                "an inferred hypothesis may not be long_term: inference cannot create "
                "durable preference"
            )
        if payload.strength is BeliefStrength.STRONG:
            raise UnsupportedTransitionError(
                "an inferred hypothesis may not be STRONG: that strength is reserved for "
                "user-stated or user-confirmed beliefs"
            )
        target_value = _normalised(payload.value)
        beliefs = list(state.preference_beliefs)

        # An explicit user truth of the same kind/value always wins: same polarity keeps
        # the explicit record (provenance only); opposite polarity is a contradiction.
        for index, belief in enumerate(beliefs):
            if (
                not belief.is_active
                or belief.scope is not scope
                or belief.kind is not payload.kind
                or _normalised(belief.value) != target_value
            ):
                continue
            if belief.lifecycle not in (
                PreferenceLifecycle.EXPLICIT,
                PreferenceLifecycle.CONFIRMED,
            ):
                continue
            if belief.polarity == payload.polarity:
                beliefs[index] = _updated_belief(belief, event_id=event.event_id)
                return {"preference_beliefs": tuple(beliefs)}
            raise UnsupportedTransitionError(
                f"hypothesis {payload.kind.value!r}={payload.value!r} contradicts an "
                f"active explicit user statement of the opposite polarity"
            )

        # A competing inference of the same kind/value and opposite polarity is withdrawn
        # before the new hypothesis becomes active.
        for index, belief in enumerate(beliefs):
            if (
                belief.is_active
                and belief.scope is scope
                and belief.kind is payload.kind
                and _normalised(belief.value) == target_value
                and belief.polarity != payload.polarity
                and belief.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS
            ):
                beliefs[index] = _retracted_belief(belief, event_id=event.event_id)

        index = _find_belief_index(
            beliefs,
            scope=scope,
            kind=payload.kind,
            value=payload.value,
            polarity=payload.polarity,
        )
        if index is None:
            beliefs.append(
                PreferenceBelief(
                    belief_id=preference_belief_id(
                        user_key=state.user_key,
                        task_id=state.task_id,
                        scope=scope,
                        kind=payload.kind,
                        value=payload.value,
                        polarity=payload.polarity,
                    ),
                    kind=payload.kind,
                    value=payload.value,
                    polarity=payload.polarity,
                    lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
                    status=PreferenceStatus.ACTIVE,
                    scope=scope,
                    strength=payload.strength,
                    origin=BeliefOrigin.SYSTEM_INFERRED,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            existing = beliefs[index]
            if existing.lifecycle in (
                PreferenceLifecycle.EXPLICIT,
                PreferenceLifecycle.CONFIRMED,
            ):
                # Only an inactive record can reach this branch (active ones were handled
                # above); overwriting user evidence with an inference is never allowed.
                raise UnsupportedTransitionError(
                    "a hypothesis may not overwrite an explicit or confirmed belief "
                    "record of the same identity"
                )
            beliefs[index] = _updated_belief(
                existing,
                event_id=event.event_id,
                lifecycle=PreferenceLifecycle.INFERRED_HYPOTHESIS,
                status=PreferenceStatus.ACTIVE,
                strength=payload.strength,
                origin=BeliefOrigin.SYSTEM_INFERRED,
            )
        return {"preference_beliefs": tuple(beliefs)}

    def _apply_preference_confirmed(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Promote exactly one active inferred hypothesis to a confirmed belief."""
        payload = _expect(event, PreferenceConfirmed)
        beliefs = list(state.preference_beliefs)
        index = _find_belief_index(
            beliefs,
            scope=payload.scope,
            kind=payload.kind,
            value=payload.value,
            polarity=payload.polarity,
        )
        if index is None:
            raise UnsupportedTransitionError(
                f"cannot confirm {payload.kind.value!r}={payload.value!r}: no belief "
                f"with that identity exists"
            )
        belief = beliefs[index]
        if (
            not belief.is_active
            or belief.lifecycle is not PreferenceLifecycle.INFERRED_HYPOTHESIS
        ):
            raise UnsupportedTransitionError(
                f"only an active inferred hypothesis can be confirmed, but "
                f"{payload.kind.value!r}={payload.value!r} is "
                f"{belief.lifecycle.value!r}/{belief.status.value!r}"
            )
        beliefs[index] = _updated_belief(
            belief,
            event_id=event.event_id,
            lifecycle=PreferenceLifecycle.CONFIRMED,
            status=PreferenceStatus.ACTIVE,
            strength=BeliefStrength.STRONG,
            origin=BeliefOrigin.USER_CONFIRMED,
        )
        return {"preference_beliefs": tuple(beliefs)}

    def _apply_preference_retracted(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Withdraw active beliefs of one scope+kind, optionally narrowed by value.

        A statement that was both a preference belief and an explicitly authorized hard
        constraint is one user requirement with two projections.  A targeted retraction
        must withdraw both projections; leaving the constraint active would make the
        retracted value keep filtering candidates.  The match is restricted to the same
        scope, kind, optional value and optional polarity, so unrelated feature beliefs
        and constraints remain active.
        """
        payload = _expect(event, PreferenceRetracted)
        scope = payload.scope
        target_value = (
            _normalised(payload.value) if payload.value is not None else None
        )
        beliefs = list(state.preference_beliefs)
        matched = 0
        for index, belief in enumerate(beliefs):
            if not belief.is_active or belief.scope is not scope:
                continue
            if belief.kind is not payload.kind:
                continue
            if target_value is not None and _normalised(belief.value) != target_value:
                continue
            if payload.polarity is not None and belief.polarity != payload.polarity:
                continue
            beliefs[index] = _retracted_belief(belief, event_id=event.event_id)
            matched += 1

        constraints = list(state.hard_constraints)
        matched_constraints = 0
        for index, constraint in enumerate(constraints):
            if not constraint.is_active or constraint.scope is not scope:
                continue
            if constraint.kind is not payload.kind:
                continue
            if target_value is not None and _normalised(constraint.value) != target_value:
                continue
            if payload.polarity is not None and constraint.polarity != payload.polarity:
                continue
            constraints[index] = _updated_constraint(
                constraint,
                event_id=event.event_id,
                status=PreferenceStatus.SUPERSEDED,
            )
            matched_constraints += 1

        if not matched and not matched_constraints:
            raise UnsupportedTransitionError(
                f"retraction of {payload.kind.value!r} in scope {scope.value!r} "
                f"matched no active belief or constraint"
            )
        return {
            "preference_beliefs": tuple(beliefs),
            "hard_constraints": tuple(constraints),
        }

    # -- hard constraints -------------------------------------------------- #

    def _apply_user_constraint_stated(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Upsert an explicitly authorized hard constraint (the only way to create one).

        A task/user-scoped numeric budget of the same kind is a lifecycle replacement:
        the previous active value is retained as SUPERSEDED for provenance but can no
        longer enter an effective constraint projection.  Other kinds/values are left
        alone because multiple brands/categories may legitimately coexist.
        """
        payload = _expect(event, UserConstraintStated)
        constraints = list(state.hard_constraints)
        if payload.kind in {PreferenceKind.PRICE_MAX, PreferenceKind.PRICE_MIN}:
            for index, constraint in enumerate(constraints):
                if (
                    constraint.is_active
                    and constraint.scope is payload.scope
                    and constraint.kind is payload.kind
                    and _normalised(constraint.value) != _normalised(payload.value)
                ):
                    constraints[index] = _updated_constraint(
                        constraint,
                        event_id=event.event_id,
                        status=PreferenceStatus.SUPERSEDED,
                    )
        index = _find_constraint_index(
            constraints,
            scope=payload.scope,
            kind=payload.kind,
            value=payload.value,
            polarity=payload.polarity,
        )
        if index is None:
            constraints.append(
                DecisionConstraint(
                    constraint_id=decision_constraint_id(
                        user_key=state.user_key,
                        task_id=state.task_id,
                        scope=payload.scope,
                        kind=payload.kind,
                        value=payload.value,
                        polarity=payload.polarity,
                    ),
                    kind=payload.kind,
                    value=payload.value,
                    polarity=payload.polarity,
                    scope=payload.scope,
                    authority=ConstraintAuthority.USER_EXPLICIT,
                    status=PreferenceStatus.ACTIVE,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            constraints[index] = _updated_constraint(
                constraints[index],
                event_id=event.event_id,
                authority=ConstraintAuthority.USER_EXPLICIT,
                status=PreferenceStatus.ACTIVE,
            )
        return {"hard_constraints": tuple(constraints)}

    # -- relative feature importance --------------------------------------- #

    def _apply_preference_priority_stated(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Upsert a relative priority; a reverse statement wins over its reverse pair."""
        payload = _expect(event, PreferencePriorityStated)
        priorities = list(state.preference_priorities)
        scope = payload.scope
        reverse_identity = (
            scope.value,
            payload.kind.value,
            _normalised(payload.lower),
            _normalised(payload.higher),
        )
        target_identity = (
            scope.value,
            payload.kind.value,
            _normalised(payload.higher),
            _normalised(payload.lower),
        )
        for index, priority in enumerate(priorities):
            if priority.is_active and priority.identity() == reverse_identity:
                priorities[index] = _updated_priority(
                    priority,
                    event_id=event.event_id,
                    status=PreferenceStatus.SUPERSEDED,
                )
        index = _find_priority_index(priorities, target_identity)
        if index is None:
            priorities.append(
                FeaturePriority(
                    priority_id=feature_priority_id(
                        user_key=state.user_key,
                        task_id=state.task_id,
                        scope=scope,
                        kind=payload.kind,
                        higher=payload.higher,
                        lower=payload.lower,
                    ),
                    kind=payload.kind,
                    higher=payload.higher,
                    lower=payload.lower,
                    scope=scope,
                    status=PreferenceStatus.ACTIVE,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            priorities[index] = _updated_priority(
                priorities[index],
                event_id=event.event_id,
                status=PreferenceStatus.ACTIVE,
            )
        return {"preference_priorities": tuple(priorities)}

    # -- observations ------------------------------------------------------ #

    def _apply_product_comparison_feedback(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Append a comparison observation.  No belief, constraint or priority change."""
        payload = _expect(event, ProductComparisonFeedback)
        observation = DecisionObservation(
            observation_id=comparison_observation_id(event_id=event.event_id),
            kind=ObservationKind.PRODUCT_COMPARISON,
            compared_refs=tuple(payload.compared_refs),
            selected_ref=payload.selected_ref,
            feedback_text=payload.feedback_text,
            event_id=event.event_id,
        )
        return {"observations": state.observations + (observation,)}

    # -- knowledge beliefs ------------------------------------------------- #

    def _apply_feature_explanation_requested(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Mark an attribute as needing explanation without ever downgrading a proof."""
        payload = _expect(event, FeatureExplanationRequested)
        knowledge = list(state.knowledge_beliefs)
        index = _find_knowledge_index(knowledge, payload.attribute)
        if index is None:
            knowledge.append(
                KnowledgeBelief(
                    knowledge_id=knowledge_belief_id(
                        user_key=state.user_key, attribute=payload.attribute
                    ),
                    attribute=payload.attribute,
                    status=KnowledgeStatus.NEEDS_EXPLANATION,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            existing = knowledge[index]
            status = existing.status
            if status is not KnowledgeStatus.UNDERSTANDING_DEMONSTRATED:
                status = KnowledgeStatus.NEEDS_EXPLANATION
            knowledge[index] = _updated_knowledge(
                existing, event_id=event.event_id, status=status
            )
        return {"knowledge_beliefs": tuple(knowledge)}

    def _apply_feature_explanation_shown(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Record a displayed explanation: evidence of display, never of understanding."""
        payload = _expect(event, FeatureExplanationShown)
        knowledge = list(state.knowledge_beliefs)
        index = _find_knowledge_index(knowledge, payload.attribute)
        if index is None:
            knowledge.append(
                KnowledgeBelief(
                    knowledge_id=knowledge_belief_id(
                        user_key=state.user_key, attribute=payload.attribute
                    ),
                    attribute=payload.attribute,
                    status=KnowledgeStatus.EXPLANATION_PROVIDED,
                    supporting_event_ids=(event.event_id,),
                    explanation_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            existing = knowledge[index]
            status = existing.status
            if status in (
                KnowledgeStatus.UNKNOWN,
                KnowledgeStatus.NEEDS_EXPLANATION,
            ):
                status = KnowledgeStatus.EXPLANATION_PROVIDED
            knowledge[index] = _updated_knowledge(
                existing,
                event_id=event.event_id,
                status=status,
                explanation_event_ids=_append_unique(
                    existing.explanation_event_ids, event.event_id
                ),
            )
        return {"knowledge_beliefs": tuple(knowledge)}

    def _apply_user_understanding_demonstrated(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """The user demonstrated understanding; this is the only path to that status."""
        payload = _expect(event, UserUnderstandingDemonstrated)
        knowledge = list(state.knowledge_beliefs)
        index = _find_knowledge_index(knowledge, payload.attribute)
        if index is None:
            knowledge.append(
                KnowledgeBelief(
                    knowledge_id=knowledge_belief_id(
                        user_key=state.user_key, attribute=payload.attribute
                    ),
                    attribute=payload.attribute,
                    status=KnowledgeStatus.UNDERSTANDING_DEMONSTRATED,
                    supporting_event_ids=(event.event_id,),
                    last_event_id=event.event_id,
                )
            )
        else:
            knowledge[index] = _updated_knowledge(
                knowledge[index],
                event_id=event.event_id,
                status=KnowledgeStatus.UNDERSTANDING_DEMONSTRATED,
            )
        return {"knowledge_beliefs": tuple(knowledge)}

    # -- task context and read-only references ----------------------------- #

    def _apply_task_context_updated(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Replace the typed task context with the controller-supplied one."""
        payload = _expect(event, TaskContextUpdated)
        return {
            "task_context": TaskContext(
                intent=payload.intent,
                request_text=payload.request_text,
                updated_by_event_id=event.event_id,
            )
        }

    def _apply_candidate_pool_referenced(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Replace the read-only candidate-pool reference (latest trusted reference wins)."""
        payload = _expect(event, CandidatePoolReferenced)
        return {
            "candidate_pool_ref": CandidatePoolRef(
                ref=payload.ref,
                source=payload.source,
                candidate_count=payload.candidate_count,
                digest=payload.digest,
                snapshot_schema_version=payload.snapshot_schema_version,
                referenced_by_event_id=event.event_id,
            )
        }

    def _apply_ranking_snapshot_referenced(
        self, state: UnifiedDecisionState, event: StateEvent
    ) -> dict[str, Any]:
        """Replace the read-only ranking-snapshot reference (latest trusted wins)."""
        payload = _expect(event, RankingSnapshotReferenced)
        return {
            "ranking_snapshot_ref": RankingSnapshotRef(
                ref=payload.ref,
                source=payload.source,
                method=payload.method,
                candidate_count=payload.candidate_count,
                digest=payload.digest,
                snapshot_schema_version=payload.snapshot_schema_version,
                referenced_by_event_id=event.event_id,
            )
        }
