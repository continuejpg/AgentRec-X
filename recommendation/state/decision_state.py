"""Typed, versioned state models for the M5-MVP unified decision state.

``UnifiedDecisionState`` is the central state abstraction of AgentRec-X 3.0.  Memory is
**not** the centre of the agent; it is a projection built from validated user
information, and this model is what the decision agent and the recommendation layers are
given instead of an opaque profile string.

What the state owns
-------------------
* preference beliefs -- what the system currently believes the user prefers, with
  lifecycle, scope, discrete strength and event provenance;
* feature priorities -- relative statements such as "comfort matters more than weight";
* knowledge beliefs -- whether the user understands a product attribute;
* the typed task context;
* authorized hard constraints;
* interaction observations (comparison feedback), which are evidence and never beliefs;
* read-only references to trusted candidate-pool and ranking-snapshot outputs.

What the state deliberately cannot own
--------------------------------------
There is no field for:
* a product identity, a product row or fabricated catalogue data;
* a model score, a fused score or a rank from the recommendation engine; candidate and
  ranking outputs are opaque references with an optional integrity digest;
* the trusted chronological purchase history SASRec is trained on (that data belongs to
  the application and is never written here);
* raw memory-store records (the legacy adapter projects them read-only).

Every belief carries the event ids that support it, so a decision can always be traced
back to the user statement or the trusted evidence that produced it.  Retraction and
supersession change status; nothing is silently deleted.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import (
    BeliefOrigin,
    BeliefStrength,
    ConstraintAuthority,
    DIGEST_LENGTH,
    HARD_CONSTRAINT_AUTHORITIES,
    KnowledgeStatus,
    MAX_REF_LENGTH,
    MAX_REQUEST_TEXT_LENGTH,
    MAX_SESSION_ID_LENGTH,
    MAX_SOURCE_LENGTH,
    MAX_TASK_ID_LENGTH,
    MAX_TASK_INTENT_LENGTH,
    MAX_USER_KEY_LENGTH,
    ObservationKind,
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
    STATE_SCHEMA_VERSION,
    bounded_identifier,
    bounded_text,
    canonical_json,
    clean_attribute,
    clean_preference_value,
    deterministic_id,
    redacted_text,
)

__all__ = [
    "CandidatePoolRef",
    "DecisionConstraint",
    "DecisionObservation",
    "FeaturePriority",
    "InteractionMetadata",
    "KnowledgeBelief",
    "PreferenceBelief",
    "RankingSnapshotRef",
    "TaskContext",
    "UnifiedDecisionState",
    "comparison_observation_id",
    "decision_constraint_id",
    "empty_state",
    "feature_priority_id",
    "knowledge_belief_id",
    "preference_belief_id",
]

_INTENT_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,%d}$" % (MAX_TASK_INTENT_LENGTH - 1))
_DIGEST = re.compile(r"^[0-9a-f]{%d}$" % DIGEST_LENGTH)


# --------------------------------------------------------------------------- #
# Preference beliefs
# --------------------------------------------------------------------------- #


class PreferenceBelief(BaseModel):
    """One belief about what the user prefers, with lifecycle and provenance.

    Identity is ``(scope, kind, value, polarity)``.  Updating a belief keeps its
    ``belief_id`` and its position in the state tuple; a material change (e.g. the user
    corrects an inferred hypothesis) produces a lifecycle/status transition on the same
    record so the audit trail stays continuous.

    ``strength`` is ordinal (see :class:`~recommendation.state.schemas.BeliefStrength`),
    never a calibrated probability.  An ``INFERRED_HYPOTHESIS`` is soft evidence: it may
    order candidates or trigger a clarification, and it may never be read as a hard
    constraint or as a durable explicit preference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    direction: PreferenceDirection | None = None
    lifecycle: PreferenceLifecycle
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    scope: PreferenceScope = PreferenceScope.TASK
    strength: BeliefStrength
    origin: BeliefOrigin
    #: Event ids that support this belief, in application order, de-duplicated.
    supporting_event_ids: tuple[str, ...] = ()
    #: The last event that changed this belief (audit convenience).
    last_event_id: str | None = None
    #: Set only by the read-only legacy-memory adapter, for traceability.
    legacy_memory_id: str | None = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return clean_preference_value(value)

    @field_validator("belief_id")
    @classmethod
    def _validate_belief_id(cls, value: object) -> str:
        return bounded_identifier(value, field="belief_id", max_length=MAX_REF_LENGTH)

    @property
    def is_active(self) -> bool:
        """True when this belief currently contributes to the state."""
        return self.status is PreferenceStatus.ACTIVE

    @property
    def is_soft_signal(self) -> bool:
        """True when this belief may only order candidates, never filter them."""
        return self.lifecycle is PreferenceLifecycle.INFERRED_HYPOTHESIS

    @property
    def may_be_committed(self) -> bool:
        """True when this belief could later be proposed for durable memory."""
        return (
            self.is_active
            and self.scope is PreferenceScope.LONG_TERM
            and self.lifecycle.may_be_committed
        )

    def identity(self) -> tuple[str, str, str, str]:
        """Deterministic identity used for update/conflict resolution."""
        return (
            self.scope.value,
            self.kind.value,
            " ".join(self.value.split()).casefold(),
            self.polarity.value,
        )


# --------------------------------------------------------------------------- #
# Feature priorities
# --------------------------------------------------------------------------- #


class FeaturePriority(BaseModel):
    """A relative preference: ``higher`` matters more than ``lower`` within a scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority_id: str
    kind: PreferenceKind = PreferenceKind.FEATURE
    higher: str
    lower: str
    scope: PreferenceScope = PreferenceScope.TASK
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    supporting_event_ids: tuple[str, ...] = ()
    last_event_id: str | None = None

    @field_validator("higher", "lower")
    @classmethod
    def _validate_feature(cls, value: object) -> str:
        return clean_attribute(value)

    @field_validator("priority_id")
    @classmethod
    def _validate_priority_id(cls, value: object) -> str:
        return bounded_identifier(value, field="priority_id", max_length=MAX_REF_LENGTH)

    @property
    def is_active(self) -> bool:
        """True when this priority currently contributes to the state."""
        return self.status is PreferenceStatus.ACTIVE

    def identity(self) -> tuple[str, str, str, str]:
        """Deterministic identity used for update/conflict resolution."""
        return (
            self.scope.value,
            self.kind.value,
            " ".join(self.higher.split()).casefold(),
            " ".join(self.lower.split()).casefold(),
        )


# --------------------------------------------------------------------------- #
# Knowledge beliefs
# --------------------------------------------------------------------------- #


class KnowledgeBelief(BaseModel):
    """Estimate of whether the user understands one product attribute.

    An explanation being *shown* is recorded (it is useful context for the agent), but
    it only ever moves the status to ``EXPLANATION_PROVIDED``.  Only
    ``USER_UNDERSTANDING_DEMONSTRATED`` may produce
    ``UNDERSTANDING_DEMONSTRATED``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_id: str
    attribute: str
    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    supporting_event_ids: tuple[str, ...] = ()
    explanation_event_ids: tuple[str, ...] = ()
    last_event_id: str | None = None

    @field_validator("attribute")
    @classmethod
    def _validate_attribute(cls, value: object) -> str:
        return clean_attribute(value)

    @field_validator("knowledge_id")
    @classmethod
    def _validate_knowledge_id(cls, value: object) -> str:
        return bounded_identifier(value, field="knowledge_id", max_length=MAX_REF_LENGTH)

    @property
    def is_understood(self) -> bool:
        """True only when the user demonstrated understanding explicitly."""
        return self.status is KnowledgeStatus.UNDERSTANDING_DEMONSTRATED

    def identity(self) -> str:
        """Deterministic identity: the normalised attribute."""
        return " ".join(self.attribute.split()).casefold()


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #


class DecisionConstraint(BaseModel):
    """A hard constraint authorized by an explicit user statement.

    Only :data:`~recommendation.state.schemas.HARD_CONSTRAINT_AUTHORITIES` may define
    feasibility.  In M5-MVP that is ``USER_EXPLICIT`` alone -- an inferred hypothesis
    has no path to this model, because only ``USER_CONSTRAINT_STATED`` creates one.
    A belief (even a confirmed one) is a ranking signal, not a filter, so it is stored
    in ``preference_beliefs`` and never here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str
    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    scope: PreferenceScope = PreferenceScope.TASK
    authority: ConstraintAuthority = ConstraintAuthority.USER_EXPLICIT
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    supporting_event_ids: tuple[str, ...] = ()
    last_event_id: str | None = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return clean_preference_value(value)

    @field_validator("constraint_id")
    @classmethod
    def _validate_constraint_id(cls, value: object) -> str:
        return bounded_identifier(value, field="constraint_id", max_length=MAX_REF_LENGTH)

    @property
    def is_active(self) -> bool:
        """True when this constraint currently contributes to the state."""
        return self.status is PreferenceStatus.ACTIVE

    @property
    def is_authorized_hard(self) -> bool:
        """True only for an active, explicitly user-authorized constraint."""
        return (
            self.status is PreferenceStatus.ACTIVE
            and self.authority in HARD_CONSTRAINT_AUTHORITIES
        )

    def identity(self) -> tuple[str, str, str, str]:
        """Deterministic identity used for update/conflict resolution."""
        return (
            self.scope.value,
            self.kind.value,
            " ".join(self.value.split()).casefold(),
            self.polarity.value,
        )


# --------------------------------------------------------------------------- #
# Observations and read-only references
# --------------------------------------------------------------------------- #


class DecisionObservation(BaseModel):
    """A recorded interaction observation (e.g. a product comparison choice).

    An observation is evidence, not a conclusion.  It records what happened and keeps
    the event id; it carries no preference field and exposes
    :attr:`implied_preferences` as an always-empty tuple so that "a choice implies a
    feature preference" cannot be read out of this model by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    kind: ObservationKind = ObservationKind.PRODUCT_COMPARISON
    compared_refs: tuple[str, ...] = ()
    selected_ref: str | None = None
    feedback_text: str = ""
    event_id: str

    @field_validator("observation_id", "event_id")
    @classmethod
    def _validate_ids(cls, value: object, info) -> str:
        return bounded_identifier(value, field=info.field_name, max_length=MAX_REF_LENGTH)

    @field_validator("compared_refs")
    @classmethod
    def _validate_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            bounded_identifier(ref, field="compared_refs item", max_length=MAX_REF_LENGTH)
            for ref in value
        )

    @field_validator("feedback_text")
    @classmethod
    def _validate_feedback(cls, value: object) -> str:
        return bounded_text(value, field="feedback_text", max_length=500)

    @property
    def implied_preferences(self) -> tuple[()]:
        """Always empty: an observation cannot prove a preference by itself."""
        return ()


class CandidatePoolRef(BaseModel):
    """Read-only reference to a candidate pool produced by a trusted component.

    The decision state stores an opaque reference, the producing component name, the
    size and an optional integrity digest -- never a product identity or a score.
    ``read_only`` is typed as ``Literal[True]``, so the model cannot represent a
    writable reference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    source: str
    candidate_count: int = Field(ge=0)
    digest: str | None = None
    #: Version of the trusted candidate-snapshot contract this reference points at.
    snapshot_schema_version: int = Field(default=1, ge=1, le=1000)
    referenced_by_event_id: str | None = None
    read_only: Literal[True] = True

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: object) -> str:
        return bounded_identifier(value, field="ref", max_length=MAX_REF_LENGTH)

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return bounded_identifier(value, field="source", max_length=MAX_SOURCE_LENGTH)

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ValueError("digest must be a lowercase 64-character SHA-256 hex digest")
        return value


class RankingSnapshotRef(BaseModel):
    """Read-only reference to a ranking snapshot produced by the trusted engine.

    ``method`` names the producing contract (e.g. ``rrf_rank_fusion``).  There is no
    field for a per-item rank or score: the snapshot itself stays in the trusted
    recommendation layer, and this state only points at it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    source: str
    method: str
    candidate_count: int = Field(ge=0)
    digest: str | None = None
    #: Version of the trusted ranking-snapshot contract this reference points at.
    snapshot_schema_version: int = Field(default=1, ge=1, le=1000)
    referenced_by_event_id: str | None = None
    read_only: Literal[True] = True

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: object) -> str:
        return bounded_identifier(value, field="ref", max_length=MAX_REF_LENGTH)

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return bounded_identifier(value, field="source", max_length=MAX_SOURCE_LENGTH)

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: object) -> str:
        return bounded_identifier(value, field="method", max_length=MAX_SOURCE_LENGTH)

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ValueError("digest must be a lowercase 64-character SHA-256 hex digest")
        return value


# --------------------------------------------------------------------------- #
# Context and interaction metadata
# --------------------------------------------------------------------------- #


class TaskContext(BaseModel):
    """Typed context of the current task.

    ``intent`` is a bounded label whose vocabulary is owned by the accepted
    ``TaskIntent`` enum; it is stored as a label rather than a duplicated enum so this
    module stays importable without the model stack.  ``request_text`` is untrusted
    user words, bounded and credential-redacted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = "unknown"
    request_text: str = ""
    updated_by_event_id: str | None = None

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, value: object) -> str:
        if not isinstance(value, str) or not _INTENT_LABEL.fullmatch(value.strip()):
            raise ValueError("intent must be a lowercase label such as 'recommend'")
        return value.strip()

    @field_validator("request_text")
    @classmethod
    def _validate_request_text(cls, value: object) -> str:
        return redacted_text(
            value, field="request_text", max_length=MAX_REQUEST_TEXT_LENGTH
        )

    @field_validator("updated_by_event_id")
    @classmethod
    def _validate_event_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return bounded_identifier(value, field="updated_by_event_id", max_length=MAX_REF_LENGTH)


class InteractionMetadata(BaseModel):
    """Deterministic interaction counters and the last applied event marker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_count: int = Field(default=0, ge=0)
    user_turn_count: int = Field(default=0, ge=0)
    last_event_id: str | None = None
    last_event_sequence: int = Field(default=0, ge=0)

    @field_validator("last_event_id")
    @classmethod
    def _validate_last_event_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return bounded_identifier(value, field="last_event_id", max_length=MAX_REF_LENGTH)


# --------------------------------------------------------------------------- #
# The unified state
# --------------------------------------------------------------------------- #


class UnifiedDecisionState(BaseModel):
    """The versioned, immutable state every downstream decision is projected from.

    The state is produced only by replaying events through
    :class:`~recommendation.state.belief_updater.BeliefUpdater`.  It is frozen, so a
    consumer cannot patch a belief in place; a transition returns a new state.  Its
    tuple fields preserve first-appearance order and are updated in place on repeat
    events, which keeps replay deterministic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Version of the state contract (``STATE_SCHEMA_VERSION``).
    state_version: int = STATE_SCHEMA_VERSION
    user_key: str
    session_id: str | None = None
    task_id: str
    preference_beliefs: tuple[PreferenceBelief, ...] = ()
    preference_priorities: tuple[FeaturePriority, ...] = ()
    knowledge_beliefs: tuple[KnowledgeBelief, ...] = ()
    task_context: TaskContext = Field(default_factory=TaskContext)
    hard_constraints: tuple[DecisionConstraint, ...] = ()
    observations: tuple[DecisionObservation, ...] = ()
    candidate_pool_ref: CandidatePoolRef | None = None
    ranking_snapshot_ref: RankingSnapshotRef | None = None
    interaction: InteractionMetadata = Field(default_factory=InteractionMetadata)

    @field_validator("state_version")
    @classmethod
    def _validate_version(cls, value: object) -> int:
        if value != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"state version {value} is not supported "
                f"(expected {STATE_SCHEMA_VERSION})"
            )
        return value

    @field_validator("user_key")
    @classmethod
    def _validate_user_key(cls, value: object) -> str:
        return bounded_identifier(value, field="user_key", max_length=MAX_USER_KEY_LENGTH)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: object) -> str:
        return bounded_identifier(value, field="task_id", max_length=MAX_TASK_ID_LENGTH)

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return bounded_identifier(value, field="session_id", max_length=MAX_SESSION_ID_LENGTH)

    # -- projections over the state --------------------------------------- #

    @property
    def active_preference_beliefs(self) -> tuple[PreferenceBelief, ...]:
        """Active preference beliefs, in first-appearance order."""
        return tuple(belief for belief in self.preference_beliefs if belief.is_active)

    @property
    def inferred_hypotheses(self) -> tuple[PreferenceBelief, ...]:
        """Active inferred hypotheses (soft evidence only)."""
        return tuple(
            belief for belief in self.active_preference_beliefs if belief.is_soft_signal
        )

    @property
    def explicit_or_confirmed_beliefs(self) -> tuple[PreferenceBelief, ...]:
        """Active beliefs the user stated or confirmed."""
        return tuple(
            belief
            for belief in self.active_preference_beliefs
            if belief.lifecycle.may_be_committed
        )

    @property
    def active_priorities(self) -> tuple[FeaturePriority, ...]:
        """Active relative feature priorities."""
        return tuple(priority for priority in self.preference_priorities if priority.is_active)

    @property
    def active_knowledge_beliefs(self) -> tuple[KnowledgeBelief, ...]:
        """Knowledge beliefs that are no longer ``UNKNOWN``."""
        return tuple(
            belief for belief in self.knowledge_beliefs if belief.status is not KnowledgeStatus.UNKNOWN
        )

    @property
    def attributes_needing_explanation(self) -> tuple[str, ...]:
        """Attributes whose knowledge state calls for an explanation next."""
        return tuple(
            belief.attribute
            for belief in self.knowledge_beliefs
            if belief.status is KnowledgeStatus.NEEDS_EXPLANATION
        )

    @property
    def authorized_hard_constraints(self) -> tuple[DecisionConstraint, ...]:
        """Only constraints authorized to define feasibility."""
        return tuple(
            constraint for constraint in self.hard_constraints if constraint.is_authorized_hard
        )

    # -- serialization ----------------------------------------------------- #

    def to_canonical_json(self) -> str:
        """Return the canonical JSON encoding of this state."""
        return canonical_json(self.model_dump(mode="json"))

    def state_digest(self) -> str:
        """SHA-256 of the canonical state encoding.

        Two replays of the same event stream must produce the same digest; the digest
        is the deterministic-replay check used by tests and the smoke example.
        """
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Construction and deterministic ids
# --------------------------------------------------------------------------- #


def empty_state(
    *, user_key: str, task_id: str, session_id: str | None = None
) -> UnifiedDecisionState:
    """Return the empty state for a task, validating identity bounds."""
    return UnifiedDecisionState(
        user_key=user_key,
        session_id=session_id,
        task_id=task_id,
    )


def _scope_identity_part(scope: PreferenceScope, task_id: str) -> str:
    """Task part of a deterministic id: empty for long-term, task id otherwise."""
    return "*" if scope is PreferenceScope.LONG_TERM else task_id


def preference_belief_id(
    *,
    user_key: str,
    task_id: str,
    scope: PreferenceScope,
    kind: PreferenceKind,
    value: str,
    polarity: PreferencePolarity,
) -> str:
    """Deterministic id of a preference belief identity."""
    return deterministic_id(
        "pb",
        user_key,
        _scope_identity_part(scope, task_id),
        scope.value,
        kind.value,
        " ".join(value.split()).casefold(),
        polarity.value,
    )


def feature_priority_id(
    *,
    user_key: str,
    task_id: str,
    scope: PreferenceScope,
    kind: PreferenceKind,
    higher: str,
    lower: str,
) -> str:
    """Deterministic id of a feature-priority identity."""
    return deterministic_id(
        "fp",
        user_key,
        _scope_identity_part(scope, task_id),
        scope.value,
        kind.value,
        " ".join(higher.split()).casefold(),
        " ".join(lower.split()).casefold(),
    )


def knowledge_belief_id(*, user_key: str, attribute: str) -> str:
    """Deterministic id of a knowledge belief (knowledge is per user and attribute)."""
    return deterministic_id(
        "kb", user_key, " ".join(attribute.split()).casefold()
    )


def decision_constraint_id(
    *,
    user_key: str,
    task_id: str,
    scope: PreferenceScope,
    kind: PreferenceKind,
    value: str,
    polarity: PreferencePolarity,
) -> str:
    """Deterministic id of a hard-constraint identity."""
    return deterministic_id(
        "dc",
        user_key,
        _scope_identity_part(scope, task_id),
        scope.value,
        kind.value,
        " ".join(value.split()).casefold(),
        polarity.value,
    )


def comparison_observation_id(*, event_id: str) -> str:
    """Deterministic id of a comparison observation, derived from its event id."""
    return deterministic_id("ob", event_id)
