"""Typed events and payloads for the M5-MVP unified decision state engine.

Every state change is an *event*.  The event envelope is versioned, user/task scoped
and actor attributed; the payload is a closed Pydantic model selected by
:class:`EventType`.  An event is the only input the reducer accepts, so a caller cannot
mutate a state object directly and cannot smuggle a product id, a model score or a
preference into a field the payload does not define (``extra="forbid"``).

Design rules frozen here
------------------------
* **One event type, one payload model.**  :data:`EVENT_PAYLOAD_MODELS` is the single
  registry; :class:`StateEvent` coerces and then verifies the payload type, so a
  mismatched ``(event_type, payload)`` pair is a hard validation error.
* **Actor authority is declarative.**  :data:`EVENT_ACTORS` names the actors allowed to
  emit each event type.  A user cannot "propose a system hypothesis"; only the agent,
  the system or a trusted component can.  A user *can* state a preference, confirm a
  hypothesis, retract a belief, state a constraint, or demonstrate understanding.
* **Opaque references only.**  Candidate-pool and ranking-snapshot events carry an
  opaque reference, a source label, a count and an optional SHA-256 digest.  They have
  **no field for a product identity or a model score**, so a fabricated product row or
  score cannot be stored in the decision state.
* **Raw text is bounded and redacted.**  User-authored spans are truncated to the
  accepted M9 bound and replaced with ``[redacted]`` when they look like a credential.
* **Ordering is not time.**  ``timestamp`` is event metadata recorded for audit; the
  deterministic order is the store-assigned ``sequence``.  A timestamp can never change
  reduction order or state identity.
"""

from __future__ import annotations

import hashlib
import math
import re
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from recommendation.memory.schemas import MAX_SOURCE_TEXT_LENGTH

from .schemas import (
    BeliefStrength,
    HypothesisBasis,
    MAX_COMPARED_REFS,
    MAX_EVENT_ID_LENGTH,
    MAX_EVIDENCE_REFS,
    MAX_FEEDBACK_TEXT_LENGTH,
    MAX_REF_LENGTH,
    MAX_REASON_LENGTH,
    MAX_REQUEST_TEXT_LENGTH,
    MAX_SESSION_ID_LENGTH,
    MAX_SOURCE_LENGTH,
    MAX_TASK_ID_LENGTH,
    MAX_TASK_INTENT_LENGTH,
    MAX_USER_KEY_LENGTH,
    MIN_COMPARED_REFS,
    PreferenceDirection,
    PreferenceKind,
    PreferencePolarity,
    PreferenceScope,
    bounded_identifier,
    bounded_text,
    canonical_json,
    clean_attribute,
    clean_preference_value,
    redacted_text,
)

__all__ = [
    "EVENT_ACTORS",
    "EVENT_PAYLOAD_MODELS",
    "EVENT_SCHEMA_VERSION",
    "CandidatePoolReferenced",
    "EventActor",
    "EventType",
    "FeatureExplanationRequested",
    "FeatureExplanationShown",
    "PreferenceConfirmed",
    "PreferenceHypothesisProposed",
    "PreferencePriorityStated",
    "PreferenceRetracted",
    "ProductComparisonFeedback",
    "RankingSnapshotReferenced",
    "StateEvent",
    "StateEventPayload",
    "TaskContextUpdated",
    "UserConstraintStated",
    "UserPreferenceStated",
    "UserUnderstandingDemonstrated",
    "new_event",
    "payload_model_for",
]

#: Version of the event envelope/payload contract.  Bumped when a stored payload shape
#: changes; the event store records it per row and refuses an unknown value on read.
EVENT_SCHEMA_VERSION = 1

#: Bound on the number of compared candidate references in one comparison observation.
_MAX_COMPARISON_REFS = MAX_COMPARED_REFS


class EventActor(str, Enum):
    """Who emitted an event.  Recorded for authority and audit, never inferred."""

    #: The user, through a validated interpretation step.
    USER = "user"
    #: The recommendation agent (may propose hypotheses, describe the task, show text).
    AGENT = "agent"
    #: A trusted deterministic component (recommendation engine, catalogue, ledger).
    TRUSTED_COMPONENT = "trusted_component"
    #: The system itself (e.g. offline replay bookkeeping); never user-authored.
    SYSTEM = "system"


class EventType(str, Enum):
    """The closed event vocabulary of M5-MVP.

    Every member has clear semantics and a reducer rule.  The first nine are the
    minimum required by the milestone; the remainder exist because the state model has
    fields that must come from somewhere, and inventing them by side effect would be
    worse than naming them:

    * ``PREFERENCE_PRIORITY_STATED`` -- a relative statement such as "comfort matters
      more than weight".  It is not a new preference value, so it gets its own event.
    * ``TASK_CONTEXT_UPDATED`` -- the typed task context (intent label, bounded request
      text) is supplied by the controller, not derived from a preference event.
    * ``CANDIDATE_POOL_REFERENCED`` / ``RANKING_SNAPSHOT_REFERENCED`` -- read-only
      references to trusted recommendation-engine outputs.
    """

    # -- preference beliefs ------------------------------------------------ #
    USER_PREFERENCE_STATED = "user_preference_stated"
    PREFERENCE_HYPOTHESIS_PROPOSED = "preference_hypothesis_proposed"
    PREFERENCE_CONFIRMED = "preference_confirmed"
    PREFERENCE_RETRACTED = "preference_retracted"
    # -- hard constraints -------------------------------------------------- #
    USER_CONSTRAINT_STATED = "user_constraint_stated"
    # -- relative feature importance --------------------------------------- #
    PREFERENCE_PRIORITY_STATED = "preference_priority_stated"
    # -- observations (never beliefs) -------------------------------------- #
    PRODUCT_COMPARISON_FEEDBACK = "product_comparison_feedback"
    # -- knowledge state --------------------------------------------------- #
    FEATURE_EXPLANATION_REQUESTED = "feature_explanation_requested"
    FEATURE_EXPLANATION_SHOWN = "feature_explanation_shown"
    USER_UNDERSTANDING_DEMONSTRATED = "user_understanding_demonstrated"
    # -- task context / trusted references --------------------------------- #
    TASK_CONTEXT_UPDATED = "task_context_updated"
    CANDIDATE_POOL_REFERENCED = "candidate_pool_referenced"
    RANKING_SNAPSHOT_REFERENCED = "ranking_snapshot_referenced"


class StateEventPayload(BaseModel):
    """Base class for every payload model.  Closed: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Shared preference target
# --------------------------------------------------------------------------- #


class _PreferenceTarget(StateEventPayload):
    """A preference identity: kind + cleaned value + polarity."""

    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity = PreferencePolarity.PREFER

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return clean_preference_value(value)


class _SourceText(StateEventPayload):
    """A user-authored support span, bounded and credential-redacted."""

    source_text: str

    @field_validator("source_text")
    @classmethod
    def _validate_source_text(cls, value: object, info: ValidationInfo) -> str:
        return redacted_text(
            value,
            field=info.field_name or "source_text",
            max_length=MAX_SOURCE_TEXT_LENGTH,
        )


# --------------------------------------------------------------------------- #
# Preference events
# --------------------------------------------------------------------------- #


class UserPreferenceStated(_PreferenceTarget, _SourceText):
    """The user explicitly stated a preference.

    Default scope is ``TASK`` and **not** ``LONG_TERM``: a normal request must not
    silently become durable memory.  A long-term statement has to say so.  This is the
    reversible default the milestone asks for.

    ``replaces_value`` is the only way a statement can supersede an earlier value
    without a separate retraction; the corrector must name what it corrects, so a
    second independent preference is never mistaken for a correction.
    """

    scope: PreferenceScope = PreferenceScope.TASK
    direction: PreferenceDirection | None = None
    replaces_value: str | None = None
    #: M5.2: the accepted M9 ``REPLACE`` mode never names the corrected value when the
    #: statement is a general correction ("actually I prefer blue instead").  When true,
    #: the reducer supersedes the active same-kind/same-polarity belief(s) with a different
    #: value.  It is mutually exclusive with the named ``replaces_value`` selector.
    replace_same_kind: bool = False

    @field_validator("replaces_value")
    @classmethod
    def _validate_replaces(cls, value: object) -> str | None:
        if value is None:
            return None
        return clean_preference_value(value)

    @model_validator(mode="after")
    def _validate_correction_selector(self) -> UserPreferenceStated:
        if self.replace_same_kind and self.replaces_value is not None:
            raise ValueError(
                "a statement carries either a named replaces_value or replace_same_kind, "
                "not both"
            )
        return self


class PreferenceHypothesisProposed(_PreferenceTarget):
    """The system proposed a *hypothesis*; it is not truth and must not become hard.

    The scope is restricted to ``SESSION``/``TASK`` and the strength to
    ``WEAK``/``MODERATE``.  A long-term inference would silently create durable
    preference and a ``STRONG`` inference would pretend to be calibrated evidence, so
    both are refused by the payload itself (and re-checked by the reducer).
    """

    scope: PreferenceScope = PreferenceScope.TASK
    strength: BeliefStrength = BeliefStrength.WEAK
    basis: HypothesisBasis
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: object, info: ValidationInfo) -> str:
        return redacted_text(value, field=info.field_name or "reason", max_length=MAX_REASON_LENGTH)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > MAX_EVIDENCE_REFS:
            raise ValueError(f"at most {MAX_EVIDENCE_REFS} evidence refs are allowed")
        return tuple(
            bounded_identifier(ref, field="evidence_refs item", max_length=MAX_REF_LENGTH)
            for ref in value
        )

    @model_validator(mode="after")
    def _validate_inference_limits(self) -> PreferenceHypothesisProposed:
        if self.scope is PreferenceScope.LONG_TERM:
            raise ValueError(
                "an inferred hypothesis must not be long_term: inference cannot create "
                "durable preference"
            )
        if self.strength is BeliefStrength.STRONG:
            raise ValueError(
                "an inferred hypothesis must not be strong: STRONG is reserved for "
                "user-stated or user-confirmed beliefs"
            )
        return self


class PreferenceConfirmed(_PreferenceTarget, _SourceText):
    """The user confirmed an existing inferred hypothesis (same identity **and scope**).

    ``scope`` is required for the target identity: a hypothesis is task/session scoped,
    so confirmation must name which scoped hypothesis it confirms.  The accepted default
    is ``TASK`` (the same reversible default as the other preference events).
    """

    scope: PreferenceScope = PreferenceScope.TASK


class PreferenceRetracted(_SourceText):
    """The user withdrew a belief.

    A retraction is recorded, not deleted: the belief keeps its provenance and moves to
    the retracted lifecycle.  ``value`` is optional, so "colour does not matter any
    more" retracts the whole kind; ``polarity`` is an optional narrowing filter.
    """

    kind: PreferenceKind
    value: str | None = None
    polarity: PreferencePolarity | None = None
    scope: PreferenceScope = PreferenceScope.TASK

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str | None:
        if value is None:
            return None
        return clean_preference_value(value)


# --------------------------------------------------------------------------- #
# Constraints and relative importance
# --------------------------------------------------------------------------- #


class UserConstraintStated(_PreferenceTarget, _SourceText):
    """The user stated a constraint that may define feasibility.

    This is the **only** M5-MVP event that can create a hard constraint.  A hypothesis
    cannot express a constraint, which is what structurally prevents an inference from
    becoming a filter.
    """

    scope: PreferenceScope = PreferenceScope.TASK


class PreferencePriorityStated(_SourceText):
    """The user ranked one feature above another within a scope."""

    higher: str
    lower: str
    kind: PreferenceKind = PreferenceKind.FEATURE
    scope: PreferenceScope = PreferenceScope.TASK

    @field_validator("higher", "lower")
    @classmethod
    def _validate_feature(cls, value: object, info: ValidationInfo) -> str:
        return clean_attribute(value)

    @model_validator(mode="after")
    def _validate_distinct(self) -> PreferencePriorityStated:
        if self.higher.casefold() == self.lower.casefold():
            raise ValueError("higher and lower feature must be different")
        return self


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


class ProductComparisonFeedback(StateEventPayload):
    """The user chose between products or gave comparison feedback.

    This is an **observation**.  It deliberately carries no feature, no preference and
    no attribute mention: recording a choice must not, by itself, prove a specific
    feature preference.  ``compared_refs``/``selected_ref`` are opaque references to a
    candidate pool the trusted layer produced, not product identities owned here.
    """

    compared_refs: tuple[str, ...]
    selected_ref: str | None = None
    feedback_text: str = ""

    @field_validator("compared_refs")
    @classmethod
    def _validate_compared_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not (MIN_COMPARED_REFS <= len(value) <= _MAX_COMPARISON_REFS):
            raise ValueError(
                f"a comparison carries between {MIN_COMPARED_REFS} and "
                f"{_MAX_COMPARISON_REFS} candidate refs"
            )
        cleaned = tuple(
            bounded_identifier(ref, field="compared_refs item", max_length=MAX_REF_LENGTH)
            for ref in value
        )
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("compared_refs must be unique")
        return cleaned

    @field_validator("selected_ref")
    @classmethod
    def _validate_selected_ref(cls, value: object) -> str | None:
        if value is None:
            return None
        return bounded_identifier(value, field="selected_ref", max_length=MAX_REF_LENGTH)

    @field_validator("feedback_text")
    @classmethod
    def _validate_feedback(cls, value: object, info: ValidationInfo) -> str:
        return redacted_text(
            value, field=info.field_name or "feedback_text", max_length=MAX_FEEDBACK_TEXT_LENGTH
        )

    @model_validator(mode="after")
    def _validate_selection(self) -> ProductComparisonFeedback:
        if self.selected_ref is not None and self.selected_ref not in self.compared_refs:
            raise ValueError("selected_ref must be one of compared_refs")
        return self


# --------------------------------------------------------------------------- #
# Knowledge-state events
# --------------------------------------------------------------------------- #


class FeatureExplanationRequested(StateEventPayload):
    """The user asked what an attribute means; the attribute needs explanation."""

    attribute: str
    question_text: str = ""

    @field_validator("attribute")
    @classmethod
    def _validate_attribute(cls, value: object) -> str:
        return clean_attribute(value)

    @field_validator("question_text")
    @classmethod
    def _validate_question(cls, value: object, info: ValidationInfo) -> str:
        return redacted_text(
            value, field=info.field_name or "question_text", max_length=MAX_FEEDBACK_TEXT_LENGTH
        )


class FeatureExplanationShown(StateEventPayload):
    """An explanation was displayed.

    A display is evidence that the user *saw* something, not that they understood it.
    The reducer moves the attribute to ``EXPLANATION_PROVIDED`` and never further.
    ``explanation_ref`` references the explanation artifact; the explanation text is
    not copied into the decision state.
    """

    attribute: str
    explanation_ref: str

    @field_validator("attribute")
    @classmethod
    def _validate_attribute(cls, value: object) -> str:
        return clean_attribute(value)

    @field_validator("explanation_ref")
    @classmethod
    def _validate_explanation_ref(cls, value: object) -> str:
        return bounded_identifier(value, field="explanation_ref", max_length=MAX_REF_LENGTH)


class UserUnderstandingDemonstrated(StateEventPayload):
    """The user demonstrated understanding of an attribute in their own words.

    This is the only event that may move a knowledge belief to
    ``UNDERSTANDING_DEMONSTRATED``.  The evidence span is required: a demonstration
    without evidence would be a system assertion, not a user demonstration.
    """

    attribute: str
    evidence_text: str

    @field_validator("attribute")
    @classmethod
    def _validate_attribute(cls, value: object) -> str:
        return clean_attribute(value)

    @field_validator("evidence_text")
    @classmethod
    def _validate_evidence(cls, value: object, info: ValidationInfo) -> str:
        cleaned = redacted_text(
            value, field=info.field_name or "evidence_text", max_length=MAX_SOURCE_TEXT_LENGTH
        )
        if not cleaned:
            raise ValueError("evidence_text must not be blank")
        return cleaned


# --------------------------------------------------------------------------- #
# Task context and read-only trusted references
# --------------------------------------------------------------------------- #

_INTENT_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,%d}$" % (MAX_TASK_INTENT_LENGTH - 1))
_METHOD_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,%d}$" % (MAX_SOURCE_LENGTH - 1))
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class TaskContextUpdated(StateEventPayload):
    """The controller supplied the typed task context.

    ``intent`` is stored as a bounded label, not as a second intent enum: the vocabulary
    is owned by the accepted ``TaskIntent``, and this layer must stay importable without
    the model stack.  ``request_text`` is untrusted user words, bounded and redacted.
    """

    intent: str = "unknown"
    request_text: str = ""

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, value: object) -> str:
        if not isinstance(value, str) or not _INTENT_LABEL.fullmatch(value.strip()):
            raise ValueError("intent must be a lowercase label such as 'recommend'")
        return value.strip()

    @field_validator("request_text")
    @classmethod
    def _validate_request_text(cls, value: object, info: ValidationInfo) -> str:
        return redacted_text(
            value, field=info.field_name or "request_text", max_length=MAX_REQUEST_TEXT_LENGTH
        )


class CandidatePoolReferenced(StateEventPayload):
    """A read-only reference to a candidate pool produced by a trusted component.

    No product identity and no score: the reference is opaque, the digest is an
    optional integrity check, and ``read_only`` is fixed true on the state model.
    """

    ref: str
    source: str
    candidate_count: int = Field(ge=0)
    digest: str | None = None
    #: Version of the trusted candidate-snapshot contract carried by this reference.
    #: It is metadata for the trusted engine's snapshot schema, not the event schema.
    snapshot_schema_version: int = Field(default=1, ge=1, le=1000)

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


class RankingSnapshotReferenced(StateEventPayload):
    """A read-only reference to a ranking snapshot produced by the trusted engine."""

    ref: str
    source: str
    method: str
    candidate_count: int = Field(ge=0)
    digest: str | None = None
    #: Version of the trusted ranking-snapshot contract carried by this reference.
    snapshot_schema_version: int = Field(default=1, ge=1, le=1000)

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
        if not isinstance(value, str) or not _METHOD_LABEL.fullmatch(value.strip()):
            raise ValueError("method must be a lowercase label such as 'rrf_rank_fusion'")
        return value.strip()

    @field_validator("digest")
    @classmethod
    def _validate_digest(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ValueError("digest must be a lowercase 64-character SHA-256 hex digest")
        return value


#: The payload union.  The concrete member is selected by ``event_type`` before
#: validation; the union annotation keeps ordinary Pydantic serialization typed.
StateEventPayloadUnion = Union[
    UserPreferenceStated,
    PreferenceHypothesisProposed,
    PreferenceConfirmed,
    PreferenceRetracted,
    UserConstraintStated,
    PreferencePriorityStated,
    ProductComparisonFeedback,
    FeatureExplanationRequested,
    FeatureExplanationShown,
    UserUnderstandingDemonstrated,
    TaskContextUpdated,
    CandidatePoolReferenced,
    RankingSnapshotReferenced,
]

#: The single payload registry: one event type maps to exactly one payload model.
EVENT_PAYLOAD_MODELS: dict[EventType, type[StateEventPayload]] = {
    EventType.USER_PREFERENCE_STATED: UserPreferenceStated,
    EventType.PREFERENCE_HYPOTHESIS_PROPOSED: PreferenceHypothesisProposed,
    EventType.PREFERENCE_CONFIRMED: PreferenceConfirmed,
    EventType.PREFERENCE_RETRACTED: PreferenceRetracted,
    EventType.USER_CONSTRAINT_STATED: UserConstraintStated,
    EventType.PREFERENCE_PRIORITY_STATED: PreferencePriorityStated,
    EventType.PRODUCT_COMPARISON_FEEDBACK: ProductComparisonFeedback,
    EventType.FEATURE_EXPLANATION_REQUESTED: FeatureExplanationRequested,
    EventType.FEATURE_EXPLANATION_SHOWN: FeatureExplanationShown,
    EventType.USER_UNDERSTANDING_DEMONSTRATED: UserUnderstandingDemonstrated,
    EventType.TASK_CONTEXT_UPDATED: TaskContextUpdated,
    EventType.CANDIDATE_POOL_REFERENCED: CandidatePoolReferenced,
    EventType.RANKING_SNAPSHOT_REFERENCED: RankingSnapshotReferenced,
}

#: Declarative actor authority.  An event whose actor is not listed is refused.
EVENT_ACTORS: dict[EventType, frozenset[EventActor]] = {
    EventType.USER_PREFERENCE_STATED: frozenset({EventActor.USER}),
    EventType.PREFERENCE_HYPOTHESIS_PROPOSED: frozenset(
        {EventActor.AGENT, EventActor.SYSTEM, EventActor.TRUSTED_COMPONENT}
    ),
    EventType.PREFERENCE_CONFIRMED: frozenset({EventActor.USER}),
    EventType.PREFERENCE_RETRACTED: frozenset({EventActor.USER}),
    EventType.USER_CONSTRAINT_STATED: frozenset({EventActor.USER}),
    EventType.PREFERENCE_PRIORITY_STATED: frozenset({EventActor.USER}),
    EventType.PRODUCT_COMPARISON_FEEDBACK: frozenset({EventActor.USER}),
    EventType.FEATURE_EXPLANATION_REQUESTED: frozenset({EventActor.USER}),
    EventType.FEATURE_EXPLANATION_SHOWN: frozenset({EventActor.AGENT, EventActor.SYSTEM}),
    EventType.USER_UNDERSTANDING_DEMONSTRATED: frozenset({EventActor.USER}),
    EventType.TASK_CONTEXT_UPDATED: frozenset({EventActor.AGENT, EventActor.SYSTEM}),
    EventType.CANDIDATE_POOL_REFERENCED: frozenset({EventActor.TRUSTED_COMPONENT}),
    EventType.RANKING_SNAPSHOT_REFERENCED: frozenset({EventActor.TRUSTED_COMPONENT}),
}


def payload_model_for(event_type: EventType | str) -> type[StateEventPayload]:
    """Return the payload model for ``event_type`` (``ValueError`` if unknown)."""
    return EVENT_PAYLOAD_MODELS[EventType(event_type)]


# --------------------------------------------------------------------------- #
# Event envelope
# --------------------------------------------------------------------------- #


class StateEvent(BaseModel):
    """One immutable, versioned, attributed decision-state event.

    The envelope is intentionally small.  It carries identity and ordering metadata;
    every semantic field lives in the typed payload.  ``sequence`` is ``None`` until
    the event store assigns it inside the append transaction, and is then immutable.

    Duplicate-id semantics are defined on the *semantic* fields
    (:meth:`semantic_fields`).  ``timestamp`` and ``sequence`` are metadata: a retried
    append with the same semantic content and a fresh timestamp is the same event.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    user_key: str
    session_id: str | None = None
    task_id: str
    event_type: EventType
    schema_version: int = EVENT_SCHEMA_VERSION
    payload: StateEventPayloadUnion
    actor: EventActor
    source: str = "unspecified"
    timestamp: float
    sequence: int | None = Field(default=None, ge=1)

    # -- construction ----------------------------------------------------- #

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data: Any) -> Any:
        """Select and validate the payload model named by ``event_type``."""
        if not isinstance(data, dict):
            return data
        raw_type = data.get("event_type")
        raw_payload = data.get("payload")
        if raw_type is None or raw_payload is None:
            return data
        try:
            event_type = EventType(raw_type)
        except ValueError:
            return data  # let normal enum validation report the bad type
        model = EVENT_PAYLOAD_MODELS[event_type]
        if isinstance(raw_payload, model):
            return data
        if isinstance(raw_payload, BaseModel):
            raise ValueError(
                f"payload for {event_type.value!r} must be {model.__name__}, "
                f"not {type(raw_payload).__name__}"
            )
        copied = dict(data)
        copied["payload"] = model.model_validate(raw_payload)
        return copied

    @field_validator("event_id")
    @classmethod
    def _validate_event_id(cls, value: object) -> str:
        return bounded_identifier(value, field="event_id", max_length=MAX_EVENT_ID_LENGTH)

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

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return bounded_text(value, field="source", max_length=MAX_SOURCE_LENGTH) or "unspecified"

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("timestamp must be a number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        return numeric

    @model_validator(mode="after")
    def _validate_contract(self) -> StateEvent:
        expected = EVENT_PAYLOAD_MODELS[self.event_type]
        if type(self.payload) is not expected:
            raise ValueError(
                f"event type {self.event_type.value!r} requires payload "
                f"{expected.__name__}, got {type(self.payload).__name__}"
            )
        allowed_actors = EVENT_ACTORS[self.event_type]
        if self.actor not in allowed_actors:
            allowed = ", ".join(sorted(actor.value for actor in allowed_actors))
            raise ValueError(
                f"event type {self.event_type.value!r} requires actor in {{{allowed}}}, "
                f"got {self.actor.value!r}"
            )
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"event schema version {self.schema_version} is not supported "
                f"(expected {EVENT_SCHEMA_VERSION})"
            )
        return self

    # -- derived properties ----------------------------------------------- #

    @property
    def scope(self) -> PreferenceScope | None:
        """The payload scope, when the event type carries one."""
        return getattr(self.payload, "scope", None)

    @property
    def applies_long_term(self) -> bool:
        """True when this event must be replayed for every task of the user."""
        return self.scope is PreferenceScope.LONG_TERM

    def canonical_payload(self) -> dict[str, Any]:
        """Return the payload as canonical JSON-compatible data."""
        return self.payload.model_dump(mode="json")

    def payload_hash(self) -> str:
        """SHA-256 of the canonical payload encoding."""
        return hashlib.sha256(
            canonical_json(self.canonical_payload()).encode("utf-8")
        ).hexdigest()

    def semantic_fields(self) -> tuple[Any, ...]:
        """The fields that define event identity for duplicate-id idempotency.

        ``timestamp`` and ``sequence`` are excluded: ordering and recording time are
        metadata, not content.
        """
        return (
            self.event_id,
            self.user_key,
            self.session_id,
            self.task_id,
            self.event_type.value,
            self.schema_version,
            self.payload_hash(),
            self.actor.value,
            self.source,
        )

    def is_same_submission(self, other: StateEvent) -> bool:
        """True when ``other`` is the same semantic event as this one."""
        return isinstance(other, StateEvent) and self.semantic_fields() == other.semantic_fields()

    def stored(self, sequence: int) -> StateEvent:
        """Return a copy with the store-assigned ``sequence`` set."""
        if not isinstance(sequence, int) or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        return self.model_copy(update={"sequence": sequence})


def new_event(
    *,
    event_type: EventType | str,
    event_id: str,
    user_key: str,
    task_id: str,
    actor: EventActor | str,
    timestamp: float,
    payload: dict[str, Any] | StateEventPayload | None = None,
    session_id: str | None = None,
    source: str = "unspecified",
    **payload_fields: Any,
) -> StateEvent:
    """Build a validated :class:`StateEvent` from a payload dictionary or keyword fields.

    ``payload`` and ``**payload_fields`` are alternatives; supplying both is an error.
    The event id is caller-supplied (the store cannot invent identity), so a test or a
    retry can deliberately re-submit the same id.
    """
    if payload is not None and payload_fields:
        raise TypeError("pass either payload=... or payload keyword fields, not both")
    resolved_type = EventType(event_type)
    if payload is None:
        payload = EVENT_PAYLOAD_MODELS[resolved_type](**payload_fields)
    return StateEvent(
        event_id=event_id,
        user_key=user_key,
        session_id=session_id,
        task_id=task_id,
        event_type=resolved_type,
        payload=payload,
        actor=EventActor(actor),
        source=source,
        timestamp=timestamp,
    )
