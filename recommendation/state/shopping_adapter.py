"""Trusted adapter from validated shopping-turn facts to M5 state events (M5.1/M5.2).

This module is intentionally **light**: it imports only ``recommendation.state`` (Pydantic
plus the standard library) and never the model, API or demo stack.  That keeps the event
construction rule testable without PyTorch, and it keeps the trust direction explicit:

```text
existing trusted parses            this adapter                 M5 state engine
user message -> ShoppingIntent  -> validated facts -> StateEvent -> append-only store
             -> PreferenceExtraction
             -> clarification option resolution
             -> response candidate/ranking views
```

What it will and will not accept
--------------------------------
The adapter only turns **already validated, explicit** facts into events:

* ``USER_PREFERENCE_STATED`` from the accepted M9 extraction's explicit candidates, from
  an explicit M9 ``REPLACE`` correction, and from a trusted clarification option the user
  selected (M5.2);
* ``PREFERENCE_RETRACTED`` from an accepted M9 ``REMOVE`` removal (M5.2) - the removal's
  selector is resolved against the current task state so a value-only removal retracts
  exactly one identified preference instead of guessing;
* ``USER_CONSTRAINT_STATED`` from the existing trusted shopping-intent hard constraints;
* ``TASK_CONTEXT_UPDATED`` from the validated task context;
* ``CANDIDATE_POOL_REFERENCED`` / ``RANKING_SNAPSHOT_REFERENCED`` (M5.2) from the trusted
  recommendation pipeline's own candidate/ranking views: opaque references + counts +
  digests + snapshot schema version only, never product rows or scores.

It has no function that accepts an inference, a model-generated suggestion or arbitrary
text-with-an-intent.  A soft shopping-intent entry therefore cannot become a hard
constraint here, and an inferred hypothesis cannot become an explicit preference.

Turn identity (M5.2)
--------------------
:func:`resolve_turn_identity` prefers, in order, a client-supplied idempotency key, the
server-owned turn id, and only then a content digest fallback.  Event ids are derived from
that logical turn identity **plus event content**, so a genuine retry reuses the same ids
while two distinct turns never collapse; a content-only fallback is clearly reported as
such and is never used by the HTTP shopping route, which always supplies server identity.

Bounds
------
At most :data:`MAX_TURN_PREFERENCES` explicit preferences,
:data:`MAX_TURN_CONSTRAINTS` explicit constraints and :data:`MAX_TURN_RETRACTIONS`
retractions are accepted per turn; more is refused rather than silently truncated.
Values and provenance spans are bounded and credential-redacted by the M5 event payload
models, which the adapter reuses unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .decision_state import UnifiedDecisionState
from .events import StateEvent, new_event
from .schemas import (
    PreferenceDirection,
    PreferenceKind,
    PreferencePolarity,
    canonical_json,
    deterministic_id,
)
from .state_projector import for_constraints

__all__ = [
    "ADAPTER_SOURCE",
    "MAX_TURN_CONSTRAINTS",
    "MAX_TURN_PREFERENCES",
    "MAX_TURN_RETRACTIONS",
    "ExplicitConstraint",
    "ExplicitPreference",
    "ExplicitRetraction",
    "SnapshotReference",
    "TurnIdentity",
    "authorized_constraint_pairs",
    "build_grounded_interaction_events",
    "build_snapshot_events",
    "build_turn_events",
    "candidate_pool_snapshot",
    "events_digest",
    "merge_constraint_pairs",
    "ranking_snapshot",
    "request_turn_key",
    "resolve_retraction",
    "resolve_turn_identity",
]

#: Per-turn bounds.  A malformed/oversized input is refused, never truncated silently.
MAX_TURN_PREFERENCES = 8
MAX_TURN_CONSTRAINTS = 8
MAX_TURN_RETRACTIONS = 8

#: Source label recorded on every event this adapter emits.
ADAPTER_SOURCE = "shopping_state_adapter"

#: Source labels for trusted snapshot references (the producing component, not a user fact).
CANDIDATE_SNAPSHOT_SOURCE = "shopping_candidate_pool"
RANKING_SNAPSHOT_SOURCE = "shopping_presented_order"


@dataclass(frozen=True)
class TurnIdentity:
    """Server-owned identity of one shopping turn.

    ``user_key`` is the preference-memory namespace, ``session_id`` the concrete demo
    session and ``task_id`` the stable per-session task key.  They are separate because
    several sessions may share a memory namespace; a task must never be keyed by the user
    key alone.
    """

    user_key: str
    session_id: str
    task_id: str
    turn_key: str


@dataclass(frozen=True)
class ExplicitPreference:
    """One explicit preference the accepted extractor (or selected option) found.

    ``direction`` carries a trusted directional choice (e.g. a clarification option
    "lower price" mapped to ``MINIMIZE``); it is not inferred from free text.
    """

    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    source_text: str = ""
    #: Set only when the statement explicitly corrects an earlier value (named target).
    replaces_value: str | None = None
    #: M5.2: the accepted M9 ``REPLACE`` mode without a named target; the reducer
    #: supersedes the active same-kind/same-polarity belief(s) with a different value.
    replace_same_kind: bool = False
    direction: PreferenceDirection | None = None


@dataclass(frozen=True)
class ExplicitConstraint:
    """One hard constraint the accepted shopping intent already classified as hard."""

    kind: PreferenceKind
    value: str
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    source_text: str = ""


@dataclass(frozen=True)
class ExplicitRetraction:
    """One accepted M9 removal, ready to become a ``PREFERENCE_RETRACTED`` event.

    M9 removals select by ``kind`` or by ``value`` (exactly one).  A value-only removal
    cannot be represented by the M5 retraction payload without a kind, so
    :func:`resolve_retraction` resolves it against the current task state and returns
    ``None`` when it is unmatched or ambiguous.  The integration layer reports that as a
    bounded legacy/M5 disagreement instead of guessing.
    """

    kind: PreferenceKind | None = None
    value: str | None = None
    polarity: PreferencePolarity | None = None
    source_text: str = ""

    def __post_init__(self) -> None:
        if self.kind is None and not _normalised_text(self.value):
            raise ValueError("a retraction needs a kind or a value selector")


@dataclass(frozen=True)
class SnapshotReference:
    """Opaque, read-only reference to a trusted candidate/ranking snapshot."""

    ref: str
    digest: str
    count: int


def _normalised_text(value: object) -> str:
    """Whitespace-collapsed text used for identity/digest comparison."""
    return " ".join(str(value or "").split())


def _coerce_kind(value: object) -> PreferenceKind:
    """Return an accepted :class:`PreferenceKind` or raise ``ValueError``.

    An unknown kind is a malformed input for the state engine, not something to invent a
    fallback for: reinterpreting it as ``FREE_FORM_CONSTRAINT`` would change the meaning
    the trusted layer already gave it.
    """
    if isinstance(value, PreferenceKind):
        return value
    try:
        return PreferenceKind(str(value))
    except ValueError as exc:
        raise ValueError(f"unsupported preference kind: {value!r}") from exc


def _coerce_polarity(value: object) -> PreferencePolarity:
    if isinstance(value, PreferencePolarity):
        return value
    try:
        return PreferencePolarity(str(value))
    except ValueError as exc:
        raise ValueError(f"unsupported preference polarity: {value!r}") from exc


def request_turn_key(
    *,
    message: str,
    k: int,
    sources: Sequence[str],
    constraints: Sequence[tuple[str, str]] = (),
    clarification_id: str | None = None,
) -> str:
    """Deterministic logical-turn key for one shopping request.

    The key is content-addressed: two transports that carry the same logical request
    produce the same key (and therefore idempotent events), while any change to the
    message, requested count, source plan, stated constraints or clarification answer
    produces a different one.
    """
    payload = {
        "message": _normalised_text(message),
        "k": int(k),
        "sources": [str(source).strip().lower() for source in sources],
        "constraints": sorted(
            [
                [str(kind).strip().lower(), _normalised_text(value).casefold()]
                for kind, value in constraints
            ]
        ),
        "clarification_id": str(clarification_id) if clarification_id else None,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"turn_{digest[:24]}"


def _dedupe_preferences(
    preferences: Iterable[ExplicitPreference],
) -> tuple[ExplicitPreference, ...]:
    """Coerce, validate and de-duplicate explicit preferences deterministically."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[ExplicitPreference] = []
    for preference in preferences:
        kind = _coerce_kind(preference.kind)
        polarity = _coerce_polarity(preference.polarity)
        direction = _coerce_direction(preference.direction)
        value = preference.value
        if not _normalised_text(value):
            raise ValueError("explicit preference value must not be blank")
        signature = (kind.value, _normalised_text(value).casefold(), polarity.value)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(
            ExplicitPreference(
                kind=kind,
                value=value,
                polarity=polarity,
                source_text=preference.source_text,
                replaces_value=preference.replaces_value,
                replace_same_kind=bool(preference.replace_same_kind),
                direction=direction,
            )
        )
    if len(kept) > MAX_TURN_PREFERENCES:
        raise ValueError(
            f"a shopping turn carries at most {MAX_TURN_PREFERENCES} explicit preferences"
        )
    kept.sort(key=lambda item: (item.kind.value, _normalised_text(item.value).casefold(), item.polarity.value))
    return tuple(kept)


def _dedupe_constraints(
    constraints: Iterable[ExplicitConstraint],
) -> tuple[ExplicitConstraint, ...]:
    """Coerce, validate and de-duplicate explicit constraints deterministically."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[ExplicitConstraint] = []
    for constraint in constraints:
        kind = _coerce_kind(constraint.kind)
        polarity = _coerce_polarity(constraint.polarity)
        value = constraint.value
        if not _normalised_text(value):
            raise ValueError("explicit constraint value must not be blank")
        signature = (kind.value, _normalised_text(value).casefold(), polarity.value)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(
            ExplicitConstraint(
                kind=kind,
                value=value,
                polarity=polarity,
                source_text=constraint.source_text,
            )
        )
    if len(kept) > MAX_TURN_CONSTRAINTS:
        raise ValueError(
            f"a shopping turn carries at most {MAX_TURN_CONSTRAINTS} explicit constraints"
        )
    kept.sort(key=lambda item: (item.kind.value, _normalised_text(item.value).casefold(), item.polarity.value))
    return tuple(kept)


def _dedupe_retractions(
    retractions: Iterable[ExplicitRetraction],
) -> tuple[ExplicitRetraction, ...]:
    """Coerce, validate and de-duplicate resolved retractions deterministically.

    The adapter only emits a retraction whose selector was resolved to a concrete kind by
    :func:`resolve_retraction`; a value-only selector is refused here because the M5
    retraction payload must identify the preference kind.  Skipping instead would hide a
    state/legacy disagreement, so the integration resolves first and reports the rest.
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[ExplicitRetraction] = []
    for retraction in retractions:
        if retraction.kind is None:
            raise ValueError(
                "a retraction must be resolved to a preference kind before it becomes an event"
            )
        kind = _coerce_kind(retraction.kind)
        polarity = (
            _coerce_polarity(retraction.polarity)
            if retraction.polarity is not None
            else None
        )
        value = retraction.value
        signature = (
            kind.value,
            _normalised_text(value).casefold() if value is not None else "*",
            polarity.value if polarity is not None else "*",
        )
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(
            ExplicitRetraction(
                kind=kind,
                value=value,
                polarity=polarity,
                source_text=retraction.source_text,
            )
        )
    if len(kept) > MAX_TURN_RETRACTIONS:
        raise ValueError(
            f"a shopping turn carries at most {MAX_TURN_RETRACTIONS} retractions"
        )
    kept.sort(
        key=lambda item: (
            item.kind.value if item.kind is not None else "",
            _normalised_text(item.value).casefold(),
            item.polarity.value if item.polarity is not None else "",
        )
    )
    return tuple(kept)


def _coerce_direction(value: object) -> PreferenceDirection | None:
    """Return an accepted :class:`PreferenceDirection` or ``None``."""
    if value is None:
        return None
    if isinstance(value, PreferenceDirection):
        return value
    try:
        return PreferenceDirection(str(value))
    except ValueError as exc:
        raise ValueError(f"unsupported preference direction: {value!r}") from exc


def _event_id(identity: TurnIdentity, event_type: str, *parts: object) -> str:
    """Deterministic event id: turn identity plus event content."""
    return deterministic_id(
        "m5evt",
        identity.user_key,
        identity.task_id,
        identity.turn_key,
        event_type,
        *parts,
    )


def build_turn_events(
    *,
    identity: TurnIdentity,
    message: str,
    intent_label: str,
    preferences: Sequence[ExplicitPreference] = (),
    constraints: Sequence[ExplicitConstraint] = (),
    retractions: Sequence[ExplicitRetraction] = (),
    timestamp: float,
) -> tuple[StateEvent, ...]:
    """Build the validated M5 events for one shopping turn.

    The order is deterministic (context, then preferences, then constraints, then
    retractions, each group sorted by kind/value/polarity) so replays and diagnostics are
    comparable run to run.  One event per identity is emitted even if the caller passes
    duplicates.  A retraction produced here must already be resolved to a concrete kind
    (see :func:`resolve_retraction`).

    Raises
    ------
    ValueError
        A field is malformed (unknown kind/polarity/direction, blank value, over-bound
        input).  The caller decides whether that is a shadow diagnostic or a fail-closed
        error.
    """
    if not isinstance(identity, TurnIdentity):
        raise TypeError("identity must be a TurnIdentity")
    for name, value in (
        ("user_key", identity.user_key),
        ("session_id", identity.session_id),
        ("task_id", identity.task_id),
        ("turn_key", identity.turn_key),
    ):
        if not _normalised_text(value):
            raise ValueError(f"{name} must not be blank")

    events: list[StateEvent] = [
        new_event(
            event_type="task_context_updated",
            event_id=_event_id(identity, "task_context_updated"),
            user_key=identity.user_key,
            task_id=identity.task_id,
            session_id=identity.session_id,
            actor="agent",
            source=ADAPTER_SOURCE,
            timestamp=timestamp,
            intent=intent_label,
            request_text=message,
        )
    ]
    for preference in _dedupe_preferences(preferences):
        events.append(
            new_event(
                event_type="user_preference_stated",
                event_id=_event_id(
                    identity,
                    "user_preference_stated",
                    preference.kind.value,
                    _normalised_text(preference.value).casefold(),
                    preference.polarity.value,
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="user",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                kind=preference.kind,
                value=preference.value,
                polarity=preference.polarity,
                direction=preference.direction,
                scope="task",
                source_text=preference.source_text,
                replaces_value=preference.replaces_value,
                replace_same_kind=preference.replace_same_kind,
            )
        )
    for constraint in _dedupe_constraints(constraints):
        events.append(
            new_event(
                event_type="user_constraint_stated",
                event_id=_event_id(
                    identity,
                    "user_constraint_stated",
                    constraint.kind.value,
                    _normalised_text(constraint.value).casefold(),
                    constraint.polarity.value,
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="user",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                kind=constraint.kind,
                value=constraint.value,
                polarity=constraint.polarity,
                scope="task",
                source_text=constraint.source_text,
            )
        )
    for retraction in _dedupe_retractions(retractions):
        resolved_kind = retraction.kind
        assert resolved_kind is not None  # guaranteed by _dedupe_retractions
        events.append(
            new_event(
                event_type="preference_retracted",
                event_id=_event_id(
                    identity,
                    "preference_retracted",
                    resolved_kind.value,
                    _normalised_text(retraction.value).casefold()
                    if retraction.value is not None
                    else "*",
                    retraction.polarity.value
                    if retraction.polarity is not None
                    else "*",
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="user",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                kind=resolved_kind,
                value=retraction.value,
                polarity=retraction.polarity,
                scope="task",
                source_text=retraction.source_text,
            )
        )
    return tuple(events)


def resolve_turn_identity(
    *,
    task_id: str,
    idempotency_key: str | None = None,
    server_turn_id: str | None = None,
    fallback_content_key: str | None = None,
) -> tuple[str, str]:
    """Resolve the logical turn identity and report which source produced it.

    Precedence (strongest retry semantics first):

    1. ``idempotency_key`` - a client-supplied stable key; a retried request reuses the
       same event ids.  Scoped by task id so two tasks can never collide.
    2. ``server_turn_id`` - the server-owned turn allocation.  Distinct turns get distinct
       ids even when their content is identical, but a transport retry allocates a new
       turn and therefore is not deduplicated (documented fallback semantics).
    3. ``fallback_content_key`` - only for direct service callers/tests that have no
       server identity; this is a content digest and is explicitly reported, never used by
       the HTTP shopping route.
    """
    task = _normalised_text(task_id)
    if not task:
        raise ValueError("task_id must not be blank")
    key = _normalised_text(idempotency_key)
    if key:
        return f"{task}:idem:{key}", "idempotency_key"
    server = _normalised_text(server_turn_id)
    if server:
        return server, "server_turn_id"
    fallback = _normalised_text(fallback_content_key)
    if fallback:
        return fallback, "content_fallback"
    raise ValueError("a turn identity needs an idempotency key, a server turn id or a fallback")


def candidate_pool_snapshot(identities: Sequence[str]) -> SnapshotReference:
    """Opaque reference + canonical digest of a trusted grounded candidate set.

    The digest is order-insensitive (the pool is a set) and the reference is stable for
    the same identity set.  No identity is stored in the decision state; only the digest,
    the reference and the count cross the boundary.
    """
    cleaned = sorted({str(identity) for identity in identities if str(identity)})
    digest = hashlib.sha256(canonical_json(cleaned).encode("utf-8")).hexdigest()
    return SnapshotReference(ref=f"cp_{digest[:24]}", digest=digest, count=len(cleaned))


def ranking_snapshot(ordered_identities: Sequence[str]) -> SnapshotReference:
    """Opaque reference + canonical digest of a trusted presented ranking order.

    The digest is order-sensitive: it captures exactly the order the user was shown.
    """
    cleaned = [str(identity) for identity in ordered_identities if str(identity)]
    digest = hashlib.sha256(canonical_json(cleaned).encode("utf-8")).hexdigest()
    return SnapshotReference(ref=f"rs_{digest[:24]}", digest=digest, count=len(cleaned))


def build_grounded_interaction_events(
    *,
    identity: TurnIdentity,
    timestamp: float,
    compared_refs: Sequence[str] = (),
    comparison_feedback: str = "",
    explanation_requested: str | None = None,
    question_text: str = "",
    explanation_shown: str | None = None,
    explanation_ref: str | None = None,
) -> tuple[StateEvent, ...]:
    """Build the real interaction events for one M7 grounded action.

    Semantics are kept separate:

    * ``PRODUCT_COMPARISON_FEEDBACK`` is only emitted when the **user** actually answered
      a comparison (``compared_refs`` are the candidates that were compared); merely
      displaying a comparison is not user feedback.
    * ``FEATURE_EXPLANATION_REQUESTED`` is only emitted when the user's own message asked
      for an explanation.
    * ``FEATURE_EXPLANATION_SHOWN`` records that an explanation was displayed; it never
      asserts understanding (the M5 reducer itself stops at ``EXPLANATION_PROVIDED``).

    Event ids are deterministic from the frozen turn identity and the event content, so a
    retry appends nothing and a replay is byte-identical.
    """
    if not isinstance(identity, TurnIdentity):
        raise TypeError("identity must be a TurnIdentity")
    events: list[StateEvent] = []
    refs = tuple(dict.fromkeys(str(item) for item in compared_refs if str(item)))
    if refs:
        feedback = str(comparison_feedback or "")
        events.append(
            new_event(
                event_type="product_comparison_feedback",
                event_id=_event_id(
                    identity,
                    "product_comparison_feedback",
                    *sorted(refs),
                    feedback.casefold(),
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="user",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                compared_refs=refs,
                selected_ref=None,
                feedback_text=feedback,
            )
        )
    if explanation_requested:
        events.append(
            new_event(
                event_type="feature_explanation_requested",
                event_id=_event_id(
                    identity,
                    "feature_explanation_requested",
                    _normalised_text(explanation_requested).casefold(),
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="user",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                attribute=explanation_requested,
                question_text=question_text,
            )
        )
    if explanation_shown:
        ref = str(explanation_ref or "")
        if not ref:
            raise ValueError("explanation_ref is required when an explanation was shown")
        events.append(
            new_event(
                event_type="feature_explanation_shown",
                event_id=_event_id(
                    identity,
                    "feature_explanation_shown",
                    _normalised_text(explanation_shown).casefold(),
                    ref,
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="agent",
                source=ADAPTER_SOURCE,
                timestamp=timestamp,
                attribute=explanation_shown,
                explanation_ref=ref,
            )
        )
    return tuple(events)


def build_snapshot_events(
    *,
    identity: TurnIdentity,
    candidate_pool: SnapshotReference | None = None,
    ranking: SnapshotReference | None = None,
    candidate_source: str = CANDIDATE_SNAPSHOT_SOURCE,
    ranking_source: str = RANKING_SNAPSHOT_SOURCE,
    ranking_method: str = "presented_order",
    snapshot_schema_version: int = 1,
    timestamp: float,
) -> tuple[StateEvent, ...]:
    """Build the trusted candidate/ranking reference events for one turn.

    Only a trusted component may produce these events, so they are actor
    ``trusted_component``.  They carry an opaque reference, a source label, a count, an
    optional canonical digest and the snapshot schema version - never a product row, a
    per-item rank or a score.
    """
    if not isinstance(identity, TurnIdentity):
        raise TypeError("identity must be a TurnIdentity")
    events: list[StateEvent] = []
    if candidate_pool is not None:
        events.append(
            new_event(
                event_type="candidate_pool_referenced",
                event_id=_event_id(
                    identity,
                    "candidate_pool_referenced",
                    candidate_pool.ref,
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="trusted_component",
                source=candidate_source,
                timestamp=timestamp,
                payload={
                    "ref": candidate_pool.ref,
                    "source": candidate_source,
                    "candidate_count": candidate_pool.count,
                    "digest": candidate_pool.digest,
                    "snapshot_schema_version": snapshot_schema_version,
                },
            )
        )
    if ranking is not None:
        events.append(
            new_event(
                event_type="ranking_snapshot_referenced",
                event_id=_event_id(
                    identity,
                    "ranking_snapshot_referenced",
                    ranking.ref,
                ),
                user_key=identity.user_key,
                task_id=identity.task_id,
                session_id=identity.session_id,
                actor="trusted_component",
                source=ranking_source,
                timestamp=timestamp,
                payload={
                    "ref": ranking.ref,
                    "source": ranking_source,
                    "method": ranking_method,
                    "candidate_count": ranking.count,
                    "digest": ranking.digest,
                    "snapshot_schema_version": snapshot_schema_version,
                },
            )
        )
    return tuple(events)


def resolve_retraction(
    state: UnifiedDecisionState,
    retraction: ExplicitRetraction,
) -> ExplicitRetraction | None:
    """Resolve a value-only M9 removal against the current task state.

    * An explicit kind selector is passed through unchanged (with its optional value and
      polarity filters).
    * A value-only selector is resolved only when exactly one active task belief currently
      has that value (and matches the optional polarity).  Zero matches means the legacy
      memory and the M5 task state disagree (e.g. a long-term-only entry); more than one
      distinct kind means the removal is ambiguous.  In both cases ``None`` is returned so
      the caller can report the mismatch instead of guessing.
    """
    if retraction.kind is not None:
        return retraction
    value = _normalised_text(retraction.value).casefold()
    if not value:
        return None
    exact = [
        belief
        for belief in state.active_preference_beliefs
        if _normalised_text(belief.value).casefold() == value
        and (
            retraction.polarity is None
            or belief.polarity is retraction.polarity
        )
    ]
    # A user can withdraw a remembered requirement by naming only its leading
    # noun phrase ("forget the replacement mouthpiece" for
    # "replacement mouthpiece for a water filter").  Resolve only when that
    # phrase identifies exactly one active belief; zero or multiple matches stay
    # UNKNOWN and are reported as a legacy/M5 disagreement.
    matches = exact
    resolved_value = retraction.value
    if not matches:
        matches = [
            belief
            for belief in state.active_preference_beliefs
            if _value_prefix_match(value, _normalised_text(belief.value).casefold())
            and (
                retraction.polarity is None
                or belief.polarity is retraction.polarity
            )
        ]
        if len(matches) == 1:
            # Persist the canonical remembered value so the reducer can apply its
            # exact retraction transition; the user's source_text is already kept
            # separately in the event.
            resolved_value = matches[0].value
    if not matches and value in {"weight", "lighter", "lighter option", "lightest", "lightweight"}:
        # The rule extractor emits the attribute name for "forget the weight
        # preference" while the confirmed M5 belief may carry the surface value
        # ("lighter option").  Resolve only a unique feature belief, never a goal.
        feature_matches = [
            belief
            for belief in state.active_preference_beliefs
            if belief.kind is PreferenceKind.FEATURE
            and any(term in _normalised_text(belief.value).casefold() for term in ("weight", "light"))
        ]
        if len(feature_matches) == 1:
            matches = feature_matches
            resolved_value = feature_matches[0].value
    kinds = {belief.kind for belief in matches}
    if len(matches) == 1 and len(kinds) == 1:
        return ExplicitRetraction(
            kind=next(iter(kinds)),
            value=resolved_value,
            polarity=retraction.polarity,
            source_text=retraction.source_text,
        )
    # The same explicit requirement may survive only as a hard constraint when its
    # preference projection was already superseded by an earlier correction.  Resolve
    # that constraint when it is the unique match; never guess across multiple kinds.
    constraint_matches = [
        constraint
        for constraint in state.hard_constraints
        if constraint.is_active
        and (
            retraction.polarity is None
            or constraint.polarity is retraction.polarity
        )
        and (
            _normalised_text(constraint.value).casefold() == value
            or _value_prefix_match(value, _normalised_text(constraint.value).casefold())
        )
    ]
    constraint_kinds = {constraint.kind for constraint in constraint_matches}
    if len(constraint_matches) == 1 and len(constraint_kinds) == 1:
        return ExplicitRetraction(
            kind=next(iter(constraint_kinds)),
            value=constraint_matches[0].value,
            polarity=retraction.polarity,
            source_text=retraction.source_text,
        )
    return None


def _value_prefix_match(selector: str, candidate: str) -> bool:
    """True when a withdrawal selector names a whole leading phrase.

    This is deliberately stricter than a substring test: the selector must be a
    word-prefix of the remembered value, so "water" does not match
    "fresh water filter" and "filter" does not match "water filter".
    """
    words = selector.split()
    if not words or not candidate.startswith(selector):
        return False
    if len(candidate) == len(selector):
        return True
    return candidate[len(selector)] in " \t,;:-"


def events_digest(events: Sequence[StateEvent]) -> str:
    """Deterministic digest of an event batch, used only for bounded diagnostics."""
    material = "\x1f".join(event.event_id for event in events)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def authorized_constraint_pairs(
    state: UnifiedDecisionState,
) -> tuple[tuple[str, str], ...]:
    """The authorized hard constraints of ``state`` as ``(kind, value)`` pairs.

    This is exactly :func:`recommendation.state.state_projector.for_constraints`: only
    user-authorized explicit constraints, with task-over-long-term precedence applied and
    inferred hypotheses structurally excluded.  The demo layer converts these pairs into
    its existing trusted ``AgentConstraintItem`` inputs; this module never filters a
    candidate itself.
    """
    projection = for_constraints(state)
    return tuple(
        (constraint.kind.value, constraint.value) for constraint in projection.constraints
    )


def merge_constraint_pairs(
    trusted: Sequence[tuple[str, str]],
    authorized: Sequence[tuple[str, str]],
    *,
    current_turn_kinds: Sequence[str] = (),
) -> tuple[tuple[str, str], ...]:
    """Union of existing trusted constraints and M5 authorized constraints.

    The existing trusted order comes first and is never rewritten.  A kind present in
    the **current turn's** trusted hard set takes precedence over an older M5-authorized
    constraint of the same kind, even when the values differ: the current statement is
    the newer authority, and appending both would silently over-constrain the run.
    Remembered-vs-remembered conflicts keep the existing union behavior; callers that
    do not pass ``current_turn_kinds`` get the historical merge semantics.
    """
    current_kinds = {str(kind).strip().lower() for kind in current_turn_kinds}
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str]] = []
    trusted_kinds: set[str] = set()
    for kind, value in trusted:
        signature = (str(kind).strip().lower(), _normalised_text(value).casefold())
        if not signature[0] or not signature[1] or signature in seen:
            continue
        seen.add(signature)
        trusted_kinds.add(signature[0])
        merged.append((str(kind).strip().lower(), _normalised_text(value)))
    for kind, value in authorized:
        signature = (str(kind).strip().lower(), _normalised_text(value).casefold())
        if not signature[0] or not signature[1] or signature in seen:
            continue
        if signature[0] in current_kinds and signature[0] in trusted_kinds:
            # Only an actually-present current-turn trusted constraint may suppress
            # the M5 value.  A bare kind in current_turn_kinds is not enough: the
            # proposal path deliberately has no trusted pair and must keep its
            # authorized constraint.
            continue
        seen.add(signature)
        merged.append((str(kind).strip().lower(), _normalised_text(value)))
    return tuple(merged)
