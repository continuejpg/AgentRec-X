"""Read-only bridge from the accepted M9 preference memory to M5 long-term beliefs.

Milestone 9 memory (``recommendation.memory``) is the accepted Serving V1 preference
writer.  M5-MVP does **not** migrate it, does **not** write to it, and does **not**
change its schema or service contract.  This module only reads an already-produced
``PreferenceMemorySnapshot`` (duck-typed: ``user_key`` + ``entries``) and projects the
active explicit entries as ``LONG_TERM`` / ``EXPLICIT`` / ``STRONG`` beliefs, each
carrying the originating ``legacy_memory_id`` for traceability.

The two-step split is deliberate:

* :func:`project_legacy_memory` is a pure, deterministic read: active entries only, one
  stable order (``logical_seq`` then ``memory_id``), a receipt of the entries it had to
  skip, and no mutation of the snapshot;
* :func:`merge_legacy_memory` combines a projection with an event-sourced state.  The
  event-sourced state always wins: a belief whose identity already exists is not
  overridden.  Interaction counters, observations and events are never touched, because
  a read-only adapter must not fabricate user turns or provenance events.

Only ``ACTIVE`` entries participate.  A ``SUPERSEDED``/``REMOVED`` entry historically
matters to M9's audit, but it is not current user truth, so it is not a belief here.  An
entry whose value fails the accepted M9 validation (too long, blank, or credential-like)
cannot be represented in a :class:`~recommendation.state.decision_state.PreferenceBelief`;
such an entry is skipped and reported in ``skipped_memory_ids`` instead of aborting the
whole projection or storing a redacted lie.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .decision_state import (
    PreferenceBelief,
    UnifiedDecisionState,
    preference_belief_id,
)
from .errors import CrossUserStateError
from .schemas import (
    MAX_REF_LENGTH,
    MAX_TASK_ID_LENGTH,
    MAX_USER_KEY_LENGTH,
    BeliefOrigin,
    BeliefStrength,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
    bounded_identifier,
)

__all__ = [
    "LegacyMemoryProjection",
    "merge_legacy_memory",
    "project_legacy_memory",
]


class LegacyMemoryProjection(BaseModel):
    """The read-only result of projecting an M9 snapshot.

    ``beliefs`` are valid :class:`PreferenceBelief` records in deterministic order;
    ``skipped_memory_ids`` names every active entry that could not be projected, so the
    omission is reported rather than silent.  The model is frozen: downstream code
    cannot edit the projection it was handed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_key: str
    beliefs: tuple[PreferenceBelief, ...] = ()
    skipped_memory_ids: tuple[str, ...] = ()

    @field_validator("user_key")
    @classmethod
    def _validate_user_key(cls, value: object) -> str:
        return bounded_identifier(
            value, field="user_key", max_length=MAX_USER_KEY_LENGTH
        )

    @field_validator("skipped_memory_ids")
    @classmethod
    def _validate_skipped_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            bounded_identifier(memory_id, field="skipped_memory_ids item", max_length=MAX_REF_LENGTH)
            for memory_id in value
        )


def _snapshot_user_key(snapshot: Any) -> str:
    """Read and validate the snapshot's owning ``user_key`` (duck-typed)."""
    raw_user_key = getattr(snapshot, "user_key", None)
    if not isinstance(raw_user_key, str):
        raise ValueError("preference-memory snapshot must expose a string user_key")
    return bounded_identifier(
        raw_user_key, field="snapshot.user_key", max_length=MAX_USER_KEY_LENGTH
    )


def _entry_memory_id(entry: Any) -> str:
    """Read and validate one entry's opaque ``memory_id``."""
    raw_memory_id = getattr(entry, "memory_id", None)
    if not isinstance(raw_memory_id, str):
        raise ValueError("a preference-memory entry must expose a string memory_id")
    return bounded_identifier(
        raw_memory_id, field="memory_id", max_length=MAX_REF_LENGTH
    )


def _entry_is_active(entry: Any) -> bool:
    """True when the duck-typed entry reports the accepted ``ACTIVE`` status."""
    # ``PreferenceStatus`` is a ``str`` enum, so this also accepts a plain "active"
    # string from a serialized snapshot without accepting any other value.
    return getattr(entry, "status", None) == PreferenceStatus.ACTIVE


def _belief_from_entry(
    entry: Any,
    *,
    user_key: str,
    task_id: str,
    memory_id: str,
) -> PreferenceBelief:
    """Convert one active M9 entry into a long-term explicit belief.

    Raises ``ValueError``/``TypeError`` (including the Pydantic ``ValidationError``
    subclass) when the entry cannot be represented; the caller records the skip.
    """
    raw_kind = getattr(entry, "kind", None)
    raw_value = getattr(entry, "value", None)
    raw_polarity = getattr(entry, "polarity", PreferencePolarity.PREFER)
    kind = PreferenceKind(raw_kind)
    polarity = PreferencePolarity(raw_polarity)
    if not isinstance(raw_value, str):
        raise ValueError("a preference-memory entry value must be a string")
    return PreferenceBelief(
        belief_id=preference_belief_id(
            user_key=user_key,
            task_id=task_id,
            scope=PreferenceScope.LONG_TERM,
            kind=kind,
            value=raw_value,
            polarity=polarity,
        ),
        kind=kind,
        value=raw_value,
        polarity=polarity,
        lifecycle=PreferenceLifecycle.EXPLICIT,
        status=PreferenceStatus.ACTIVE,
        scope=PreferenceScope.LONG_TERM,
        strength=BeliefStrength.STRONG,
        origin=BeliefOrigin.LEGACY_MEMORY,
        supporting_event_ids=(),
        legacy_memory_id=memory_id,
    )


def project_legacy_memory(
    snapshot: Any, *, task_id: str
) -> LegacyMemoryProjection:
    """Project an M9 preference-memory snapshot into read-only long-term beliefs.

    ``snapshot`` is duck-typed: it must expose a ``user_key`` and an iterable
    ``entries``; each entry must expose ``memory_id``, ``kind``, ``value``,
    ``polarity``, ``status`` and ``logical_seq``.  The accepted service and store are
    not imported and the snapshot is not mutated.

    Ordering is total and reproducible (``logical_seq`` then ``memory_id``).  Only
    ``ACTIVE`` entries participate, every projected belief is
    ``LONG_TERM``/``EXPLICIT``/``ACTIVE``/``STRONG``/``LEGACY_MEMORY``, and entries that
    fail :class:`PreferenceBelief` validation are skipped and reported.  A repeated
    identity can never produce two projected beliefs: the earliest entry wins and the
    later duplicate is reported in ``skipped_memory_ids``.
    """
    resolved_task_id = bounded_identifier(
        task_id, field="task_id", max_length=MAX_TASK_ID_LENGTH
    )
    user_key = _snapshot_user_key(snapshot)
    entries = tuple(getattr(snapshot, "entries", ()) or ())

    def _order_key(entry: Any) -> tuple[int, str]:
        raw_sequence = getattr(entry, "logical_seq", 0)
        try:
            sequence = int(raw_sequence)
        except (TypeError, ValueError):
            sequence = 0
        return (sequence, str(getattr(entry, "memory_id", "")))

    beliefs: list[PreferenceBelief] = []
    skipped: list[str] = []
    seen_identities: set[tuple[str, str, str, str]] = set()
    for entry in sorted(entries, key=_order_key):
        if not _entry_is_active(entry):
            continue
        try:
            memory_id = _entry_memory_id(entry)
            belief = _belief_from_entry(
                entry,
                user_key=user_key,
                task_id=resolved_task_id,
                memory_id=memory_id,
            )
        except (TypeError, ValueError):
            # Report the skip when the id itself is at least a usable string; an entry
            # without a valid id cannot be named in the receipt, so it is dropped.
            raw_memory_id = getattr(entry, "memory_id", None)
            if isinstance(raw_memory_id, str) and raw_memory_id.strip():
                try:
                    skipped.append(
                        bounded_identifier(
                            raw_memory_id,
                            field="memory_id",
                            max_length=MAX_REF_LENGTH,
                        )
                    )
                except ValueError:
                    pass
            continue
        identity = belief.identity()
        if identity in seen_identities:
            skipped.append(memory_id)
            continue
        seen_identities.add(identity)
        beliefs.append(belief)
    return LegacyMemoryProjection(
        user_key=user_key,
        beliefs=tuple(beliefs),
        skipped_memory_ids=tuple(skipped),
    )


def merge_legacy_memory(
    state: UnifiedDecisionState, projection: LegacyMemoryProjection
) -> UnifiedDecisionState:
    """Return ``state`` with the projection's non-conflicting long-term beliefs appended.

    The event-sourced state wins: a legacy belief whose identity
    ``(scope, kind, normalised value, polarity)`` is already present is ignored, so the
    adapter can never override a belief the user expressed through events.  Only the
    ``preference_beliefs`` tuple changes; interaction counters, observations, task
    context and trusted references are untouched, and nothing is written back to the M9
    store.  The input state and projection are not mutated.
    """
    if projection.user_key != state.user_key:
        raise CrossUserStateError(
            f"legacy-memory projection belongs to user {projection.user_key!r}, "
            f"not {state.user_key!r}"
        )
    known_identities = {
        belief.identity() for belief in state.preference_beliefs
    }
    additions: list[PreferenceBelief] = []
    for belief in projection.beliefs:
        identity = belief.identity()
        if identity in known_identities:
            continue
        known_identities.add(identity)
        additions.append(belief)
    if not additions:
        return state
    return state.model_copy(
        update={"preference_beliefs": state.preference_beliefs + tuple(additions)}
    )
