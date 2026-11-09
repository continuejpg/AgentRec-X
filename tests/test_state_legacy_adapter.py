"""Tests for the read-only M9 preference-memory -> M5 belief adapter.

The adapter is a compatibility bridge, not a migration: it reads an already-produced
``PreferenceMemorySnapshot`` (duck-typed) and exposes the active explicit entries as
long-term beliefs.  These tests pin its contract:

* only ``ACTIVE`` entries project, each as
  ``LONG_TERM``/``EXPLICIT``/``ACTIVE``/``STRONG``/``LEGACY_MEMORY`` with
  ``legacy_memory_id`` and a deterministic belief id;
* ordering is deterministic (``logical_seq`` then ``memory_id``) and the snapshot is not
  mutated;
* an entry the state model refuses (e.g. a credential-looking value) is skipped and
  reported instead of failing the projection;
* merging never overrides an event-sourced belief of the same identity, never touches
  interaction metadata or events, and is idempotent;
* a projection for another user is refused with ``CrossUserStateError``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from recommendation.memory.schemas import (
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
)
from recommendation.state.belief_updater import BeliefUpdater
from recommendation.state.decision_state import (
    empty_state,
    preference_belief_id,
)
from recommendation.state.errors import CrossUserStateError
from recommendation.state.legacy_adapter import (
    LegacyMemoryProjection,
    merge_legacy_memory,
    project_legacy_memory,
)
from recommendation.state.schemas import (
    BeliefOrigin,
    BeliefStrength,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
)
from tests.state_fixtures import (
    DEFAULT_TASK,
    DEFAULT_USER,
    comparison_event,
    preference_stated_event,
)

UPDATER = BeliefUpdater()
FIXED_READ_AT = 1_700_000_000.0


def _snapshot(*entries: PreferenceMemoryEntry, user_key: str = DEFAULT_USER):
    """A trusted M9 snapshot with a fixed read time (no wall clock in tests)."""
    return PreferenceMemorySnapshot(
        user_key=user_key, entries=tuple(entries), read_at=FIXED_READ_AT
    )


def _entry(
    memory_id: str,
    kind: PreferenceKind,
    value: str,
    *,
    status: PreferenceStatus = PreferenceStatus.ACTIVE,
    logical_seq: int = 1,
    polarity: PreferencePolarity = PreferencePolarity.PREFER,
    source_text: str = "I like that.",
) -> PreferenceMemoryEntry:
    """One accepted M9 memory entry."""
    return PreferenceMemoryEntry(
        memory_id=memory_id,
        user_key=DEFAULT_USER,
        kind=kind,
        value=value,
        polarity=polarity,
        source_text=source_text,
        source_turn_id="turn-1",
        status=status,
        logical_seq=logical_seq,
        created_at=FIXED_READ_AT,
    )


@dataclass(frozen=True)
class _FakeEntry:
    """Minimal duck-typed entry, used where the accepted M9 model would refuse the value."""

    memory_id: str
    kind: object
    value: str
    status: object = "active"
    logical_seq: int = 1
    polarity: object = "prefer"


@dataclass(frozen=True)
class _FakeSnapshot:
    """Minimal duck-typed snapshot (the adapter must not require the accepted types)."""

    user_key: str
    entries: tuple[_FakeEntry, ...]


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def test_active_entries_become_long_term_explicit_beliefs() -> None:
    snapshot = _snapshot(
        _entry("mem-2", PreferenceKind.BRAND, "acme", logical_seq=2),
        _entry("mem-1", PreferenceKind.COLOR, "black", logical_seq=1),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert projection.user_key == DEFAULT_USER
    assert projection.skipped_memory_ids == ()
    assert [belief.value for belief in projection.beliefs] == ["black", "acme"]
    for belief in projection.beliefs:
        assert belief.scope is PreferenceScope.LONG_TERM
        assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
        assert belief.status is PreferenceStatus.ACTIVE
        assert belief.strength is BeliefStrength.STRONG
        assert belief.origin is BeliefOrigin.LEGACY_MEMORY
        assert belief.supporting_event_ids == ()
        assert belief.last_event_id is None
        assert belief.direction is None

    first, second = projection.beliefs
    assert first.legacy_memory_id == "mem-1"
    assert second.legacy_memory_id == "mem-2"
    assert first.belief_id == preference_belief_id(
        user_key=DEFAULT_USER,
        task_id=DEFAULT_TASK,
        scope=PreferenceScope.LONG_TERM,
        kind=PreferenceKind.COLOR,
        value="black",
        polarity=PreferencePolarity.PREFER,
    )


def test_projection_order_is_logical_seq_then_memory_id() -> None:
    snapshot = _snapshot(
        _entry("mem-b", PreferenceKind.FEATURE, "b", logical_seq=2),
        _entry("mem-a", PreferenceKind.FEATURE, "a", logical_seq=2),
        _entry("mem-first", PreferenceKind.FEATURE, "first", logical_seq=1),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert [belief.legacy_memory_id for belief in projection.beliefs] == [
        "mem-first",
        "mem-a",
        "mem-b",
    ]


def test_superseded_and_removed_entries_are_excluded() -> None:
    snapshot = _snapshot(
        _entry(
            "mem-superseded",
            PreferenceKind.COLOR,
            "red",
            status=PreferenceStatus.SUPERSEDED,
            logical_seq=1,
        ),
        _entry(
            "mem-removed",
            PreferenceKind.COLOR,
            "green",
            status=PreferenceStatus.REMOVED,
            logical_seq=2,
        ),
        _entry("mem-active", PreferenceKind.COLOR, "black", logical_seq=3),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert [belief.value for belief in projection.beliefs] == ["black"]
    # Inactive entries are not current truth; they are not "skipped" either.
    assert projection.skipped_memory_ids == ()


def test_secret_like_and_oversized_values_are_skipped_and_reported() -> None:
    snapshot = _FakeSnapshot(
        user_key=DEFAULT_USER,
        entries=(
            _FakeEntry(
                "mem-secret",
                PreferenceKind.FEATURE,
                "sk_live_abcdefgh12345678",
                logical_seq=1,
            ),
            _FakeEntry(
                "mem-too-long",
                PreferenceKind.FEATURE,
                "x" * 200,
                logical_seq=2,
            ),
            _FakeEntry("mem-ok", PreferenceKind.COLOR, "black", logical_seq=3),
        ),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert [belief.legacy_memory_id for belief in projection.beliefs] == ["mem-ok"]
    assert projection.skipped_memory_ids == ("mem-secret", "mem-too-long")


def test_duplicate_identity_is_projected_once_and_reported() -> None:
    snapshot = _snapshot(
        _entry("mem-first", PreferenceKind.COLOR, "black", logical_seq=1),
        _entry("mem-duplicate", PreferenceKind.COLOR, "black", logical_seq=2),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert [belief.legacy_memory_id for belief in projection.beliefs] == ["mem-first"]
    assert projection.skipped_memory_ids == ("mem-duplicate",)


def test_long_term_belief_id_does_not_depend_on_task() -> None:
    snapshot = _snapshot(_entry("mem-1", PreferenceKind.COLOR, "black"))

    for_task_one = project_legacy_memory(snapshot, task_id="task-1")
    for_task_two = project_legacy_memory(snapshot, task_id="task-2")

    assert for_task_one.beliefs == for_task_two.beliefs


def test_projection_duck_types_the_snapshot() -> None:
    snapshot = _FakeSnapshot(
        user_key=DEFAULT_USER,
        entries=(_FakeEntry("mem-1", "brand", "acme", logical_seq=1),),
    )

    projection = project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    (belief,) = projection.beliefs
    assert belief.kind is PreferenceKind.BRAND
    assert belief.polarity is PreferencePolarity.PREFER
    assert belief.scope is PreferenceScope.LONG_TERM


def test_projection_does_not_mutate_the_snapshot() -> None:
    snapshot = _snapshot(
        _entry("mem-1", PreferenceKind.COLOR, "black"),
        _entry("mem-2", PreferenceKind.BRAND, "acme", logical_seq=2),
    )
    before = snapshot.model_dump()

    project_legacy_memory(snapshot, task_id=DEFAULT_TASK)

    assert snapshot.model_dump() == before


def test_invalid_snapshot_user_key_is_refused() -> None:
    with pytest.raises(ValueError):
        project_legacy_memory(_FakeSnapshot(user_key="  ", entries=()), task_id=DEFAULT_TASK)


def test_adapter_module_does_not_import_memory_service_or_store() -> None:
    import recommendation.state.legacy_adapter as adapter_module

    source = inspect.getsource(adapter_module)
    assert "PreferenceMemoryService" not in source
    assert "memory.service" not in source
    assert "memory.store" not in source


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def test_merge_appends_legacy_beliefs_without_touching_interaction() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    state = UPDATER.apply(
        state,
        preference_stated_event(
            event_id="evt-user", kind="feature", value="lightweight"
        ),
    )
    state = UPDATER.apply(state, comparison_event(event_id="evt-comparison"))
    projection = project_legacy_memory(
        _snapshot(_entry("mem-1", PreferenceKind.COLOR, "black")),
        task_id=DEFAULT_TASK,
    )

    merged = merge_legacy_memory(state, projection)

    assert len(state.preference_beliefs) == 1
    assert len(merged.preference_beliefs) == 2
    event_sourced, legacy = merged.preference_beliefs
    assert event_sourced == state.preference_beliefs[0]
    assert event_sourced.scope is PreferenceScope.TASK
    assert legacy.scope is PreferenceScope.LONG_TERM
    assert legacy.origin is BeliefOrigin.LEGACY_MEMORY
    assert legacy.legacy_memory_id == "mem-1"
    # A read-only adapter never fabricates interaction metadata, events or observations.
    assert merged.interaction == state.interaction
    assert merged.observations == state.observations
    assert merged.session_id == state.session_id
    assert merged.task_context == state.task_context
    # The input state is not mutated.
    assert len(state.preference_beliefs) == 1


def test_merge_never_overrides_event_sourced_long_term_identity() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    state = UPDATER.apply(
        state,
        preference_stated_event(
            event_id="evt-long-term",
            scope="long_term",
            kind="color",
            value="black",
        ),
    )
    projection = project_legacy_memory(
        _snapshot(
            _entry("mem-black", PreferenceKind.COLOR, "black", logical_seq=1),
            _entry("mem-blue", PreferenceKind.COLOR, "blue", logical_seq=2),
        ),
        task_id=DEFAULT_TASK,
    )

    merged = merge_legacy_memory(state, projection)

    assert len(merged.preference_beliefs) == 2
    black, blue = merged.preference_beliefs
    # Event-sourced state wins: the explicit user statement is untouched.
    assert black.value == "black"
    assert black.origin is BeliefOrigin.USER_STATED
    assert black.lifecycle is PreferenceLifecycle.EXPLICIT
    assert black.supporting_event_ids == ("evt-long-term",)
    assert black.legacy_memory_id is None
    # Only the non-conflicting legacy identity is appended.
    assert blue.value == "blue"
    assert blue.origin is BeliefOrigin.LEGACY_MEMORY


def test_merge_is_idempotent() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    projection = project_legacy_memory(
        _snapshot(
            _entry("mem-1", PreferenceKind.COLOR, "black"),
            _entry("mem-2", PreferenceKind.BRAND, "acme", logical_seq=2),
        ),
        task_id=DEFAULT_TASK,
    )

    first = merge_legacy_memory(state, projection)
    second = merge_legacy_memory(first, projection)

    assert second == first
    assert second.state_digest() == first.state_digest()


def test_merge_with_empty_projection_returns_state_unchanged() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    projection = project_legacy_memory(_snapshot(), task_id=DEFAULT_TASK)

    assert merge_legacy_memory(state, projection) == state
    assert projection.beliefs == ()


def test_merge_refuses_projection_for_another_user() -> None:
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    projection = project_legacy_memory(
        _snapshot(
            _entry("mem-1", PreferenceKind.COLOR, "black"), user_key="user-b"
        ),
        task_id=DEFAULT_TASK,
    )

    with pytest.raises(CrossUserStateError):
        merge_legacy_memory(state, projection)


def test_merge_accepts_hand_built_projection() -> None:
    """The merge step only needs a ``LegacyMemoryProjection`` value."""
    state = empty_state(user_key=DEFAULT_USER, task_id=DEFAULT_TASK)
    projection = project_legacy_memory(
        _snapshot(_entry("mem-1", PreferenceKind.COLOR, "black")),
        task_id=DEFAULT_TASK,
    )
    hand_built = LegacyMemoryProjection(
        user_key=projection.user_key, beliefs=projection.beliefs
    )

    merged = merge_legacy_memory(state, hand_built)

    assert merged.preference_beliefs == projection.beliefs
