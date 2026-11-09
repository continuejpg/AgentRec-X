"""Tests for the M5-MVP append-only SQLite event store.

Covers the storage contract only: per-user sequence assignment, idempotent duplicate ids,
conflicting duplicate ids, deterministic sequence ordering, long-term event visibility,
user isolation, persistence across reopen and the recorded store schema version.  State
reduction and manager orchestration are covered by ``tests/test_state_manager.py``.

Every test uses ``tests.state_fixtures.temporary_database`` (never ``tmp_path``: 0700
temporary directories are unusable on the confined Windows host this project is developed
on) and fixed timestamps, so no test depends on a wall clock or on the order in which
temp directories happen to be created.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.errors import (  # noqa: E402
    EventConflictError,
    EventStoreError,
)
from recommendation.state.event_store import (  # noqa: E402
    EVENT_STORE_SCHEMA_VERSION,
    STATE_DB_ENV_VAR,
    SQLiteEventStore,
)
from tests.state_fixtures import (  # noqa: E402
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_TS,
    DEFAULT_USER,
    candidate_pool_event,
    context_event,
    preference_stated_event,
    temporary_database,
)

OTHER_USER = "user-b"
OTHER_TASK = "task-2"


@pytest.fixture()
def store():
    """Open a store on a fresh temporary database and close it afterwards."""
    with temporary_database() as database:
        instance = SQLiteEventStore(database)
        try:
            yield instance
        finally:
            instance.close()


def _sorted_event_ids(events) -> list[str]:
    """Return event ids in the order the store returned them."""
    return [event.event_id for event in events]


# --------------------------------------------------------------------------- #
# Sequence assignment
# --------------------------------------------------------------------------- #


def test_append_assigns_sequences_per_user_independently(store) -> None:
    first = store.append(
        preference_stated_event(event_id="evt-a1", user_key="user-a")
    )
    second = store.append(
        preference_stated_event(event_id="evt-a2", user_key="user-a", value="blue")
    )
    third = store.append(
        preference_stated_event(event_id="evt-a3", user_key="user-a", value="green")
    )
    other = store.append(
        preference_stated_event(event_id="evt-b1", user_key=OTHER_USER)
    )

    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
    # Another user starts its own sequence, unaffected by the first user's events.
    assert other.sequence == 1
    assert [event.sequence for event in store.get_all_events("user-a")] == [1, 2, 3]
    assert [event.sequence for event in store.get_all_events(OTHER_USER)] == [1]
    # The stored copy preserves the caller's metadata (timestamp/session are not order).
    assert first.timestamp == DEFAULT_TS
    assert first.session_id == DEFAULT_SESSION
    assert first.user_key == "user-a"


# --------------------------------------------------------------------------- #
# Idempotent duplicate ids
# --------------------------------------------------------------------------- #


def test_duplicate_id_with_matching_semantics_is_idempotent(store) -> None:
    original = store.append(
        preference_stated_event(event_id="evt-dup", timestamp=DEFAULT_TS)
    )
    # Same semantic event, retried with a different recording time.
    retry = preference_stated_event(event_id="evt-dup", timestamp=DEFAULT_TS + 500)
    assert retry.is_same_submission(original)

    returned = store.append(retry)

    assert returned == original
    assert returned.sequence == 1
    # The stored copy wins: the retry timestamp did not overwrite recorded metadata.
    assert returned.timestamp == DEFAULT_TS
    assert store.count("user-a") == 1
    assert _sorted_event_ids(store.get_all_events("user-a")) == ["evt-dup"]
    # The duplicate did not consume a sequence number.
    following = store.append(
        preference_stated_event(event_id="evt-next", value="blue")
    )
    assert following.sequence == 2


def test_duplicate_id_with_different_payload_raises_conflict(store) -> None:
    stored = store.append(
        preference_stated_event(
            event_id="evt-conflict",
            value="lightweight",
            source_text="the stored secret payload text",
        )
    )
    changed = preference_stated_event(event_id="evt-conflict", value="heavy")

    with pytest.raises(EventConflictError) as raised:
        store.append(changed)

    # The message names the conflict, never echoes the stored payload.
    message = str(raised.value)
    assert "lightweight" not in message
    assert "the stored secret payload text" not in message
    # Nothing was written and the existing stream is untouched.
    assert store.count("user-a") == 1
    replayed = store.get_all_events("user-a")
    assert len(replayed) == 1
    assert replayed[0] == stored
    assert replayed[0].payload.value == "lightweight"
    assert replayed[0].sequence == 1


def test_duplicate_id_submitted_by_another_user_is_a_conflict(store) -> None:
    stored = store.append(
        preference_stated_event(event_id="evt-shared", user_key="user-a")
    )

    with pytest.raises(EventConflictError):
        store.append(
            preference_stated_event(event_id="evt-shared", user_key=OTHER_USER)
        )

    # A duplicate id never leaks one user's event into the other's scope.
    assert store.count("user-a") == 1
    assert store.count(OTHER_USER) == 0
    assert store.get_all_events(OTHER_USER) == ()
    assert store.get_event("evt-shared", user_key=OTHER_USER) is None
    owned = store.get_event("evt-shared", user_key="user-a")
    assert owned is not None
    assert owned == stored


# --------------------------------------------------------------------------- #
# Cross-user isolation
# --------------------------------------------------------------------------- #


def test_user_scoped_reads_are_isolated(store) -> None:
    store.append(preference_stated_event(event_id="evt-a", user_key="user-a"))
    store.append(context_event(event_id="evt-a2", user_key="user-a"))
    store.append(preference_stated_event(event_id="evt-b", user_key=OTHER_USER))

    assert _sorted_event_ids(store.get_events(DEFAULT_TASK, "user-a")) == [
        "evt-a",
        "evt-a2",
    ]
    assert _sorted_event_ids(store.get_events(DEFAULT_TASK, OTHER_USER)) == ["evt-b"]
    assert _sorted_event_ids(store.get_all_events("user-a")) == ["evt-a", "evt-a2"]
    assert _sorted_event_ids(store.get_all_events(OTHER_USER)) == ["evt-b"]
    assert store.count("user-a") == 2
    assert store.count(OTHER_USER) == 1
    assert store.count("unknown-user") == 0
    # A user-scoped id lookup never discloses another user's event.
    assert store.get_event("evt-b", user_key="user-a") is None
    assert store.get_event("evt-a", user_key=OTHER_USER) is None
    # The explicit global lookup by id is the only cross-user read surface.
    assert store.get_event("evt-b") is not None


def test_another_users_events_never_appear_in_user_scoped_reads(store) -> None:
    store.append(preference_stated_event(event_id="evt-a-task", user_key="user-a"))
    store.append(
        preference_stated_event(
            event_id="evt-a-long",
            user_key="user-a",
            task_id=OTHER_TASK,
            scope="long_term",
            value="blue",
        )
    )
    store.append(preference_stated_event(event_id="evt-b-task", user_key=OTHER_USER))
    store.append(
        preference_stated_event(
            event_id="evt-b-long",
            user_key=OTHER_USER,
            task_id=OTHER_TASK,
            scope="long_term",
            value="green",
        )
    )

    foreign_ids = {"evt-b-task", "evt-b-long"}
    for read in (
        store.get_events(DEFAULT_TASK, "user-a"),
        store.get_events(OTHER_TASK, "user-a"),
        store.get_all_events("user-a"),
        store.replay(DEFAULT_TASK, "user-a"),
    ):
        assert {event.event_id for event in read}.isdisjoint(foreign_ids)
    assert store.count("user-a") == 2
    for event_id in foreign_ids:
        assert store.get_event(event_id, user_key="user-a") is None


# --------------------------------------------------------------------------- #
# Deterministic ordering
# --------------------------------------------------------------------------- #


def test_ordering_is_by_sequence_not_timestamp(store) -> None:
    appended: list[str] = []
    # Timestamps decrease with each append; the store must still order by sequence.
    for index, timestamp in enumerate(
        (DEFAULT_TS, DEFAULT_TS - 100.0, DEFAULT_TS - 200.0), start=1
    ):
        event = preference_stated_event(
            event_id=f"evt-order-{index}",
            value=f"value-{index}",
            timestamp=timestamp,
        )
        appended.append(store.append(event).event_id)

    assert _sorted_event_ids(store.get_all_events("user-a")) == appended
    assert [event.sequence for event in store.get_all_events("user-a")] == [1, 2, 3]
    task_stream = store.get_events(DEFAULT_TASK, "user-a")
    assert _sorted_event_ids(task_stream) == appended
    assert [event.sequence for event in task_stream] == [1, 2, 3]
    # ``replay`` is the named read a reducer folds; it must expose the same ordering.
    assert store.replay(DEFAULT_TASK, "user-a") == task_stream


# --------------------------------------------------------------------------- #
# Long-term events cross tasks; other tasks' task-scoped events do not
# --------------------------------------------------------------------------- #


def test_get_events_includes_user_long_term_events_from_other_tasks(store) -> None:
    store.append(
        preference_stated_event(
            event_id="evt-task-1", task_id=DEFAULT_TASK, value="lightweight"
        )
    )
    other_task_event = store.append(
        preference_stated_event(
            event_id="evt-task-2", task_id=OTHER_TASK, value="blue"
        )
    )
    long_term_event = store.append(
        preference_stated_event(
            event_id="evt-long-term",
            task_id=OTHER_TASK,
            scope="long_term",
            value="linen only",
        )
    )
    foreign_long_term = store.append(
        preference_stated_event(
            event_id="evt-foreign-long-term",
            user_key=OTHER_USER,
            task_id=OTHER_TASK,
            scope="long_term",
            value="wool only",
        )
    )

    stream = store.get_events(DEFAULT_TASK, "user-a")
    assert _sorted_event_ids(stream) == ["evt-task-1", "evt-long-term"]
    # Ordering key is the user's global sequence, not the task-local order.
    assert [event.sequence for event in stream] == [1, long_term_event.sequence]
    assert long_term_event.sequence == 3
    # A task-scoped event from another task is not part of this task's stream.
    assert other_task_event.event_id not in {event.event_id for event in stream}
    # Another user's long-term event is never included, even for the same task id.
    assert foreign_long_term.event_id not in {event.event_id for event in stream}
    # Long-term inclusion is explicit and can be disabled.
    assert _sorted_event_ids(
        store.get_events(DEFAULT_TASK, "user-a", include_long_term=False)
    ) == ["evt-task-1"]
    # The long-term event is still available when replaying its own task.
    assert "evt-long-term" in _sorted_event_ids(
        store.get_events(OTHER_TASK, "user-a")
    )


# --------------------------------------------------------------------------- #
# Single-event reads
# --------------------------------------------------------------------------- #


def test_get_event_round_trips_payload_and_sequence(store) -> None:
    stored = store.append(
        candidate_pool_event(event_id="evt-pool", candidate_count=7)
    )

    fetched = store.get_event("evt-pool")
    assert fetched == stored
    assert fetched is not None
    assert fetched.sequence == 1
    assert type(fetched.payload) is type(stored.payload)
    assert fetched.payload.candidate_count == 7
    assert fetched.scope is None
    assert store.get_event("evt-missing") is None
    assert store.get_event("evt-pool", user_key=DEFAULT_USER) == stored
    assert store.get_event("evt-pool", user_key=OTHER_USER) is None


# --------------------------------------------------------------------------- #
# Persistence and schema version
# --------------------------------------------------------------------------- #


def test_reopening_a_file_database_preserves_every_event() -> None:
    with temporary_database() as database:
        first = SQLiteEventStore(database)
        try:
            stored = [
                first.append(preference_stated_event(event_id="evt-a1")),
                first.append(
                    preference_stated_event(
                        event_id="evt-a2", task_id=OTHER_TASK, value="blue"
                    )
                ),
                first.append(
                    preference_stated_event(
                        event_id="evt-a3",
                        task_id=OTHER_TASK,
                        scope="long_term",
                        value="green",
                    )
                ),
                first.append(
                    preference_stated_event(event_id="evt-b1", user_key=OTHER_USER)
                ),
            ]
        finally:
            first.close()

        second = SQLiteEventStore(database)
        try:
            assert second.schema_version == EVENT_STORE_SCHEMA_VERSION
            assert second.get_all_events("user-a") == tuple(stored[:3])
            assert second.get_all_events(OTHER_USER) == (stored[3],)
            # The task-1 stream is preserved, including the cross-task long-term event.
            assert second.get_events(DEFAULT_TASK, "user-a") == (
                stored[0],
                stored[2],
            )
            assert second.get_event("evt-a3") == stored[2]
            assert second.count("user-a") == 3
            # Sequence assignment continues after reopen instead of restarting.
            next_event = second.append(
                preference_stated_event(event_id="evt-a4", value="heavy")
            )
            assert next_event.sequence == 4
        finally:
            second.close()


def test_schema_version_is_recorded(store) -> None:
    assert store.schema_version == EVENT_STORE_SCHEMA_VERSION
    assert EVENT_STORE_SCHEMA_VERSION == 1
    assert STATE_DB_ENV_VAR == "AGENTRECX_STATE_DB"


def test_schema_version_is_written_to_the_meta_table() -> None:
    with temporary_database() as database:
        with SQLiteEventStore(database) as store:
            assert store.schema_version == EVENT_STORE_SCHEMA_VERSION
        connection = sqlite3.connect(database)
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'event_store_schema_version'"
            ).fetchone()
        finally:
            connection.close()
    assert row is not None
    assert int(row[0]) == EVENT_STORE_SCHEMA_VERSION


def test_unknown_stored_schema_version_is_refused() -> None:
    with temporary_database() as database:
        with SQLiteEventStore(database):
            pass
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE meta SET value = '2' WHERE key = 'event_store_schema_version'"
            )
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(EventStoreError):
            SQLiteEventStore(database)


def test_unsupported_stored_event_schema_version_is_a_typed_error() -> None:
    """A row written for another event contract must never be partially trusted."""
    with temporary_database() as database:
        with SQLiteEventStore(database) as store:
            store.append(preference_stated_event(event_id="evt-1"))
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE events SET schema_version = 2 WHERE event_id = 'evt-1'"
            )
            connection.commit()
        finally:
            connection.close()

        with SQLiteEventStore(database) as store:
            with pytest.raises(EventStoreError):
                store.get_event("evt-1")


# --------------------------------------------------------------------------- #
# Thread safety (the store is safe to share across threads)
# --------------------------------------------------------------------------- #


def test_store_is_usable_from_a_different_thread() -> None:
    """Regression for the M9 failure mode: create on one thread, use on another."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        created_on = threading.get_ident()
        results: dict[str, object] = {}

        def use_from_worker() -> None:
            try:
                results["thread"] = threading.get_ident()
                store.append(preference_stated_event(event_id="evt-worker"))
                results["count"] = store.count("user-a")
                results["ids"] = _sorted_event_ids(store.get_all_events("user-a"))
            except BaseException as exc:  # noqa: BLE001 - reported through ``results``
                results["error"] = exc

        worker = threading.Thread(target=use_from_worker)
        worker.start()
        worker.join()
        try:
            assert "error" not in results, results.get("error")
            assert results["thread"] != created_on
            assert results["count"] == 1
            assert results["ids"] == ["evt-worker"]
            # The creating thread can still use the store afterwards.
            assert store.count("user-a") == 1
        finally:
            store.close()


def test_concurrent_appends_get_unique_sequential_sequences() -> None:
    """Concurrent writers are serialised by the store lock and one transaction each."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        errors: list[BaseException] = []
        barrier = threading.Barrier(6)

        def write(index: int) -> None:
            try:
                barrier.wait(timeout=10)
                store.append(
                    preference_stated_event(
                        event_id=f"evt-concurrent-{index}",
                        value=f"value-{index}",
                        timestamp=DEFAULT_TS + index,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)

        threads = [
            threading.Thread(target=write, args=(index,)) for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        try:
            assert errors == []
            events = store.get_all_events("user-a")
            assert len(events) == 6
            assert sorted(event.sequence for event in events) == [1, 2, 3, 4, 5, 6]
            assert len({event.event_id for event in events}) == 6
            # Reads are always in sequence order, never thread-scheduling order.
            sequences = [event.sequence for event in events]
            assert sequences == sorted(sequences)
        finally:
            store.close()


# --------------------------------------------------------------------------- #
# Argument validation and lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_read_arguments_are_refused(store, blank: str) -> None:
    with pytest.raises(ValueError):
        store.get_events(blank, "user-a")
    with pytest.raises(ValueError):
        store.get_events(DEFAULT_TASK, blank)
    with pytest.raises(ValueError):
        store.get_all_events(blank)
    with pytest.raises(ValueError):
        store.count(blank)
    with pytest.raises(ValueError):
        store.get_event(blank)
    with pytest.raises(ValueError):
        store.get_event("evt-1", user_key=blank)


def test_context_manager_closes_the_store() -> None:
    with temporary_database() as database:
        with SQLiteEventStore(database) as store:
            store.append(preference_stated_event(event_id="evt-1"))
            assert store.count("user-a") == 1
        # The closed store reports a typed error rather than a driver exception.
        with pytest.raises(EventStoreError):
            store.count("user-a")
