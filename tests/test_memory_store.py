"""Milestone 9 store tests: in-memory and SQLite preference stores.

Both backends are exercised through the same parametrised fixture, so a behaviour that
only one of them implements fails the suite.

Fully offline and deterministic: SQLite runs in a temporary directory (and ``:memory:``
where useful), no database file is committed, and no test depends on wall-clock time
or hash iteration order.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.memory import (  # noqa: E402
    MEMORY_SCHEMA_VERSION,
    InMemoryPreferenceStore,
    PreferenceKind,
    PreferenceMemoryEntry,
    PreferencePolarity,
    PreferenceStatus,
    SQLitePreferenceStore,
    deterministic_memory_id,
)


def make_entry(
    user_key: str = "alice",
    turn_id: str = "t1",
    kind: PreferenceKind = PreferenceKind.COLOR,
    value: str = "black",
    polarity: PreferencePolarity = PreferencePolarity.PREFER,
    seq: int = 1,
    status: PreferenceStatus = PreferenceStatus.ACTIVE,
    created_at: float = 1000.0,
    **overrides: object,
) -> PreferenceMemoryEntry:
    """Build a valid entry with a deterministic identity."""
    payload: dict[str, object] = {
        "memory_id": deterministic_memory_id(
            user_key, turn_id, kind.value, value, polarity.value
        ),
        "user_key": user_key,
        "kind": kind,
        "value": value,
        "polarity": polarity,
        "source_text": f"user said {value}",
        "source_turn_id": turn_id,
        "extractor": "test",
        "status": status,
        "logical_seq": seq,
        "created_at": created_at,
    }
    payload.update(overrides)
    return PreferenceMemoryEntry(**payload)  # type: ignore[arg-type]


@pytest.fixture(params=["in_memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path):
    """One store instance per backend, closed automatically."""
    if request.param == "in_memory":
        yield InMemoryPreferenceStore()
        return
    instance = SQLitePreferenceStore(tmp_path / "memory.db")
    yield instance
    instance.close()


# --------------------------------------------------------------------------- #
# Basic writes and reads
# --------------------------------------------------------------------------- #


def test_add_and_read_back_an_entry(store) -> None:
    entry = make_entry()
    store.add_entry(entry)
    fetched = store.get_entry("alice", entry.memory_id)
    assert fetched == entry


def test_unknown_entry_returns_none(store) -> None:
    assert store.get_entry("alice", "nope") is None


def test_entries_are_returned_in_deterministic_order(store) -> None:
    third = make_entry(turn_id="t3", value="green", seq=3)
    first = make_entry(turn_id="t1", value="black", seq=1)
    second = make_entry(turn_id="t2", value="blue", seq=2)
    for entry in (third, first, second):
        store.add_entry(entry)
    assert [e.value for e in store.get_entries("alice")] == ["black", "blue", "green"]
    # Order is by logical sequence, not insertion or hash order.
    assert [e.logical_seq for e in store.get_entries("alice")] == [1, 2, 3]


def test_active_only_filter(store) -> None:
    active = make_entry(turn_id="t1", value="black", seq=1)
    superseded = make_entry(
        turn_id="t2",
        value="blue",
        seq=2,
        status=PreferenceStatus.SUPERSEDED,
        superseded_by="later",
    )
    removed = make_entry(
        turn_id="t3", value="green", seq=3, status=PreferenceStatus.REMOVED
    )
    for entry in (active, superseded, removed):
        store.add_entry(entry)

    assert [e.value for e in store.get_entries("alice", active_only=True)] == ["black"]
    assert [e.value for e in store.get_entries("alice")] == ["black", "blue", "green"]


def test_duplicate_memory_id_is_rejected(store) -> None:
    entry = make_entry()
    store.add_entry(entry)
    with pytest.raises(ValueError):
        store.add_entry(entry)


def test_max_logical_seq_tracks_the_user(store) -> None:
    assert store.max_logical_seq("alice") == 0
    store.add_entry(make_entry(seq=1))
    store.add_entry(make_entry(turn_id="t2", value="blue", seq=7))
    assert store.max_logical_seq("alice") == 7
    # Another user is unaffected.
    assert store.max_logical_seq("bob") == 0


def test_count_reports_active_and_total(store) -> None:
    store.add_entry(make_entry(seq=1))
    store.add_entry(
        make_entry(turn_id="t2", value="blue", seq=2, status=PreferenceStatus.SUPERSEDED)
    )
    assert store.count("alice") == 2
    assert store.count("alice", active_only=True) == 1
    assert store.count("bob") == 0


# --------------------------------------------------------------------------- #
# Update (no in-place provenance destruction)
# --------------------------------------------------------------------------- #


def test_update_entry_replaces_but_keeps_identity(store) -> None:
    entry = make_entry(seq=1)
    store.add_entry(entry)
    updated = entry.model_copy(
        update={"status": PreferenceStatus.SUPERSEDED, "superseded_by": "newer-id"}
    )
    store.update_entry(updated)
    fetched = store.get_entry("alice", entry.memory_id)
    assert fetched is not None
    assert fetched.status is PreferenceStatus.SUPERSEDED
    assert fetched.superseded_by == "newer-id"
    # The original provenance is still present on the record.
    assert fetched.source_text == entry.source_text
    assert fetched.source_turn_id == entry.source_turn_id


def test_update_unknown_entry_raises(store) -> None:
    with pytest.raises(KeyError):
        store.update_entry(make_entry())


# --------------------------------------------------------------------------- #
# User isolation (mandatory)
# --------------------------------------------------------------------------- #


def test_users_are_isolated(store) -> None:
    alice = make_entry(user_key="alice", value="black", seq=1)
    bob = make_entry(user_key="bob", value="green", seq=1)
    store.add_entry(alice)
    store.add_entry(bob)

    assert [e.value for e in store.get_entries("alice")] == ["black"]
    assert [e.value for e in store.get_entries("bob")] == ["green"]
    # A lookup for one user never returns the other's entry, even by exact id.
    assert store.get_entry("bob", alice.memory_id) is None
    assert store.get_entry("alice", bob.memory_id) is None


def test_identical_memory_ids_across_users_do_not_collide(store) -> None:
    """The primary key is (user_key, memory_id), so ids may repeat across users."""
    alice = make_entry(user_key="alice", value="black", seq=1)
    bob = alice.model_copy(update={"user_key": "bob"})
    store.add_entry(alice)
    store.add_entry(bob)
    assert store.count("alice") == 1
    assert store.count("bob") == 1


def test_memory_id_is_deterministic_and_origin_keyed() -> None:
    first = deterministic_memory_id("alice", "t1", "color", "black", "prefer")
    again = deterministic_memory_id("alice", "t1", "color", "black", "prefer")
    assert first == again
    # Different origin -> different id.
    assert first != deterministic_memory_id("alice", "t2", "color", "black", "prefer")
    assert first != deterministic_memory_id("alice", "t1", "color", "blue", "prefer")
    assert first != deterministic_memory_id("alice", "t1", "color", "black", "avoid")
    assert first != deterministic_memory_id("bob", "t1", "color", "black", "prefer")


# --------------------------------------------------------------------------- #
# Store containment: no interaction-event API
# --------------------------------------------------------------------------- #


def test_store_exposes_no_interaction_event_api(store) -> None:
    """Structural proof that trusted interaction history cannot be persisted here."""
    for forbidden in (
        "add_interaction",
        "add_interaction_from_text",
        "add_event",
        "append_history",
        "add_parent_asin",
        "record_click",
        "record_purchase",
    ):
        assert not hasattr(store, forbidden), f"store must not expose {forbidden}"


def test_entry_schema_has_no_interaction_or_item_fields() -> None:
    fields = set(PreferenceMemoryEntry.model_fields)
    assert fields.isdisjoint({"parent_asin", "item_id", "asin", "interaction", "event"})


# --------------------------------------------------------------------------- #
# SQLite specifics
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sqlite_store(tmp_path: Path):
    instance = SQLitePreferenceStore(tmp_path / "memory.db")
    yield instance
    instance.close()


def test_sqlite_records_schema_version(sqlite_store: SQLitePreferenceStore) -> None:
    assert sqlite_store.schema_version == MEMORY_SCHEMA_VERSION


def test_sqlite_reopen_preserves_everything(tmp_path: Path) -> None:
    """The M9 persistence requirement: close, reopen, state and provenance survive."""
    path = tmp_path / "memory.db"
    first = SQLitePreferenceStore(path)
    active = make_entry(turn_id="t1", value="black", seq=1)
    superseded = make_entry(
        turn_id="t2",
        value="blue",
        seq=2,
        status=PreferenceStatus.SUPERSEDED,
        superseded_by=active.memory_id,
    )
    first.add_entry(active)
    first.add_entry(superseded)
    first.close()

    second = SQLitePreferenceStore(path)
    try:
        assert second.count("alice") == 2
        assert [e.value for e in second.get_entries("alice", active_only=True)] == ["black"]
        reopened = second.get_entry("alice", active.memory_id)
        assert reopened is not None
        assert reopened.source_text == active.source_text
        assert reopened.source_turn_id == "t1"
        assert reopened.logical_seq == 1
        assert reopened.created_at == active.created_at
        superseded_row = second.get_entry("alice", superseded.memory_id)
        assert superseded_row is not None
        assert superseded_row.status is PreferenceStatus.SUPERSEDED
        assert superseded_row.superseded_by == active.memory_id
        # Idempotency-relevant read surface also survives.
        assert [e.value for e in second.get_entries_for_turn("alice", "t1")] == ["black"]
        # Isolation survives too.
        assert second.count("bob") == 0
    finally:
        second.close()


def test_sqlite_rejects_a_mismatched_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLitePreferenceStore(path)
    store.close()
    connection = sqlite3.connect(path)
    connection.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError):
        SQLitePreferenceStore(path)


def test_sqlite_in_memory_backend_works() -> None:
    store = SQLitePreferenceStore(":memory:")
    try:
        store.add_entry(make_entry())
        assert store.count("alice", active_only=True) == 1
    finally:
        store.close()


def test_sqlite_duplicate_id_is_rejected_inside_one_transaction(
    sqlite_store: SQLitePreferenceStore,
) -> None:
    entry = make_entry()
    sqlite_store.add_entry(entry)
    with pytest.raises(ValueError):
        sqlite_store.add_entry(entry)
    assert sqlite_store.count("alice") == 1


def test_sqlite_atomic_write_rolls_back_on_failure(
    sqlite_store: SQLitePreferenceStore,
) -> None:
    """A failure inside a transaction leaves no partial state behind."""
    sqlite_store.add_entry(make_entry(seq=1))

    with pytest.raises(RuntimeError):
        with sqlite_store._atomic_write():  # noqa: SLF001 - transaction semantics test
            sqlite_store.add_entry(make_entry(turn_id="t2", value="blue", seq=2))
            raise RuntimeError("simulated failure mid-update")

    # The pre-existing entry survives and the new one was rolled back.
    assert [e.value for e in sqlite_store.get_entries("alice")] == ["black"]


def test_sqlite_nested_atomic_write_rolls_back_the_outer_unit(
    sqlite_store: SQLitePreferenceStore,
) -> None:
    """A nested failure discarded by the caller still rolls back the outer unit."""
    sqlite_store.add_entry(make_entry(seq=1))

    with sqlite_store._atomic_write():  # noqa: SLF001 - transaction semantics test
        sqlite_store.add_entry(make_entry(turn_id="t2", value="blue", seq=2))
        try:
            with sqlite_store._atomic_write():  # noqa: SLF001
                sqlite_store.add_entry(make_entry(turn_id="t3", value="green", seq=3))
                raise RuntimeError("inner failure, caught by the caller")
        except RuntimeError:
            pass

    # Nothing from the outer unit survived, because the inner failure poisoned it.
    assert [e.value for e in sqlite_store.get_entries("alice")] == ["black"]


def test_sqlite_convenience_read_of_unknown_turn_is_empty(
    sqlite_store: SQLitePreferenceStore,
) -> None:
    assert sqlite_store.get_entries_for_turn("alice", "never") == ()


# --------------------------------------------------------------------------- #
# Cross-thread usability (Milestone 11 integration regression)
# --------------------------------------------------------------------------- #


def test_sqlite_store_is_usable_from_a_different_thread(tmp_path: Path) -> None:
    """Regression: the store must work when created and used on different threads.

    Milestone 11 serves the demo from a multi-threaded ASGI server, which creates the
    store during startup and then touches it from request worker threads.  SQLite's
    default ``check_same_thread`` affinity made every such request fail with
    ``sqlite3.ProgrammingError``.  The store already serialises every access behind its
    own ``RLock``, so sharing the connection is safe and is what this test pins.
    """
    import threading

    database = tmp_path / "cross_thread.sqlite3"
    store = SQLitePreferenceStore(database)
    creator = threading.get_ident()
    results: dict[str, object] = {}

    def use_from_another_thread() -> None:
        try:
            results["thread"] = threading.get_ident()
            store.add_entry(make_entry(seq=1))
            store.add_entry(make_entry(turn_id="t2", value="blue", seq=2))
            results["values"] = [entry.value for entry in store.get_entries("alice", active_only=True)]
            results["count"] = store.count("alice", active_only=True)
        except BaseException as exc:  # noqa: BLE001 - reported through ``results``
            results["error"] = exc

    worker = threading.Thread(target=use_from_another_thread)
    worker.start()
    worker.join()

    assert "error" not in results, results.get("error")
    assert results["thread"] != creator, "the fixture did not leave the creating thread"
    assert results["values"] == ["black", "blue"]
    assert results["count"] == 2
    # The creating thread can still use it afterwards.
    assert store.count("alice", active_only=True) == 2
    store.close()


def test_sqlite_store_serialises_concurrent_writers(tmp_path: Path) -> None:
    """Concurrent writers from many threads leave a consistent store, with no lost rows."""
    import threading

    store = SQLitePreferenceStore(tmp_path / "concurrent.sqlite3")
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            store.add_entry(make_entry(turn_id=f"t{index}", value=f"v{index}", seq=index))
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(1, 13)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.count("alice") == 12
    assert {entry.value for entry in store.get_entries("alice")} == {
        f"v{index}" for index in range(1, 13)
    }
    # Deterministic ordering is by ``logical_seq``, not by thread scheduling.
    assert [entry.logical_seq for entry in store.get_entries("alice")] == list(range(1, 13))
    store.close()
