"""Versioned Cold Resume snapshot: build, bind, validate, migrate (M1-D S6).

These tests do not start MCP: they pin the durable contract that makes a cold resume safe.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_web_request_store import _new_store, _setup  # noqa: E402
from web_demo.continuation import (  # noqa: E402
    CONTINUATION_EXPIRED,
    CONTINUATION_GENERATION_MISMATCH,
    CONTINUATION_MALFORMED,
    CONTINUATION_MISSING,
    CONTINUATION_MODE_UNAVAILABLE,
    CONTINUATION_PENDING_MISMATCH,
    CONTINUATION_SCHEMA,
    CONTINUATION_SCHEMA_MISMATCH,
    CONTINUATION_SESSION_MISMATCH,
    CONTINUATION_TASK_MISMATCH,
    CONTINUATION_UNREADABLE,
    CONTINUATION_VERSION_MISMATCH,
    CONTINUATION_VERSION,
    MCP_CONTINUATION_SCHEMA,
    build_web_continuation,
    validate_continuation,
)
from web_demo.state_store import (  # noqa: E402
    _SCHEMA_V1_STATEMENTS,
    _SCHEMA_V2_STATEMENTS,
    _SCHEMA_V3_STATEMENTS,
    _SCHEMA_V4_STATEMENTS,
    _SCHEMA_V5_STATEMENTS,
    SQLiteWebStateStore,
    StateDataCorruption,
    WEB_STATE_SCHEMA_VERSION,
)

PROFILE = "profile_0057"
USER_KEY = "p14e2_lite_0057"
QUERY = "I need some new headphones. Not sure what matters yet."
TASK = {"task_id": USER_KEY, "user_id": "p14e2_user_0057", "current_query": QUERY}


def _pending():
    store = _new_store(Path(tempfile.mkdtemp(prefix="continuation_")))
    try:
        _, pending = _setup(store, sid="web-r")
        return pending
    finally:
        store.close()


def _mcp_block(
    pending, *, expires_at: float, query: str = "q", session_id: str = "web-r", **overrides
) -> dict:
    values = {
        "schema": MCP_CONTINUATION_SCHEMA,
        "version": 1,
        "mode": "cpu_live",
        "user_key": USER_KEY,
        "session_id": session_id,
        "clarification_id": pending.clarification_id,
        "question": "What matters most for this purchase?",
        "reason": "missing_key_need",
        "current_query": query,
        "task": dict(TASK, current_query=query),
        "task_constraints": {
            "schema": "agentrecx.current_task_constraints.v1",
            "hard_exclusions": [],
            "soft_preferences": [],
            "source_turns": [],
        },
        "created_at": expires_at - 900.0,
        "expires_at": expires_at,
    }
    values.update(overrides)
    return values


def _snapshot(pending, *, expires_at: float | None = None, **overrides) -> dict:
    expiry = pending.expires_at if expires_at is None else expires_at
    values = {
        "schema": CONTINUATION_SCHEMA,
        "version": CONTINUATION_VERSION,
        "mode": "cpu_live",
        "profile_id": PROFILE,
        "session_id": "web-r",
        "web_task_id": pending.task_id,
        "pending_id": pending.pending_id,
        "clarification_id": pending.clarification_id,
        "reset_generation": 0,
        "query": "q",
        "question": "What matters most for this purchase?",
        "reason": "missing_key_need",
        "expires_at": expiry,
        "mcp": _mcp_block(pending, expires_at=expiry),
    }
    values.update(overrides)
    return values


def _check(snapshot, pending, **overrides):
    values = {
        "session_id": "web-r",
        "profile_id": PROFILE,
        "task_id": pending.task_id,
        "pending_id": pending.pending_id,
        "clarification_id": pending.clarification_id,
        "reset_generation": 0,
        "expires_at": pending.expires_at,
        "query": "q",
        "expected_user_key": USER_KEY,
    }
    values.update(overrides)
    return validate_continuation(snapshot, **values)


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


def test_builder_wraps_the_child_snapshot() -> None:
    pending = _pending()
    mcp = _mcp_block(pending, expires_at=pending.expires_at)
    snapshot = build_web_continuation(
        mcp_continuation=mcp, query="q", profile_id=PROFILE
    )
    assert snapshot is not None
    assert snapshot["schema"] == CONTINUATION_SCHEMA
    assert snapshot["version"] == CONTINUATION_VERSION
    assert snapshot["mode"] == "cpu_live"
    assert snapshot["profile_id"] == PROFILE
    assert snapshot["session_id"] == "web-r"
    assert snapshot["clarification_id"] == pending.clarification_id
    assert snapshot["query"] == "q"
    assert snapshot["question"].startswith("What matters")
    assert snapshot["expires_at"] == pending.expires_at
    assert snapshot["mcp"]["task"]["task_id"] == USER_KEY
    # The Web binding is stamped by the store, never by the builder.
    assert snapshot["pending_id"] == ""
    assert snapshot["web_task_id"] == ""
    assert snapshot["reset_generation"] == 0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda block: None,
        lambda block: {**block, "schema": "other"},
        lambda block: {**block, "mode": "qwen_live"},
        lambda block: {**block, "session_id": ""},
        lambda block: {**block, "question": ""},
        lambda block: {**block, "current_query": ""},
        lambda block: {**block, "task": {}},
        lambda block: {**block, "expires_at": "soon"},
        lambda block: {**block, "created_at": "now"},
    ],
)
def test_builder_returns_none_when_the_child_context_is_not_restorable(mutator) -> None:
    pending = _pending()
    mcp = _mcp_block(pending, expires_at=pending.expires_at)
    mutated = mutator(mcp)
    assert build_web_continuation(
        mcp_continuation=mutated, query="q", profile_id=PROFILE
    ) is None


# --------------------------------------------------------------------------- #
# validator
# --------------------------------------------------------------------------- #


def test_valid_snapshot_is_accepted_and_returns_the_mcp_block() -> None:
    pending = _pending()
    snapshot = _snapshot(pending)
    check = _check(snapshot, pending)
    assert check.ok
    assert check.payload == snapshot["mcp"]


def test_missing_or_unreadable_snapshot_is_rejected() -> None:
    pending = _pending()
    assert _check(None, pending).code == CONTINUATION_MISSING
    assert _check("not-an-object", pending).code == CONTINUATION_UNREADABLE


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"schema": "agentrecx.web.unknown"}, CONTINUATION_SCHEMA_MISMATCH),
        ({"version": 2}, CONTINUATION_VERSION_MISMATCH),
        ({"mode": "qwen_live"}, CONTINUATION_MODE_UNAVAILABLE),
        ({"profile_id": "profile_0049"}, "continuation_profile_mismatch"),
        ({"session_id": "web-other"}, CONTINUATION_SESSION_MISMATCH),
        ({"web_task_id": "task-other"}, CONTINUATION_TASK_MISMATCH),
        ({"pending_id": "pend-other"}, CONTINUATION_PENDING_MISMATCH),
        ({"clarification_id": "clar-other"}, "continuation_clarification_mismatch"),
        ({"reset_generation": 1}, CONTINUATION_GENERATION_MISMATCH),
        ({"expires_at": 0.0}, CONTINUATION_MALFORMED),
        ({"query": "different"}, CONTINUATION_MALFORMED),
    ],
)
def test_outer_binding_mismatches_are_rejected(overrides, code) -> None:
    pending = _pending()
    check = _check(_snapshot(pending, **overrides), pending)
    assert not check.ok
    assert check.code == code


def test_expired_snapshot_is_rejected() -> None:
    pending = _pending()
    past = time.time() - 10.0
    snapshot = _snapshot(pending, expires_at=past)
    # The pending row it is compared against is the same expired deadline.
    check = _check(snapshot, pending, expires_at=past)
    assert check.code == CONTINUATION_EXPIRED


@pytest.mark.parametrize(
    "mutator",
    [
        lambda mcp: {**mcp, "schema": "other"},
        lambda mcp: {**mcp, "mode": "qwen_live"},
        lambda mcp: {**mcp, "user_key": "someone_else"},
        lambda mcp: {**mcp, "session_id": "web-other"},
        lambda mcp: {**mcp, "clarification_id": "clar-other"},
        lambda mcp: {**mcp, "current_query": "other"},
        lambda mcp: {**mcp, "question": ""},
        lambda mcp: {**mcp, "task": {}},
        lambda mcp: {**mcp, "task": {"task_id": "", "current_query": "q"}},
    ],
)
def test_inner_mcp_block_mismatches_are_rejected(mutator) -> None:
    pending = _pending()
    snapshot = _snapshot(pending)
    snapshot["mcp"] = mutator(snapshot["mcp"])
    check = _check(snapshot, pending)
    assert not check.ok
    assert check.payload is None


def test_missing_inner_block_is_rejected() -> None:
    pending = _pending()
    snapshot = _snapshot(pending)
    snapshot.pop("mcp")
    assert _check(snapshot, pending).code == CONTINUATION_MALFORMED


# --------------------------------------------------------------------------- #
# store binding + migration
# --------------------------------------------------------------------------- #


def _commit_with_continuation(store, *, expires_at: float, snapshot_overrides=None):
    mcp = _mcp_block(
        SimpleNamespace(clarification_id="clar-cold"), expires_at=expires_at,
        query=QUERY, session_id="web-c",
    )
    snapshot = build_web_continuation(
        mcp_continuation=mcp, query=QUERY, profile_id=PROFILE
    )
    snapshot.update(snapshot_overrides or {})
    store.create_session(session_id="web-c", profile_id=PROFILE)
    task = store.create_task("web-c", initial_requirements={"raw_query": QUERY})
    turn_id = store.allocate_turn_id("web-c")
    return store.commit_ask(
        "web-c", task.task_id, expected_task_version=task.task_version,
        clarification_id="clar-cold", source_turn_id=turn_id, query=QUERY,
        payload={"question": "?"}, expires_at=expires_at,
        continuation=snapshot,
    )


def test_store_stamps_the_web_binding_authoritatively(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = SQLiteWebStateStore(db_path)
    try:
        expires_at = time.time() + 600
        # A caller cannot smuggle a different session/task/generation into the row.
        pending = _commit_with_continuation(
            store, expires_at=expires_at,
            snapshot_overrides={
                "session_id": "web-somewhere-else",
                "web_task_id": "task-elsewhere",
                "pending_id": "pend-elsewhere",
                "reset_generation": 99,
            },
        )

        stored = pending.continuation
        assert stored is not None
        assert stored["session_id"] == "web-c"
        assert stored["web_task_id"] == pending.task_id
        assert stored["pending_id"] == pending.pending_id
        assert stored["clarification_id"] == "clar-cold"
        assert stored["profile_id"] == PROFILE
        assert stored["reset_generation"] == 0
        assert stored["expires_at"] == expires_at == pending.expires_at
        assert stored["mcp"]["task"]["task_id"] == USER_KEY

        # Every reader returns the same snapshot.
        assert store.get_pending(pending.pending_id).continuation == stored
        assert store.get_open_pending_for_session("web-c").continuation == stored
        assert store.get_pending_for_clarification("web-c", "clar-cold").continuation == stored
        # And the stored snapshot validates against its own row.
        check = validate_continuation(
            stored, session_id="web-c", profile_id=PROFILE, task_id=pending.task_id,
            pending_id=pending.pending_id, clarification_id="clar-cold",
            reset_generation=0, expires_at=pending.expires_at, query=QUERY,
            expected_user_key=USER_KEY,
        )
        assert check.ok
    finally:
        store.close()


def test_pending_without_a_snapshot_has_none(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        assert pending.continuation is None
        assert store.get_open_pending_for_session("web-r").continuation is None
    finally:
        store.close()


def _build_v5_database(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        for statement in (
            _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS + _SCHEMA_V3_STATEMENTS
            + _SCHEMA_V4_STATEMENTS + _SCHEMA_V5_STATEMENTS
        ):
            connection.execute(statement)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '5')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-v5', 'profile_0057', 'v5', 1.0, 2.0, 1, 1, 1, 0, 0)"
        )
        connection.execute(
            "INSERT INTO web_task(task_id, session_id, status, task_version, reset_generation, "
            "current_requirements_json, requirements_version, created_at, updated_at) "
            "VALUES ('task-v5', 'web-v5', 'WAITING_FOR_USER', 2, 0, '{}', 0, 1.0, 2.0)"
        )
        connection.execute(
            "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, turn_id, "
            "reply_turn_id, mode, payload_json, status, created_at, expires_at, consumed_at) "
            "VALUES ('pend-v5', 'task-v5', 'clar-v5', 'q', 'web-v5-t1', 'web-v5-r1', "
            "'cpu_live', '{\"question\": \"?\"}', 'OPEN', 1.0, 9999999999.0, NULL)"
        )
        connection.commit()
    finally:
        connection.close()


def test_v5_database_migrates_and_old_pendings_have_no_continuation(tmp_path: Path) -> None:
    db_path = tmp_path / "v5.sqlite3"
    _build_v5_database(db_path)
    store = SQLiteWebStateStore(db_path)
    try:
        assert WEB_STATE_SCHEMA_VERSION == 6
        assert store.schema_version == 6
        pending = store.get_open_pending("task-v5")
        assert pending is not None and pending.status == "OPEN"
        assert pending.continuation is None
        raw = sqlite3.connect(str(db_path))
        try:
            columns = {
                row[1] for row in raw.execute("PRAGMA table_info(web_pending)").fetchall()
            }
        finally:
            raw.close()
        assert "continuation_json" in columns
    finally:
        store.close()


def test_unparseable_continuation_is_data_corruption(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = SQLiteWebStateStore(db_path)
    try:
        store.create_session(session_id="web-bad", profile_id=PROFILE)
        task = store.create_task("web-bad", initial_requirements={"raw_query": "q"})
        turn_id = store.allocate_turn_id("web-bad")
        pending = store.commit_ask(
            "web-bad", task.task_id, expected_task_version=task.task_version,
            clarification_id="clar-bad", source_turn_id=turn_id, query="q",
            payload={"question": "?"}, expires_at=time.time() + 600,
        )
    finally:
        store.close()

    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute(
            "UPDATE web_pending SET continuation_json = 'not json' WHERE pending_id = ?",
            (pending.pending_id,),
        )
        raw.commit()
    finally:
        raw.close()

    reopened = SQLiteWebStateStore(db_path)
    try:
        with pytest.raises(StateDataCorruption):
            reopened.get_pending(pending.pending_id)
    finally:
        reopened.close()
