"""Durable external-execution stages and result checkpoints (S5b-P2a)."""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_web_request_store import REPLY, _new_store, _setup  # noqa: E402
from web_demo.state_store import (  # noqa: E402
    REPLY_RESULT_SCHEMA,
    SQLiteWebStateStore,
    RequestAmbiguous,
    RequestConflict,
    RequestFenced,
    StateStoreError,
)

RESULT = {"recommendation": {"items": [{"item_id": "B0TEST"}]}, "narrative": "ok"}


def _turn(pending) -> dict:
    return {"turn_id": pending.reply_turn_id, "user_text": "For gaming.",
            "assistant_text": "ok"}


def test_claim_persists_reply_payload_and_never_rewrites_it(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert claim.request.reply_payload == {
            "message": "For gaming.", "confirmed_by_user": True,
            "confirmation_source": "web_ui_explicit_user_reply",
        }
        # A duplicate claim cannot rewrite the original message.
        with pytest.raises(RequestConflict):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload={**REPLY, "message": "changed"}
            )
        assert store.get_request("web-r", pending.reply_turn_id).reply_payload == {
            "message": "For gaming.", "confirmed_by_user": True,
            "confirmation_source": "web_ui_explicit_user_reply",
        }
    finally:
        store.close()


def test_external_started_blocks_automatic_retry(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        started = store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        assert started.execution_stage == "EXTERNAL_STARTED"
        # Even after the lease expires, an ambiguous external call is never auto-retried.
        with pytest.raises(RequestAmbiguous):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload=REPLY, lease_duration=60, now=now + 60
            )
        assert store.list_turns("web-r") == ()
    finally:
        store.close()


def test_checkpoint_survives_restart_and_enables_mcp_free_finalize(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.sqlite3"
    store = SQLiteWebStateStore(db)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        checkpoint = store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            result=RESULT, now=now + 1,
        )
        assert checkpoint.execution_stage == "RESULT_DURABLE"
        assert checkpoint.result["payload"] == RESULT
    finally:
        store.close()

    reopened = SQLiteWebStateStore(db)
    try:
        # A new executor takes over a durable checkpoint without calling MCP again.
        takeover = reopened.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=60, now=now + 10
        )
        assert takeover.outcome == "result_ready"
        task, consumed, request = reopened.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=takeover.claim_token,
            outcome="completed", turn=_turn(pending), now=now + 11,
        )
        assert (task.status, consumed.status, request.status) == (
            "COMPLETED", "CONSUMED", "COMPLETED"
        )
        assert request.result["payload"] == RESULT
        assert reopened.list_turns("web-r")[-1].turn_id == pending.reply_turn_id
    finally:
        reopened.close()


def test_checkpoint_rejects_stale_token_conflicting_result_and_reset(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=30, now=now
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        with pytest.raises(RequestFenced):
            store.checkpoint_reply_result(
                "web-r", pending.reply_turn_id, claim_token="claim-bogus",
                result=RESULT, now=now + 1,
            )
        store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            result=RESULT, now=now + 1,
        )
        with pytest.raises(RequestConflict):
            store.checkpoint_reply_result(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                result={**RESULT, "narrative": "different"}, now=now + 2,
            )
        store.reset_session("web-r")
        with pytest.raises(RequestFenced):
            store.checkpoint_reply_result(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                result=RESULT, now=now + 3,
            )
    finally:
        store.close()


def test_checkpoint_failure_does_not_become_durable(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=30, now=now
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        raw = sqlite3.connect(store.path)
        try:
            raw.execute(
                "CREATE TRIGGER fail_checkpoint BEFORE UPDATE ON web_request "
                "BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises((StateStoreError, sqlite3.Error)):
            store.checkpoint_reply_result(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                result=RESULT, now=now + 1,
            )
        record = store.get_request("web-r", pending.reply_turn_id)
        assert record.execution_stage == "EXTERNAL_STARTED"
        assert record.result is None
    finally:
        store.close()


def test_ambiguous_mark_is_fail_closed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=30, now=now
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        record = store.mark_request_ambiguous(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now + 1
        )
        assert (record.status, record.execution_stage) == ("FAILED", "AMBIGUOUS")
        with pytest.raises(RequestAmbiguous):
            store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="completed", turn=_turn(pending),
            )
        assert store.list_turns("web-r") == ()
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
    finally:
        store.close()


def test_lease_keepalive_during_slow_call_and_old_executor_fenced(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, now=now
        )
        stop = threading.Event()

        def slow_call() -> None:
            current = now
            while not stop.is_set():
                time.sleep(0.05)
                current += 2
                store.renew_request_lease(
                    "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                    lease_duration=5, now=current,
                )

        thread = threading.Thread(target=slow_call)
        thread.start()
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=5)
        assert thread.is_alive() is False
        # Another executor cannot steal a kept-alive lease.
        with pytest.raises(RequestAmbiguous):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now + 1
            )
    finally:
        store.close()


def test_v4_database_migrates_to_v5_with_conservative_stage(tmp_path: Path) -> None:
    from web_demo.state_store import (
        _SCHEMA_V1_STATEMENTS,
        _SCHEMA_V2_STATEMENTS,
        _SCHEMA_V3_STATEMENTS,
        _SCHEMA_V4_STATEMENTS,
    )

    db = tmp_path / "v4.sqlite3"
    connection = sqlite3.connect(str(db))
    try:
        for statement in (
            _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS
            + _SCHEMA_V3_STATEMENTS + _SCHEMA_V4_STATEMENTS
        ):
            connection.execute(statement)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '4')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-v4', 'profile_0057', 'v4', 1.0, 2.0, 1, 1, 1, 0, 0)"
        )
        for rid, status in (("web-v4-r1", "COMPLETED"), ("web-v4-r2", "PROCESSING")):
            connection.execute(
                "INSERT INTO web_request(session_id, request_id, pending_id, task_id, "
                "request_kind, payload_fingerprint, status, claim_token, claim_attempt, "
                "lease_expires_at, task_version_at_claim, reset_generation_at_claim, "
                "result_json, created_at, updated_at) "
                "VALUES ('web-v4', ?, 'p', 't', 'clarification_reply', 'fp', ?, 'tok', 1, "
                "9999999999.0, 1, 0, '{\"schema\": \"agentrecx.web.reply_result.v1\", "
                "\"payload\": null}', 1.0, 2.0)",
                (rid, status),
            )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteWebStateStore(db)
    try:
        assert store.schema_version == 6
        completed = store.get_request("web-v4", "web-v4-r1")
        pending_req = store.get_request("web-v4", "web-v4-r2")
        assert completed.execution_stage == "COMPLETED"
        assert completed.result["schema"] == REPLY_RESULT_SCHEMA
        # Historical PROCESSING requests are not declared safe to retry.
        assert pending_req.execution_stage == "AMBIGUOUS"
    finally:
        store.close()
