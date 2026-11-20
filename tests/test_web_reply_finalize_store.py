"""Atomic reply Turn + result persistence through finalize_reply_request (S5b-P1)."""

from __future__ import annotations

import sqlite3
import sys
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
    RequestConflict,
    RequestFenced,
    StateDataCorruption,
    StateStoreError,
)

RESULT = {"recommendation": {"items": [{"item_id": "B0TEST", "title": "Headphones"}]},
          "narrative": "Here is one match."}


def _turn(pending, text: str = "For gaming.") -> dict:
    return {
        "turn_id": pending.reply_turn_id,
        "user_text": text,
        "assistant_text": "Here is one match.",
        "products": {"items": [{"item_id": "B0TEST", "title": "Headphones"}]},
    }


def _trigger(db_path: str, table: str, event: str) -> None:
    raw = sqlite3.connect(db_path)
    try:
        raw.execute(
            f"CREATE TRIGGER fail_{table}_{event} BEFORE {event} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        raw.commit()
    finally:
        raw.close()


def test_finalize_persists_turn_result_and_consumes_everything(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        ledger_before = store.get_session("web-r").reply_counter
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        task, consumed, request = store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            outcome="completed", turn=_turn(pending), result=RESULT,
        )
        assert (task.status, task.task_version) == ("COMPLETED", 3)
        assert consumed.status == "CONSUMED"
        assert request.status == "COMPLETED"
        assert request.result["schema"] == REPLY_RESULT_SCHEMA
        assert request.result["turn_id"] == pending.reply_turn_id
        assert request.result["payload"] == RESULT

        turns = store.list_turns("web-r")
        assert [row.turn_id for row in turns] == [pending.reply_turn_id]
        assert turns[0].payload["turn"]["assistant_text"] == "Here is one match."
        assert store.get_session("web-r").reply_counter == ledger_before
    finally:
        store.close()


def test_reply_turn_sequence_does_not_collide_with_next_chat_turn(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            outcome="resumed", turn=_turn(pending),
        )
        next_turn = store.allocate_turn_id("web-r")
        store.append_turn("web-r", turn={"turn_id": next_turn, "user_text": "next"})
        assert [row.turn_id for row in store.list_turns("web-r")] == [pending.reply_turn_id, next_turn]
    finally:
        store.close()


def test_reply_turn_and_result_survive_reopen(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    store = SQLiteWebStateStore(db)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            outcome="completed", turn=_turn(pending), result=RESULT,
        )
    finally:
        store.close()

    reopened = SQLiteWebStateStore(db)
    try:
        assert reopened.list_turns("web-r")[-1].turn_id == pending.reply_turn_id
        replay = reopened.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert replay.outcome == "replay"
        assert replay.request.result["payload"] == RESULT
        assert len(reopened.list_turns("web-r")) == 1
    finally:
        reopened.close()


def test_conflict_stale_token_and_expired_lease_leave_no_turn(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        with pytest.raises(RequestConflict):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload={**REPLY, "message": "other"}
            )
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token="claim-bogus",
                outcome="completed", turn=_turn(pending), now=now + 1,
            )
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="completed", turn=_turn(pending), now=now + 10,
            )
        assert store.list_turns("web-r") == ()
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.get_active_task("web-r").status == "WAITING_FOR_USER"
    finally:
        store.close()


def test_turn_insert_failure_rolls_back_task_pending_request(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        _trigger(store.path, "web_turn", "INSERT")
        with pytest.raises((StateStoreError, sqlite3.Error)):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="completed", turn=_turn(pending), result=RESULT,
            )
        assert store.list_turns("web-r") == ()
        assert store.get_active_task("web-r").status == "WAITING_FOR_USER"
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
    finally:
        store.close()


def test_request_update_failure_rolls_back_turn_and_pending(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        _trigger(store.path, "web_request", "UPDATE")
        with pytest.raises((StateStoreError, sqlite3.Error)):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="completed", turn=_turn(pending), result=RESULT,
            )
        assert store.list_turns("web-r") == ()
        assert store.get_active_task("web-r").status == "WAITING_FOR_USER"
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
    finally:
        store.close()


def test_reset_fences_finalize_and_keeps_history_clean(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.reset_session("web-r")
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="completed", turn=_turn(pending),
            )
        assert store.list_turns("web-r") == ()
        assert store.get_open_pending_for_session("web-r") is None
    finally:
        store.close()


def test_corrupt_result_json_is_reported_not_replaced(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            outcome="completed", turn=_turn(pending), result=RESULT,
        )
        raw = sqlite3.connect(store.path)
        try:
            raw.execute(
                "UPDATE web_request SET result_json = '{\"broken\": true}' "
                "WHERE session_id = 'web-r'"
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises(StateDataCorruption):
            store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
    finally:
        store.close()


def test_s5a_finalize_without_turn_still_supported(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        task, consumed, request = store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="resumed"
        )
        assert (task.status, consumed.status, request.status) == (
            "RUNNING", "CONSUMED", "COMPLETED"
        )
        assert request.result["schema"] == REPLY_RESULT_SCHEMA
        assert request.result["turn_id"] is None
        assert store.list_turns("web-r") == ()
    finally:
        store.close()
