"""M4 unit/regression tests for durable multi-round ASK (CPU coordinator + SQLite)."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.preference.clarification_policy import (  # noqa: E402
    next_laptop_clarification,
)
from recommendation.preference.task_requirements import (  # noqa: E402
    extract_task_requirements,
)
from tests.test_web_demo_unit import (  # noqa: E402
    preferences_payload,
    reply_payload,
)
from tests.test_web_reply_executor import (  # noqa: E402
    MESSAGE,
    FakeTools,
    _executor,
    _run,
    _session,
)
from tests.test_web_request_store import REPLY, _new_store, _setup  # noqa: E402
from web_demo.continuation import (  # noqa: E402
    MCP_CONTINUATION_SCHEMA,
    CONTINUATION_VERSION,
)
from web_demo.reply_result_contract import build_clarification_payload  # noqa: E402
from web_demo.state_store import (  # noqa: E402
    PendingStateConflict,
    RequestFenced,
    TaskStateConflict,
)

from web_demo.viewmodels import clarification_payload  # noqa: E402

ASK_TWO_ID = "clar_ask_two"


def _requirements(*, budget: float | None = None, use_cases=("gaming",)) -> dict:
    return {
        "schema": "agentrecx.task_requirements.v1",
        "budget": None if budget is None else {
            "max_amount": budget, "min_amount": None, "currency": "CNY",
            "source_text": "8000 元",
        },
        "category": "laptop",
        "use_cases": list(use_cases),
        "source_text": ["q"],
    }


def _continuation(pending, *, session: str = "web-r", ask_id: str = ASK_TWO_ID) -> dict:
    now = time.time()
    return {
        "schema": MCP_CONTINUATION_SCHEMA,
        "version": CONTINUATION_VERSION,
        "mode": "cpu_live",
        "user_key": "p14e2_lite_0057",
        "session_id": session,
        "clarification_id": ask_id,
        "question": "您的预算是多少？",
        "reason": "missing_laptop_budget",
        "current_query": str(pending.query),
        "task": {
            "task_id": "task-canonical",
            "user_id": "p14e2_lite_0057",
            "current_query": str(pending.query),
            "catalog_namespace": "amazon_electronics_c4",
        },
        "task_constraints": {"hard_exclusions": [], "soft_preferences": [], "source_turns": []},
        "task_requirements": _requirements(),
        "created_at": now,
        "expires_at": now + 600.0,
    }


def _ask_two_payload(pending, *, ask_id: str = ASK_TWO_ID) -> dict:
    raw = {
        "clarification_id": ask_id,
        "question": "您的预算是多少？",
        "reason": "missing_laptop_budget",
        "expires_in_seconds": 600.0,
        "memory_write_enabled": True,
        "continuation": _continuation(pending, ask_id=ask_id),
    }
    return {
        "schema": "agentrecx.p17.recommend_products.v1",
        "status": "awaiting_clarification",
        "effective_query": "q laptop gaming",
        "selected_product_id": None,
        "selected_product": None,
        "gate": {
            "action": "NEED_USER_CLARIFICATION",
            "allow_finalize": False,
            "reason_code": "missing_laptop_budget",
            "reason": "one task requirement is still missing",
            "ask_question": "您的预算是多少？",
        },
        "clarification": raw,
    }


def _reply_with_requirements() -> dict:
    return {**reply_payload(), "task_requirements": _requirements()}


def _durable_clarification_result(pending) -> dict:
    ask = _ask_two_payload(pending)["clarification"]
    view = clarification_payload(ask)
    view["memory_write_enabled"] = True
    continuation = _continuation(pending)
    reply = build_clarification_payload(
        reply_turn_id=pending.reply_turn_id,
        pending_id=pending.pending_id,
        user_message=MESSAGE,
        question="您的预算是多少？",
        reason="missing_laptop_budget",
        ranking_source="cpu_live:clarification_resume+recommend_products+gate",
        query=pending.query,
        task_requirements=_requirements(),
    )
    return {
        "reply": reply,
        "preferences": None,
        "memory_update": None,
        "steps": [],
        "task_requirements": _requirements(),
        "clarification": view,
        "clarification_continuation": {
            "schema": "agentrecx.web.clarification_continuation.v1",
            "version": 1,
            "mode": "cpu_live",
            "profile_id": "profile_0057",
            "session_id": "web-r",
            "web_task_id": "task-canonical",
            "pending_id": "",
            "clarification_id": ASK_TWO_ID,
            "reset_generation": 0,
            "query": str(pending.query),
            "question": "您的预算是多少？",
            "reason": "missing_laptop_budget",
            "expires_at": time.time() + 600.0,
            "mcp": continuation,
        },
        "clarification_expires_at": time.time() + 600.0,
        "clarification_mode": "cpu_live",
    }


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #


def test_laptop_policy_asks_use_case_then_budget_then_stops() -> None:
    first = next_laptop_clarification(extract_task_requirements("推荐一台笔记本"), "推荐一台笔记本")
    assert first is not None and first.slot == "use_case"
    second = next_laptop_clarification(
        extract_task_requirements("推荐一台笔记本").merged(extract_task_requirements("游戏")),
        "推荐一台笔记本",
    )
    assert second is not None and second.slot == "budget"
    ready = extract_task_requirements("推荐一台笔记本").merged(
        extract_task_requirements("游戏，8000 元")
    )
    assert next_laptop_clarification(ready, "推荐一台笔记本") is None


def test_policy_does_not_change_headphones_or_unknown_categories() -> None:
    for text in ("I need some new headphones.", "推荐一款耳机", "推荐一台平板"):
        requirements = extract_task_requirements(text)
        assert next_laptop_clarification(requirements, text) is None


# --------------------------------------------------------------------------- #
# executor + store
# --------------------------------------------------------------------------- #


def test_reply_that_asks_again_commits_next_pending_atomically(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools(
            recommend=_ask_two_payload(pending),
            reply=_reply_with_requirements(),
        )
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "completed", outcome.detail
        assert outcome.next_pending is not None
        assert outcome.next_pending.status == "OPEN"
        assert outcome.next_pending.clarification_id == ASK_TWO_ID
        assert outcome.next_pending.turn_id == pending.reply_turn_id
        assert outcome.next_pending.pending_id != pending.pending_id
        assert outcome.next_pending.reply_turn_id != pending.reply_turn_id
        assert outcome.result["clarification"]["clarification_id"] == ASK_TWO_ID
        stored_next = store.get_pending(outcome.next_pending.pending_id)
        assert stored_next is not None and stored_next.continuation is not None
        assert stored_next.continuation["pending_id"] == outcome.next_pending.pending_id
        assert [tool for tool, _ in tools.calls] == [
            "submit_clarification_reply",
            "get_active_preferences",
            "recommend_products",
        ]

        consumed = store.get_pending(pending.pending_id)
        assert consumed is not None and consumed.status == "CONSUMED"
        task = store.get_task(pending.task_id)
        assert task is not None and task.status == "WAITING_FOR_USER"
        assert task.task_version > 1
        assert task.current_requirements["use_cases"] == ["gaming"]
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request is not None and request.status == "COMPLETED"
        assert request.result["outcome"] == "awaiting_clarification"
        assert request.result["next_pending_id"] == outcome.next_pending.pending_id
        assert [row.turn_id for row in store.list_turns("web-r")] == [
            pending.reply_turn_id,
        ]
        assert len(store.list_pending(pending.task_id)) == 2
    finally:
        store.close()


def test_replayed_reply_returns_the_same_next_ask_without_mcp(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        first = asyncio.run(_run(
            _executor(store, FakeTools(
                recommend=_ask_two_payload(pending),
                reply=_reply_with_requirements(),
            )),
            _session(), pending, clarification_id=pending.clarification_id,
        ))
        assert first.outcome == "completed"
        replay_tools = FakeTools()
        replay = asyncio.run(_run(
            _executor(store, replay_tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert replay.outcome == "replayed"
        assert replay_tools.calls == []
        assert replay.result == first.result
        assert len(store.list_pending(pending.task_id)) == 2
    finally:
        store.close()


def test_checkpointed_next_ask_is_finalized_without_mcp(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=0.05,
        )
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
        )
        result = _durable_clarification_result(pending)
        store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            result=result, now=time.time() - 10.0,
        )
        time.sleep(0.05)
        tools = FakeTools()
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "result_reused", outcome.detail
        assert tools.calls == []
        assert outcome.next_pending is not None
        assert outcome.next_pending.clarification_id == ASK_TWO_ID
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request is not None and request.status == "COMPLETED"
    finally:
        store.close()


def test_finalize_with_ask_is_one_transaction_and_fences_reset(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
        )
        turn = {
            "turn_id": pending.reply_turn_id, "user_text": MESSAGE,
            "assistant_text": "您的预算是多少？", "products": {"items": []},
        }
        store.reset_session("web-r")
        with pytest.raises((RequestFenced, TaskStateConflict)):
            store.finalize_reply_with_clarification(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                requirements=_requirements(), turn=turn,
                ranking_source="test", result={"reply": {}},
                next_clarification_id=ASK_TWO_ID,
                next_payload={"clarification_id": ASK_TWO_ID, "question": "?"},
                next_continuation=_continuation(pending),
                next_expires_at=time.time() + 600.0,
            )
        assert store.get_open_pending_for_session("web-r") is None
        assert len(store.list_turns("web-r")) == 0
    finally:
        store.close()


def test_invalid_next_ask_rolls_back_without_consuming_the_reply(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
        )
        turn = {
            "turn_id": pending.reply_turn_id, "user_text": MESSAGE,
            "assistant_text": "您的预算是多少？", "products": {"items": []},
        }
        with pytest.raises(ValueError):
            store.finalize_reply_with_clarification(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                requirements=_requirements(), turn=turn,
                ranking_source="test", result={"reply": {}},
                next_clarification_id=pending.clarification_id,  # reused: invalid
                next_payload={"clarification_id": pending.clarification_id, "question": "?"},
                next_continuation=_continuation(pending),
                next_expires_at=time.time() + 600.0,
            )
        old = store.get_pending(pending.pending_id)
        assert old is not None and old.status == "OPEN"
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
        assert store.list_turns("web-r") == ()
        assert len(store.list_pending(pending.task_id)) == 1
    finally:
        store.close()


def test_reset_cancels_next_ask_and_its_reply(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        first = asyncio.run(_run(
            _executor(store, FakeTools(
                recommend=_ask_two_payload(pending),
                reply=_reply_with_requirements(),
            )),
            _session(), pending, clarification_id=pending.clarification_id,
        ))
        assert first.next_pending is not None
        ask_two = first.next_pending
        store.reset_session("web-r")
        assert store.get_pending(pending.pending_id).status == "CONSUMED"
        assert store.get_pending(ask_two.pending_id).status == "CANCELLED"
        with pytest.raises((PendingStateConflict, TaskStateConflict)):
            store.claim_reply_request(
                "web-r", ask_two.pending_id, payload=REPLY,
            )
        assert store.get_request("web-r", ask_two.reply_turn_id) is None
    finally:
        store.close()


def test_external_failure_stays_ambiguous_and_creates_no_next_ask(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools(
            recommend=_ask_two_payload(pending),
            reply=_reply_with_requirements(),
        )
        tools.fail["submit_clarification_reply"] = RuntimeError("child died")
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "ambiguous"
        old = store.get_pending(pending.pending_id)
        assert old is not None and old.status == "OPEN"
        assert len(store.list_pending(pending.task_id)) == 1
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request is not None
        assert (request.status, request.execution_stage) == ("FAILED", "AMBIGUOUS")
    finally:
        store.close()


def test_concurrent_replies_have_one_executor_and_one_next_ask(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools(
            recommend=_ask_two_payload(pending),
            reply=_reply_with_requirements(),
        )
        tools.delay["submit_clarification_reply"] = 0.2
        barrier = threading.Barrier(2)
        outcomes = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=30)
                outcomes.append(asyncio.run(_run(
                    _executor(store, tools), _session(), pending,
                    clarification_id=pending.clarification_id,
                )))
            except BaseException as exc:  # noqa: BLE001 - collected and asserted
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert errors == []
        assert len(outcomes) == 2
        assert [item.outcome for item in outcomes].count("completed") == 1
        assert {
            item.outcome for item in outcomes if item.outcome != "completed"
        } <= {"busy", "ambiguous", "replayed"}
        submits = [call for call in tools.calls
                   if call[0] == "submit_clarification_reply"]
        assert len(submits) == 1, "only the winner may perform the M9 write"
        assert len(store.list_turns("web-r")) == 1
        open_pending = store.get_open_pending_for_session("web-r")
        assert open_pending is not None
        assert open_pending.clarification_id == ASK_TWO_ID
        assert len(store.list_pending(pending.task_id)) == 2
    finally:
        store.close()
