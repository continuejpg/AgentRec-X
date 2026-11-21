"""Coordinator tests for fenced warm reply execution (S5b-P2b-B + M1-C Round B/C)."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_web_demo_unit import (  # noqa: E402
    info_payload,
    preferences_payload,
    recommend_payload,
    reply_payload,
)
from tests.test_web_request_store import REPLY, _new_store, _setup  # noqa: E402
from web_demo.handle_lifecycle import (  # noqa: E402
    HANDLE_MISSING,
    HANDLE_OK,
    HandleCheck,
)
from web_demo.reply_assembly import (  # noqa: E402
    ReplyAssemblyRequest,
    assemble_reply_presentation,
    reply_response_from_result,
)
from web_demo.reply_executor import (  # noqa: E402
    CPU_RESUME_RANKING_SOURCE,
    WarmReplyExecutor,
)
from web_demo.reply_result_contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    build_reply_payload,
)

PROFILE = "profile_0057"
CATALOGUE_SIZE = 76263
MESSAGE = "For gaming."


def _handle_ok(ref) -> HandleCheck:  # noqa: ARG001 - fake live child process
    return HandleCheck(ok=True, code=HANDLE_OK)


def _handle_missing(ref) -> HandleCheck:  # noqa: ARG001 - no recorded handle
    return HandleCheck(ok=False, code=HANDLE_MISSING, detail="no live MCP handle")


class FakeTools:
    """Real-shaped MCP payloads for the four calls a warm reply performs."""

    def __init__(self, *, on_call=None, recommend=None, reply=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.on_call = on_call
        self.fail: dict[str, Exception] = {}
        self.delay: dict[str, float] = {}
        self.recommend = recommend or recommend_payload()
        self.reply = reply or reply_payload()

    async def __call__(self, profile: str, tool: str, arguments: dict) -> dict:
        self.calls.append((tool, dict(arguments)))
        if tool in self.fail:
            raise self.fail[tool]
        if tool in self.delay:
            await asyncio.sleep(self.delay[tool])
        if self.on_call is not None:
            self.on_call(tool)
        if tool == "submit_clarification_reply":
            return self.reply
        if tool == "get_active_preferences":
            return preferences_payload()
        if tool == "recommend_products":
            return self.recommend
        if tool == "get_product_info":
            return info_payload(str(arguments["product_ref"]))
        raise AssertionError(f"unexpected tool {tool}")


def _assemble(request: ReplyAssemblyRequest) -> dict:
    return assemble_reply_presentation(request, catalogue_size=CATALOGUE_SIZE)


def _run(executor, session, pending, **kwargs):
    kwargs.setdefault("message", MESSAGE)
    return executor.execute(session, pending, **kwargs)


def _session():
    return SimpleNamespace(session_id="web-r", profile_id=PROFILE)


def _executor(store, tools, **kwargs):
    options = {
        "heartbeat_interval": 3600.0,
        "lease_duration": 30.0,
        "handle_check": _handle_ok,
        "assemble": _assemble,
    }
    options.update(kwargs)
    return WarmReplyExecutor(store, call_tool=tools, profile_id=PROFILE, **options)


def _payload(message: str = MESSAGE) -> dict:
    return {**REPLY, "message": message}


def _durable_result(pending) -> dict:
    """A complete durable result in the exact shape the coordinator checkpoints."""
    recommendation = recommend_payload()
    presentation = assemble_reply_presentation(
        ReplyAssemblyRequest(
            recommendation=recommendation,
            product_infos={recommendation["selected_product_id"]: info_payload()},
            reply=reply_payload(),
            preference_payload=preferences_payload(),
            query=pending.query,
            recommendation_id="rec-checkpoint",
            ranking_source=CPU_RESUME_RANKING_SOURCE,
            profile_id=PROFILE,
            turn_id=pending.reply_turn_id,
        ),
        catalogue_size=CATALOGUE_SIZE,
    )
    contract = build_reply_payload(
        reply_turn_id=pending.reply_turn_id,
        pending_id=pending.pending_id,
        user_message=MESSAGE,
        recommendation=recommendation,
        narrative=presentation["narrative"],
        recommendation_id=presentation["recommendation_id"],
        ranking_source=presentation["ranking_source"],
        products=presentation["products"],
    )
    return {
        "reply": contract,
        "preferences": presentation["preferences"],
        "memory_update": presentation["memory_update"],
        "steps": [],
    }


# --------------------------------------------------------------------------- #
# happy path / replay / durable reuse
# --------------------------------------------------------------------------- #


def test_happy_path_executes_real_call_order_and_finalizes(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "completed"
        assert [tool for tool, _ in tools.calls] == [
            "submit_clarification_reply",
            "get_active_preferences",
            "recommend_products",
            "get_product_info",
        ]
        assert tools.calls[0][1]["turn_id"] == pending.reply_turn_id
        assert tools.calls[0][1]["confirmation_source"] == "web_ui_explicit_user_reply"
        assert tools.calls[3][1]["product_ref"] == "B004K6R43G"

        assert outcome.pending.status == "CONSUMED"
        assert outcome.task.status == "COMPLETED"
        assert outcome.request.status == "COMPLETED"
        assert outcome.request.execution_stage == "COMPLETED"

        result = outcome.result
        assert set(result) == {
            "reply", "preferences", "memory_update", "steps", "task_requirements",
        }
        reply = result["reply"]
        assert reply["schema"] == CONTRACT_SCHEMA
        assert reply["reply_turn_id"] == pending.reply_turn_id
        assert reply["pending_id"] == pending.pending_id
        assert reply["assistant_text"].startswith("Memory update (profile_0057)")
        assert reply["recommendation"]["ranking_source"] == CPU_RESUME_RANKING_SOURCE
        assert reply["recommendation"]["id"].startswith("rec-")
        items = reply["products"]["items"]
        assert [item["product_id"] for item in items] == ["B004K6R43G"]
        assert reply["products"]["recommendation_id"] == reply["recommendation"]["id"]
        assert reply["products"]["request_id"] == pending.reply_turn_id
        assert result["memory_update"]["added"] == 2
        assert [step["tool"] for step in result["steps"]] == [
            "submit_clarification_reply", "get_active_preferences",
            "recommend_products", "get_product_info",
        ]

        turns = store.list_turns("web-r")
        assert [row.turn_id for row in turns] == [pending.reply_turn_id]
        assert turns[0].payload["turn"]["products"]["items"][0]["product_id"] == "B004K6R43G"
        assert store.get_session("web-r").reply_counter == 1
    finally:
        store.close()


def test_replay_returns_saved_result_without_any_mcp_call(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
        first = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert first.outcome == "completed"
        calls_after_first = len(tools.calls)

        replay_tools = FakeTools()
        second = asyncio.run(_run(
            _executor(store, replay_tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert second.outcome == "replayed"
        assert replay_tools.calls == []
        assert second.result == first.result
        assert second.turn["turn_id"] == pending.reply_turn_id
        assert len(store.list_turns("web-r")) == 1
        assert len(tools.calls) == calls_after_first
    finally:
        store.close()


def test_result_ready_reuses_checkpoint_without_mcp(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=_payload())
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token
        )
        store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            result=_durable_result(pending),
        )
        raw = sqlite3.connect(store.path)
        try:
            raw.execute("UPDATE web_request SET lease_expires_at = 0")
            raw.commit()
        finally:
            raw.close()
        tools = FakeTools()
        outcome = asyncio.run(_run(
            _executor(store, tools, handle_check=_handle_missing), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "result_reused"
        assert tools.calls == []
        assert outcome.pending.status == "CONSUMED"
        assert store.list_turns("web-r")[-1].turn_id == pending.reply_turn_id
        reply = outcome.result["reply"]
        assert reply["recommendation"]["id"] == "rec-checkpoint"
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# fail closed: no handle / busy / conflict / non-final / timeout / checkpoint
# --------------------------------------------------------------------------- #


def test_no_live_handle_never_claims_or_calls_mcp(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
        outcome = asyncio.run(_run(
            _executor(store, tools, handle_check=_handle_missing),
            _session(), pending, clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "warm_unavailable"
        assert outcome.handle is not None and outcome.handle.code == HANDLE_MISSING
        assert tools.calls == []
        assert store.get_request("web-r", pending.reply_turn_id) is None
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.list_turns("web-r") == ()
    finally:
        store.close()


def test_busy_and_conflict_do_not_execute_mcp(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        store.claim_reply_request("web-r", pending.pending_id, payload=_payload())
        tools = FakeTools()
        busy = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert busy.outcome == "busy"
        assert tools.calls == []

        conflicting = FakeTools()
        outcome = asyncio.run(_executor(store, conflicting).execute(
            _session(), pending, message="Something else entirely.",
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "conflict"
        assert conflicting.calls == []
    finally:
        store.close()


def test_non_final_resume_is_ambiguous_and_never_finalized(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools(recommend=recommend_payload(status="awaiting_clarification"))
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "ambiguous"
        # The M9 write really happened; it is reported honestly and never retried.
        assert [tool for tool, _ in tools.calls][:1] == ["submit_clarification_reply"]
        assert store.list_turns("web-r") == ()
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request.execution_stage == "AMBIGUOUS"
        assert request.status == "FAILED"
        assert store.get_open_pending_for_session("web-r").status == "OPEN"

        retry_tools = FakeTools()
        retry = asyncio.run(_run(
            _executor(store, retry_tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert retry.outcome == "ambiguous"
        assert retry_tools.calls == []
    finally:
        store.close()


def test_timeout_is_ambiguous_and_never_finalized(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
        tools.delay["submit_clarification_reply"] = 0.5
        outcome = asyncio.run(_run(
            _executor(store, tools, tool_timeout=0.05), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "ambiguous"
        assert store.list_turns("web-r") == ()
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request.execution_stage in {"EXTERNAL_STARTED", "AMBIGUOUS"}
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        retry = asyncio.run(_run(
            _executor(store, FakeTools()), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert retry.outcome == "ambiguous"
    finally:
        store.close()


def test_checkpoint_failure_does_not_finalize(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)

        def install_trigger(tool: str) -> None:
            if tool == "submit_clarification_reply":
                raw = sqlite3.connect(store.path)
                try:
                    raw.execute(
                        "CREATE TRIGGER fail_ckpt BEFORE UPDATE ON web_request "
                        "BEGIN SELECT RAISE(ABORT, 'injected'); END"
                    )
                    raw.commit()
                finally:
                    raw.close()

        tools = FakeTools(on_call=install_trigger)
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "checkpoint_failed"
        assert len(tools.calls) == 4  # reply + prefs + recommendation + product info
        assert store.list_turns("web-r") == ()
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.get_request("web-r", pending.reply_turn_id).status != "COMPLETED"
    finally:
        store.close()


def test_reset_during_execution_fences_finalize(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)

        def reset(tool: str) -> None:
            if tool == "recommend_products":
                store.reset_session("web-r")

        tools = FakeTools(on_call=reset)
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "fenced"
        assert store.list_turns("web-r") == ()
        assert store.get_open_pending_for_session("web-r") is None
    finally:
        store.close()


def test_completed_request_replays_even_after_handle_loss(tmp_path: Path) -> None:
    """A committed reply is durable: replaying it needs no live handle and no MCP."""
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        first = asyncio.run(_run(
            _executor(store, FakeTools()), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert first.outcome == "completed"
        replay_tools = FakeTools()
        replay = asyncio.run(_run(
            _executor(store, replay_tools, handle_check=_handle_missing),
            _session(), pending, clarification_id=pending.clarification_id,
        ))
        assert replay.outcome == "replayed"
        assert replay_tools.calls == []
        assert replay.result == first.result
    finally:
        store.close()


def test_expired_lease_fences_finalize_without_a_turn(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
        tools.delay["recommend_products"] = 0.15
        outcome = asyncio.run(_run(
            _executor(store, tools, lease_duration=0.05, heartbeat_interval=3600.0),
            _session(), pending, clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "fenced"
        assert store.list_turns("web-r") == ()
        request = store.get_request("web-r", pending.reply_turn_id)
        assert request is not None
        assert request.status == "PROCESSING"
        assert request.execution_stage == "EXTERNAL_STARTED"
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
    finally:
        store.close()


def test_concurrent_executors_have_exactly_one_winner(tmp_path: Path) -> None:
    import threading

    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        tools = FakeTools()
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
        } <= {"busy", "ambiguous"}
        submits = [call for call in tools.calls if call[0] == "submit_clarification_reply"]
        assert len(submits) == 1, "only the winning executor may perform the M9 write"
        assert len(store.list_turns("web-r")) == 1
        assert store.get_open_pending_for_session("web-r") is None
    finally:
        store.close()


def test_known_budget_refusal_is_durable_without_a_product(tmp_path: Path) -> None:
    """A real budget refusal is a known outcome: durable, replayable, and never a product."""
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        requirements = {
            "schema": "agentrecx.task_requirements.v1",
            "budget": {"max_amount": 8000.0, "min_amount": None, "currency": "CNY",
                       "source_text": "游戏，预算改成 8000 元"},
            "category": "laptop",
            "use_cases": ["gaming"],
            "source_text": ["推荐一台 5000 元的笔记本", "游戏，预算改成 8000 元"],
        }
        reply = {**reply_payload(), "task_requirements": requirements}
        recommendation = {
            "schema": "agentrecx.p17.recommend_products.v1",
            "status": "budget_unverifiable",
            "selected_product_id": None,
            "selected_product": None,
            "gate": {
                "action": "ALLOW_FINALIZE",
                "allow_finalize": False,
                "reason_code": "budget_unverifiable",
                "reason": "the catalogue provides no price for this candidate",
            },
            "budget_verification": {
                "budget": requirements["budget"],
                "label": "<= 8000 CNY",
                "candidates_checked": 3,
                "within": 0,
                "over": 0,
                "unknown": 3,
                "currency_mismatch": 0,
                "verified": False,
            },
            "error": None,
        }
        tools = FakeTools(recommend=recommendation, reply=reply)
        outcome = asyncio.run(_run(
            _executor(store, tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "completed"
        assert [tool for tool, _ in tools.calls] == [
            "submit_clarification_reply",
            "get_active_preferences",
            "recommend_products",
        ], "a refusal must not fetch product details or assemble a product"
        assert tools.calls[2][1]["task_requirements"] == requirements

        result = outcome.result
        assert result["refusal"]["code"] == "budget_unverifiable"
        assert result["reply"]["products"]["items"] == []
        assert result["reply"]["recommendation"]["id"] is None
        assert result["task_requirements"] == requirements
        response = reply_response_from_result(
            result, clarification_id=pending.clarification_id
        )
        assert response["recommendation"] is None
        assert response["refusal"]["code"] == "budget_unverifiable"
        assert "No product recommendation" in response["message"]
        assert "8000 CNY" in response["message"]
        assert "price UNKNOWN" in response["message"]
        assert outcome.pending.status == "CONSUMED"

        replay_tools = FakeTools()
        replay = asyncio.run(_run(
            _executor(store, replay_tools), _session(), pending,
            clarification_id=pending.clarification_id,
        ))
        assert replay.outcome == "replayed"
        assert replay_tools.calls == []
        assert replay.result == result
    finally:
        store.close()
