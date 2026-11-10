"""BFF Qwen-Live flow contract tests (fake MCP manager, no model)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from web_demo import demo_flow
from web_demo.config import Settings
from web_demo.sessions import PendingClarification, Session

PRODUCT_ID = "B000TEST01"
INFO = {
    "status": "ok",
    "product_id": PRODUCT_ID,
    "selected_product": {
        "item_id": PRODUCT_ID,
        "namespaced_id": f"amazon_electronics_c4:{PRODUCT_ID}",
        "title": "Wireless earbuds test record",
        "product_type": "earbuds",
        "form_factor": "in_ear",
    },
    "product_info": {"title": "Wireless earbuds test record", "price": None},
    "product_evidence": [{
        "evidence_id": "pev_01",
        "raw_field": "title",
        "span_start": 0,
        "span_end": 8,
        "span_text": "Wireless",
        "raw_excerpt": "Wireless earbuds test record",
        "attribute": "title",
        "verification_status": "verified_span",
        "source_record": "data/test.jsonl",
    }],
    "unknown_fields": [{"attribute": "price", "reason": "not in catalogue"}],
}
GATE = {
    "action": "ALLOW_FINALIZE", "reason_code": "evidence_sufficient",
    "allow_finalize": True, "selected_candidate_id": PRODUCT_ID,
    "unknown_fields": ["price"], "product_role": "primary",
}
PLANNER = {
    "kind": "qwen_base", "model": "Qwen3.5-9B-Base", "calls": 2,
    "actions": [
        {"step": 1, "action": "RETRIEVE", "arguments": {"query": "earbuds"}, "rationale": "need candidates"},
        {"step": 2, "action": "FINALIZE", "arguments": {}, "rationale": "ready"},
    ],
    "latency_ms": [111.5, 222.5], "input_tokens": 100, "output_tokens": 20,
    "parser_errors": [],
}


def _qwen_result(status: str = "finalized") -> dict[str, Any]:
    return {
        "schema": "agentrecx.p17.run_agent_turn.v1",
        "status": status,
        "query": "I need earbuds for working out",
        "planner": PLANNER,
        "selected_product_id": PRODUCT_ID if status == "finalized" else None,
        "candidate_ids": [PRODUCT_ID] if status == "finalized" else [],
        "gate": GATE if status == "finalized" else {"action": "NEED_USER_CLARIFICATION", "reason_code": "ambiguity"},
        "evidence_bundle": None,
        "planner_steps": {"steps": [], "final_ids": [PRODUCT_ID]},
        "clarification": (
            {"clarification_id": "qclar_test", "question": "Do you prefer over-ear or in-ear?",
             "reason": "qwen_planner_ask", "expires_in_seconds": 1800, "memory_write_enabled": True}
            if status == "awaiting_clarification" else None
        ),
        "search": {"search_mode": "bm25", "score_kind": "bm25", "catalogue_size": 76263},
        "rag": {"status": "validated", "final_answer": "A validated P15 answer.", "bundle": {}, "claim_validation": {}},
        "error": None,
        "latency_ms": 444.0,
    }


class _Health:
    def mark_unavailable(self, detail: str) -> None:
        self.unavailable = detail


class _Manager:
    def __init__(self) -> None:
        self.settings = Settings(search_mode="bm25", qwen_live_enabled=True, qwen_model_endpoint="http://127.0.0.1:9")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.health = _Health()

    def ensure_ready(self, profile_id: str):  # noqa: ANN001
        class _Ready:
            writes_enabled = True
        return _Ready()

    def catalogue_size(self) -> int:
        return 76263

    async def call_tool(self, profile_id: str, tool: str, arguments: dict[str, Any] | None = None,
                        **kwargs: Any) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append((tool, args))
        callback = kwargs.get("progress_callback")
        if tool == "run_agent_turn":
            if callback is not None:
                await callback(10, 100, json.dumps({"tool": "qwen_planner", "label": "Qwen Planner Request", "status": "running", "detail": "Qwen3.5-9B-Base; step 1"}))
                await callback(20, 100, json.dumps({"tool": "qwen_planner", "label": "Qwen Planner Request", "status": "ok", "detail": "Qwen3.5-9B-Base; step 1", "latency_ms": 111.5}))
                await callback(30, 100, json.dumps({"tool": "planner_action", "label": "Planner Action: RETRIEVE", "status": "ok", "detail": "step 1"}))
                await callback(40, 100, json.dumps({"tool": "retrieve_1", "label": "RETRIEVE", "status": "ok", "detail": "20 candidates", "latency_ms": 2.0}))
            return _qwen_result()
        if tool == "get_product_info":
            return INFO
        if tool == "submit_clarification_reply":
            return {
                "update": {"added": 2, "superseded": 0, "removed": 0, "skipped_duplicates": 0,
                           "already_processed": False},
                "confirmation_source": "web_ui_explicit_user_reply",
                "active_preferences": [{"memory_id": "mem_1", "value": "wireless", "polarity": "positive",
                                        "kind": "feature", "status": "active", "source_text": "wireless"}],
                "inactive_preferences": [],
            }
        if tool == "get_active_preferences":
            return {"authorized_user_key": "p14e2_lite_0057", "history_status": "ok",
                    "active_preferences": [{"memory_id": "mem_1", "value": "wireless", "polarity": "positive",
                                            "kind": "feature", "status": "active", "source_text": "wireless"}],
                    "inactive_preferences": []}
        if tool == "recommend_products":
            raise AssertionError("Qwen resume must not call the Rule recommendation tool")
        raise AssertionError(f"unexpected tool {tool}")


def _session() -> Session:
    return Session(session_id="web-test", profile_id="profile_0049")


def test_qwen_live_stream_renders_real_progress_products_evidence_and_planner_metadata():
    manager = _Manager()
    session = _session()
    settings = manager.settings
    events: list[dict[str, Any]] = []
    health = _Health()

    async def collect() -> None:
        async for item in demo_flow.chat_events(
            manager=manager, session=session, settings=settings,
            message="I need earbuds for working out", mode="qwen_live", qwen_health=health,
        ):
            events.append(item)

    asyncio.run(collect())
    names = [item["event"] for item in events]
    assert "products" in names and "evidence" in names and "text" in names and "done" in names
    tools = [item["data"]["tool"] for item in events if item["event"] == "tool_status"]
    assert "qwen_planner" in tools and "planner_action" in tools and "retrieve_1" in tools
    done = next(item["data"] for item in events if item["event"] == "done")
    assert done["planner"] == "qwen_base"
    assert done["planner_calls"] == 2
    assert done["planner_latency_ms"] == 334.0
    assert session.turns[-1].planner == "qwen_base"
    assert session.turns[-1].planner_model == "Qwen3.5-9B-Base"
    assert session.turns[-1].view()["planner_calls"] == 2


def test_qwen_ask_creates_a_qwen_pending_clarification_without_fake_finalize():
    manager = _Manager()
    session = _session()
    settings = manager.settings

    async def fake_call(profile_id: str, tool: str, arguments: dict[str, Any] | None = None, **kwargs: Any):
        del profile_id, arguments, kwargs
        assert tool == "run_agent_turn"
        return _qwen_result("awaiting_clarification")

    manager.call_tool = fake_call  # type: ignore[method-assign]
    events: list[dict[str, Any]] = []

    async def collect() -> None:
        async for item in demo_flow.chat_events(
            manager=manager, session=session, settings=settings,
            message="I need earbuds", mode="qwen_live", qwen_health=_Health(),
        ):
            events.append(item)

    asyncio.run(collect())
    names = [item["event"] for item in events]
    assert "clarification" in names
    assert "text" in names
    done = next(item["data"] for item in events if item["event"] == "done")
    assert done["stop_reason"] == "awaiting_clarification"
    assert session.pending_clarification is not None
    assert session.pending_clarification.mode == "qwen_live"
    assert session.pending_clarification.payload["question"].startswith("Do you prefer")
    assert session.turns[-1].products is not None
    assert session.turns[-1].products["recommended_product_id"] is None


def test_qwen_clarification_reply_resumes_with_qwen_not_rule():
    manager = _Manager()
    session = _session()
    session.pending_clarification = PendingClarification(
        payload={"clarification_id": "qclar_test", "question": "Do you prefer over-ear or in-ear?",
                 "memory_write_enabled": True},
        query="I need earbuds", turn_id="web-test-t1", reply_turn_id="web-test-r1",
        mode="qwen_live",
    )
    response = asyncio.run(demo_flow.clarification_reply(
        manager=manager, session=session, clarification_id="qclar_test",
        message="I prefer wireless and noise cancelling.", qwen_health=_Health(),
    ))
    tools = [tool for tool, _ in manager.calls]
    assert "submit_clarification_reply" in tools
    assert "run_agent_turn" in tools
    assert "recommend_products" not in tools
    assert response["planner"] == "qwen_base"
    assert response["memory_update"]["confirmation_source"] == "web_ui_explicit_user_reply"
    assert session.turns[-1].planner == "qwen_base"
