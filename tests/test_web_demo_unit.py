"""Unit tests for the Web Demo BFF using a fake MCP manager.

These tests never start a real MCP server.  They cover request validation (extra fields are
forbidden), session isolation, SSE event ordering, real tool-error mapping, writes-disabled
behaviour, clarification reply construction and replay.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from web_demo import viewmodels as vm
from web_demo.app import create_app
from web_demo.config import PROFILE_BY_ID, REPLAY_PATH, Settings
from web_demo.mcp_manager import MCPManagerError, MCPToolError, MCPUnavailableError
from web_demo.sessions import SessionStore

REPO = Path(__file__).resolve().parents[1]
NAMESPACE = "amazon_electronics_c4"
SOURCE = "data/p13r/electronics/catalog_meta.jsonl"


# ---------------------------------------------------------------------------
# Fake MCP manager and real-shaped payload fixtures
# ---------------------------------------------------------------------------


class FakeRuntime:
    def __init__(self, profile_id: str, *, writes_enabled: bool) -> None:
        self.profile_id = profile_id
        self.writes_enabled = writes_enabled


class FakeManager:
    """Minimal duck-typed replacement for :class:`web_demo.mcp_manager.MCPManager`."""

    def __init__(
        self,
        *,
        ready: bool = True,
        writes_enabled: bool = True,
        catalogue_size: int = 76263,
        search_mode: str = "bm25",
    ) -> None:
        self.ready = ready
        self.writes_enabled = writes_enabled
        self._catalogue_size = catalogue_size
        self.search_mode = search_mode
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {}
        self.errors: dict[str, MCPManagerError] = {}
        self.started = False
        self.stopped = False
        self.restarted: list[str | None] = []

    # -- lifecycle / health --------------------------------------------------
    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def restart(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        self.restarted.append(profile_id)
        return [
            {"profile_id": pid, "status": "starting", "tools": None}
            for pid in ([profile_id] if profile_id else PROFILE_BY_ID)
        ]

    def ensure_ready(self, profile_id: str) -> FakeRuntime:
        if profile_id not in PROFILE_BY_ID:
            raise MCPManagerError(
                f"unknown profile_id {profile_id!r}", code="unknown_profile_id", http_status=400
            )
        if not self.ready:
            raise MCPUnavailableError(
                f"MCP server for {profile_id} is not ready (status=starting)",
                detail={"profile_id": profile_id, "status": "starting"},
            )
        return FakeRuntime(profile_id, writes_enabled=self.writes_enabled)

    def health_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "profile_id": profile_id,
                "user_key": PROFILE_BY_ID[profile_id].user_key,
                "status": "ready" if self.ready else "starting",
                "search_mode": self.search_mode,
                "writes_enabled": self.writes_enabled,
                "startup_ms": 1234.5 if self.ready else None,
                "error": None,
                "tools": ["search_products"] if self.ready else None,
            }
            for profile_id in PROFILE_BY_ID
        ]

    def aggregate_status(self) -> tuple[str, bool, bool]:
        if self.ready:
            return "ready", True, True
        return "starting", False, False

    def catalogue_size(self) -> int:
        return self._catalogue_size

    # -- tools ---------------------------------------------------------------
    async def call_tool(
        self, profile_id: str, tool: str, arguments: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append((profile_id, tool, args))
        if tool in self.errors:
            raise self.errors[tool]
        handler = self.responses.get(tool)
        if handler is None:
            raise MCPToolError(f"fake manager has no response for {tool}", tool=tool)
        value = handler(profile_id, args) if callable(handler) else handler
        if isinstance(value, Exception):
            raise value
        return json.loads(json.dumps(value))


def product_title(item_id: str) -> str:
    return f"Real catalogue product {item_id}"


def info_payload(item_id: str = "B004K6R43G", *, title: str | None = None) -> dict[str, Any]:
    title = title or product_title(item_id)
    namespaced = f"{NAMESPACE}:{item_id}"
    return {
        "schema": "agentrecx.p17.get_product_info.v1",
        "status": "ok",
        "namespace": NAMESPACE,
        "product_id": item_id,
        "product_info": {
            "status": "ok",
            "product_id": item_id,
            "namespaced_id": namespaced,
            "product_type": "earbuds",
            "title": title,
            "form_factor": "in_ear",
            "facts": {
                "title": {"status": "unsupported", "value": None},
                "price": {
                    "status": "unknown", "value": None, "source": SOURCE,
                    "reason": "catalog_meta.jsonl has no price field", "verification": "absent",
                },
                "brand": {
                    "status": "unknown", "value": None, "source": SOURCE,
                    "reason": "catalog_meta.jsonl has no brand field", "verification": "absent",
                },
                "material": {
                    "status": "unknown", "value": None, "source": SOURCE,
                    "reason": "catalog_meta.jsonl has no material field", "verification": "absent",
                },
                "stock": {
                    "status": "unknown", "value": None, "source": SOURCE,
                    "reason": "catalog_meta.jsonl has no stock field", "verification": "absent",
                },
                "warranty": {"status": "unsupported", "value": None},
            },
            "unknown_fields": ["price", "brand", "material", "stock"],
            "unsupported_fields": ["title", "warranty"],
            "source": SOURCE,
        },
        "selected_product": {
            "item_id": item_id,
            "namespaced_id": namespaced,
            "title": title,
            "product_type": "earbuds",
            "form_factor": "in_ear",
            "attributes": {
                "in_ear": {
                    "status": "present",
                    "evidence": title,
                    "source": "catalog_meta:text",
                }
            },
            "text_chars": len(title),
            "source": SOURCE,
            "source_record": f"{SOURCE}:{namespaced}",
        },
        "product_evidence": [
            {
                "evidence_id": "pev_01",
                "namespace": NAMESPACE,
                "product_id": item_id,
                "source_record": f"{SOURCE}:{namespaced}",
                "raw_field": "title",
                "span_start": 0,
                "span_end": len(title),
                "span_text": title,
                "raw_excerpt": title,
                "attribute": None,
                "verification_status": "lexical_span_exact",
            },
            {
                "evidence_id": "pev_02",
                "namespace": NAMESPACE,
                "product_id": item_id,
                "source_record": f"{SOURCE}:{namespaced}",
                "raw_field": "title",
                "span_start": 0,
                "span_end": len("in_ear"),
                "span_text": "in_ear",
                "raw_excerpt": title,
                "attribute": "in_ear",
                "verification_status": "lexical_span_exact",
            },
        ],
        "unknown_fields": [
            {"attribute": "price", "reason": "no reliable numeric price in the catalogue record"},
            {"attribute": "brand", "reason": "the catalogue record has no trusted brand/store field"},
            {"attribute": "stock", "reason": "the catalogue snapshot is not a live availability feed"},
            {"attribute": "material", "reason": "no trusted material field in the catalogue record"},
            {"attribute": "size", "reason": "no trusted size field in the catalogue record"},
            {"attribute": "warranty", "reason": "the catalogue record carries no warranty statement"},
        ],
        "latency_ms": 1.5,
        "notes": ["product fields are catalogue facts; UNKNOWN fields are not filled from model knowledge"],
    }


def search_payload(query: str = "wireless earbuds") -> dict[str, Any]:
    hits = []
    for index, item_id in enumerate(("B004K6R43G", "B00008VF63"), start=1):
        hits.append(
            {
                "item_id": item_id,
                "namespaced_id": f"{NAMESPACE}:{item_id}",
                "title": product_title(item_id),
                "score": 1.0 - index * 0.1,
                "source": "catalog_search",
                "namespace": NAMESPACE,
            }
        )
    return {
        "schema": "agentrecx.p17.search_products.v1",
        "status": "ok",
        "query": query,
        "search_mode": "bm25",
        "score_kind": "bm25",
        "catalogue_size": 76263,
        "hits": hits,
        "latency_ms": 2.5,
        "notes": ["real catalogue search; no target/evaluator data is read"],
    }


def gate_payload(action: str = "ALLOW_FINALIZE") -> dict[str, Any]:
    return {
        "action": action,
        "allow_finalize": action == "ALLOW_FINALIZE",
        "reason_code": "evidence_sufficient",
        "reason": "hard constraints verified; no explicit preference conflict",
        "selected_candidate_id": "B004K6R43G",
        "candidate_presentable_ids": ["B004K6R43G", "B00008VF63"],
        "hard_constraint_status": {"hard_constraints": "verified"},
        "preference_conflict_status": "none",
        "unknown_fields": ["over_ear"],
        "product_type": None,
        "product_role": "main",
        "evidence_refs": [],
        "ask_question": None,
        "bounded_scope": True,
        "budget_used": {},
        "gate_version": "decision_gate_v0.1",
        "reasons": [],
    }


def recommend_payload(*, status: str = "finalized") -> dict[str, Any]:
    if status == "awaiting_clarification":
        return {
            "schema": "agentrecx.p17.recommend_products.v1",
            "status": "awaiting_clarification",
            "query": "I need some new headphones. Not sure what matters yet.",
            "selected_product_id": None,
            "selected_product": None,
            "gate": gate_payload("ASK_CLARIFICATION"),
            "evidence_bundle": None,
            "planner_steps": {},
            "search_mode": "bm25",
            "error": None,
            "latency_ms": 10.0,
            "notes": [],
        }
    return {
        "schema": "agentrecx.p17.recommend_products.v1",
        "status": "finalized",
        "query": "wireless earbuds",
        "selected_product_id": "B004K6R43G",
        "selected_product": {
            "item_id": "B004K6R43G",
            "namespaced_id": f"{NAMESPACE}:B004K6R43G",
            "title": product_title("B004K6R43G"),
            "product_type": "earbuds",
            "form_factor": "in_ear",
            "attributes": {
                "in_ear": {"status": "present", "evidence": "title", "source": "catalog_meta:text"}
            },
            "source": SOURCE,
            "source_record": f"{SOURCE}:{NAMESPACE}:B004K6R43G",
        },
        "gate": gate_payload(),
        "evidence_bundle": {"schema": "agentrecx.p15.evidence_bundle.v1"},
        "planner_steps": {},
        "search_mode": "bm25",
        "error": None,
        "latency_ms": 120.0,
        "notes": [],
    }


def clarification_payload(*, memory_write_enabled: bool = True) -> dict[str, Any]:
    return {
        "schema": "agentrecx.p16.preference_clarification.v1",
        "skill": "preference-clarification",
        "status": "awaiting_clarification",
        "current_query": "I need some new headphones. Not sure what matters yet.",
        "user_key": "p14e2_lite_0057",
        "question": "What matters most for this purchase - for example the use case, a budget, or a feature you need?",
        "reason": "missing_key_need",
        "requires_user_reply": True,
        "clarification_id": "clar_fake_handle",
        "authorized_user_key": "p14e2_lite_0057",
        "expires_in_seconds": 1800.0,
        "memory_write_enabled": memory_write_enabled,
        "notes": [],
    }


def preferences_payload(user_key: str = "p14e2_lite_0057") -> dict[str, Any]:
    entry = {
        "memory_id": "b577cd49debda88ab3b3b8a4",
        "kind": "feature",
        "value": "wireless",
        "polarity": "prefer",
        "status": "active",
        "logical_seq": 2,
        "source_turn_id": "web-abc-r1",
        "source_text": "I prefer noise cancelling and wireless.",
        "created_at": 1794206326.979659,
        "query_token_overlap": 0,
    }
    inactive = dict(entry, memory_id="m_inactive", value="over-ear", status="inactive")
    return {
        "schema": "agentrecx.p17.get_active_preferences.v1",
        "status": "ok",
        "authorized_user_key": user_key,
        "history_status": "active_preferences",
        "active_preferences": [entry],
        "inactive_preferences": [inactive],
        "audit_source": "M9 PreferenceMemoryService via HistoryStateReader",
        "notes": [],
    }


def reply_payload(*, added: int = 2) -> dict[str, Any]:
    return {
        "schema": "agentrecx.p16.preference_clarification.v1",
        "status": "memory_updated",
        "user_message": "I prefer noise cancelling and wireless.",
        "turn_id": "web-abc-r1",
        "already_processed": False,
        "update": {
            "added": added,
            "superseded": 0,
            "removed": 0,
            "skipped_duplicates": 0,
            "removal_directives": 0,
            "already_processed": False,
        },
        "active_preferences": preferences_payload()["active_preferences"],
        "inactive_preferences": [],
        "history_status": "active_preferences",
        "clarification_id": "clar_fake_handle",
        "authorized_user_key": "p14e2_lite_0057",
        "confirmation_source": "web_ui_explicit_user_reply",
        "latency_ms": 0.43,
        "notes": [],
    }


def comparison_payload() -> dict[str, Any]:
    products = []
    for item_id in ("B004K6R43G", "B00008VF63"):
        namespaced = f"{NAMESPACE}:{item_id}"
        products.append(
            {
                "item_id": item_id,
                "namespace": NAMESPACE,
                "title": product_title(item_id),
                "product_type": "earbuds",
                "role_status": "MATCH",
                "product_info": info_payload(item_id)["product_info"],
                "evidence_bundle": {
                    "schema": "agentrecx.p15.evidence_bundle.v1",
                    "selected_product": {"item_id": item_id, "source_record": f"{SOURCE}:{namespaced}"},
                    "product_evidence": info_payload(item_id)["product_evidence"],
                    "unknown_fields": [
                        {"attribute": "price", "reason": "no reliable numeric price in the catalogue record"},
                        {"attribute": "brand", "reason": "the catalogue record has no trusted brand/store field"},
                    ],
                },
                "claims": [
                    {
                        "claim_type": "product_fact",
                        "claim_text": f"Catalogue title: {product_title(item_id)}",
                        "product_id": item_id,
                        "attribute": "title",
                        "evidence_refs": ["pev_01"],
                    }
                ],
                "claim_validation": {"valid_claims": [], "rejected_claims": []},
                "preference_relations": [
                    {
                        "memory_id": "b577cd49debda88ab3b3b8a4",
                        "value": "wireless",
                        "polarity": "prefer",
                        "kind": "feature",
                        "observed_in_catalogue_text": False,
                        "relation": "not_evidenced",
                        "source_text": "I prefer wireless.",
                    }
                ],
                "unknown_fields": ["price", "brand"],
            }
        )
    return {
        "schema": "agentrecx.p16.shopping_comparison.v1",
        "status": "comparison_ready",
        "current_query": "compare these",
        "products": products,
        "errors": [],
        "conditional_best_supported": None,
        "final_answer": "Comparison of 2 products (catalogue evidence only):\n\nNo unique conditional preference winner.",
        "qwen_used": False,
        "tool_calls": ["catalogue_identity_resolution", "product_info", "product_info"],
        "claims_valid": 16,
        "claims_rejected": 0,
        "unknown_count": 12,
        "latency_ms": 1.0,
        "notes": ["catalogue facts describe only the product record; no quality/stock/rating claim is inferred"],
    }


def invalid_comparison_payload() -> dict[str, Any]:
    return {
        "schema": "agentrecx.p16.shopping_comparison.v1",
        "status": "invalid_product_reference",
        "current_query": "compare these",
        "products": [],
        "errors": [
            {
                "code": "product_not_found",
                "ref": "B000NOTREAL",
                "message": "product identity is not in the current Electronics catalogue",
            }
        ],
        "conditional_best_supported": None,
        "final_answer": "I can't verify those product references.",
        "qwen_used": False,
        "claims_valid": 0,
        "claims_rejected": 0,
        "unknown_count": 0,
        "latency_ms": 0.5,
        "notes": [],
    }


def make_app(tmp_path: Path, manager: FakeManager, *, writes: bool = True, replay_path: Path | None = None):
    settings = Settings(
        search_mode="bm25",
        allow_memory_writes=writes,
        log_dir=tmp_path / "logs",
        replay_path=replay_path or REPLAY_PATH,
        python_executable=sys.executable,
        mcp_timeout_s=30.0,
    )
    return create_app(settings=settings, manager=manager, sessions=SessionStore())


def parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(":"):
            continue
        if not line:
            if current_event is not None:
                events.append({"event": current_event, "data": json.loads("\n".join(data_lines))})
                current_event = None
                data_lines = []
            continue
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
    if current_event is not None:
        events.append({"event": current_event, "data": json.loads("\n".join(data_lines))})
    return events


def event_names(events: list[dict[str, Any]]) -> list[str]:
    return [event["event"] for event in events]


def create_session(client: TestClient, profile_id: str = "profile_0049") -> dict[str, Any]:
    response = client.post("/api/sessions", json={"profile_id": profile_id})
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Health / config / replay
# ---------------------------------------------------------------------------


def test_health_reports_real_servers_and_modes(tmp_path):
    manager = FakeManager(ready=True)
    with TestClient(make_app(tmp_path, manager)) as client:
        payload = client.get("/api/health").json()
    assert manager.started is True
    assert payload["status"] == "ready" and payload["ready"] is True
    assert payload["catalogue_size"] == 76263
    assert {server["profile_id"] for server in payload["servers"]} == set(PROFILE_BY_ID)
    assert all(server["status"] == "ready" for server in payload["servers"])
    assert payload["modes"]["cpu_live"]["available"] is True
    assert payload["modes"]["qwen_live"]["available"] is False
    assert payload["memory_writes_enabled"] is True


def test_health_still_available_while_starting(tmp_path):
    manager = FakeManager(ready=False)
    with TestClient(make_app(tmp_path, manager)) as client:
        payload = client.get("/api/health").json()
        assert payload["status"] == "starting"
        assert payload["ready"] is False
        assert payload["modes"]["cpu_live"]["available"] is False
        assert all(server["startup_ms"] is None for server in payload["servers"])


def test_demo_config_uses_engine_profiles_only(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        config = client.get("/api/demo/config").json()
    assert [profile["profile_id"] for profile in config["profiles"]] == [
        "profile_0049",
        "profile_0057",
    ]
    assert config["default_profile_id"] == "profile_0049"
    assert config["limits"]["compare_min"] == 2 and config["limits"]["compare_max"] == 5
    assert config["verified_comparison_ids"] == ["B004K6R43G", "B00008VF63"]
    assert config["profiles"][0]["memory_writes"] is False
    assert config["profiles"][1]["memory_writes"] is True


def test_replay_is_real_historical_artifact(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        response = client.get("/api/replay")
        assert response.status_code == 200
        payload = response.json()
        missing = client.get("/api/replay")
    assert payload["mode"] == "Qwen_Agent_Replay"
    assert payload["replay"] is True
    assert payload["live_call"] is False
    assert payload["badge"] == "Historical Replay — Not a Live Model Call"
    assert payload["examples"], "expected the existing P18 real trajectories"
    assert missing.status_code == 200  # same app; file exists


def test_replay_missing_artifact_returns_503(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager, replay_path=tmp_path / "missing.json")) as client:
        response = client.get("/api/replay")
    assert response.status_code == 503
    assert response.json()["code"] == "replay_unavailable"


# ---------------------------------------------------------------------------
# Sessions / validation
# ---------------------------------------------------------------------------


def test_sessions_are_isolated_and_profile_bound(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        first = create_session(client, "profile_0049")
        second = create_session(client, "profile_0057")
        assert first["session_id"].startswith("web-")
        assert first["session_id"] != second["session_id"]
        assert first["profile_id"] == "profile_0049"
        assert second["profile_id"] == "profile_0057"
        listing = client.get("/api/sessions?limit=20").json()["sessions"]
        assert len(listing) == 2
        fetched = client.get(f"/api/sessions/{first['session_id']}").json()
        assert fetched["session_id"] == first["session_id"]
        assert fetched["turns"] == []


def test_session_unknown_and_bad_profile(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        missing = client.get("/api/sessions/web-doesnotexist")
        bad_profile = client.post("/api/sessions", json={"profile_id": "profile_9999"})
    assert missing.status_code == 404
    assert missing.json()["code"] == "session_not_found"
    assert bad_profile.status_code == 400
    assert bad_profile.json()["code"] == "unknown_profile_id"


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/sessions", {"profile_id": "profile_0049", "user_id": "p14e2_lite_0057"}),
        ("/api/chat/stream", {"session_id": "web-x", "message": "hi", "confirmed_by_user": True}),
        ("/api/products/details", {"session_id": "web-x", "product_ref": "B004K6R43G", "user_key": "x"}),
        (
            "/api/clarification/reply",
            {
                "session_id": "web-x",
                "clarification_id": "clar_x",
                "message": "reply",
                "confirmation_source": "trusted_local_client",
            },
        ),
    ],
)
def test_client_cannot_supply_identity_or_confirmation(tmp_path, path, body):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        response = client.post(path, json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert manager.calls == []


def test_reset_session_clears_turns_not_memory(tmp_path):
    manager = FakeManager()
    manager.responses = {
        "search_products": search_payload(),
        "get_product_info": lambda profile_id, args: info_payload(args["product_ref"]),
        "recommend_products": recommend_payload(),
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "wireless earbuds"},
        )
        before = client.get(f"/api/sessions/{session['session_id']}").json()
        reset = client.post(f"/api/sessions/{session['session_id']}/reset").json()
    assert len(before["turns"]) == 1
    assert reset["turns"] == []
    assert reset["pending_clarification"] is None
    # reset never touches the MCP-side M9 memory: no tool call is made by reset.
    tools = [call[1] for call in manager.calls]
    assert "get_active_preferences" not in tools


# ---------------------------------------------------------------------------
# Chat SSE: finalized recommendation
# ---------------------------------------------------------------------------


def test_chat_sse_event_order_and_real_payload_mapping(tmp_path):
    manager = FakeManager(writes_enabled=False)
    manager.responses = {
        "search_products": search_payload(),
        "get_product_info": lambda profile_id, args: info_payload(args["product_ref"]),
        "recommend_products": recommend_payload(),
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        response = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "wireless earbuds for workouts"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        session_view = client.get(f"/api/sessions/{session['session_id']}").json()

    names = event_names(events)
    assert names[0] == "tool_status" and events[0]["data"]["status"] == "running"
    assert names[-1] == "done"
    assert events[-1]["data"]["stop_reason"] == "finalized"
    assert names.index("products") > names.index("tool_status")
    assert names.index("evidence") > names.index("products")
    assert names.index("text") > names.index("evidence")
    assert names.count("done") == 1

    products = events[names.index("products")]["data"]
    assert products["catalogue_size"] == 76263
    assert products["search_mode"] == "bm25"
    assert [card["product_id"] for card in products["items"]] == ["B004K6R43G", "B00008VF63"]
    assert products["recommended_product_id"] == "B004K6R43G"
    assert products["items"][0]["recommended"] is True
    assert products["gate"]["action"] == "ALLOW_FINALIZE"
    assert products["gate"]["candidate_presentable_ids"] == ["B004K6R43G", "B00008VF63"]
    first_card = products["items"][0]
    assert first_card["price"]["status"] == "UNKNOWN"
    assert first_card["price"]["display"] == "Price unavailable / UNKNOWN"
    assert first_card["image_url"] is None
    assert first_card["evidence_available"] is True
    assert {item["attribute"] for item in first_card["attributes"]} == {"in_ear"}
    assert "price" in {item["attribute"] for item in first_card["unknown_fields"]}

    evidence = events[names.index("evidence")]["data"]
    assert evidence["product_id"] == "B004K6R43G"
    assert evidence["evidence"][0]["span_text"] == product_title("B004K6R43G")
    assert evidence["source_record"].endswith("B004K6R43G")

    narrative = events[names.index("text")]["data"]["text"]
    assert "76,263" in narrative
    assert "ALLOW_FINALIZE" in narrative
    assert "B004K6R43G" in narrative
    assert "UNKNOWN" in narrative

    assert len(session_view["turns"]) == 1
    turn = session_view["turns"][0]
    assert turn["products"]["recommended_product_id"] == "B004K6R43G"
    assert turn["error"] is None
    assert [item["tool"] for item in turn["tool_activity"]] == [
        "search_products",
        "get_product_info",
        "recommend_products",
    ]


def test_chat_comparison_path_uses_selected_ids_only(tmp_path):
    manager = FakeManager()
    manager.responses = {"compare_products": comparison_payload()}
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        response = client.post(
            "/api/chat/stream",
            json={
                "session_id": session["session_id"],
                "message": "Please compare these two products",
                "selected_product_ids": ["B004K6R43G", "B00008VF63"],
            },
        )
        events = parse_sse(response.text)
    assert event_names(events) == ["tool_status", "tool_status", "comparison", "text", "done"]
    comparison = events[2]["data"]
    assert comparison["status"] == "comparison_ready"
    assert [product["product_id"] for product in comparison["products"]] == [
        "B004K6R43G", "B00008VF63",
    ]
    assert comparison["claims_summary"] == {"valid": 16, "rejected": 0, "unknown": 12}
    assert comparison["dimensions"]
    assert comparison["preference_relations"][0]["per_product"]["B004K6R43G"] == "not_evidenced"
    assert events[-1]["data"]["stop_reason"] == "comparison"
    tools = [call[1] for call in manager.calls]
    assert tools == ["compare_products"], tools


# ---------------------------------------------------------------------------
# Chat SSE: clarification / memory reply / errors
# ---------------------------------------------------------------------------


def _clarification_setup(manager: FakeManager) -> None:
    manager.responses = {
        "search_products": search_payload(),
        "get_product_info": lambda profile_id, args: info_payload(args["product_ref"]),
        "recommend_products": recommend_payload(status="awaiting_clarification"),
        "request_preference_clarification": clarification_payload(),
    }


def test_chat_clarification_flow_stores_pending_handle(tmp_path):
    manager = FakeManager(writes_enabled=False)
    _clarification_setup(manager)
    with TestClient(make_app(tmp_path, manager, writes=False)) as client:
        session = create_session(client)
        response = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "I need some new headphones."},
        )
        events = parse_sse(response.text)
        session_view = client.get(f"/api/sessions/{session['session_id']}").json()
    names = event_names(events)
    assert "clarification" in names
    clarification = events[names.index("clarification")]["data"]
    assert clarification["clarification_id"] == "clar_fake_handle"
    assert clarification["quick_options"] == []
    assert clarification["memory_write_enabled"] is True
    assert events[-1]["data"]["stop_reason"] == "awaiting_clarification"
    assert session_view["pending_clarification"]["clarification_id"] == "clar_fake_handle"
    assert session_view["turns"][0]["clarification"]["reason"] == "missing_key_need"


def test_reply_denied_when_writes_disabled(tmp_path):
    manager = FakeManager(writes_enabled=False)
    _clarification_setup(manager)
    with TestClient(make_app(tmp_path, manager, writes=False)) as client:
        session = create_session(client, "profile_0057")
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "I need some new headphones."},
        )
        _ = chat.text  # drain the SSE body so the turn (and pending handle) is finalised
        response = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session["session_id"],
                "clarification_id": "clar_fake_handle",
                "message": "I prefer noise cancelling and wireless.",
            },
        )
        session_view = client.get(f"/api/sessions/{session['session_id']}").json()
    assert response.status_code == 409
    assert response.json()["code"] == "memory_writes_disabled"
    # No success turn was faked; the clarification turn remains and the pending handle (with its
    # stable reply turn id) is preserved so the user can retry after writes are enabled.
    assert len(session_view["turns"]) == 1
    assert session_view["turns"][0]["clarification"]["clarification_id"] == "clar_fake_handle"
    assert session_view["turns"][0]["error"] is None
    assert session_view["pending_clarification"]["clarification_id"] == "clar_fake_handle"
    assert all(call[1] != "submit_clarification_reply" for call in manager.calls)


def test_reply_persists_memory_updates_preferences_and_continues(tmp_path):
    manager = FakeManager(writes_enabled=True)
    _clarification_setup(manager)
    recommend_calls = {"count": 0}

    def recommend_handler(profile_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        recommend_calls["count"] += 1
        if recommend_calls["count"] == 1:
            return recommend_payload(status="awaiting_clarification")
        return recommend_payload()

    manager.responses.update(
        {
            "recommend_products": recommend_handler,
            "submit_clarification_reply": reply_payload(),
            "get_active_preferences": preferences_payload(),
        }
    )
    with TestClient(make_app(tmp_path, manager, writes=True)) as client:
        session = create_session(client, "profile_0057")
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "I need some new headphones."},
        )
        _ = chat.text  # drain the SSE body so the pending clarification is stored
        response = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session["session_id"],
                "clarification_id": "clar_fake_handle",
                "message": "I prefer noise cancelling and wireless.",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        # A second reply for the now-cleared clarification is refused, not replayed.
        second = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session["session_id"],
                "clarification_id": "clar_fake_handle",
                "message": "I prefer noise cancelling and wireless.",
            },
        )
        session_view = client.get(f"/api/sessions/{session['session_id']}").json()

    assert payload["memory_update"]["added"] == 2
    assert payload["memory_update"]["already_processed"] is False
    assert payload["memory_update"]["confirmation_source"] == "web_ui_explicit_user_reply"
    assert payload["preferences"]["profile_id"] == "profile_0057"
    assert len(payload["preferences"]["active"]) == 1
    assert len(payload["preferences"]["inactive"]) == 1
    assert payload["recommendation"]["recommended_product_id"] == "B004K6R43G"
    assert payload["recommendation"]["items"][0]["match_status"] == "recommended"
    assert payload["error"] is None
    assert {step["tool"] for step in payload["steps"]} >= {
        "submit_clarification_reply",
        "get_active_preferences",
        "recommend_products",
    }

    submit_calls = [call for call in manager.calls if call[1] == "submit_clarification_reply"]
    assert len(submit_calls) == 1
    profile_id, _tool, arguments = submit_calls[0]
    assert profile_id == "profile_0057"
    # The BFF (not the browser) supplies the confirmation, and it labels the real POST reply.
    assert arguments["confirmed_by_user"] is True
    assert arguments["confirmation_source"] == "web_ui_explicit_user_reply"
    assert arguments["clarification_id"] == "clar_fake_handle"
    assert arguments["turn_id"] == session_view["turns"][-1]["turn_id"]

    assert session_view["pending_clarification"] is None
    assert len(session_view["turns"]) == 2
    reply_turn = session_view["turns"][-1]
    assert reply_turn["memory_update"]["added"] == 2
    assert reply_turn["products"]["recommended_product_id"] == "B004K6R43G"

    # The duplicate reply is refused without creating another MCP write.
    assert second.status_code == 409
    assert second.json()["code"] == "clarification_not_pending"
    assert len([call for call in manager.calls if call[1] == "submit_clarification_reply"]) == 1


def test_reply_rejects_mismatched_clarification_id(tmp_path):
    manager = FakeManager(writes_enabled=True)
    _clarification_setup(manager)
    manager.responses.update({"submit_clarification_reply": reply_payload()})
    with TestClient(make_app(tmp_path, manager, writes=True)) as client:
        session = create_session(client, "profile_0057")
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "I need some new headphones."},
        )
        _ = chat.text
        response = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session["session_id"],
                "clarification_id": "clar_other_handle",
                "message": "I prefer wireless.",
            },
        )
    assert response.status_code == 409
    assert response.json()["code"] == "clarification_not_pending"


def test_mcp_tool_error_is_surfaced_in_sse(tmp_path):
    manager = FakeManager()
    manager.errors["search_products"] = MCPToolError(
        "catalogue unavailable (real failure)", code="tool_error", tool="search_products"
    )
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        response = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "wireless earbuds"},
        )
        events = parse_sse(response.text)
        session_view = client.get(f"/api/sessions/{session['session_id']}").json()
    names = event_names(events)
    assert "products" not in names
    assert "error" in names
    error = events[names.index("error")]["data"]
    assert "catalogue unavailable" in error["message"]
    assert error["tool"] == "search_products"
    assert names[-1] == "done" and events[-1]["data"]["stop_reason"] == "error"
    assert session_view["turns"][0]["error"]["tool"] == "search_products"


def test_readiness_503_before_servers_ready(tmp_path):
    manager = FakeManager(ready=False)
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": "hi"},
        )
        details = client.post(
            "/api/products/details",
            json={"session_id": session["session_id"], "product_ref": "B004K6R43G"},
        )
        prefs = client.get(f"/api/preferences?session_id={session['session_id']}")
    for response in (chat, details, prefs):
        assert response.status_code == 503
        assert response.json()["code"] == "mcp_not_ready"


# ---------------------------------------------------------------------------
# Products endpoints
# ---------------------------------------------------------------------------


def test_product_details_maps_real_evidence_and_unknown(tmp_path):
    manager = FakeManager()
    manager.responses = {"get_product_info": lambda profile_id, args: info_payload(args["product_ref"])}
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        payload = client.post(
            "/api/products/details",
            json={"session_id": session["session_id"], "product_ref": "B004K6R43G"},
        ).json()
    assert payload["product_id"] == "B004K6R43G"
    assert payload["facts"][0]["attribute"] == "title"
    assert payload["facts"][1]["attribute"] == "price"
    assert payload["facts"][1]["status"] == "unknown"
    assert payload["evidence"][0]["raw_field"] == "title"
    assert payload["evidence"][0]["span_text"] == product_title("B004K6R43G")
    assert payload["unknown_fields"][0]["attribute"] == "price"
    assert payload["raw_fields"]["title"] == [product_title("B004K6R43G")]
    assert payload["evidence_available"] is True
    assert any("semantic_truth_verified=false" in note for note in payload["notes"])


def test_compare_endpoint_and_real_error(tmp_path):
    manager = FakeManager()
    manager.responses = {"compare_products": comparison_payload()}
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        ok = client.post(
            "/api/products/compare",
            json={
                "session_id": session["session_id"],
                "product_refs": ["B004K6R43G", "B00008VF63"],
            },
        )
        manager.responses["compare_products"] = invalid_comparison_payload()
        bad = client.post(
            "/api/products/compare",
            json={"session_id": session["session_id"], "product_refs": ["B004K6R43G", "B000NOTREAL"]},
        )
        too_many = client.post(
            "/api/products/compare",
            json={
                "session_id": session["session_id"],
                "product_refs": ["A", "B", "C", "D", "E", "F"],
            },
        )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["status"] == "comparison_ready"
    assert payload["products"][1]["role_status"] == "MATCH"
    assert payload["unknown_fields"][0]["products"] == ["B004K6R43G", "B00008VF63"]
    assert payload["conflicts"] == []
    assert bad.status_code == 400
    assert bad.json()["code"] == "invalid_product_reference"
    assert bad.json()["detail"]["errors"][0]["code"] == "product_not_found"
    assert too_many.status_code == 422


def test_preferences_endpoint_active_inactive_separation(tmp_path):
    manager = FakeManager()
    manager.responses = {"get_active_preferences": preferences_payload()}
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client, "profile_0057")
        payload = client.get(f"/api/preferences?session_id={session['session_id']}").json()
    assert payload["profile_id"] == "profile_0057"
    assert payload["user_key"] == "p14e2_lite_0057"
    assert payload["history_status"] == "active_preferences"
    assert [entry["value"] for entry in payload["active"]] == ["wireless"]
    assert [entry["value"] for entry in payload["inactive"]] == ["over-ear"]
    assert payload["fetched_at"].endswith("Z")


def test_restart_endpoint_calls_manager(tmp_path):
    manager = FakeManager()
    with TestClient(make_app(tmp_path, manager)) as client:
        payload = client.post("/api/demo/restart", json={"profile_id": "profile_0049"}).json()
    assert payload["restarting"] is True
    assert manager.restarted == ["profile_0049"]


# ---------------------------------------------------------------------------
# View-model unit checks (no HTTP)
# ---------------------------------------------------------------------------


def test_price_is_never_invented_from_boolean_presence_fact():
    info = info_payload()
    info["product_info"]["facts"]["price"] = {
        "status": "present", "value": True, "source": SOURCE, "verification": "text_evidence",
    }
    card = vm.product_card(
        info, source="search", retrieval_source=vm.RETRIEVAL_BM25,
        match_status="search_result", recommended=False,
    )
    assert card["price"]["status"] == "UNKNOWN"
    assert card["price"]["display"] == "Price unavailable / UNKNOWN"


def test_comparison_statuses_pass_through_without_recompute():
    raw = comparison_payload()
    raw["products"][1]["role_status"] = "NOT_REQUIRED"
    view = vm.comparison_payload(raw)
    assert [product["role_status"] for product in view["products"]] == ["MATCH", "NOT_REQUIRED"]
    assert view["dimensions"][0]["cells"][0]["status"] == "unsupported"
    assert view["claims_summary"]["valid"] == 16
    assert view["conditional_recommendation"] == {"supported": None, "text": None}


def test_memory_update_flag_prefers_nested_update():
    view = vm.memory_update_payload(
        {"update": {"added": 2, "already_processed": True}, "already_processed": False,
         "confirmation_source": "web_ui_explicit_user_reply"},
        None,
    )
    assert view["added"] == 2
    assert view["already_processed"] is True
    assert view["confirmation_source"] == "web_ui_explicit_user_reply"


def test_profiles_are_isolated_and_memory_identity_is_server_bound(tmp_path):
    manager = FakeManager(writes_enabled=True)
    _clarification_setup(manager)
    manager.responses["get_active_preferences"] = preferences_payload("p14e2_lite_0057")
    with TestClient(make_app(tmp_path, manager, writes=True)) as client:
        session_49 = create_session(client, "profile_0049")
        session_57 = create_session(client, "profile_0057")

        # 0057 clarification handle exists only on the 0057 session.
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session_57["session_id"], "message": "I need some new headphones."},
        )
        _ = chat.text
        pending_57 = client.get(f"/api/sessions/{session_57['session_id']}").json()[
            "pending_clarification"
        ]
        assert pending_57["clarification_id"] == "clar_fake_handle"

        # The 0049 session cannot use 0057's handle: it has no pending clarification of its own.
        hijack = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session_49["session_id"],
                "clarification_id": pending_57["clarification_id"],
                "message": "I prefer wireless.",
            },
        )
        # 0049 is also read-only, so writes are refused before any MCP call.
        assert hijack.status_code == 409
        assert hijack.json()["code"] in {"memory_writes_disabled", "clarification_not_pending"}

        prefs = client.get(f"/api/preferences?session_id={session_49['session_id']}")
        assert prefs.status_code == 200

        session_57_view = client.get(f"/api/sessions/{session_57['session_id']}").json()
        session_49_view = client.get(f"/api/sessions/{session_49['session_id']}").json()
    assert session_49_view["profile_id"] == "profile_0049"
    assert session_49_view["turns"] == []
    assert session_57_view["profile_id"] == "profile_0057"
    assert len(session_57_view["turns"]) == 1
    # Preference reads were executed against the session-bound profile, never a browser-supplied id.
    prefs_calls = [call for call in manager.calls if call[1] == "get_active_preferences"]
    assert prefs_calls and prefs_calls[-1][0] == "profile_0049"
    assert "p14e2_lite_0057" not in json.dumps(
        [call[2] for call in manager.calls if call[1] == "get_active_preferences"]
    )


def test_cancelled_sse_generator_leaves_no_turn_or_pending_state(tmp_path):
    import anyio

    from web_demo import demo_flow

    manager = FakeManager()
    manager.responses = {
        "search_products": search_payload(),
        "get_product_info": lambda profile_id, args: info_payload(args["product_ref"]),
        "recommend_products": recommend_payload(),
    }
    app = make_app(tmp_path, manager)

    async def scenario() -> None:
        session = app.state.sessions.create("profile_0049")
        generator = demo_flow.chat_events(
            manager=manager,
            session=session,
            settings=app.state.settings,
            message="wireless earbuds",
        )
        first = await generator.__anext__()
        assert first["event"] == "tool_status" and first["data"]["status"] == "running"
        await generator.aclose()  # simulate the browser disconnecting mid-turn
        assert session.turns == []
        assert session.pending_clarification is None

    anyio.run(scenario)

