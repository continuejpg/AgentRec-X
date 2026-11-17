"""Behavior Feedback V1 tests: event validation, source policy, profile and shadow.

These tests use a fake MCP manager exactly like ``tests/test_web_demo_unit.py``: no
catalogue, BGE, GPU or network is needed.  The browser visibility itself is covered by
the real browser E2E run; here we test the server-side contract the browser must satisfy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from web_demo.app import create_app
from web_demo.behavior import BehaviorStore
from web_demo.config import PROFILE_BY_ID, REPLAY_PATH, Settings
from web_demo.mcp_manager import MCPManagerError
from web_demo.sessions import SessionStore

REPO = Path(__file__).resolve().parents[1]
NAMESPACE = "amazon_electronics_c4"
SOURCE = "data/p13r/electronics/catalog_meta.jsonl"


class FakeManager:
    def __init__(self, *, ready: bool = True, writes_enabled: bool = True) -> None:
        self.ready = ready
        self.writes_enabled = writes_enabled
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {}
        self.search_mode = "bm25"
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def restart(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        return []

    def ensure_ready(self, profile_id: str) -> Any:
        if profile_id not in PROFILE_BY_ID:
            raise MCPManagerError("unknown profile", code="unknown_profile_id", http_status=400)
        if not self.ready:
            raise MCPManagerError("not ready", code="mcp_unavailable", http_status=503)
        return type("Runtime", (), {"profile_id": profile_id, "writes_enabled": self.writes_enabled})()

    def health_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "profile_id": pid,
                "user_key": PROFILE_BY_ID[pid].user_key,
                "status": "ready",
                "search_mode": self.search_mode,
                "writes_enabled": self.writes_enabled,
                "startup_ms": 1.0,
                "error": None,
                "tools": [],
            }
            for pid in PROFILE_BY_ID
        ]

    def aggregate_status(self) -> tuple[str, bool, bool]:
        return ("ready", True, True) if self.ready else ("starting", False, False)

    def catalogue_size(self) -> int:
        return 76263

    async def call_tool(
        self, profile_id: str, tool: str, arguments: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append((profile_id, tool, args))
        handler = self.responses.get(tool)
        if handler is None:
            raise MCPManagerError(f"no response for {tool}", code="tool_unavailable", http_status=503)
        value = handler(profile_id, args) if callable(handler) else handler
        return json.loads(json.dumps(value))


def _info(
    item_id: str,
    *,
    attributes: tuple[str, ...] | None = None,
    evidence: bool = True,
) -> dict[str, Any]:
    # P10 is deliberately unique so a behavior boost cannot leak to every card through a
    # shared attribute; the other fixtures keep the common earbud attributes.
    if attributes is None:
        attributes = ("special_p10",) if item_id == "P10" else ("in_ear", "wireless")
    selected_attributes = {name: {"status": "present"} for name in attributes}
    product_evidence = [
        {
            "attribute": name,
            "evidence_id": f"pev_{item_id}_{name}",
            "raw_field": "title",
            "span_text": f"{item_id} has {name}",
            "source_record": f"{SOURCE}:{NAMESPACE}:{item_id}",
        }
        for name in attributes
    ] if evidence else []
    return {
        "schema": "agentrecx.p17.get_product_info.v1",
        "status": "ok",
        "namespace": NAMESPACE,
        "product_id": item_id,
        "product_info": {
            "status": "ok", "product_id": item_id, "namespaced_id": f"{NAMESPACE}:{item_id}",
            "product_type": "earbuds", "title": f"Product {item_id}", "form_factor": "in_ear",
            "facts": {"price": {"status": "unknown", "value": None, "reason": "no price field"},
                      "brand": {"status": "unknown", "value": None, "reason": "no brand field"},
                      "material": {"status": "unknown", "value": None, "reason": "no material field"},
                      "stock": {"status": "unknown", "value": None, "reason": "no stock field"}},
            "unknown_fields": ["price", "brand", "material", "stock"],
            "source": SOURCE,
        },
        "selected_product": {
            "item_id": item_id,
            "namespaced_id": f"{NAMESPACE}:{item_id}",
            "title": f"Product {item_id}",
            "product_type": "earbuds",
            "form_factor": "in_ear",
            "attributes": selected_attributes,
            "source": SOURCE,
            "source_record": f"{SOURCE}:{NAMESPACE}:{item_id}",
        },
        "product_evidence": product_evidence,
        "unknown_fields": [],
    }


def _search(item_ids: tuple[str, ...] = ("P01", "P02", "P03")) -> dict[str, Any]:
    return {
        "schema": "agentrecx.p17.search_products.v1", "status": "ok", "query": "earbuds",
        "search_mode": "bm25", "score_kind": "bm25", "catalogue_size": 76263,
        "hits": [
            {"item_id": item_id, "namespaced_id": f"{NAMESPACE}:{item_id}",
             "title": f"Product {item_id}", "score": 1.0 - index * 0.01}
            for index, item_id in enumerate(item_ids, start=1)
        ],
        "latency_ms": 1.0,
    }


def _gate(presentable: tuple[str, ...] = ("P01", "P02", "P03")) -> dict[str, Any]:
    return {
        "action": "ALLOW_FINALIZE", "reason_code": "evidence_sufficient",
        "reason": "hard constraints verified", "selected_candidate_id": "P01",
        "candidate_presentable_ids": list(presentable),
        "unknown_fields": [], "product_role": "main",
    }


def _recommend(item_ids: tuple[str, ...] = ("P01", "P02", "P03")) -> dict[str, Any]:
    return {
        "schema": "agentrecx.p17.recommend_products.v1", "status": "finalized",
        "query": "earbuds", "selected_product_id": "P01",
        "selected_product": _info("P01")["selected_product"],
        "gate": _gate(tuple(item_ids)), "evidence_bundle": {}, "planner_steps": {},
        "search_mode": "bm25", "error": None, "latency_ms": 1.0,
    }


def _preferences(active: list[dict[str, Any]] | None = None, inactive: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "agentrecx.p17.get_active_preferences.v1", "status": "ok",
        "history_status": "active_preferences",
        "active_preferences": active or [],
        "inactive_preferences": inactive or [],
        "audit_source": "M9", "notes": [],
    }


def make_app(tmp_path: Path, manager: FakeManager) -> Any:
    settings = Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=tmp_path / "logs",
        behavior_db_path=tmp_path / "behavior.sqlite3",
        replay_path=REPLAY_PATH,
        mcp_timeout_s=30.0,
        python_executable=sys.executable,
    )
    return create_app(settings=settings, manager=manager, sessions=SessionStore())


def create_session(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/sessions", json={"profile_id": "profile_0049"})
    assert response.status_code == 200, response.text
    return response.json()


def parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: str | None = None
    data: list[str] = []
    for line in text.splitlines():
        if line.startswith(":") or not line:
            if current is not None and not line:
                events.append({"event": current, "data": json.loads("\n".join(data))})
                current, data = None, []
            continue
        if line.startswith("event: "):
            current = line[len("event: "):]
        elif line.startswith("data: "):
            data.append(line[len("data: "):])
    if current is not None:
        events.append({"event": current, "data": json.loads("\n".join(data))})
    return events


def send_query(client: TestClient, session_id: str, message: str = "wireless earbuds") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = client.post("/api/chat/stream", json={"session_id": session_id, "message": message})
    assert response.status_code == 200, response.text
    events = parse_sse(response.text)
    products = next(event["data"] for event in events if event["event"] == "products")
    return events, products


def setup_manager(item_ids: tuple[str, ...] = ("P01", "P02", "P03")) -> FakeManager:
    manager = FakeManager()
    manager.responses.update({
        "search_products": _search(item_ids),
        "get_product_info": lambda _profile, args: _info(args["product_ref"]),
        "recommend_products": _recommend(item_ids),
        "get_active_preferences": _preferences(),
    })
    return manager


def event(event_id: str, event_type: str, rec_id: str, product_id: str, position: int,
          *, source: str = "LIVE_USER_INTERACTION", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": event_id,
        "event_type": event_type,
        "recommendation_id": rec_id,
        "product_id": product_id,
        "displayed_position": position,
        "event_source": source,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def post_events(client: TestClient, session_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post("/api/behavior/events", json={"session_id": session_id, "events": events})
    assert response.status_code == 200, response.text
    return response.json()


def test_impression_validation_and_idempotency(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        positions = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        assert rec and positions == {"P01": 1, "P02": 2, "P03": 3}

        first = post_events(client, session["session_id"], [event("evt-imp-0001", "product_impression", rec, "P01", 1)])
        assert first["accepted_count"] == 1
        duplicate = post_events(client, session["session_id"], [event("evt-imp-0002", "product_impression", rec, "P01", 1)])
        assert duplicate["duplicate_count"] == 1
        wrong_position = post_events(client, session["session_id"], [event("evt-click-0001", "product_click", rec, "P02", 1)])
        assert wrong_position["rejected_count"] == 1
        assert wrong_position["rejected"][0]["code"] == "invalid_position"
        wrong_product = post_events(client, session["session_id"], [event("evt-click-0002", "product_click", rec, "NOPE", 1)])
        assert wrong_product["rejected"][0]["code"] == "unknown_product"
        wrong_rec = post_events(client, session["session_id"], [event("evt-click-0003", "product_click", "rec-other", "P01", 1)])
        assert wrong_rec["rejected"][0]["code"] == "unknown_recommendation"


def test_session_isolation_and_delete(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        first = create_session(client)
        second = create_session(client)
        _events_a, products_a = send_query(client, first["session_id"])
        _events_b, products_b = send_query(client, second["session_id"])
        rec_a, rec_b = products_a["recommendation_id"], products_b["recommendation_id"]
        assert rec_a != rec_b

        hijack = post_events(client, second["session_id"], [
            event("evt-hijack-01", "product_click", rec_a, "P01", 1)
        ])
        assert hijack["rejected_count"] == 1
        assert hijack["rejected"][0]["code"] == "unknown_recommendation"

        post_events(client, first["session_id"], [event("evt-a-01", "product_click", rec_a, "P03", 3)])
        post_events(client, second["session_id"], [event("evt-b-01", "product_click", rec_b, "P02", 2)])
        profile_a = client.get(f"/api/behavior/profile?session_id={first['session_id']}").json()
        profile_b = client.get(f"/api/behavior/profile?session_id={second['session_id']}").json()
        assert profile_a["counts"]["product_click"] == 1
        assert profile_b["counts"]["product_click"] == 1
        assert [item["product_id"] for item in profile_a["recent_viewed"]] == ["P03"]
        assert [item["product_id"] for item in profile_b["recent_viewed"]] == ["P02"]

        deleted = client.delete(f"/api/behavior?session_id={first['session_id']}").json()
        assert deleted["deleted_events"] == 1
        assert client.get(f"/api/behavior/profile?session_id={first['session_id']}").json()["counts"]["live_events"] == 0
        assert client.get(f"/api/behavior/profile?session_id={second['session_id']}").json()["counts"]["product_click"] == 1


def test_only_live_source_enters_profile(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        post_events(client, session["session_id"], [
            event("evt-live-0001", "product_click", rec, "P01", 1),
            event("evt-replay-01", "product_click", rec, "P02", 2, source="RECORDED_REPLAY"),
            event("evt-synth-001", "product_click", rec, "P03", 3, source="SYNTHETIC_TEST"),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["live_events"] == 1
        assert profile["counts"]["recorded_replay"] == 1
        assert profile["counts"]["synthetic_test"] == 1
        assert profile["source_counts"]["recorded_replay"] == 1
        attrs = {item["attribute"] for item in profile["attribute_interests"]}
        assert attrs == {"in_ear", "wireless"}  # only P01, never the replayed P02/P03
        enters = {item["event_id"]: item["enters_profile"] for item in profile["recent_events"]}
        assert enters["evt-live-0001"] is True
        assert enters["evt-replay-01"] is False
        assert enters["evt-synth-001"] is False


def test_attribute_interests_require_verified_evidence(tmp_path):
    manager = FakeManager()
    manager.responses.update({
        "search_products": _search(("P09",)),
        "get_product_info": lambda _profile, args: _info(args["product_ref"], evidence=False),
        "recommend_products": _recommend(("P09",)),
        "get_active_preferences": _preferences(),
    })
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        post_events(client, session["session_id"], [event("evt-noev-001", "product_click", rec, "P09", 1)])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["product_click"] == 1
        assert profile["attribute_interests"] == []
        assert profile["product_interests"][0]["product_id"] == "P09"


def test_no_behavior_data_makes_shadow_identical(tmp_path):
    manager = setup_manager()
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={products['recommendation_id']}"
        ).json()
        assert shadow["applied"] is False
        assert shadow["order_changed"] is False
        assert [row["product_id"] for row in shadow["base_ranking"]] == [
            row["product_id"] for row in shadow["behavior_ranking"]
        ]


def test_shadow_uses_point_in_time_events_and_can_move_tail(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 11))
    manager = setup_manager(item_ids)
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events1, products1 = send_query(client, session["session_id"])
        rec1 = products1["recommendation_id"]
        positions = {item["product_id"]: item["displayed_position"] for item in products1["items"]}
        assert positions["P10"] == 10

        # These events happen *after* recommendation 1; its shadow must ignore them.
        post_events(client, session["session_id"], [
            event("evt-tail-001", "product_click", rec1, "P10", 10),
            event("evt-tail-002", "product_click", rec1, "P10", 10),
            event("evt-tail-003", "explicit_item_feedback", rec1, "P10", 10, metadata={"feedback": "like"}),
        ])
        old_shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={rec1}"
        ).json()
        assert old_shadow["applied"] is False
        assert old_shadow["rows"][-1]["behavior_contribution"] == 0

        _events2, products2 = send_query(client, session["session_id"], "more earbuds")
        rec2 = products2["recommendation_id"]
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={rec2}"
        ).json()
        assert shadow["enabled"] is True
        assert shadow["applied"] is True
        assert shadow["order_changed"] is True
        row10 = next(row for row in shadow["rows"] if row["product_id"] == "P10")
        assert row10["behavior_contribution"] > 0
        assert row10["shadow_rank"] < row10["base_rank"]


def test_explicit_negative_preference_suppresses_behavior(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 11))
    manager = setup_manager(item_ids)
    negative = [{
        "memory_id": "m_avoid_special", "kind": "feature", "value": "special_p10",
        "polarity": "avoid", "status": "active", "logical_seq": 1,
        "source_turn_id": "t1", "source_text": "avoid special p10", "created_at": 1.0,
    }]
    manager.responses["get_active_preferences"] = _preferences(negative)
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events1, products1 = send_query(client, session["session_id"])
        rec1 = products1["recommendation_id"]
        post_events(client, session["session_id"], [
            event("evt-neg-0001", "product_click", rec1, "P10", 10),
            event("evt-neg-0002", "product_click", rec1, "P10", 10),
            event("evt-neg-0003", "explicit_item_feedback", rec1, "P10", 10, metadata={"feedback": "like"}),
        ])
        _events2, products2 = send_query(client, session["session_id"], "more earbuds")
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={products2['recommendation_id']}"
        ).json()
        row10 = next(row for row in shadow["rows"] if row["product_id"] == "P10")
        assert row10["behavior_contribution"] == 0
        assert row10["suppressed_reason"] == "explicit_negative_preference"
        assert shadow["active_preferences"][0]["polarity"] == "negative"


def test_disabled_session_rejects_events_and_hides_profile(tmp_path):
    manager = setup_manager()
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        disabled = client.post("/api/behavior/enabled", json={"session_id": session["session_id"], "enabled": False})
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        rejected = client.post(
            "/api/behavior/events",
            json={"session_id": session["session_id"], "events": [
                event("evt-off-0001", "product_click", rec, "P01", 1),
            ]},
        )
        assert rejected.status_code == 409
        assert rejected.json()["code"] == "behavior_disabled"
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["enabled"] is False
        assert profile["counts"]["live_events"] == 0


def test_suggestion_requires_confirmation_and_never_calls_m9_writer(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    manager.responses["remember_preference"] = {
        "schema": "agentrecx.p52.remember_preference.v1", "status": "ok",
        "update": {"added": 1, "superseded": 0, "removed": 0, "skipped_duplicates": 0},
        "active_preferences": [], "inactive_preferences": [],
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        post_events(client, session["session_id"], [
            event("evt-sug-0001", "product_click", rec, "P01", 1),
            event("evt-sug-0002", "product_click", rec, "P01", 1),
            event("evt-sug-0003", "product_click", rec, "P02", 2),
            event("evt-sug-0004", "product_click", rec, "P02", 2),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        suggestion = profile["suggestion"]
        assert suggestion is not None
        assert suggestion["requires_confirmation"] is True
        assert suggestion["inferred_from_behavior"] is True
        assert not any(call[1] == "remember_preference" for call in manager.calls)

        written = client.post(
            "/api/memory/remember",
            json={"session_id": session["session_id"], "message": suggestion["message"]},
        )
        assert written.status_code == 200, written.text
        assert any(call[1] == "remember_preference" for call in manager.calls)


def test_hard_constraint_presentable_set_is_never_changed(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 11))
    manager = setup_manager(item_ids)
    # Only P01 is presentable in this gate.  The shadow may re-order the already-displayed
    # pool but must copy the Gate selection and eligibility flags unchanged.
    manager.responses["recommend_products"] = lambda _profile, _args: {
        **_recommend(item_ids), "gate": _gate(("P01",))
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events1, products1 = send_query(client, session["session_id"])
        rec1 = products1["recommendation_id"]
        post_events(client, session["session_id"], [
            event("evt-hard-0001", "product_click", rec1, "P10", 10),
            event("evt-hard-0002", "product_click", rec1, "P10", 10),
        ])
        _events2, products2 = send_query(client, session["session_id"], "more earbuds")
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={products2['recommendation_id']}"
        ).json()
        row10 = next(row for row in shadow["rows"] if row["product_id"] == "P10")
        assert shadow["gate"]["selected_candidate_id"] == "P01"
        assert shadow["gate"]["candidate_presentable_ids"] == ["P01"]
        assert shadow["hard_constraint_violations"] == 0
        assert shadow["eligibility_unchanged"] is True
        assert row10["gate_eligible"] is False
        assert row10["recommended"] is False
        assert row10["behavior_contribution"] > 0  # presentation-only shadow, not a Gate change
