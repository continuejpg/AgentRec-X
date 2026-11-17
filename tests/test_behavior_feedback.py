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
from web_demo.behavior import (
    BehaviorStore,
    build_behavior_profile,
    build_recommendation_snapshot,
    build_shadow,
)
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


def test_human_and_automated_browser_sources_enter_profile_with_provenance(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        post_events(client, session["session_id"], [
            event("evt-live-0001", "product_click", rec, "P01", 1),
            event("evt-auto-0001", "product_click", rec, "P01", 1, source="AUTOMATED_BROWSER_E2E"),
            event("evt-replay-01", "product_click", rec, "P02", 2, source="RECORDED_REPLAY"),
            event("evt-synth-001", "product_click", rec, "P03", 3, source="SYNTHETIC_TEST"),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["live_events"] == 2
        assert profile["counts"]["human_live_events"] == 1
        assert profile["counts"]["automated_browser_e2e_events"] == 1
        assert profile["counts"]["recorded_replay"] == 1
        assert profile["counts"]["synthetic_test"] == 1
        assert profile["source_counts"]["automated_browser_e2e"] == 1
        assert profile["source_counts"]["recorded_replay"] == 1
        assert profile["profile_provenance"]["human_preference_statistics"] == "LIVE_USER_INTERACTION only"
        attrs = {item["attribute"] for item in profile["attribute_interests"]}
        assert attrs == {"in_ear", "wireless"}  # only P01, never the replayed P02/P03
        enters = {item["event_id"]: item["enters_profile"] for item in profile["recent_events"]}
        assert enters["evt-live-0001"] is True
        assert enters["evt-auto-0001"] is True
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
        assert row10["behavior_contribution"] > 0  # diagnostic only
        assert row10["eligible_behavior_contribution"] == 0
        assert shadow["eligible_candidate_count"] == 1
        assert [row["product_id"] for row in shadow["eligible_base_ranking"]] == ["P01"]
        assert "P10" not in shadow["eligible_ranking_ids"]

def test_impression_is_exposure_context_with_zero_interest(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        post_events(client, session["session_id"], [
            event("evt-ctx-0001", "product_impression", rec, "P01", pos["P01"]),
            event("evt-ctx-0002", "product_impression", rec, "P02", pos["P02"]),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["product_impression"] == 2
        assert profile["counts"]["product_click"] == 0
        assert profile["product_interests"] == []
        assert profile["attribute_interests"] == []
        assert profile["recent_impressions"][0]["clicked"] is False
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={rec}"
        ).json()
        assert shadow["behavior_signal_present"] is False
        assert shadow["eligible_behavior_applied"] is False
        assert shadow["diagnostic_behavior_applied"] is False
        assert [row["product_id"] for row in shadow["eligible_base_ranking"]] == [
            row["product_id"] for row in shadow["eligible_behavior_ranking"]
        ]


def test_click_weak_interest_like_and_dislike_are_item_level(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        post_events(client, session["session_id"], [
            event("evt-click-1001", "product_click", rec, "P01", pos["P01"]),
            event("evt-like-1002", "explicit_item_feedback", rec, "P02", pos["P02"], metadata={"feedback": "like"}),
            event("evt-dislike-1003", "explicit_item_feedback", rec, "P03", pos["P03"], metadata={"feedback": "dislike"}),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["product_click"] == 1
        assert profile["counts"]["explicit_like"] == 1
        assert profile["counts"]["explicit_dislike"] == 1
        scores = {item["product_id"]: item["score"] for item in profile["product_interests"]}
        assert scores["P01"] > 0
        assert scores["P02"] > scores["P01"]
        assert scores["P03"] < 0
        assert profile["recent_viewed"][0]["product_id"] == "P01"
        # Attribute interests are positive only and carry source/provenance.
        assert profile["attribute_interests"]
        assert "product_click" in profile["attribute_interests"][0]["source_event_types"]
        assert profile["attribute_interests"][0]["evidence"][0]["event_type"] in {"product_click", "explicit_item_feedback"}


def test_compare_lifecycle_attempt_completed_failed_are_distinct(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        base = {"compared_product_ids": ["P01", "P02"], "compare_id": "cmp-v11-001", "action_status": "attempted"}
        post_events(client, session["session_id"], [
            event("evt-cmp-1001", "product_compare", rec, "P01", pos["P01"], metadata=base),
        ])
        post_events(client, session["session_id"], [
            event("evt-cmp-1002", "product_compare", rec, "P01", pos["P01"], metadata={**base, "action_status": "succeeded"}),
            event("evt-cmp-1003", "product_compare", rec, "P02", pos["P02"],
                  metadata={"compared_product_ids": ["P02", "P03"], "compare_id": "cmp-v11-002",
                            "action_status": "failed", "failure_code": "role_mismatch",
                            "failure_message": "accessory cannot be compared as main product"}),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        counts = profile["counts"]
        assert counts["product_compare_attempted"] == 1
        assert counts["product_compare_completed"] == 1
        assert counts["product_compare_failed"] == 1
        assert profile["product_interests"] == []
        assert profile["attribute_interests"] == []
        assert {item["status"] for item in profile["recent_compared"]} == {"failed", "succeeded"}
        failure = next(item for item in profile["recent_compared"] if item["status"] == "failed")
        assert failure["failure_code"] == "role_mismatch"
        assert failure["failure_message"]
        events = {item["event_id"]: item for item in profile["recent_events"]}
        assert events["evt-cmp-1002"]["metadata"]["action_status"] == "succeeded"
        assert events["evt-cmp-1003"]["metadata"]["failure_code"] == "role_mismatch"


def test_compare_attempt_does_not_create_interest(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        post_events(client, session["session_id"], [
            event("evt-cmp-2001", "product_compare", rec, "P01", pos["P01"],
                  metadata={"compared_product_ids": ["P01", "P02"], "compare_id": "cmp-attempt-1",
                            "action_status": "attempted"}),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["product_compare_attempted"] == 1
        assert profile["product_interests"] == []
        assert profile["attribute_interests"] == []
        assert profile["recent_compared"][0]["status"] == "attempted"


def test_repeated_clicks_are_capped(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        post_events(client, session["session_id"], [
            event(f"evt-loop-{index:04d}", "product_click", rec, "P01", pos["P01"])
            for index in range(5)
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        assert profile["counts"]["product_click"] == 5
        assert profile["product_interests"][0]["counted_events"] == 2
        assert profile["product_interests"][0]["score"] <= 2.0


def test_eligible_ranking_excludes_ineligible_and_diagnostic_shows_signal(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 11))
    manager = setup_manager(item_ids)
    manager.responses["recommend_products"] = lambda _profile, _args: {
        **_recommend(item_ids), "gate": _gate(("P01", "P02"))
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events1, products1 = send_query(client, session["session_id"])
        rec1 = products1["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products1["items"]}
        post_events(client, session["session_id"], [
            event("evt-inelig-0001", "product_click", rec1, "P10", pos["P10"]),
            event("evt-inelig-0002", "product_click", rec1, "P10", pos["P10"]),
            event("evt-inelig-0003", "explicit_item_feedback", rec1, "P10", pos["P10"], metadata={"feedback": "like"}),
        ])
        _events2, products2 = send_query(client, session["session_id"], "more")
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={products2['recommendation_id']}"
        ).json()
        assert shadow["eligible_candidate_count"] == 2
        assert shadow["eligible_candidate_ids"] == ["P01", "P02"]
        assert [row["product_id"] for row in shadow["eligible_base_ranking"]] == ["P01", "P02"]
        assert all(row["product_id"] not in {"P03", "P10"} for row in shadow["eligible_behavior_ranking"])
        row10 = next(row for row in shadow["diagnostic_candidates"] if row["product_id"] == "P10")
        assert row10["gate_eligible"] is False
        assert row10["behavior_contribution"] > 0
        assert any(item["product_id"] == "P10" for item in shadow["ineligible_candidates_with_behavior_signal"])
        assert shadow["eligible_ranking_changed"] is False
        assert shadow["diagnostic_behavior_applied"] is True
        assert shadow["gate_context"]["eligible_candidate_count"] == 2


def test_gate_not_finalizable_has_no_eligible_ranking(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 4))
    payload = {
        "recommendation_id": "rec-unit-001",
        "ranker_version": "existing_ranking_v1",
        "query": "unit",
        "items": [
            {"product_id": item_id, "displayed_position": index, "title": item_id, "position": index,
             "verified_attributes": [{"attribute": f"attr_{item_id}", "value": "true", "evidence_id": f"pev_{item_id}"}]}
            for index, item_id in enumerate(item_ids, start=1)
        ],
        "gate": {"action": "ASK_CLARIFICATION", "allow_finalize": False,
                 "selected_candidate_id": None, "candidate_presentable_ids": ["P01", "P02"]},
    }
    snapshot = build_recommendation_snapshot(payload)
    profile = build_behavior_profile([], session_id="web-unit", enabled=True, now_ts=1000.0,
                                     reference_ts=1000.0, window_s=3600, half_life_s=1800)
    shadow = build_shadow(snapshot=snapshot, profile=profile, enabled=True)
    assert shadow["eligible_candidate_count"] == 0
    assert shadow["eligible_recommendation_available"] is False
    assert shadow["eligible_base_ranking"] == []
    assert shadow["eligible_behavior_ranking"] == []
    assert shadow["gate_context"]["no_eligible_reason"] == "gate_action_does_not_allow_finalize"
    assert shadow["diagnostic_ranking"]


def test_time_decay_halves_a_click_score(tmp_path):
    store = BehaviorStore(tmp_path / "behavior.sqlite3", clock=lambda: 1000.0)
    try:
        payload = {
            "recommendation_id": "rec-decay-001",
            "ranker_version": "existing_ranking_v1",
            "query": "unit",
            "items": [{"product_id": "P01", "displayed_position": 1, "title": "P01", "position": 1,
                       "verified_attributes": [{"attribute": "attr", "value": "true", "evidence_id": "pev_1"}]}],
            "gate": {"action": "ALLOW_FINALIZE", "allow_finalize": True,
                     "selected_candidate_id": "P01", "candidate_presentable_ids": ["P01"]},
        }
        snapshot = build_recommendation_snapshot(payload, now_ts=1000.0)
        store.record("web-decay", [
            {"event_id": "evt-decay-0001", "event_type": "product_click", "event_source": "LIVE_USER_INTERACTION",
             "recommendation_id": "rec-decay-001", "product_id": "P01", "displayed_position": 1,
             "metadata": {}},
        ], contexts={"rec-decay-001": snapshot}, enabled=True)
        immediate = store.profile("web-decay", now=1000.0)
        after_half_life = store.profile("web-decay", now=1000.0 + 1800.0)
        assert immediate["product_interests"][0]["score"] == 1.0
        assert after_half_life["product_interests"][0]["score"] == 0.5
    finally:
        store.close()


def test_withdrawn_preference_is_not_resurrected_by_clicks(tmp_path):
    item_ids = tuple(f"P{index:02d}" for index in range(1, 11))
    manager = setup_manager(item_ids)
    inactive = [{
        "memory_id": "m_withdrawn", "kind": "feature", "value": "special_p10",
        "polarity": "prefer", "status": "removed", "logical_seq": 1,
        "source_turn_id": "t1", "source_text": "withdrawn", "created_at": 1.0,
    }]
    manager.responses["get_active_preferences"] = _preferences(inactive=inactive)
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events1, products1 = send_query(client, session["session_id"])
        rec1 = products1["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products1["items"]}
        post_events(client, session["session_id"], [
            event("evt-wd-0001", "product_click", rec1, "P10", pos["P10"]),
            event("evt-wd-0002", "product_click", rec1, "P10", pos["P10"]),
        ])
        _events2, products2 = send_query(client, session["session_id"], "more")
        shadow = client.get(
            f"/api/behavior/shadow?session_id={session['session_id']}&recommendation_id={products2['recommendation_id']}"
        ).json()
        row10 = next(row for row in shadow["diagnostic_candidates"] if row["product_id"] == "P10")
        assert row10["gate_eligible"] is True
        assert row10["eligible_behavior_contribution"] == 0
        assert row10["suppressed_reason"] == "withdrawn_preference_not_restored"
        assert shadow["eligible_ranking_changed"] is False

def test_compare_passes_session_query_to_adapter(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    manager.responses["compare_products"] = {
        "schema": "agentrecx.p16.shopping_comparison.v1",
        "status": "invalid_product_reference",
        "current_query": "wireless earbuds",
        "products": [],
        "errors": [{"code": "role_mismatch", "item_id": "P01", "message": "role mismatch"}],
        "conditional_best_supported": None,
        "final_answer": "cannot compare",
        "qwen_used": False,
        "tool_calls": [], "claims_valid": 0, "claims_rejected": 0,
        "unknown_count": 0, "latency_ms": 0.1, "notes": [],
    }
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        send_query(client, session["session_id"], "I need wireless earbuds for the gym.")
        client.post("/api/products/compare", json={
            "session_id": session["session_id"], "product_refs": ["P01", "P02"],
        })
    calls = [call for call in manager.calls if call[1] == "compare_products"]
    assert calls
    assert calls[-1][2]["query"] == "I need wireless earbuds for the gym."

def test_compare_result_is_not_downgraded_by_late_attempt(tmp_path):
    manager = setup_manager(("P01", "P02", "P03"))
    with TestClient(make_app(tmp_path, manager)) as client:
        session = create_session(client)
        _events, products = send_query(client, session["session_id"])
        rec = products["recommendation_id"]
        pos = {item["product_id"]: item["displayed_position"] for item in products["items"]}
        # The result event id sorts before the attempt id at the same server timestamp.
        post_events(client, session["session_id"], [
            event("evt-cmp-0001", "product_compare", rec, "P01", pos["P01"],
                  metadata={"compared_product_ids": ["P01", "P02"], "compare_id": "cmp-order-1",
                            "action_status": "succeeded"}),
            event("evt-cmp-9999", "product_compare", rec, "P01", pos["P01"],
                  metadata={"compared_product_ids": ["P01", "P02"], "compare_id": "cmp-order-1",
                            "action_status": "attempted"}),
        ])
        profile = client.get(f"/api/behavior/profile?session_id={session['session_id']}").json()
        compared = profile["recent_compared"][0]
        assert compared["compare_id"] == "cmp-order-1"
        assert compared["status"] == "succeeded"
        assert profile["counts"]["product_compare_attempted"] == 1
        assert profile["counts"]["product_compare_completed"] == 1
