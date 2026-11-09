from __future__ import annotations

import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROPOSALS = {
    "I need a water filter under $30 and I prefer the lighter option.": {
        "schema_version": "agentrecx.preference_operation.v1", "clarification": None,
        "operations": [
            {"op": "set_goal", "attribute": "product_type", "value": "water_filter", "direction": None, "scope": "task", "evidence": "water filter"},
            {"op": "update_constraint", "attribute": "price_max", "value": 30, "direction": None, "scope": "task", "evidence": "$30"},
            {"op": "add_preference", "attribute": "weight", "value": None, "direction": "minimize", "scope": "task", "evidence": "lighter option"},
        ],
    },
    "Actually make the budget $50 and forget the lighter preference.": {
        "schema_version": "agentrecx.preference_operation.v1", "clarification": None,
        "operations": [
            {"op": "update_constraint", "attribute": "price_max", "value": 50, "direction": None, "scope": "task", "evidence": "$50"},
            {"op": "retract_preference", "attribute": "weight", "value": None, "direction": None, "scope": "task", "evidence": "forget the lighter preference"},
        ],
    },
    "Remove the budget requirement.": {
        "schema_version": "agentrecx.preference_operation.v1", "clarification": None,
        "operations": [{"op": "retract_constraint", "attribute": "price_max", "value": None, "direction": None, "scope": "task", "evidence": "Remove the budget requirement."}],
    },
    "Show more options.": {
        "schema_version": "agentrecx.preference_operation.v1", "clarification": None,
        "operations": [{"op": "noop", "attribute": None, "value": None, "direction": None, "scope": None, "evidence": "Show more options.", "reason": "no_change"}],
    },
}


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        current = str(json.loads(raw.decode()).get("current") or "")
        payload = PROPOSALS.get(current, {"schema_version": "agentrecx.preference_operation.v1", "clarification": None, "operations": []})
        body = json.dumps({"payload": payload, "text": "stub", "errors": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def test_four_turn_budget_reaches_r1_and_eligibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AGENTRECX_QWEN_PROPOSAL", "on")
    monkeypatch.setenv("AGENTRECX_QWEN_PROPOSAL_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("AGENTRECX_STATE_MODE", "apply_constraints")
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    monkeypatch.setenv("AGENTRECX_M5_SOFT_RERANK", "on")
    monkeypatch.setenv("AGENTRECX_STATE_DB", str(tmp_path / "state.sqlite3"))
    from recommendation.api.app import create_app

    app = create_app(enable_demo=True)
    try:
        with TestClient(app) as client:
            profile_id = client.get("/v1/demo/profiles").json()["profiles"][0]["profile_id"]
            session_id = client.post("/v1/demo/sessions", json={"profile_id": profile_id}).json()["session_id"]
            service = app.state.demo_runtime.agent_service()
            seen = []
            for message in PROPOSALS:
                response = client.post(f"/v1/demo/sessions/{session_id}/agent/shopping", json={"message": message, "k": 5})
                assert response.status_code == 200, response.text
                task_id = app.state.demo_runtime.sessions.get(session_id).session_id
                trace = service.constraint_trace(task_id) or {}
                actual = (trace.get("r1_actual") or {}).get("constraints")
                seen.append(actual)
            assert seen[0] == [["price_max", "30"]]
            assert seen[1] == [["price_max", "50"]]
            assert not seen[2]
            assert seen[3] in ([], None)
    finally:
        server.shutdown()
