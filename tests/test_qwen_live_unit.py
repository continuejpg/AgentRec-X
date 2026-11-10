"""Qwen Base Live config/health contract tests (no model: test HTTP double)."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from web_demo.config import Settings, load_settings, modes_public
from web_demo.qwen_health import QwenHealthMonitor


class _PlannerDouble(BaseHTTPRequestHandler):
    action = "RETRIEVE"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path != "/plan":
            self.send_error(404)
            return
        action = self.action
        arguments = {"query": body.get("query") or "wireless earbuds"}
        payload = {
            "payload": {"action": action, "arguments": arguments, "rationale": "test double"},
            "text": json.dumps({"action": action}),
            "errors": [],
            "latency_ms": 12.5,
            "input_tokens": 10,
            "output_tokens": 5,
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def _server(action: str = "RETRIEVE") -> tuple[ThreadingHTTPServer, str, type]:
    handler = type("Handler", (_PlannerDouble,), {"action": action})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", handler


def test_qwen_settings_default_disabled_and_aliases():
    settings = load_settings({})
    assert settings.qwen_live_enabled is False
    assert settings.qwen_live_configured is False
    modes = modes_public(cpu_live_available=True)
    assert modes["cpu_live"]["available"] is True
    assert modes["qwen_live"]["available"] is False
    assert "No live Qwen service detected" in modes["qwen_live"]["disabled_reason"]

    enabled = load_settings({
        "QWEN_LIVE_ENABLED": "1",
        "QWEN_MODEL_ENDPOINT": "http://127.0.0.1:9",
        "QWEN_MODEL_NAME": "Qwen3.5-9B-Base",
        "QWEN_LIVE_MAX_STEPS": "6",
    })
    assert enabled.qwen_live_configured is True
    assert enabled.qwen_max_steps == 6
    assert enabled.qwen_model_endpoint == "http://127.0.0.1:9"


def test_health_requires_real_planner_action_and_does_not_leak_endpoint():
    server, endpoint, _ = _server("RETRIEVE")
    try:
        settings = Settings(qwen_live_enabled=True, qwen_model_endpoint=endpoint)
        monitor = QwenHealthMonitor(settings)
        result = asyncio.run(monitor.check())
        assert result["available"] is True
        assert result["status"] == "available"
        assert result["planner_action"] == "RETRIEVE"
        assert result["latency_ms"] >= 0
        serialized = json.dumps(result)
        assert endpoint not in serialized
    finally:
        server.shutdown()
        server.server_close()


def test_health_rejects_invalid_or_missing_action():
    server, endpoint, _ = _server("NOT_A_PLANNER_ACTION")
    try:
        settings = Settings(qwen_live_enabled=True, qwen_model_endpoint=endpoint)
        monitor = QwenHealthMonitor(settings)
        result = asyncio.run(monitor.check())
        assert result["available"] is False
        assert result["status"] == "unavailable"
        assert "PlannerAction" not in (result["detail"] or "") or "invalid" in (result["detail"] or "")
        modes = modes_public(cpu_live_available=True, qwen_health=result)
        assert modes["qwen_live"]["available"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_health_unreachable_is_unavailable_not_rule():
    settings = Settings(
        qwen_live_enabled=True,
        qwen_model_endpoint="http://127.0.0.1:9",
        qwen_health_timeout_s=0.2,
    )
    monitor = QwenHealthMonitor(settings)
    result = asyncio.run(monitor.check())
    assert result["available"] is False
    assert result["status"] == "unavailable"
    assert result["planner_action"] is None
    assert "No live Qwen service detected" in modes_public(
        cpu_live_available=True, qwen_health=result
    )["qwen_live"]["disabled_reason"]
