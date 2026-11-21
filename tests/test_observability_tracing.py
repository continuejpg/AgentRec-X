"""M3-A observability regression tests.

These tests exercise the tracing *boundary*, not a vendor UI: safe attributes, exception and
exporter containment, HTTP request spans, and W3C context propagation into an MCP request.
The real Phoenix/stdio E2E is intentionally a separate opt-in test because it needs a running
collector and the real 76k-record catalogue.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
from opentelemetry.sdk.trace.export import SpanExporter  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import tests.test_web_demo_unit as twu  # noqa: E402
from tests.test_web_cold_resume_e2e import (  # noqa: E402
    REPLY_TEXT,
    _ask,
    _create,
    _durable_result,
    _fake_settings,
    _reply,
)
from web_demo.state_store import SQLiteWebStateStore  # noqa: E402
from observability import (  # noqa: E402
    Tracing,
    configure_tracing,
    set_active_tracing,
)
from web_demo.app import create_app  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.mcp_manager import MCPManager, MCPToolError  # noqa: E402


@pytest.fixture()
def exported() -> tuple[Tracing, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracing = configure_tracing(
        enabled=True,
        project_name="agentrecx-test",
        service_name="agentrecx-test-bff",
        span_exporter=exporter,
        use_batch=False,
    )
    try:
        yield tracing, exporter
    finally:
        tracing.shutdown()
        set_active_tracing(Tracing())


class _FailingExporter(SpanExporter):
    def export(self, spans):  # noqa: ANN001 - protocol signature
        raise RuntimeError("collector is unavailable")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        raise RuntimeError("collector is unavailable")


def test_disabled_tracing_is_a_noop_without_sdk_side_effects() -> None:
    tracing = Tracing()
    assert tracing.health()["enabled"] is False
    assert tracing.current_trace_id() is None
    carrier: dict[str, object] = {}
    tracing.inject(carrier)
    assert carrier == {}
    with tracing.span("never-exported", attributes={"secret": "not recorded"}) as span:
        span.set_attribute("any", "value")
        assert span.is_recording() is False
    assert tracing.health()["enabled"] is False


def test_safe_scalar_attributes_and_exception_type_only(
    exported: tuple[Tracing, InMemorySpanExporter],
) -> None:
    tracing, exporter = exported

    class BusinessFailure(RuntimeError):
        pass

    with pytest.raises(BusinessFailure):
        with tracing.span(
            "business",
            attributes={
                "agentrecx.session_id": "web-safe",
                "nested": {"message": "must be omitted"},
                "numbers": [1, 2, 3],
            },
        ) as span:
            span.set_attribute("agentrecx.pending_id", "web-pending")
            raise BusinessFailure("user text must not be exported")

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes or {})
    assert attrs["agentrecx.session_id"] == "web-safe"
    assert attrs["agentrecx.pending_id"] == "web-pending"
    assert attrs["numbers"] == (1, 2, 3)
    assert attrs["error.type"] == "BusinessFailure"
    assert "nested" not in attrs
    assert "user text must not be exported" not in json.dumps(attrs, default=str)
    assert finished[0].status.status_code.name == "ERROR"


def test_exporter_failure_does_not_break_the_business_block() -> None:
    tracing = configure_tracing(
        enabled=True,
        project_name="agentrecx-test",
        span_exporter=_FailingExporter(),
        use_batch=False,
    )
    try:
        result: list[str] = []
        with tracing.span("business") as span:
            result.append("ok")
            span.set_attribute("business.ok", True)
        assert result == ["ok"]
        # force_flush/shutdown must also contain the failing collector.
        tracing.shutdown()
        assert tracing.health()["enabled"] is False
    finally:
        set_active_tracing(Tracing())


def test_fastapi_request_span_records_safe_request_and_session_ids(
    exported: tuple[Tracing, InMemorySpanExporter], tmp_path: Path
) -> None:
    tracing, exporter = exported
    manager = twu.warm_reply_manager()
    app = create_app(
        settings=Settings(
            log_dir=tmp_path / "logs",
            web_state_db_path=tmp_path / "state.sqlite3",
        ),
        manager=manager,
        tracing=tracing,
    )
    with TestClient(app) as client:
        config = client.get("/api/demo/config")
        assert config.status_code == 200
        request_id = config.headers["X-AgentRecX-Request-Id"]
        created = client.post("/api/sessions", json={"profile_id": "profile_0057"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

    finished = exporter.get_finished_spans()
    server_spans = [span for span in finished if span.kind.name == "SERVER"]
    assert server_spans
    config_span = next(
        span for span in server_spans
        if span.attributes.get("http.route") == "/api/demo/config"
    )
    assert config_span.attributes["agentrecx.http_request_id"] == request_id
    assert "http.target" not in config_span.attributes or "secret" not in str(
        config_span.attributes["http.target"]
    )
    assert not any(
        span.attributes.get("http.route") == "/api/health" for span in server_spans
    )
    session_span = next(span for span in server_spans if span.attributes.get("http.route") == "/api/sessions")
    assert session_span.attributes["agentrecx.session_id"] == session_id
    # Request IDs are random and safe; no query or body is captured by the hook.
    assert session_span.attributes["agentrecx.request_kind"] == "/api/sessions"
    assert "gaming laptop under 8000" not in json.dumps(
        session_span.attributes, default=str
    )


def test_warm_reply_is_labelled_warm_and_parented_to_http(
    exported: tuple[Tracing, InMemorySpanExporter], tmp_path: Path
) -> None:
    tracing, exporter = exported
    manager = twu.warm_reply_manager()
    app = create_app(
        settings=Settings(
            allow_memory_writes=True,
            log_dir=tmp_path / "logs",
            web_state_db_path=tmp_path / "state.sqlite3",
        ),
        manager=manager,
        tracing=tracing,
    )
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={"profile_id": "profile_0057"})
        session_id = created.json()["session_id"]
        ask = client.post(
            "/api/chat/stream",
            json={
                "session_id": session_id,
                "message": "I need some new headphones. Not sure what matters yet.",
            },
        )
        clarification = next(
            event["data"] for event in twu.parse_sse(ask.text)
            if event["event"] == "clarification"
        )
        reply = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session_id,
                "clarification_id": clarification["clarification_id"],
                "message": "I prefer noise cancelling.",
            },
        )
        assert reply.status_code == 200, reply.text

    spans = list(exporter.get_finished_spans())
    reply_span = next(span for span in spans if span.name == "agent.reply")
    reply_http = next(
        span for span in spans
        if span.kind.name == "SERVER"
        and span.attributes.get("http.route") == "/api/clarification/reply"
    )
    assert reply_span.attributes["agentrecx.reply.path"] == "warm"
    assert reply_span.parent is not None
    assert reply_span.parent.span_id == reply_http.context.span_id
    chat_span = next(span for span in spans if span.name == "agent.chat")
    chat_http = next(
        span for span in spans
        if span.kind.name == "SERVER"
        and span.attributes.get("http.route") == "/api/chat/stream"
    )
    assert chat_span.parent is not None
    assert chat_span.parent.span_id == chat_http.context.span_id


def test_result_durable_finalize_is_labelled_without_tool_spans(
    exported: tuple[Tracing, InMemorySpanExporter], tmp_path: Path
) -> None:
    tracing, exporter = exported
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]

    store = SQLiteWebStateStore(tmp_path / "state.sqlite3")
    try:
        pending = store.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None
        claim = store.claim_reply_request(
            session_id, pending.pending_id,
            payload={
                "message": REPLY_TEXT,
                "confirmed_by_user": True,
                "confirmation_source": "web_ui_explicit_user_reply",
            },
        )
        store.mark_request_external_started(
            session_id, pending.reply_turn_id, claim_token=claim.claim_token
        )
        store.checkpoint_reply_result(
            session_id, pending.reply_turn_id, claim_token=claim.claim_token,
            result=_durable_result(pending),
        )
        raw = sqlite3.connect(str(tmp_path / "state.sqlite3"))
        try:
            raw.execute("UPDATE web_request SET lease_expires_at = 0")
            raw.commit()
        finally:
            raw.close()
    finally:
        store.close()

    manager2 = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager2)
    app2 = create_app(
        settings=_fake_settings(tmp_path), manager=manager2, tracing=tracing,
    )
    with TestClient(app2) as client2:
        response = _reply(client2, session_id, clarification_id)
    assert response.status_code == 200, response.text

    spans = list(exporter.get_finished_spans())
    reply_span = next(span for span in spans if span.name == "agent.reply")
    assert reply_span.attributes["agentrecx.reply.path"] == "checkpoint_finalize"
    assert reply_span.attributes["agentrecx.used_checkpoint"] is True
    finalize_span = next(span for span in spans if span.name == "reply.finalize")
    assert finalize_span.attributes["agentrecx.finalize.source"] == "checkpoint"
    assert not [
        span for span in spans
        if str(span.name).startswith(("tools/call", "mcp.call"))
    ], "a durable checkpoint must finalize without any MCP tool span"


def test_ambiguous_failure_is_observable_and_stays_a_503(
    exported: tuple[Tracing, InMemorySpanExporter], tmp_path: Path
) -> None:
    tracing, exporter = exported
    manager = twu.warm_reply_manager()
    app = create_app(
        settings=Settings(
            allow_memory_writes=True,
            log_dir=tmp_path / "logs",
            web_state_db_path=tmp_path / "state.sqlite3",
        ),
        manager=manager,
        tracing=tracing,
    )
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        manager.errors["recommend_products"] = MCPToolError(
            "post-write recommendation failed", tool="recommend_products",
        )
        response = _reply(client, session_id, clarification_id)
        assert response.status_code == 503
        assert response.json()["code"] == "clarification_reply_ambiguous"

    spans = list(exporter.get_finished_spans())
    reply_span = next(span for span in spans if span.name == "agent.reply")
    assert reply_span.attributes["agentrecx.reply.path"] == "warm_attempt"
    assert reply_span.attributes["agentrecx.reply.outcome"] == "ambiguous"
    assert not any(span.name == "reply.checkpoint" for span in spans)
    assert not any(span.name == "reply.finalize" for span in spans)


def test_mcp_call_injects_traceparent_into_request_meta(
    exported: tuple[Tracing, InMemorySpanExporter], tmp_path: Path
) -> None:
    tracing, exporter = exported

    class RecordingClient:
        def __init__(self) -> None:
            self.meta: dict | None = None
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, tool, arguments, progress_callback=None, meta=None):  # noqa: ANN001
            self.meta = dict(meta or {})
            self.calls.append((tool, dict(arguments or {})))
            return SimpleNamespace(
                is_error=False,
                structured_content={"status": "ok", "hits": []},
                content=[],
            )

    manager = MCPManager(
        Settings(log_dir=tmp_path / "logs", repo=REPO_ROOT),
        tracing=tracing,
    )
    runtime = manager.runtimes["profile_0057"]
    client = RecordingClient()
    runtime.client = client
    runtime.status = "ready"
    runtime.process_instance_id = "mcp-process-a"
    runtime.generation = 3

    import asyncio

    asyncio.run(manager.call_tool(
        "profile_0057",
        "search_products",
        {"query": "wireless headphones", "top_k": 3, "session_id": "web-safe"},
    ))

    assert client.calls == [("search_products", {"query": "wireless headphones", "top_k": 3, "session_id": "web-safe"})]
    assert client.meta is not None
    assert client.meta["traceparent"].startswith("00-")
    assert "wireless headphones" not in json.dumps(client.meta)

    client_spans = [
        span for span in exporter.get_finished_spans()
        if span.name == "mcp.call search_products"
    ]
    assert len(client_spans) == 1
    span = client_spans[0]
    assert span.attributes["mcp.tool.name"] == "search_products"
    assert span.attributes["mcp.tool.success"] is True
    assert span.attributes["mcp.argument_keys"] == ("query", "session_id", "top_k")
    assert span.attributes["agentrecx.session_id"] == "web-safe"
    assert span.attributes["mcp.process_instance_id"] == "mcp-process-a"
    # The trace id in the W3C carrier is the real client span trace id.
    carrier_trace_id = client.meta["traceparent"].split("-")[1]
    assert carrier_trace_id == format(span.context.trace_id, "032x")


def test_disabled_manager_does_not_append_trace_meta(tmp_path: Path) -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.meta: dict | None = None

        async def call_tool(self, tool, arguments, progress_callback=None, **kwargs):  # noqa: ANN001
            self.meta = dict(kwargs.get("meta") or {})
            return SimpleNamespace(is_error=False, structured_content={"status": "ok"}, content=[])

    manager = MCPManager(Settings(log_dir=tmp_path / "logs", repo=REPO_ROOT))
    runtime = manager.runtimes["profile_0057"]
    client = RecordingClient()
    runtime.client = client
    runtime.status = "ready"
    runtime.process_instance_id = "mcp-process-a"

    import asyncio

    asyncio.run(manager.call_tool("profile_0057", "search_products", {"query": "x"}))
    assert client.meta == {}
