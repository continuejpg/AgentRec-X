"""M3-A opt-in E2E: real Cold Resume traced into a running Phoenix instance.

Run only when a local Phoenix is already accepting OTLP/HTTP:

    AGENTRECX_RUN_OBSERVABILITY_E2E=1 \\
    AGENTRECX_PHOENIX_URL=http://127.0.0.1:6006 \\
    pytest tests/test_web_observability_e2e.py -q -p no:cacheprovider

The test uses the same real BM25 MCP child, real C4 catalogue, real demo historical FX
sidecar, real SQLite ledger and real process restart as
``test_web_demo_price_e2e.py``.  It does not inject fixture traces into Phoenix.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import tests.test_web_demo_unit as twu  # noqa: E402
from tests.test_web_cold_resume_e2e import (  # noqa: E402
    _count_calls,
    _create,
    _mcp_child_pids,
    _real_settings,
    _reply,
    _wait_for_new_child,
    _wait_ready,
)
from web_demo.app import create_app  # noqa: E402
from web_demo.handle_lifecycle import HandleRef  # noqa: E402

PROFILE = "profile_0057"
ASK_QUERY = "推荐一台 5000 元的笔记本"
REPLY_TEXT = "游戏，预算改成 8000 元"
SIDECAR = REPO_ROOT / "data/enrichment/electronics_metadata_sidecar.jsonl"
PHOENIX_URL = os.environ.get("AGENTRECX_PHOENIX_URL", "http://127.0.0.1:6006").rstrip("/")
pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTRECX_RUN_OBSERVABILITY_E2E") != "1",
    reason="set AGENTRECX_RUN_OBSERVABILITY_E2E=1 with a local Phoenix running",
)


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - local demo
        return json.loads(response.read().decode("utf-8"))


def _spans(project: str) -> list[dict]:
    encoded = urllib.parse.quote(project, safe="")
    payload = _json(f"{PHOENIX_URL}/v1/projects/{encoded}/spans?limit=1000")
    return list(payload.get("data") or [])


def _wait_for_spans(project: str, predicate, timeout_s: float = 45.0) -> list[dict]:
    deadline = time.time() + timeout_s
    last: list[dict] = []
    while time.time() < deadline:
        last = _spans(project)
        if predicate(last):
            return last
        time.sleep(1.0)
    names = sorted({str(span.get("name")) for span in last})
    raise AssertionError(f"Phoenix did not receive the expected trace; spans={names}")


def _with_attrs(spans: list[dict], name: str) -> list[dict]:
    return [span for span in spans if span.get("name") == name]


def test_real_cold_resume_is_traced_into_phoenix(tmp_path: Path) -> None:
    # Fail with a clear message if the optional Phoenix receiver was not started.
    root_html = urllib.request.urlopen(PHOENIX_URL + "/", timeout=10).read().decode("utf-8", "replace")
    assert "Phoenix" in root_html
    project = f"agentrecx-m3-e2e-{int(time.time())}"

    settings = dataclasses.replace(
        _real_settings(tmp_path),
        price_mode="demo_historical_fx",
        demo_price_sidecar=SIDECAR,
        otel_enabled=True,
        otel_endpoint=PHOENIX_URL,
        otel_project_name=project,
        otel_service_name="agentrecx-web-demo",
        otel_sample_ratio=1.0,
        otel_export_timeout_s=2.0,
    )

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()

        chat = client.post(
            "/api/chat/stream", json={"session_id": session_id, "message": ASK_QUERY}
        )
        assert chat.status_code == 200, chat.text
        events = twu.parse_sse(chat.text)
        clarification = next(
            (event["data"] for event in events if event["event"] == "clarification"), None
        )
        assert clarification is not None
        clarification_id = clarification["clarification_id"]

        restart = client.post("/api/demo/restart", json={"profile_id": PROFILE})
        assert restart.status_code == 200, restart.text
        _wait_ready(client)
        pids_b = _wait_for_new_child(pids_a)
        assert set(pids_b).isdisjoint(pids_a)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).code == "stale_process"

        calls = _count_calls(manager)
        response = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendation"]["items"][0]["product_id"] == "B0BY2Z27LC"
        assert body["budget_verification"]["within"] == 1
        assert body["budget_verification"]["unknown"] == 3
        assert body["budget_verification"]["selected_price"]["fx_rate"] == "7.00"

        calls_before_replay = len(calls)
        replay = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before_replay

    spans = _wait_for_spans(
        project,
        lambda rows: (
            bool(_with_attrs(rows, "agent.chat"))
            and any(
                span.get("attributes", {}).get("agentrecx.reply.path") == "cold_resume"
                for span in _with_attrs(rows, "agent.reply")
            )
        ),
    )

    # 1. The real BFF, MCP child and business stages all arrived in Phoenix.
    names = {str(span.get("name")) for span in spans}
    required = {
        "agent.chat",
        "agent.reply",
        "reply.handle_check",
        "resume.cold_restore",
        "reply.claim",
        "reply.external_started",
        "reply.checkpoint",
        "reply.finalize",
        "tools/call restore_clarification_handle",
        "tools/call submit_clarification_reply",
        "tools/call get_active_preferences",
        "tools/call recommend_products",
        "tools/call get_product_info",
        "retrieval.catalog_search",
        "budget.assessment",
        "gate.decision",
    }
    assert required <= names, sorted(required - names)

    # 2. Cold Resume path is explicit; the old handle was not reused.
    cold_reply = next(
        span for span in _with_attrs(spans, "agent.reply")
        if span["attributes"].get("agentrecx.reply.path") == "cold_resume"
    )
    assert cold_reply["attributes"]["agentrecx.reply.outcome"] == "completed"
    assert cold_reply["attributes"]["agentrecx.execution_stage"] == "COMPLETED"
    handle_span = _with_attrs(spans, "reply.handle_check")[0]
    assert handle_span["attributes"]["agentrecx.resume.restored"] is True
    assert handle_span["attributes"]["agentrecx.handle.ok"] is True
    restore_span = next(
        span for span in _with_attrs(spans, "mcp.call restore_clarification_handle")
        if span["attributes"].get("mcp.tool.success") is True
    )
    assert restore_span["attributes"]["mcp.tool.name"] == "restore_clarification_handle"

    # 3. Real BM25 retrieval and the real budget Gate are visible, without query text.
    retrieval = _with_attrs(spans, "retrieval.catalog_search")[0]
    assert retrieval["attributes"]["retrieval.mode"] == "bm25"
    assert retrieval["attributes"]["retrieval.hit_count"] > 0
    budget = next(
        span for span in _with_attrs(spans, "budget.assessment")
        if span["attributes"].get("budget.max_amount") == 8000.0
    )
    assert budget["attributes"]["budget.price_mode"] == "demo_historical_fx"
    assert budget["attributes"]["budget.within"] == 1
    assert budget["attributes"]["budget.unknown"] == 3
    gate = next(
        span for span in _with_attrs(spans, "gate.decision")
        if span["attributes"].get("gate.allow_finalize") is True
    )
    assert gate["attributes"]["gate.action"] == "ALLOW_FINALIZE"
    assert gate["attributes"]["gate.allow_finalize"] is True
    # The Gate may prefer another role-compatible candidate; the budget hard constraint
    # then selects the verified within-budget MSI.  The final recommendation span records
    # the real selected id.
    assert gate["attributes"]["gate.candidate_count"] == 4
    recommend_span = next(
        span for span in _with_attrs(spans, "tools/call recommend_products")
        if span["attributes"].get("agentrecx.recommend.status") == "finalized"
    )
    assert recommend_span["attributes"]["agentrecx.recommend.selected_product_id"] == "B0BY2Z27LC"

    # 4. W3C propagation produced a real client -> server parent link across stdio.
    client_call = next(
        span for span in _with_attrs(spans, "mcp.call recommend_products")
        if span.get("parent_id") and span["attributes"].get("mcp.tool.success") is True
    )
    server_calls = [
        span for span in _with_attrs(spans, "tools/call recommend_products")
        if span.get("parent_id") == client_call["context"]["span_id"]
        and span["context"]["trace_id"] == client_call["context"]["trace_id"]
    ]
    assert server_calls, "the child MCP server span must be a child of the BFF client span"
    assert client_call["attributes"]["mcp.tool.name"] == "recommend_products"
    assert client_call["attributes"]["mcp.process_instance_id"]

    # 5. The ASK trace and the post-restart reply share stable business IDs, not a process id.
    chat_span = _with_attrs(spans, "agent.chat")[0]
    # Phoenix reserves ``session.id`` for its own UUID-based session feature and drops a
    # non-UUID value; the stable business key is therefore an explicit project attribute.
    assert chat_span["attributes"]["agentrecx.session_id"] == session_id
    assert cold_reply["attributes"]["agentrecx.session_id"] == session_id
    assert cold_reply["attributes"]["agentrecx.pending_id"]
    assert cold_reply["attributes"]["agentrecx.request_id"] == cold_reply["attributes"][
        "agentrecx.reply_turn_id"
    ]

    # 6. A completed replay is observed as a replay and has no tool-call children.
    replay_roots = [
        span for span in _with_attrs(spans, "agent.reply")
        if span["attributes"].get("agentrecx.reply.path") == "replay"
    ]
    assert replay_roots
    replay_children = [
        span for span in spans
        if span.get("parent_id") == replay_roots[0]["context"]["span_id"]
    ]
    assert not any(
        str(span.get("name", "")).startswith(("tools/call", "mcp.call"))
        for span in replay_children
    )

    # 7. Privacy: query/reply/narrative text must not appear in any exported attribute.
    exported = json.dumps([span.get("attributes") for span in spans], ensure_ascii=False)
    for secret in (ASK_QUERY, REPLY_TEXT, "Gaming Laptop", "演示参考价", "noise cancelling"):
        assert secret not in exported, f"private business text leaked into traces: {secret}"

    # 8. Phoenix's trace listing also contains these traces (the UI reads this store).
    traces = _json(
        f"{PHOENIX_URL}/v1/projects/{urllib.parse.quote(project, safe='')}/traces?limit=20"
    )
    assert traces.get("data"), "Phoenix trace listing is empty"
