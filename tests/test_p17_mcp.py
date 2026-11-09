"""P17 tests: official MCP v2 stdio server, independent client, real tools, memory permissions."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p17"


def load(name: str):
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def test_sdk_and_protocol_manifest():
    assert version("mcp") == "2.3.0"
    manifest = load("sdk_manifest.json")
    assert manifest["sdk"]["package"] == "mcp"
    assert manifest["sdk"]["version"] == "2.3.0"
    assert manifest["server"]["transport"] == "stdio"
    assert manifest["server"]["protocol_target"] == "2026-07-28"
    assert manifest["security"]["caller_user_id_accepted"] is False
    assert manifest["security"]["remote_http"] is False if "remote_http" in manifest["security"] else True
    source = (REPO / "recommendation/mcp/server.py").read_text(encoding="utf-8")
    assert "from mcp.server import MCPServer" in source
    assert 'transport="stdio"' in source
    assert "FastMCP" not in source
    assert "StreamableHTTPServer" not in source


def test_tools_are_discoverable_and_schemas_have_no_user_identity():
    schemas = load("tool_schemas.json")
    expected = {"search_products", "get_product_info", "recommend_products",
                "get_active_preferences", "compare_products",
                "request_preference_clarification", "submit_clarification_reply"}
    assert {tool["name"] for tool in schemas["tools"]} == expected
    serialized = json.dumps(schemas, ensure_ascii=False)
    assert "user_id" not in serialized
    assert "target_parent_asin" not in serialized
    assert "expected_decision" not in serialized
    for tool in schemas["tools"]:
        assert tool["input_schema"]["type"] == "object"
    by_name = {tool["name"]: tool for tool in schemas["tools"]}
    assert by_name["search_products"]["input_schema"]["properties"]["top_k"]["maximum"] == 50
    assert by_name["compare_products"]["input_schema"]["properties"]["product_refs"]["minItems"] == 2
    assert by_name["compare_products"]["input_schema"]["properties"]["product_refs"]["maxItems"] == 5


def test_real_mcp_client_discovery_and_core_tools():
    transcript = load("client_transcript.json")
    assert transcript["sdk"] == "mcp==2.3.0"
    assert len(transcript["servers"]) == 2
    server_a = transcript["servers"][0]
    assert server_a["protocol_version"] == "2026-07-28"
    assert len(server_a["tools"]) == 7
    calls = transcript["tool_calls"]
    successful = [call for call in calls if not call.get("is_error")]
    assert any(call["tool"] == "search_products" and call["structured_content"]["search_mode"] in {"bge", "bm25"}
               for call in successful)
    product = next(call for call in successful if call["tool"] == "get_product_info")
    assert product["structured_content"]["status"] == "ok"
    assert product["structured_content"]["product_evidence"]
    assert product["structured_content"]["unknown_fields"]
    for evidence in product["structured_content"]["product_evidence"]:
        assert evidence["raw_excerpt"][evidence["span_start"]:evidence["span_end"]] == evidence["span_text"]
    recommendation = next(call for call in successful if call["tool"] == "recommend_products")
    assert recommendation["structured_content"]["status"] == "finalized"
    assert recommendation["structured_content"]["gate"]["allow_finalize"] is True
    comparison = next(call for call in successful if call["tool"] == "compare_products")
    assert comparison["structured_content"]["status"] in {"comparison_ready", "role_mismatch", "invalid_product_reference"}
    assert comparison["structured_content"]["qwen_used"] is False
    assert any(call["tool"] == "request_preference_clarification" for call in successful)


def test_memory_provenance_and_write_confirmation():
    transcript = load("client_transcript.json")
    calls = transcript["tool_calls"]
    prefs = [call for call in calls if call["tool"] == "get_active_preferences" and not call.get("is_error")]
    assert prefs
    a_prefs = [call for call in prefs if call["structured_content"]["authorized_user_key"] == "p14e2_lite_0049"]
    assert a_prefs
    assert any(pref["value"] == "noise cancelling"
               for pref in a_prefs[-1]["structured_content"]["active_preferences"])
    submits = [call for call in calls if call["tool"] == "submit_clarification_reply" and not call.get("is_error")]
    assert len(submits) == 2
    assert submits[0]["structured_content"]["update"]["added"] == 1
    assert submits[1]["structured_content"]["already_processed"] is True
    errors = [call for call in calls if call.get("is_error")]
    messages = " ".join(" ".join(call.get("content") or []) for call in errors)
    assert "memory_writes_disabled" in messages
    assert "user_confirmation_required" in messages
    assert "clarification_id_expired" in messages
    assert "invalid_or_expired_clarification_id" in messages
    metrics = load("metrics.json")
    assert metrics["memory"]["unique_successful_writes"] == 1
    assert metrics["memory"]["duplicate_retry_no_new_entry"] is True
    assert metrics["security"]["caller_supplied_user_id_in_tools"] is False


def test_failure_cases_are_explicit_and_real():
    transcript = load("client_transcript.json")
    errors = [call for call in transcript["tool_calls"] if call.get("is_error")]
    text = " ".join(" ".join(call.get("content") or []) for call in errors)
    assert "product_not_found" in text
    assert "namespace mismatch" in text
    assert "invalid_or_expired_clarification_id" in text
    assert "memory_writes_disabled" in text
    assert "user_confirmation_required" in text
    assert "clarification_id_expired" in text
    failed_tools = {call["tool"] for call in errors}
    assert "get_product_info" in failed_tools
    assert "search_products" in failed_tools
    assert "compare_products" in failed_tools
    assert "unknown_tool" in failed_tools
    metrics = load("metrics.json")
    assert metrics["failures_are_explicit"] is True
    assert metrics["tools_discovered"] == 7


def test_no_reserved_or_evaluator_payload_and_legacy_artifacts():
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8", errors="ignore") for task_id in reserved), path
    transcript = load("client_transcript.json")
    serialized = json.dumps(transcript, ensure_ascii=False)
    assert "expected_decision" not in serialized
    assert "target_parent_asin" not in serialized
    assert (REPO / "runs/p16/metrics.json").is_file()
    assert (REPO / "runs/p15/metrics.json").is_file()
    assert (REPO / "runs/p14d2/comparison.json").is_file()
