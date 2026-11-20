"""Real MCP fixture -> durable reply contract -> SQLite checkpoint/finalize."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_web_request_store import REPLY, _new_store, _setup  # noqa: E402
from web_demo.reply_result_contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    RecommendationResultError,
    build_reply_payload,
    turn_from_payload,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "mcp_recommend_products_finalized.json").read_text()
)


def _payload(**overrides):
    values = dict(
        reply_turn_id="web-r-r1", pending_id="pend-1", user_message="For gaming.",
        recommendation=FIXTURE, narrative="Here is one match.",
        recommendation_id="rec-1", ranking_source="cpu_live:mcp_recommend",
    )
    values.update(overrides)
    return build_reply_payload(**values)


def test_real_fixture_shape() -> None:
    assert FIXTURE["status"] == "finalized"
    assert FIXTURE["gate"]["allow_finalize"] is True
    assert FIXTURE["selected_product"]["item_id"] == FIXTURE["selected_product_id"]
    assert "products" not in FIXTURE
    assert "narrative" not in FIXTURE


def test_contract_builds_durable_payload_from_real_fixture() -> None:
    payload = _payload()
    assert payload["schema"] == CONTRACT_SCHEMA
    assert payload["reply_turn_id"] == "web-r-r1"
    assert payload["assistant_text"] == "Here is one match."
    items = payload["products"]["items"]
    assert len(items) == 1
    assert items[0]["item_id"] == FIXTURE["selected_product_id"]
    assert payload["recommendation"]["selected_product_id"] == FIXTURE["selected_product_id"]
    assert payload["recommendation"]["ranking_source"] == "cpu_live:mcp_recommend"
    assert "payload" not in payload
    assert json.loads(json.dumps(payload)) == payload


def test_gate_and_identity_conflicts_are_rejected() -> None:
    broken_gate = json.loads(json.dumps(FIXTURE))
    broken_gate["gate"]["allow_finalize"] = False
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=broken_gate)

    broken_id = json.loads(json.dumps(FIXTURE))
    broken_id["selected_product"]["item_id"] = "B0OTHER"
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=broken_id)

    missing = dict(FIXTURE)
    missing.pop("selected_product")
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=missing)


def test_error_and_clarification_statuses_are_rejected() -> None:
    failed = dict(FIXTURE, status="error", error={"code": "boom"})
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=failed)
    clarify = dict(FIXTURE, status="awaiting_clarification")
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=clarify)
    unknown = dict(FIXTURE, status="something_new")
    with pytest.raises(RecommendationResultError):
        _payload(recommendation=unknown)


def test_supplied_products_keep_selected_identity_and_validate_items() -> None:
    products = {"items": [{"item_id": "B0LIST", "title": "Other"}]}
    payload = _payload(products=products)
    ids = [item["item_id"] for item in payload["products"]["items"]]
    assert ids == ["B0LIST", FIXTURE["selected_product_id"]]
    assert payload["products"]["selected_appended"] is True

    with pytest.raises(RecommendationResultError):
        _payload(products={"items": [{"title": "no id"}]})
    with pytest.raises(RecommendationResultError):
        _payload(products={"items": []})
    with pytest.raises(RecommendationResultError):
        _payload(narrative="")


def test_missing_presentation_fields_are_rejected() -> None:
    with pytest.raises(RecommendationResultError):
        _payload(recommendation_id="")
    with pytest.raises(RecommendationResultError):
        _payload(ranking_source="")
    payload = _payload()
    with pytest.raises(RecommendationResultError):
        turn_from_payload({**payload, "schema": "unknown"})


def test_checkpoint_finalize_restart_and_replay(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        payload = _payload(reply_turn_id=pending.reply_turn_id, pending_id=pending.pending_id)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token
        )
        store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, result=payload
        )
        assert store.get_request("web-r", pending.reply_turn_id).execution_stage == "RESULT_DURABLE"
    finally:
        store.close()

    reopened = _new_store(tmp_path)
    try:
        raw = sqlite3.connect(reopened.path)
        try:
            raw.execute("UPDATE web_request SET lease_expires_at = 0")
            raw.commit()
        finally:
            raw.close()
        takeover = reopened.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert takeover.outcome == "result_ready"
        stored = takeover.request.result["payload"]
        assert stored["recommendation"]["id"] == "rec-1"
        task, consumed, request = reopened.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=takeover.claim_token,
            outcome="completed", turn=turn_from_payload(stored),
        )
        assert (task.status, consumed.status, request.status) == (
            "COMPLETED", "CONSUMED", "COMPLETED"
        )
        replay = reopened.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert replay.outcome == "replay"
        assert replay.request.result["payload"]["assistant_text"] == "Here is one match."
        assert len(reopened.list_turns("web-r")) == 1
        reopened.reset_session("web-r")
        from web_demo.state_store import RequestFenced
        with pytest.raises(RequestFenced):
            reopened.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
    finally:
        reopened.close()
