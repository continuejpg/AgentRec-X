"""Tests for the independent Qwen evidence verifier (no network/GPU)."""

from __future__ import annotations

import json

from recommendation.control.evidence_verifier import (
    QwenVerifierClient,
    build_verifier_messages,
    parse_verifier_response,
    verify_claims,
)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[dict[str, str]] = []

    def generate(self, messages):
        self.messages = messages
        return {"text": self.text, "latency_ms": 12.5, "input_tokens": 100, "output_tokens": 20}


def test_prompt_is_independent_of_planner_and_p15_validation() -> None:
    messages = build_verifier_messages(
        product_id="p1",
        claims=[{"claim_id": "c1", "claim_type": "product_fact", "claim_text": "It is wireless.",
                 "product_id": "p1", "evidence_refs": ["pev_1"]}],
        evidence=[{"evidence_id": "pev_1", "product_id": "p1", "raw_field": "title",
                   "span_text": "Wireless", "raw_excerpt": "Wireless"}],
        unknown_fields=["price"],
    )
    joined = "\n".join(message["content"] for message in messages)
    assert "pev_1" in joined and "It is wireless." in joined
    assert "valid_claims" not in joined and "planner" not in joined.lower()
    assert "gate" not in joined.lower()


def test_exact_verdicts_are_parsed() -> None:
    output = json.dumps({"verdicts": [
        {"claim_id": "c1", "verdict": "SUPPORTED", "reason": "direct"},
        {"claim_id": "c2", "verdict": "INSUFFICIENT_EVIDENCE", "reason": "indirect"},
    ]})
    result = parse_verifier_response(output, ["c1", "c2"], model="qwen")
    assert result.status == "ok"
    assert [item.verdict for item in result.verdicts] == ["SUPPORTED", "INSUFFICIENT_EVIDENCE"]
    assert result.id_alignment == "exact"


def test_positional_id_alignment_and_extra_brace_repair() -> None:
    output = ('{"verdicts":['
              '{"claim_id":"p1_claim_0","verdict":"CONTRADICTED","reason":"a"},'
              '{"claim_id":"p1_claim_1","verdict":"SUPPORTED","reason":"b"}]}}')
    result = parse_verifier_response(output, ["c1", "c2"], model="qwen")
    assert result.status == "ok"
    assert [item.claim_id for item in result.verdicts] == ["c1", "c2"]
    assert [item.verdict for item in result.verdicts] == ["CONTRADICTED", "SUPPORTED"]
    assert result.id_alignment == "positional"


def test_invalid_verdict_is_reported_not_fabricated() -> None:
    output = json.dumps({"verdicts": [
        {"claim_id": "c1", "verdict": "MAYBE", "reason": "bad"},
    ]})
    result = parse_verifier_response(output, ["c1"], model="qwen")
    assert result.status == "invalid_model_payload"
    assert result.verdicts == []


def test_verify_claims_returns_transport_error_without_success() -> None:
    class Broken:
        def generate(self, messages):
            raise TimeoutError("gpu timeout")

    result = verify_claims(
        client=Broken(), product_id="p1",
        claims=[{"claim_id": "c1", "claim_type": "product_fact", "claim_text": "x"}],
        evidence=[],
    )
    assert result.status == "error"
    assert result.verdicts == []
    assert result.errors and "TimeoutError" in result.errors[0]
