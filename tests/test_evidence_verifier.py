"""Tests for the independent Qwen evidence verifier (no network/GPU)."""

from __future__ import annotations

import json

from recommendation.control.evidence_verifier import (
    QwenVerifierClient,
    _precheck_claim,
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


def test_multi_claim_id_mismatch_is_invalid_not_reordered() -> None:
    output = ('{"verdicts":['
              '{"claim_id":"p1_claim_0","verdict":"CONTRADICTED","reason":"a"},'
              '{"claim_id":"p1_claim_1","verdict":"SUPPORTED","reason":"b"}]}')
    result = parse_verifier_response(output, ["c1", "c2"], model="qwen")
    assert result.status == "invalid_model_payload"
    assert result.verdicts == []


def test_single_claim_renamed_id_is_explicitly_mapped() -> None:
    output = ('{"verdicts":[{"claim_id":"p1_claim_0","verdict":"SUPPORTED",'
              '"reason":"direct","evidence_refs":["pev_1"],"missing_support":[]}]}')
    result = parse_verifier_response(output, ["c1"], model="qwen")
    assert result.status == "ok"
    assert result.verdicts[0].claim_id == "c1"
    assert result.id_alignment == "single_claim"


def test_single_claim_missing_array_bracket_is_repaired_without_guessing() -> None:
    output = ('{"verdicts":[{"claim_id":"B004K6R43G","verdict":"SUPPORTED",'
              '"reason":"direct","evidence_refs":["pev_1"],"missing_support":[]}}')
    result = parse_verifier_response(output, ["a01"], model="qwen")
    assert result.status == "ok"
    assert result.verdicts[0].claim_id == "a01"
    assert result.verdicts[0].verdict == "SUPPORTED"


def test_invalid_verdict_is_reported_not_fabricated() -> None:
    output = json.dumps({"verdicts": [
        {"claim_id": "c1", "verdict": "MAYBE", "reason": "bad"},
    ]})
    result = parse_verifier_response(output, ["c1"], model="qwen")
    assert result.status == "invalid_model_payload"
    assert result.verdicts == []


def test_precheck_rejects_missing_or_wrong_citations() -> None:
    claim = {"claim_id": "c1", "claim_type": "product_fact", "claim_text": "It is waterproof.",
             "product_id": "p1", "attribute": "waterproof", "evidence_refs": []}
    assert _precheck_claim(claim, [], []).verdict == "INSUFFICIENT_EVIDENCE"
    wrong = {
        "claim_id": "c2", "claim_type": "preference_link",
        "claim_text": "You prefer water resistance.",
        "evidence_refs": ["mev_1"],
    }
    evidence = [{"evidence_id": "mev_1", "raw_field": "preference",
                 "value": "wireless", "polarity": "prefer", "active": True,
                 "span_text": "I prefer wireless."}]
    assert _precheck_claim(wrong, evidence, []).verdict == "INSUFFICIENT_EVIDENCE"


def test_precheck_detects_metadata_conflict_and_attribute_alias() -> None:
    evidence = [
        {"evidence_id": "pev_1", "product_id": "p1", "raw_field": "metadata_categories",
         "span_text": "Over-Ear Headphones"},
        {"evidence_id": "pev_2", "product_id": "p1", "raw_field": "metadata_details",
         "span_text": "Form Factor: On Ear"},
        {"evidence_id": "pev_3", "product_id": "p1", "raw_field": "title",
         "span_text": "IPX5 Sweatproof earbuds"},
    ]
    claim = {"claim_id": "c1", "claim_type": "product_fact", "claim_text": "It is over-ear.",
             "product_id": "p1", "attribute": "form_factor", "evidence_refs": ["pev_1"]}
    assert _precheck_claim(claim, evidence, []).verdict == "INSUFFICIENT_EVIDENCE"
    supported = {"claim_id": "c2", "claim_type": "product_fact",
                 "claim_text": "They are IPX5 sweatproof.", "product_id": "p1",
                 "attribute": "waterproof", "evidence_refs": ["pev_3"]}
    assert _precheck_claim(supported, evidence, []) is None


def test_verify_claims_returns_transport_error_without_success() -> None:
    class Broken:
        def generate(self, messages):
            raise TimeoutError("gpu timeout")

    result = verify_claims(
        client=Broken(), product_id="p1",
        claims=[{"claim_id": "c1", "claim_type": "product_fact", "claim_text": "x",
                 "product_id": "p1", "evidence_refs": ["pev_1"]}],
        evidence=[{"evidence_id": "pev_1", "product_id": "p1", "raw_field": "title",
                   "span_text": "x", "raw_excerpt": "x"}],
    )
    assert result.status == "error"
    assert result.verdicts == []
    assert result.errors and "TimeoutError" in result.errors[0]
