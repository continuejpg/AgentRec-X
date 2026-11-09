"""P15 tests: Evidence Bundle, claim validation, E2E integration, memory provenance."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p14c0.run_dev import _memory_read
from recommendation.control.rag_evidence import (
    build_bundle,
    build_product_evidence,
    render_final_answer,
    validate_claims,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p15"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task(task_id="p14e2_lite_0049"):
    return next(row for row in _jsonl(REPO / "data/p14e2/benchmark/dev.agent.jsonl")
                if row["task_id"] == task_id)


def test_product_evidence_spans_are_exact_and_independent_of_model():
    catalogue = ElectronicsCatalogue.load()
    product, evidence, unknown = build_product_evidence(
        catalogue=catalogue, item_id="B004K6R43G", namespace="amazon_electronics_c4", task_id="t")
    assert product and product["item_id"] == "B004K6R43G"
    assert evidence and all(item.verification_status == "lexical_span_exact" for item in evidence)
    for item in evidence:
        assert item.raw_excerpt[item.span_start:item.span_end] == item.span_text
        assert item.source_record.startswith("data/p13r")
    assert {item.attribute for item in unknown} >= {"price", "brand", "stock", "material", "size", "warranty"}


def test_bundle_separates_product_and_memory_evidence_and_keeps_inactive_entries():
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0055")
    preferences, history = _memory_read(catalogue, task)
    bundle = build_bundle(task=task, selected_item_id="B00B93N0PO", catalogue=catalogue,
                          history=history, gate_decision={}, search_mode="bge",
                          ledger_source="CatalogSearchSource(mode=bge)")
    assert bundle.product_evidence and bundle.preference_evidence
    assert all(item.product_id == "B00B93N0PO" for item in bundle.product_evidence)
    assert any(item.active for item in bundle.preference_evidence)
    assert any(not item.active for item in bundle.preference_evidence)
    assert all(item.memory_id and item.source_turn_id and item.source_text for item in bundle.preference_evidence)
    assert bundle.user_id == history["user_key"]


def test_claim_validation_accepts_grounded_and_rejects_unsupported():
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0049")
    preferences, history = _memory_read(catalogue, task)
    bundle = build_bundle(task=task, selected_item_id="B004K6R43G", catalogue=catalogue,
                          history=history, gate_decision={}, search_mode="bge",
                          ledger_source="x")
    product_ref = bundle.product_evidence[0].evidence_id
    active_pref = next(item for item in bundle.preference_evidence if item.active)
    inactive_pref = next((item for item in bundle.preference_evidence if not item.active), None)
    payload = {"claims": [
        {"claim_type": "product_fact", "claim_text": "The title says wireless.",
         "product_id": "B004K6R43G", "evidence_refs": [product_ref]},
        {"claim_type": "preference_link", "claim_text": "You stated a wireless preference.",
         "preference_ref": active_pref.evidence_id, "evidence_refs": [active_pref.evidence_id]},
        {"claim_type": "unknown", "claim_text": "Price is not available.", "attribute": "price"},
        {"claim_type": "product_fact", "claim_text": "The price is $10.",
         "product_id": "B004K6R43G", "attribute": "price", "evidence_refs": [product_ref]},
        {"claim_type": "product_fact", "claim_text": "A fabricated fact.",
         "product_id": "B004K6R43G", "evidence_refs": ["pev_does_not_exist"]},
        {"claim_type": "preference_link", "claim_text": "Memories are product facts.",
         "evidence_refs": [product_ref]},
        {"claim_type": "unknown", "claim_text": "Wireless is unknown.", "attribute": "wireless"},
    ]}
    if inactive_pref is not None:
        payload["claims"].append({"claim_type": "preference_link", "claim_text": "An old preference.",
                                  "preference_ref": inactive_pref.evidence_id,
                                  "evidence_refs": [inactive_pref.evidence_id]})
    validation = validate_claims(bundle, payload)
    assert validation["automated_validation"] is True
    assert validation["semantic_truth_verified"] is False
    assert validation["counts"]["product_fact_valid"] == 1
    assert validation["counts"]["preference_link_valid"] == 1
    assert validation["counts"]["unknown_valid"] == 1
    assert validation["counts"]["rejected"] >= 4
    reasons = " ".join(reason for claim in validation["rejected_claims"] for reason in claim["reasons"])
    assert "does not exist" in reasons
    assert "promotes an UNKNOWN field" in reasons
    assert "cites preference/memory evidence" in reasons or "cites product evidence" in reasons
    if inactive_pref is not None:
        assert "inactive/retracted preference" in reasons


def test_final_answer_uses_validated_claims_only():
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0049")
    preferences, history = _memory_read(catalogue, task)
    bundle = build_bundle(task=task, selected_item_id="B004K6R43G", catalogue=catalogue,
                          history=history, gate_decision={}, search_mode="bge", ledger_source="x")
    product_ref = bundle.product_evidence[0].evidence_id
    validation = validate_claims(bundle, {"claims": [
        {"claim_type": "product_fact", "claim_text": "Title evidence only.",
         "product_id": "B004K6R43G", "evidence_refs": [product_ref]},
        {"claim_type": "unknown", "claim_text": "Price is unknown.", "attribute": "price"},
    ]})
    answer = render_final_answer(bundle, validation)
    assert "Title evidence only" in answer
    assert "[evidence:" in answer
    assert "cannot confirm" in answer
    assert "Catalogue evidence only" in answer
    assert "B004K6R43G" in answer


def test_e2e_has_existing_gate_bge_runtime_and_evidence():
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNS / "e2e").glob("*.json"))]
    assert len(records) == 13
    assert all(record["search_mode"] == "bge_dense" for record in records)
    assert all(record["search_backend"]["mode"] == "bge" for record in records)
    finalized = [record for record in records if record["terminal"]["status"] == "finalized"]
    nonfinal = [record for record in records if record["terminal"]["status"] != "finalized"]
    assert len(finalized) == 9 and len(nonfinal) == 4
    assert all(record["terminal"].get("selected_candidate_id") for record in finalized)
    assert all(record["terminal"].get("selected_candidate_id") is None for record in nonfinal)
    for record in finalized:
        explanation = record["rag_explanation"]
        assert explanation["bundle"]["selected_product"]["item_id"] == record["terminal"]["selected_candidate_id"]
        assert explanation["bundle"]["product_evidence"]
        assert explanation["raw_output"]
        assert explanation["final_answer"]
        assert explanation["validation"]["automated_validation"] is True
        assert explanation["validation"]["semantic_truth_verified"] is False
        assert explanation["validation"]["valid_claims"]
        for claim in explanation["validation"]["valid_claims"]:
            for ref in claim.get("evidence_refs") or []:
                if claim["claim_type"] == "product_fact":
                    assert any(item["evidence_id"] == ref for item in explanation["bundle"]["product_evidence"])
                if claim["claim_type"] == "preference_link":
                    assert any(item["evidence_id"] == ref and item["active"] for item in explanation["bundle"]["preference_evidence"])
        assert "Catalogue evidence only" in explanation["final_answer"]
    metrics = json.loads((RUNS / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["normal_completion_rate"] == 1.0
    assert metrics["nonfinal_selected_any"] == 0
    assert metrics["claims"]["missing_evidence_refs"] == 0
    assert metrics["claims"]["inactive_preference_used_as_active"] == 0
    assert metrics["claims"]["product_memory_confusion"] == 0
    assert metrics["claims"]["unknown_promoted_to_known"] == 0
    assert (RUNS / "schema/evidence_bundle.schema.json").is_file()
    assert (RUNS / "schema/claim_validation.schema.json").is_file()


def test_no_reserved_or_evaluator_text_in_final_answers():
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in (RUNS / "e2e").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        answer = str(record.get("final_user_answer") or "")
        assert not any(task_id in answer for task_id in reserved)
        assert "expected_decision" not in answer
        assert "target_parent_asin" not in answer
        # Input prompt may contain the selected product only, never evaluator labels.
        prompt = json.dumps(record["rag_explanation"].get("prompt_messages", []), ensure_ascii=False)
        assert "expected_decision" not in prompt
        assert "target_parent_asin" not in prompt
    assert (REPO / "runs/p14c1/comparison.json").is_file()
    assert (REPO / "runs/p14d2/comparison.json").is_file()


def test_missing_product_text_cannot_become_a_fact():
    from recommendation.control.rag_evidence import build_bundle, render_final_answer, validate_claims
    catalogue = ElectronicsCatalogue.load()
    task = _task("p14e2_lite_0049")
    preferences, history = _memory_read(catalogue, task)
    bundle = build_bundle(task=task, selected_item_id="missing_product_xyz", catalogue=catalogue,
                          history=history, gate_decision={}, search_mode="bge", ledger_source="x")
    assert bundle.selected_product is None
    assert bundle.product_evidence == []
    validation = validate_claims(bundle, {"claims": [
        {"claim_type": "product_fact", "claim_text": "This has a long battery life.",
         "product_id": "missing_product_xyz", "evidence_refs": []},
        {"claim_type": "unknown", "claim_text": "No catalogue product is available.", "attribute": "product"},
    ]})
    assert not any(claim["claim_type"] == "product_fact" for claim in validation["valid_claims"])
    assert validation["rejected_claims"]
    answer = render_final_answer(bundle, validation)
    assert "cannot verify" in answer or "can't verify" in answer
    assert "long battery life" not in answer
