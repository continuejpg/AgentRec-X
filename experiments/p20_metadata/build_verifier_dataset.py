"""Build a small real-catalogue Claim–Evidence pilot set for the independent verifier.

The bundles are produced by the existing P15 builder.  Claims are authored here as
**provisional** evaluation cases; they are not human gold labels.  Product IDs, evidence
IDs and source spans all come from the real catalogue/metadata bundle.

Output (git-ignored):
  runs/p20_metadata/verifier_bundles.json
  runs/p20_metadata/claim_evidence_cases.jsonl
  runs/p20_metadata/claim_evidence_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import prepare_hard
from experiments.p14c0.run_dev import _load, _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.control.rag_evidence import build_bundle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p20_metadata"
DEV_TASK = "p14e2_lite_0049"


def _bundle(catalogue: ElectronicsCatalogue, product_id: str, *, task_id: str | None = None):
    if task_id:
        agent = next(row for row in _load("dev", "agent") if row["task_id"] == task_id)
        _prefs, history = _memory_read(catalogue, agent)
        task = agent
    else:
        task = {"task_id": "verifier_pilot", "current_query": "I need electronics.",
                "catalog_namespace": "amazon_electronics_c4"}
        history = {"active_preferences": []}
    return build_bundle(
        task=task, selected_item_id=product_id, catalogue=catalogue, history=history,
        gate_decision={"action": "ALLOW_FINALIZE", "allow_finalize": True}, search_mode="bge",
        ledger_source="CatalogSearchSource(mode=bge)",
    )


def _pdf(bundle: Any, evidence_id: str) -> dict[str, Any]:
    for item in bundle.product_evidence:
        if item.evidence_id == evidence_id:
            return item.as_dict()
    raise KeyError(evidence_id)


def _case(case_id: str, bundle_id: str, product_id: str, claim: dict[str, Any], label: str,
          category: str, split: str = "dev", note: str = "") -> dict[str, Any]:
    return {
        "case_id": case_id, "bundle_id": bundle_id, "product_id": product_id,
        "claim": claim, "label": label, "label_source": "provisional_rule_authored",
        "category": category, "split": split, "note": note,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    bundles = {
        "earbud_pref": _bundle(catalogue, "B004K6R43G", task_id=DEV_TASK),
        "keyboard": _bundle(catalogue, "B07WZFL74B"),
        "mechanical_feeling": _bundle(catalogue, "B09H2JQD48"),
        "form_conflict": _bundle(catalogue, "B00005USA4"),
    }
    # Validate that the evidence IDs this pilot relies on are actually present.
    expected = {
        "earbud_pref": ["pev_01", "pev_04", "pev_05", "pev_08", "pev_11", "mev_487b5d38da771a69a14f57eb",
                         "mev_81b3e5aff33de3bd7c85ba11", "mev_461a92598c65e293218c2636"],
        "keyboard": ["pev_01", "pev_04", "pev_05", "pev_06", "pev_11"],
        "mechanical_feeling": ["pev_01", "pev_04", "pev_05", "pev_06", "pev_07", "pev_08"],
        "form_conflict": ["pev_01", "pev_08", "pev_09", "pev_10"],
    }
    for key, ids in expected.items():
        have = {item.evidence_id for item in bundles[key].product_evidence}
        have |= {item.evidence_id for item in bundles[key].preference_evidence}
        missing = [item for item in ids if item not in have]
        if missing:
            raise SystemExit(f"{key}: missing expected evidence refs {missing}")

    cases: list[dict[str, Any]] = []
    # A. Directly supported product facts.
    cases += [
        _case("a01", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product is in-ear.",
            "product_id": "B004K6R43G", "attribute": "in_ear", "evidence_refs": ["pev_04"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("a02", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product has a sweat-proof design.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_08"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("a03", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product is in-ear and sweat-proof.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_04", "pev_08"],
        }, "SUPPORTED", "multiple_evidence_support"),
        _case("a04", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The brand is Audio-Technica.",
            "product_id": "B004K6R43G", "attribute": "brand", "evidence_refs": ["pev_05"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("b01", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard is wireless and mechanical.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_04", "pev_06"],
        }, "SUPPORTED", "multiple_evidence_support"),
        _case("b02", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard has 10 hours of battery life.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("b03", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard has a compact 60% layout.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("c01", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard is water-resistant.",
            "product_id": "B09H2JQD48", "attribute": "waterproof", "evidence_refs": ["pev_04", "pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("c02", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard has RGB backlighting.",
            "product_id": "B09H2JQD48", "attribute": "rgb", "evidence_refs": ["pev_06", "pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("d01", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones have a 50 mm driver.",
            "product_id": "B00005USA4", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("d02", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The brand is JVC.",
            "product_id": "B00005USA4", "attribute": "brand", "evidence_refs": ["pev_09"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("d03", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones are discontinued.",
            "product_id": "B00005USA4", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        # A. Semantic contradiction with a structurally valid ref.
        _case("a05", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product is over-ear.",
            "product_id": "B004K6R43G", "attribute": "form_factor", "evidence_refs": ["pev_04"],
        }, "CONTRADICTED", "contradicted_form_factor"),
        _case("b04", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard is wired.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_04"],
        }, "CONTRADICTED", "contradicted_wireless"),
        _case("d04", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones are wireless.",
            "product_id": "B00005USA4", "attribute": None, "evidence_refs": ["pev_10"],
        }, "CONTRADICTED", "contradicted_connectivity"),
        # A. Structurally valid but semantically weak / absent support.
        _case("a06", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product has active noise cancellation.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_attribute"),
        _case("a07", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product costs $49.99 today.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "price_unknown"),
        _case("a08", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product has 20 units in stock.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_12"],
        }, "INSUFFICIENT_EVIDENCE", "historical_stock"),
        _case("b05", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard has a metal case.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_attribute"),
        _case("b06", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard includes a wireless mouse.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_accessory"),
        _case("c03", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard has mechanical switches.",
            "product_id": "B09H2JQD48", "attribute": "mechanical", "evidence_refs": ["pev_05", "pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "mechanical_feeling_is_not_switch_type"),
        _case("c04", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard is mechanical.",
            "product_id": "B09H2JQD48", "attribute": "mechanical", "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "mechanical_feeling_is_not_switch_type"),
        _case("c05", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard costs $29.99 today.",
            "product_id": "B09H2JQD48", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "historical_price"),
        _case("d05", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones are over-ear.",
            "product_id": "B00005USA4", "attribute": "form_factor", "evidence_refs": ["pev_08"],
        }, "INSUFFICIENT_EVIDENCE", "metadata_source_conflict"),
        _case("d06", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones are on-ear.",
            "product_id": "B00005USA4", "attribute": "form_factor", "evidence_refs": ["pev_10"],
        }, "INSUFFICIENT_EVIDENCE", "metadata_source_conflict"),
        _case("d07", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones have active noise cancellation.",
            "product_id": "B00005USA4", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_attribute"),
        # B. Other-product evidence / provenance failures.
        _case("x01", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product is a wireless mechanical keyboard.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_04"],
        }, "INSUFFICIENT_EVIDENCE", "other_product_evidence"),
        _case("x02", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard is in-ear.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_04"],
        }, "INSUFFICIENT_EVIDENCE", "other_product_evidence"),
        _case("x03", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard has mechanical switches.",
            "product_id": "B09H2JQD48", "attribute": "mechanical", "evidence_refs": ["pev_04"],
        }, "INSUFFICIENT_EVIDENCE", "wrong_evidence_field"),
        # Preference provenance: active refs but wrong semantic content/polarity.
        _case("p01", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer wireless.",
            "preference_ref": "mev_487b5d38da771a69a14f57eb",
            "evidence_refs": ["mev_487b5d38da771a69a14f57eb"],
        }, "SUPPORTED", "preference_supported"),
        _case("p02", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You want to avoid over-ear headphones.",
            "preference_ref": "mev_81b3e5aff33de3bd7c85ba11",
            "evidence_refs": ["mev_81b3e5aff33de3bd7c85ba11"],
        }, "SUPPORTED", "preference_supported"),
        _case("p03", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer water resistance.",
            "preference_ref": "mev_461a92598c65e293218c2636",
            "evidence_refs": ["mev_461a92598c65e293218c2636"],
        }, "SUPPORTED", "preference_supported"),
        _case("p04", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer over-ear headphones.",
            "preference_ref": "mev_81b3e5aff33de3bd7c85ba11",
            "evidence_refs": ["mev_81b3e5aff33de3bd7c85ba11"],
        }, "CONTRADICTED", "preference_polarity"),
        _case("p05", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You dislike wireless.",
            "preference_ref": "mev_487b5d38da771a69a14f57eb",
            "evidence_refs": ["mev_487b5d38da771a69a14f57eb"],
        }, "CONTRADICTED", "preference_polarity"),
        _case("p06", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer water resistance.",
            "preference_ref": "mev_487b5d38da771a69a14f57eb",
            "evidence_refs": ["mev_487b5d38da771a69a14f57eb"],
        }, "CONTRADICTED", "preference_wrong_source"),
        _case("p07", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer wireless.",
            "preference_ref": "mev_461a92598c65e293218c2636",
            "evidence_refs": ["mev_461a92598c65e293218c2636"],
        }, "CONTRADICTED", "preference_wrong_source"),
        _case("p08", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer wireless.",
            "preference_ref": "",
            "evidence_refs": [],
        }, "INSUFFICIENT_EVIDENCE", "preference_missing_ref"),
        _case("p09", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer wired headphones.",
            "preference_ref": "mev_487b5d38da771a69a14f57eb",
            "evidence_refs": ["mev_487b5d38da771a69a14f57eb"],
        }, "CONTRADICTED", "preference_wrong_value"),
        _case("p10", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You have no preference for water resistance.",
            "preference_ref": "mev_461a92598c65e293218c2636",
            "evidence_refs": ["mev_461a92598c65e293218c2636"],
        }, "CONTRADICTED", "preference_wrong_value"),
        # Additional mixed cases to reach the 40–60 pilot target.
        _case("a09", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product has a 10.7 mm driver.",
            "product_id": "B004K6R43G", "attribute": None, "evidence_refs": ["pev_08"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("a10", "earbud_pref", "B004K6R43G", {
            "claim_type": "product_fact", "claim_text": "The product is made of leather.",
            "product_id": "B004K6R43G", "attribute": "material", "evidence_refs": ["pev_12"],
        }, "INSUFFICIENT_EVIDENCE", "absent_material"),
        _case("b07", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard has tactile brown switches.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("b08", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard is water-resistant.",
            "product_id": "B07WZFL74B", "attribute": "waterproof", "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_waterproof"),
        _case("b09", "keyboard", "B07WZFL74B", {
            "claim_type": "product_fact", "claim_text": "The keyboard has a touchscreen.",
            "product_id": "B07WZFL74B", "attribute": None, "evidence_refs": ["pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_attribute"),
        _case("c06", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard has a full 104-key layout.",
            "product_id": "B09H2JQD48", "attribute": None, "evidence_refs": ["pev_01"],
        }, "SUPPORTED", "supported_product_fact"),
        _case("c07", "mechanical_feeling", "B09H2JQD48", {
            "claim_type": "product_fact", "claim_text": "The keyboard is wireless.",
            "product_id": "B09H2JQD48", "attribute": "wireless", "evidence_refs": ["pev_07", "pev_01"],
        }, "INSUFFICIENT_EVIDENCE", "absent_wireless"),
        _case("d08", "form_conflict", "B00005USA4", {
            "claim_type": "product_fact", "claim_text": "The headphones are in-ear.",
            "product_id": "B00005USA4", "attribute": "form_factor", "evidence_refs": ["pev_08"],
        }, "CONTRADICTED", "contradicted_form_factor"),
        _case("p11", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You prefer wireless and water resistance.",
            "preference_ref": "mev_487b5d38da771a69a14f57eb",
            "evidence_refs": ["mev_487b5d38da771a69a14f57eb", "mev_461a92598c65e293218c2636"],
        }, "SUPPORTED", "multiple_preference_evidence"),
        _case("p12", "earbud_pref", "B004K6R43G", {
            "claim_type": "preference_link", "claim_text": "You dislike in-ear headphones.",
            "preference_ref": "mev_81b3e5aff33de3bd7c85ba11",
            "evidence_refs": ["mev_81b3e5aff33de3bd7c85ba11"],
        }, "CONTRADICTED", "preference_wrong_value"),
    ]

    # Mark a small holdout tail, not used for prompt iteration.
    for item in cases[-8:]:
        item["split"] = "holdout"

    bundles_payload = {key: value.as_dict() for key, value in bundles.items()}
    (OUT / "verifier_bundles.json").write_text(
        json.dumps(bundles_payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    from recommendation.control.rag_evidence import validate_claims
    lines = []
    for item in cases:
        claim = item["claim"]
        bundle = bundles[item["bundle_id"]]
        baseline = validate_claims(bundle, {"claims": [claim]})
        item["baseline"] = {
            "status": baseline["status"],
            "accepted": bool(baseline.get("valid_claims")),
            "rejected": bool(baseline.get("rejected_claims")),
            "reasons": baseline.get("rejected_claims", [{}])[0].get("reasons", []) if baseline.get("rejected_claims") else [],
        }
        lines.append(json.dumps(item, ensure_ascii=False))
    (OUT / "claim_evidence_cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "agentrecx.p20.claim_evidence_pilot.v1",
        "cases": len(cases),
        "splits": {"dev": sum(1 for i in cases if i["split"] == "dev"),
                   "holdout": sum(1 for i in cases if i["split"] == "holdout")},
        "labels": {label: sum(1 for i in cases if i["label"] == label)
                   for label in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")},
        "label_status": "provisional_rule_authored_not_human_verified",
        "bundles": list(bundles_payload),
        "products": sorted({i["product_id"] for i in cases}),
        "notes": [
            "claims authored as evaluation cases over real P15 bundles; not human gold",
            "baseline is the existing deterministic P15 validator",
            "holdout tail is reserved from verifier prompt iteration",
        ],
        "sha256_cases": hashlib.sha256((OUT / "claim_evidence_cases.jsonl").read_bytes()).hexdigest(),
    }
    (OUT / "claim_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("wrote", OUT / "claim_evidence_cases.jsonl")


if __name__ == "__main__":
    main()
