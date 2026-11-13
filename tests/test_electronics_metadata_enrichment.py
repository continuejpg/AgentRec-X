"""Unit tests for the optional Raw Meta Electronics sidecar (no network, no full catalogue)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue, ElectronicsProduct
from recommendation.catalog.enrichment import MetadataSidecar, attach_metadata, read_metadata_mode
from recommendation.catalog.schemas import ProductMetadata
from recommendation.control.rag_evidence import build_product_evidence, render_final_answer, validate_claims
from recommendation.preference.current_task_constraints import (
    CurrentTaskConstraint,
    CurrentTaskConstraints,
    candidate_constraint_status,
)


def _product(item_id: str = "B000TEST01", attributes=None, form_factor=None) -> ElectronicsProduct:
    metadata = ProductMetadata(
        parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title="C4 short title",
        main_category="Electronics",
        categories=("Electronics",),
        description=("C4 text with wireless signal.",),
        source="data/p13r/electronics/catalog_meta.jsonl",
    )
    return ElectronicsProduct(
        item_id=item_id,
        namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title="C4 short title",
        text="C4 text with wireless signal.",
        product_type="headphones",
        attributes=dict(attributes or {}),
        form_factor=form_factor,
        record=metadata,
    )


def _sidecar(path: Path) -> MetadataSidecar:
    row = {
        "schema": "agentrecx.electronics_metadata_sidecar.v1",
        "item_id": "B000TEST01",
        "title": "Full upstream headphones title",
        "features": ["Bluetooth 5.3", "Over-ear design"],
        "description": ["Upstream description"],
        "categories": ["Electronics", "Headphones"],
        "store": "Acme",
        "details": {
            "Connectivity Technology": "Wireless, Bluetooth",
            "Form Factor": "Over Ear",
            "Brand": "Acme",
            "Material": "Plastic",
            "Special Feature": "Noise Cancelling",
        },
        "price": 49.99,
        "average_rating": 4.5,
        "rating_number": 120,
        "main_category": "All Electronics",
        "source": "amazon_reviews_2023:raw_meta_Electronics",
        "dataset_version": "test-rev",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return MetadataSidecar.load(path)


def test_sidecar_attaches_structured_fields_without_changing_c4_search_text(tmp_path: Path):
    product = _product()
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    original_text = product.text
    sidecar = _sidecar(tmp_path / "sidecar.jsonl")
    stats = attach_metadata(catalogue, sidecar)

    assert stats["matched_products"] == 1
    assert product.metadata is not None
    assert product.metadata.title == "Full upstream headphones title"
    assert product.metadata.brand == "Acme"
    assert product.metadata.material == "Plastic"
    assert product.metadata.price == 49.99
    assert product.text == original_text
    assert "wireless" in product.attributes
    assert "brand" in product.attributes
    assert product.form_factor == "over_ear"  # metadata form factor, no C4 conflict
    assert catalogue.metadata_mode == "original"  # direct constructor is not auto-enriched


def test_conflicting_form_factor_is_not_used_as_a_hard_fact(tmp_path: Path):
    product = _product(attributes={"in_ear": {"status": "present", "evidence": "in ear"}}, form_factor="in_ear")
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    sidecar = _sidecar(tmp_path / "sidecar.jsonl")
    attach_metadata(catalogue, sidecar)

    assert product.form_factor is None
    assert "in_ear" not in product.attributes
    assert "over_ear" not in product.attributes
    assert "form_factor" in product.metadata_conflicts


def test_inspect_and_evidence_use_upstream_fields_but_keep_price_historical(tmp_path: Path):
    product = _product()
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    sidecar = _sidecar(tmp_path / "sidecar.jsonl")
    attach_metadata(catalogue, sidecar)

    view = catalogue.inspect("B000TEST01", requested=["title", "price", "brand", "material", "wireless", "stock"])
    assert view["metadata_mode"] == "original"
    assert view["facts"]["title"]["metadata_value"] == "Full upstream headphones title"
    assert view["facts"]["price"]["status"] == "historical_metadata"
    assert view["facts"]["brand"]["value"] == "Acme"
    assert view["facts"]["material"]["value"] == "Plastic"
    assert view["facts"]["wireless"]["status"] == "present"
    assert "price" in view["unknown_fields"]
    assert "stock" in view["unknown_fields"]

    selected, evidence, unknown = build_product_evidence(
        catalogue=catalogue, item_id="B000TEST01", namespace=ELECTRONICS_NAMESPACE, task_id="t"
    )
    assert selected["metadata"]["available"] is True
    assert any(item.raw_field == "metadata_details" for item in evidence)
    assert all(item.raw_excerpt[item.span_start:item.span_end] == item.span_text for item in evidence)
    assert selected["metadata"]["price_kind"] == "historical_crawl_time"
    assert any(item.attribute == "price" for item in unknown)


def test_enriched_evidence_can_be_validated_and_rendered(tmp_path: Path):
    product = _product()
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    sidecar = _sidecar(tmp_path / "sidecar.jsonl")
    attach_metadata(catalogue, sidecar)
    selected, evidence, _ = build_product_evidence(
        catalogue=catalogue, item_id="B000TEST01", namespace=ELECTRONICS_NAMESPACE, task_id="t"
    )
    metadata_evidence = next(item for item in evidence if item.raw_field == "metadata_details")
    bundle = __import__("recommendation.control.rag_evidence", fromlist=["EvidenceBundle"]).EvidenceBundle(
        bundle_id="t", task_id="t", namespace=ELECTRONICS_NAMESPACE, current_query="q",
        selected_product=selected, product_evidence=evidence,
    )
    validation = validate_claims(bundle, {"claims": [{
        "claim_type": "product_fact", "claim_text": "The catalogue lists Wireless connectivity.",
        "product_id": "B000TEST01", "evidence_refs": [metadata_evidence.evidence_id],
    }]})
    assert validation["counts"]["product_fact_valid"] == 1
    answer = render_final_answer(bundle, validation)
    assert "Full upstream headphones title" in answer


def test_metadata_mode_validation(monkeypatch):
    monkeypatch.delenv("AGENTRECX_METADATA_MODE", raising=False)
    assert read_metadata_mode() == "original"
    monkeypatch.setenv("AGENTRECX_METADATA_MODE", "enriched")
    assert read_metadata_mode() == "enriched"
    monkeypatch.setenv("AGENTRECX_METADATA_MODE", "bogus")
    try:
        read_metadata_mode()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid metadata mode must be rejected")


def test_metadata_attribute_can_change_task_constraint_status(tmp_path: Path):
    product = _product()
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    attach_metadata(catalogue, _sidecar(tmp_path / "sidecar.jsonl"))
    constraint = CurrentTaskConstraint(
        attribute="wireless", value="wireless", kind="attribute", polarity="exclude",
        source_text="avoid wireless", source_turn_id="turn-1",
    )
    status = candidate_constraint_status(
        attributes=product.attributes, form_factor=product.form_factor,
        constraints=CurrentTaskConstraints(hard_exclusions=[constraint]),
        product_type=product.product_type,
    )
    assert status["violations"], status
    assert any(item["attribute"] == "wireless" for item in status["violations"])


def test_conflicted_form_factor_is_unknown_for_eligibility(tmp_path: Path):
    product = _product(attributes={"in_ear": {"status": "present", "evidence": "in ear"}}, form_factor="in_ear")
    catalogue = ElectronicsCatalogue({"B000TEST01": product}, {})
    attach_metadata(catalogue, _sidecar(tmp_path / "sidecar.jsonl"))
    constraint = CurrentTaskConstraint(
        attribute="in_ear", value="in_ear", kind="attribute", polarity="exclude",
        source_text="avoid in ear", source_turn_id="turn-1",
    )
    status = candidate_constraint_status(
        attributes=product.attributes, form_factor=product.form_factor,
        constraints=CurrentTaskConstraints(hard_exclusions=[constraint]),
        product_type=product.product_type,
    )
    assert not status["violations"]
    assert product.form_factor is None
