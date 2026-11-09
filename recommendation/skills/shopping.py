"""P16 shopping-comparison skill: 2–5 real products, evidence-grounded deterministic table."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.control.rag_evidence import build_bundle, validate_claims
from recommendation.preference.product_role import assess_candidate_role, infer_product_target

MAX_PRODUCTS = 5
MIN_PRODUCTS = 2


@dataclass
class ComparisonResult:
    schema: str = "agentrecx.p16.shopping_comparison.v1"
    status: str = "comparison_ready"
    current_query: str = ""
    products: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    conditional_best_supported: str | None = None
    final_answer: str = ""
    qwen_used: bool = False
    tool_calls: list[str] = field(default_factory=list)
    claims_valid: int = 0
    claims_rejected: int = 0
    unknown_count: int = 0
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "status": self.status,
            "current_query": self.current_query, "products": self.products,
            "errors": self.errors, "conditional_best_supported": self.conditional_best_supported,
            "final_answer": self.final_answer, "qwen_used": self.qwen_used,
            "tool_calls": self.tool_calls, "claims_valid": self.claims_valid,
            "claims_rejected": self.claims_rejected, "unknown_count": self.unknown_count,
            "latency_ms": self.latency_ms,
            "notes": ["catalogue facts describe only the product record; no quality/stock/rating claim is inferred",
                      "preference relations are catalogue-text observations, not proof of user satisfaction",
                      "conditional_best_supported is not an objective quality ranking"],
        }


def _resolve_refs(catalogue: ElectronicsCatalogue, refs: Iterable[str], namespace: str) -> tuple[list[Any], list[dict]]:
    products = []
    errors = []
    seen = set()
    for raw in refs:
        text = str(raw or "").strip()
        if ":" in text:
            ref_namespace, item_id = text.split(":", 1)
        else:
            ref_namespace, item_id = namespace, text
        if ref_namespace != namespace:
            errors.append({"code": "namespace_mismatch", "ref": text,
                           "expected_namespace": namespace, "got_namespace": ref_namespace})
            continue
        if not item_id or item_id in seen:
            errors.append({"code": "duplicate_or_empty_product", "ref": text})
            continue
        product = catalogue.get(item_id)
        if product is None:
            errors.append({"code": "product_not_found", "ref": text,
                           "message": "product identity is not in the current Electronics catalogue"})
            continue
        seen.add(item_id)
        products.append(product)
    return products, errors


def _preference_relations(*, bundle: Mapping[str, Any], history: Mapping[str, Any]) -> list[dict[str, Any]]:
    active = history.get("active_preferences") or []
    evidence_text = " ".join(
        [str(item.get("span_text") or "") for item in bundle.get("product_evidence") or []]
        + [str((bundle.get("selected_product") or {}).get("title") or "")]
    ).casefold()
    normalised_text = " ".join(re.sub(r"[^a-z0-9]+", " ", evidence_text).split())
    relations = []
    for pref in active:
        value = str(pref.get("value") or "").strip()
        phrase = " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())
        # Avoid false positives such as matching "in-ear" through the shared token "ear".
        observed = bool(phrase) and (
            phrase in normalised_text
            or (lambda tokens: bool(tokens) and all(token in normalised_text.split() for token in tokens))(
                [token for token in phrase.split() if len(token) >= 4]
            )
        )
        relations.append({
            "memory_id": pref.get("memory_id"), "value": value,
            "polarity": pref.get("polarity"), "kind": pref.get("kind"),
            "observed_in_catalogue_text": observed,
            "relation": "observed" if observed else "not_evidenced",
            "source_text": pref.get("source_text"),
        })
    return relations


def _product_claims(bundle: Mapping[str, Any]) -> dict[str, Any]:
    selected = (bundle.get("selected_product") or {}).get("item_id")
    product_evidence = bundle.get("product_evidence") or []
    title_refs = [item["evidence_id"] for item in product_evidence if item.get("raw_field") == "title"][:2]
    category_refs = [item["evidence_id"] for item in product_evidence
                     if item.get("raw_field") in {"main_category", "categories"}][:2]
    claims = []
    if title_refs:
        claims.append({"claim_type": "product_fact",
                       "claim_text": f"Catalogue title: {(bundle.get('selected_product') or {}).get('title')}",
                       "product_id": selected, "attribute": "title", "evidence_refs": title_refs})
    if category_refs:
        claims.append({"claim_type": "product_fact",
                       "claim_text": "The catalogue lists this product in the Electronics category.",
                       "product_id": selected, "attribute": "category", "evidence_refs": category_refs})
    for unknown in bundle.get("unknown_fields") or []:
        claims.append({"claim_type": "unknown",
                       "claim_text": f"{unknown.get('attribute')} is UNKNOWN: {unknown.get('reason')}",
                       "attribute": unknown.get("attribute")})
    validation = validate_claims_from_dict(bundle, {"claims": claims})
    return {"claims": claims, "validation": validation}


def validate_claims_from_dict(bundle: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Small adapter: reconstruct an EvidenceBundle from its JSON view for P15 validation."""
    from recommendation.control.rag_evidence import (
        EvidenceBundle, PreferenceEvidence, ProductEvidence, UnknownFact,
    )
    product = bundle.get("selected_product")
    pe = [ProductEvidence(**item) for item in bundle.get("product_evidence") or []]
    pr = [PreferenceEvidence(**item) for item in bundle.get("preference_evidence") or []]
    un = [UnknownFact(**item) for item in bundle.get("unknown_fields") or []]
    obj = EvidenceBundle(
        bundle_id=str(bundle.get("bundle_id")), task_id=str(bundle.get("task_id")),
        namespace=str(bundle.get("namespace")), current_query=str(bundle.get("current_query")),
        selected_product=product, product_evidence=pe, preference_evidence=pr,
        unknown_fields=un, ledger_source=str(bundle.get("ledger_source") or ""),
        search_mode=str(bundle.get("search_mode") or ""),
        product_type=bundle.get("product_type"), role_status=bundle.get("role_status"),
        eligibility=dict(bundle.get("eligibility") or {}),
        gate_decision=dict(bundle.get("gate_decision") or {}), user_id=str(bundle.get("user_id") or ""),
    )
    return validate_claims(obj, payload)


def shopping_comparison(
    *,
    catalogue: ElectronicsCatalogue,
    task: Mapping[str, Any],
    history: Mapping[str, Any],
    product_refs: Iterable[str],
) -> dict[str, Any]:
    """Compare 2–5 explicit real products; no Qwen call is needed for the table."""
    started = time.perf_counter()
    result = ComparisonResult(current_query=str(task.get("current_query") or ""))
    refs = list(product_refs)
    if not MIN_PRODUCTS <= len(refs) <= MAX_PRODUCTS:
        result.status = "invalid_product_count"
        result.errors.append({"code": "invalid_product_count",
                              "message": f"comparison requires {MIN_PRODUCTS}-{MAX_PRODUCTS} products, got {len(refs)}"})
        result.final_answer = f"I need {MIN_PRODUCTS}-{MAX_PRODUCTS} confirmed product references to compare."
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result.as_dict()
    namespace = str(task.get("catalog_namespace") or "amazon_electronics_c4")
    products, errors = _resolve_refs(catalogue, refs, namespace)
    result.tool_calls.append("catalogue_identity_resolution")
    if errors:
        result.status = "invalid_product_reference"
        result.errors.extend(errors)
        result.final_answer = ("I can't verify those product references. Please provide 2–5 product IDs or "
                               "namespaced references from the current Electronics catalogue.")
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result.as_dict()
    target = infer_product_target(str(task.get("current_query") or ""))
    for product in products:
        role = assess_candidate_role(product_type=product.product_type, title=product.title, target=target)
        if target.product_type and role.status.value not in {"MATCH", "NOT_REQUIRED"}:
            result.status = "role_mismatch"
            result.errors.append({"code": "role_mismatch", "item_id": product.item_id,
                                  "role_status": role.status.value,
                                  "message": "an accessory/mismatched product cannot be compared as the main product"})
            continue
        product_info = catalogue.inspect(product.item_id, requested=["title", "price", "brand", "material", "stock", "warranty"])
        result.tool_calls.append("product_info")
        bundle = build_bundle(
            task=task, selected_item_id=product.item_id, catalogue=catalogue,
            history=history, gate_decision={"skill": "shopping-comparison", "comparison_only": True},
            search_mode=catalogue.search_source.mode,
            ledger_source=f"CatalogSearchSource(mode={catalogue.search_source.mode})",
        )
        bundle.role_status = role.status.value
        claim_data = _product_claims(bundle.as_dict())
        if claim_data["validation"].get("rejected_claims"):
            result.errors.append({"code": "claim_validation_failed", "item_id": product.item_id,
                                  "rejected": claim_data["validation"]["rejected_claims"]})
        result.products.append({
            "item_id": product.item_id, "namespace": product.namespaced_id.split(":", 1)[0],
            "title": product.title, "product_type": product.product_type,
            "role_status": role.status.value,
            "product_info": product_info,
            "evidence_bundle": bundle.as_dict(),
            "claims": claim_data["claims"],
            "claim_validation": claim_data["validation"],
            "preference_relations": _preference_relations(bundle=bundle.as_dict(), history=history),
            "unknown_fields": [item.attribute for item in bundle.unknown_fields],
        })
        result.claims_valid += len(claim_data["validation"].get("valid_claims") or [])
        result.claims_rejected += len(claim_data["validation"].get("rejected_claims") or [])
        result.unknown_count += len(bundle.unknown_fields)
    if result.status == "role_mismatch" or len(result.products) != len(products):
        result.status = "invalid_product_reference" if not result.products else "role_mismatch"
        result.final_answer = ("I can't compare these as the requested main product because at least one reference "
                               "is an accessory or has a mismatched product role. Please provide main-product references.")
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result.as_dict()

    # Deterministic, preference-aware conditional ordering; not a quality claim.
    def score(product: dict[str, Any]) -> int:
        relations = product["preference_relations"]
        matched = sum(1 for rel in relations if rel["observed_in_catalogue_text"] and rel["polarity"] == "prefer")
        conflicts = sum(1 for rel in relations if rel["observed_in_catalogue_text"] and rel["polarity"] == "avoid")
        return matched - conflicts

    scores = [(score(product), product) for product in result.products]
    best = max(score_value for score_value, _product in scores)
    best_products = [product for score_value, product in scores if score_value == best]
    if best > 0 and len(best_products) == 1:
        result.conditional_best_supported = best_products[0]["item_id"]
    result.status = "comparison_ready"
    result.final_answer = _render_comparison(result)
    result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return result.as_dict()


def _render_comparison(result: ComparisonResult) -> str:
    lines = [f"Comparison of {len(result.products)} products (catalogue evidence only):"]
    for index, product in enumerate(result.products, start=1):
        lines.append(f"\n{index}. {product['title']} ({product['item_id']})")
        facts = [claim for claim in product["claims"] if claim["claim_type"] == "product_fact"]
        for claim in facts:
            lines.append(f"   - {claim['claim_text']} [evidence: {', '.join(claim['evidence_refs'])}]")
        relations = product["preference_relations"]
        if relations:
            observed = [rel for rel in relations if rel["observed_in_catalogue_text"]]
            if observed:
                lines.append("   - Catalogue text relates to active preferences: " +
                             "; ".join(f"{rel['value']} ({rel['polarity']}, {rel['memory_id']})" for rel in observed))
            else:
                lines.append("   - Active preferences are recorded, but the catalogue text does not confirm them for this product.")
        lines.append("   - UNKNOWN: " + ", ".join(product["unknown_fields"]))
    if result.conditional_best_supported:
        selected = next(product for product in result.products
                        if product["item_id"] == result.conditional_best_supported)
        lines.append(f"\nConditional choice: {selected['title']} ({selected['item_id']}) has the strongest "
                     "observed match to active stated preferences. This is not an objective quality ranking.")
    else:
        lines.append("\nNo unique conditional preference winner: the evidence is tied or insufficient; "
                     "I cannot make a reliable choice from this comparison alone.")
    lines.append("Catalogue evidence does not prove real-world quality, stock, ratings, or user satisfaction.")
    return "\n".join(lines)
