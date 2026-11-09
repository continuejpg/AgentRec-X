"""P15 evidence-grounded recommendation explanation primitives.

This is deliberately small: it turns an already-Gated FINALIZE selection and the existing
M5/M9 history read into an evidence bundle, validates model-authored claims against that bundle,
and renders a user-readable answer.  It does not retrieve products, call a planner, change a Gate,
write memory, or create a second Agent runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from experiments.p14e.catalogue import ElectronicsCatalogue, ElectronicsProduct

EVIDENCE_BUNDLE_SCHEMA = "agentrecx.p15.evidence_bundle.v1"
CLAIM_VALIDATION_SCHEMA = "agentrecx.p15.claim_validation.v1"
_ALLOWED_CLAIM_TYPES = {"product_fact", "preference_link", "unknown", "recommendation"}
_ALWAYS_UNKNOWN = {"price", "brand", "size", "material", "stock", "warranty"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProductEvidence:
    evidence_id: str
    namespace: str
    product_id: str
    source_record: str
    raw_field: str
    span_start: int
    span_end: int
    span_text: str
    raw_excerpt: str
    attribute: str | None
    verification_status: str = "lexical_span_exact"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceEvidence:
    evidence_id: str
    preference_id: str
    user_id: str
    status: str
    polarity: str
    kind: str
    value: str
    source_text: str
    source_turn_id: str
    memory_id: str
    active: bool
    source: str = "M9:PreferenceMemoryService.get_memory_history"
    verification_status: str = "memory_event_exact"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnknownFact:
    attribute: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    bundle_id: str
    task_id: str
    namespace: str
    current_query: str
    selected_product: dict[str, Any] | None
    product_evidence: list[ProductEvidence] = field(default_factory=list)
    preference_evidence: list[PreferenceEvidence] = field(default_factory=list)
    unknown_fields: list[UnknownFact] = field(default_factory=list)
    ledger_source: str = ""
    search_mode: str = ""
    product_type: str | None = None
    role_status: str | None = None
    eligibility: dict[str, Any] = field(default_factory=dict)
    gate_decision: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_BUNDLE_SCHEMA,
            "bundle_id": self.bundle_id,
            "task_id": self.task_id,
            "namespace": self.namespace,
            "current_query": self.current_query,
            "selected_product": self.selected_product,
            "product_evidence": [item.as_dict() for item in self.product_evidence],
            "preference_evidence": [item.as_dict() for item in self.preference_evidence],
            "unknown_fields": [item.as_dict() for item in self.unknown_fields],
            "ledger_source": self.ledger_source,
            "search_mode": self.search_mode,
            "product_type": self.product_type,
            "role_status": self.role_status,
            "eligibility": self.eligibility,
            "gate_decision": self.gate_decision,
            "user_id": self.user_id,
            "notes": [
                "product evidence proves only what the catalogue field says, not product quality, stock, ratings or real-world performance",
                "preference evidence is read from the authorised M9 history; inactive/removed/superseded preferences are never active evidence",
                "UNKNOWN fields are not facts and must not be filled from model knowledge",
            ],
        }


def _find_span(raw: str, needle: str) -> tuple[int, int] | None:
    if not raw or not needle:
        return None
    start = raw.casefold().find(needle.casefold())
    if start < 0:
        return None
    return start, start + len(needle)


def _raw_fields(product: ElectronicsProduct) -> dict[str, str]:
    record = product.record
    if record is None:
        return {"title": product.title, "description": product.text}
    fields: dict[str, str] = {}

    def add(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (tuple, list)):
            text = "\n".join(str(part).strip() for part in value if str(part).strip())
        else:
            return
        if text:
            fields[name] = text

    add("title", getattr(record, "title", None) or product.title)
    add("main_category", getattr(record, "main_category", None))
    add("categories", getattr(record, "categories", None))
    add("features", getattr(record, "features", None))
    add("description", getattr(record, "description", None))
    add("price_text", getattr(record, "price_text", None))
    return fields


def build_product_evidence(
    *, catalogue: ElectronicsCatalogue, item_id: str, namespace: str, task_id: str
) -> tuple[dict[str, Any] | None, list[ProductEvidence], list[UnknownFact]]:
    product = catalogue.get(item_id)
    if product is None:
        return None, [], [UnknownFact("product", "product is not present in the current catalogue")]
    raw = _raw_fields(product)
    source_record = (getattr(product.record, "source", None) or "catalogue") + f":{product.namespaced_id}"
    evidence: list[ProductEvidence] = []

    def add(field_name: str, attribute: str | None = None) -> None:
        raw_text = raw.get(field_name, "")
        if not raw_text:
            return
        # Whole-field evidence is a real span; for attributes we narrow to the lexical term.
        needle = attr_needle = attribute.replace("_", " ") if attribute else raw_text
        span = _find_span(raw_text, chosen := raw_text) if not attribute else _find_span(raw_text, attr_needle)
        if span is None and attribute:
            # Try the exact attribute token and the catalogue extraction's evidence text.
            candidates = [attribute, attribute.replace("_", "-"), attr_needle]
            extracted = (product.attributes.get(attribute) or {}).get("evidence")
            if extracted:
                candidates.append(str(extracted))
            for candidate in candidates:
                span = _find_span(raw_text, candidate)
                if span is not None:
                    chosen = raw_text[span[0]:span[1]]
                    break
        if span is None:
            return
        chosen = raw_text[span[0]:span[1]]
        evidence_id = f"pev_{len(evidence) + 1:02d}"
        evidence.append(ProductEvidence(
            evidence_id=evidence_id, namespace=namespace, product_id=item_id,
            source_record=source_record, raw_field=field_name,
            span_start=span[0], span_end=span[1], span_text=chosen,
            raw_excerpt=raw_text[:1200], attribute=attribute,
        ))

    add("title")
    add("main_category")
    add("categories")
    for attribute, fact in product.attributes.items():
        if not isinstance(fact, dict) or fact.get("status") != "present":
            continue
        for field_name in ("title", "description", "features"):
            if raw.get(field_name) and _find_span(raw[field_name], attribute.replace("_", " ")) or (
                raw.get(field_name) and _find_span(raw[field_name], attribute)
            ):
                add(field_name, attribute)
                break
    add("features")
    add("description")
    add("price_text")

    unknown: dict[str, UnknownFact] = {}

    def mark(attribute: str, reason: str) -> None:
        unknown.setdefault(attribute, UnknownFact(attribute, reason))

    if not raw.get("price_text"):
        mark("price", "no reliable numeric price in the catalogue record")
    if not getattr(product.record, "store", None):
        mark("brand", "the catalogue record has no trusted brand/store field")
    for attribute, reason in (
        ("stock", "the catalogue snapshot is not a live availability feed"),
        ("material", "no trusted material field in the catalogue record"),
        ("size", "no trusted size field in the catalogue record"),
        ("warranty", "the catalogue record carries no warranty statement"),
    ):
        mark(attribute, reason)
    return (product.as_dict() | {"source_record": source_record}), evidence, list(unknown.values())


def build_preference_evidence(*, history: Mapping[str, Any], user_id: str) -> list[PreferenceEvidence]:
    items: list[PreferenceEvidence] = []
    for status_key, active in (("active_preferences", True), ("inactive_preferences", False)):
        for entry in history.get(status_key) or ():
            memory_id = str(entry.get("memory_id") or "")
            if not memory_id:
                continue
            items.append(PreferenceEvidence(
                evidence_id=("mev_" if active else "miv_") + memory_id,
                preference_id=memory_id,
                user_id=str(history.get("user_key") or user_id),
                status="active" if active else str(entry.get("status") or "inactive"),
                polarity=str(entry.get("polarity") or "unknown"),
                kind=str(entry.get("kind") or "unknown"),
                value=str(entry.get("value") or ""),
                source_text=str(entry.get("source_text") or ""),
                source_turn_id=str(entry.get("source_turn_id") or ""),
                memory_id=memory_id,
                active=active,
            ))
    return items


def build_bundle(
    *, task: Mapping[str, Any], selected_item_id: str, catalogue: ElectronicsCatalogue,
    history: Mapping[str, Any], gate_decision: Mapping[str, Any], search_mode: str,
    ledger_source: str, bundle_prefix: str = "p15",
) -> EvidenceBundle:
    selected, product_evidence, unknown = build_product_evidence(
        catalogue=catalogue, item_id=selected_item_id,
        namespace=str(task.get("catalog_namespace") or "amazon_electronics_c4"),
        task_id=str(task["task_id"]),
    )
    user_id = str(history.get("user_key") or task.get("user_id") or "")
    preferences = build_preference_evidence(history=history, user_id=user_id)
    bundle_id = f"{bundle_prefix}_{task['task_id']}_{selected_item_id}"
    role_status = None
    eligibility = {}
    if selected:
        eligibility = {"selected_product_id": selected_item_id}
    return EvidenceBundle(
        bundle_id=bundle_id, task_id=str(task["task_id"]),
        namespace=str(task.get("catalog_namespace") or "amazon_electronics_c4"),
        current_query=str(task.get("current_query") or ""), selected_product=selected,
        product_evidence=product_evidence, preference_evidence=preferences,
        unknown_fields=unknown, ledger_source=ledger_source, search_mode=search_mode,
        product_type=selected.get("product_type") if selected else None,
        role_status=role_status, eligibility=eligibility,
        gate_decision=dict(gate_decision), user_id=user_id,
    )


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Tolerate a model that adds one short sentence around a JSON object.
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                return None, f"invalid JSON: {exc.msg}"
        else:
            return None, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "top-level JSON is not an object"
    return payload, None


def validate_claims(bundle: EvidenceBundle, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate model-authored claims against the bundle.

    This is *automated structural/lexical validation*, not semantic truth verification. It checks
    that references exist, belong to the selected product/current user, spans are exact, and that
    inactive preferences/UNKNOWN fields are not promoted to active facts.
    """
    product_refs = {item.evidence_id: item for item in bundle.product_evidence}
    pref_refs = {item.evidence_id: item for item in bundle.preference_evidence}
    unknown = {item.attribute for item in bundle.unknown_fields}
    selected = (bundle.selected_product or {}).get("item_id")
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing_refs: list[str] = []
    claims = payload.get("claims") if isinstance(payload, Mapping) else None
    if not isinstance(claims, list):
        return {"schema": CLAIM_VALIDATION_SCHEMA, "valid_claims": [], "rejected_claims": [],
                "missing_evidence_refs": [], "status": "invalid_model_payload",
                "automated_validation": True, "semantic_truth_verified": False}
    for index, claim in enumerate(claims):
        record = {"claim_id": f"claim_{index + 1}", "claim": claim}
        if not isinstance(claim, Mapping):
            record.update({"status": "rejected", "reasons": ["claim is not an object"]})
            rejected.append(record)
            continue
        claim_type = str(claim.get("claim_type") or "")
        text = str(claim.get("claim_text") or "").strip()
        refs = claim.get("evidence_refs") or []
        if not isinstance(refs, list):
            refs = []
        refs = [str(ref) for ref in refs]
        reasons: list[str] = []
        if claim_type not in _ALLOWED_CLAIM_TYPES:
            reasons.append("unsupported claim_type")
        if not text:
            reasons.append("claim_text is empty")
        if claim_type == "product_fact":
            product_id = claim.get("product_id") or selected
            if not selected or product_id != selected:
                reasons.append("product_fact does not refer to the selected product")
            claimed_attribute = str(claim.get("attribute") or "")
            if claimed_attribute and claimed_attribute in unknown:
                reasons.append(f"product_fact promotes an UNKNOWN field to a known fact: {claimed_attribute}")
            if not refs:
                reasons.append("product_fact has no evidence_refs")
            for ref in refs:
                item = product_refs.get(ref)
                if item is None:
                    missing_refs.append(ref)
                    reasons.append(f"product evidence ref does not exist: {ref}")
                    continue
                if item.product_id != selected:
                    reasons.append(f"product evidence belongs to another product: {ref}")
                field_raw = (bundle.selected_product or {}).get("source_record") if False else None
                if item.raw_excerpt[item.span_start:item.span_end] != item.span_text:
                    reasons.append(f"product evidence span is not exact: {ref}")
            # A product claim with only memory refs is a category confusion.
            if any(ref in pref_refs for ref in refs):
                reasons.append("product_fact cites preference/memory evidence")
        elif claim_type == "preference_link":
            preference_ref = str(claim.get("preference_ref") or "")
            if not refs and preference_ref:
                refs = [preference_ref]
            if not refs:
                reasons.append("preference_link has no preference evidence")
            for ref in refs:
                item = pref_refs.get(ref)
                if item is None:
                    missing_refs.append(ref)
                    reasons.append(f"preference evidence ref does not exist: {ref}")
                    continue
                if not item.active:
                    reasons.append(f"inactive/retracted preference used as active evidence: {ref}")
                if item.user_id != bundle.user_id:
                    reasons.append(f"preference belongs to another user: {ref}")
                if any(evidence_ref in product_refs for evidence_ref in [preference_ref]):
                    reasons.append("preference_link cites product evidence")
            if any(ref in product_refs for ref in refs):
                reasons.append("preference_link cites product evidence")
        elif claim_type == "unknown":
            attribute = str(claim.get("attribute") or "")
            if attribute not in unknown:
                reasons.append(f"UNKNOWN claim does not name a bundle-unknown field: {attribute or '<empty>'}")
            if refs:
                reasons.append("UNKNOWN claim must not cite product/preference evidence")
        elif claim_type == "recommendation":
            if refs:
                reasons.append("recommendation judgment must not carry fact evidence refs")
        record["status"] = "automated_validated" if not reasons else "rejected"
        record["reasons"] = reasons
        record["claim_type"] = claim_type
        record["claim_text"] = text
        record["evidence_refs"] = refs
        (valid if not reasons else rejected).append(record)
    result = {
        "schema": CLAIM_VALIDATION_SCHEMA,
        "bundle_id": bundle.bundle_id,
        "selected_product_id": selected,
        "valid_claims": valid,
        "rejected_claims": rejected,
        "missing_evidence_refs": missing_refs,
        "counts": {
            "claims": len(valid) + len(rejected), "valid": len(valid), "rejected": len(rejected),
            "product_fact_valid": sum(1 for claim in valid if claim["claim_type"] == "product_fact"),
            "preference_link_valid": sum(1 for claim in valid if claim["claim_type"] == "preference_link"),
            "unknown_valid": sum(1 for claim in valid if claim["claim_type"] == "unknown"),
            "missing_evidence_refs": len(missing_refs),
        },
        "status": "automated_validated" if valid else "no_valid_claims",
        "automated_validation": True,
        "semantic_truth_verified": False,
        "notes": ["validated references/spans and active-user scope; this is not a semantic entailment proof"],
    }
    return result


def render_final_answer(bundle: EvidenceBundle, validation: Mapping[str, Any]) -> str:
    valid = validation.get("valid_claims") or []
    product_facts = [claim for claim in valid if claim.get("claim_type") == "product_fact"]
    pref_links = [claim for claim in valid if claim.get("claim_type") == "preference_link"]
    unknowns = [claim for claim in valid if claim.get("claim_type") == "unknown"]
    recommendations = [claim for claim in valid if claim.get("claim_type") == "recommendation"]
    if not bundle.selected_product:
        return "I can't verify a product recommendation because no selected product is available."
    lines = [f"Recommended product: {bundle.selected_product.get('title')} ({bundle.selected_product.get('item_id')})."]
    if product_facts:
        lines.append("Catalogue facts:")
        for claim in product_facts:
            refs = ", ".join(claim.get("evidence_refs") or [])
            lines.append(f"- {claim.get('claim_text')} [evidence: {refs}]")
    if pref_links:
        lines.append("How this relates to your stated preferences:")
        for claim in pref_links:
            refs = ", ".join(claim.get("evidence_refs") or [])
            lines.append(f"- {claim.get('claim_text')} [memory: {refs}]")
    if unknowns:
        lines.append("What I cannot confirm:")
        for claim in unknowns:
            lines.append(f"- {claim.get('claim_text')}")
    elif bundle.unknown_fields:
        # The model may omit the limitation; the catalogue still does not provide these fields.
        attrs = ", ".join(item.attribute for item in bundle.unknown_fields)
        lines.append("What I cannot confirm: " + attrs + " are UNKNOWN in the catalogue record.")
    if recommendations:
        lines.append("Why I selected it:")
        for claim in recommendations:
            lines.append(f"- {claim.get('claim_text')}")
    else:
        basis = ("the cited catalogue facts and active preference links above" if pref_links
                 else "the cited catalogue facts above")
        lines.append("Why I selected it: the existing Decision Gate selected this candidate as the best supported "
                     "allowed candidate; " + basis + " are the only evidence basis for this recommendation.")
    if not product_facts and not pref_links and not recommendations:
        lines.append("I can't verify specific product or preference facts from the available evidence.")
    lines.append("Catalogue evidence only describes the product record; it does not prove real-world quality, availability, or user experience.")
    return "\n".join(lines)
