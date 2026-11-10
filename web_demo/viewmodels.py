"""Pure payload -> view-model mapping for the Web Demo BFF.

Every function here is side-effect free: it reads a real MCP tool payload and returns the
browser-facing view model defined in ``docs/WEB_DEMO_CONTRACT.md``.  There is no ranking,
filtering, reordering or invention of facts.  Values that the MCP server reports as UNKNOWN
stay UNKNOWN.

Payload sources (all real, from the P17 MCP tools):
    search_products                 -> search_payload
    get_product_info                -> info_payload
    recommend_products              -> recommendation_payload
    compare_products                -> comparison_payload
    get_active_preferences          -> preferences_payload
    request_preference_clarification-> clarification payload
    submit_clarification_reply      -> reply payload
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

PRICE_UNKNOWN_DISPLAY = "Price unavailable / UNKNOWN"
DEFAULT_PRICE_UNKNOWN_REASON = "no reliable numeric price in the catalogue record"
RETRIEVAL_BGE = "BGE dense retrieval"
RETRIEVAL_BM25 = "BM25 fallback"
RETRIEVAL_CATALOGUE = "catalogue direct"

GENERIC_UNKNOWN_REASON = "not established by the catalogue record"

DETAIL_FACT_ORDER = ("title", "price", "brand", "material", "stock", "warranty")
RAW_FIELD_KEYS = ("title", "description", "categories", "features")
_RAW_FIELD_ALIASES = {
    "title": "title",
    "description": "description",
    "features": "features",
    "categories": "categories",
    "main_category": "categories",
}

CATALOGUE_ONLY_NOTE = "catalogue evidence only; no real-world quality/price/stock/rating claim"
NO_HUMAN_REVIEW_NOTE = (
    "semantic_truth_verified=false: automated lexical/structural checks only, not human semantic review"
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_from_epoch(value: Any) -> str | None:
    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shared small mappers
# ---------------------------------------------------------------------------


def unknown_field_items(raw_unknown: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Normalise unknown-field payloads to ``[{attribute, reason}]`` without inventing facts."""
    items: list[dict[str, Any]] = []
    for item in raw_unknown or ():
        if isinstance(item, Mapping):
            attribute = item.get("attribute") or item.get("field")
            if not attribute:
                continue
            items.append(
                {
                    "attribute": str(attribute),
                    "reason": str(item.get("reason") or GENERIC_UNKNOWN_REASON),
                }
            )
        elif item:
            items.append({"attribute": str(item), "reason": GENERIC_UNKNOWN_REASON})
    return items


def evidence_items(product_evidence: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in product_evidence or ():
        items.append(
            {
                "evidence_id": item.get("evidence_id"),
                "raw_field": item.get("raw_field"),
                "span_start": item.get("span_start"),
                "span_end": item.get("span_end"),
                "span_text": item.get("span_text"),
                "raw_excerpt": item.get("raw_excerpt"),
                "attribute": item.get("attribute"),
                "verification_status": item.get("verification_status"),
                "source_record": item.get("source_record"),
            }
        )
    return items


def _source_record(selected_product: Mapping[str, Any] | None,
                   product_evidence: Iterable[Mapping[str, Any]] | None) -> str | None:
    if selected_product and selected_product.get("source_record"):
        return str(selected_product["source_record"])
    for item in product_evidence or ():
        if item.get("source_record"):
            return str(item["source_record"])
    return None


def _price_view(product_info: Mapping[str, Any] | None) -> dict[str, Any]:
    facts = (product_info or {}).get("facts") or {}
    price = facts.get("price") if isinstance(facts, Mapping) else None
    if isinstance(price, Mapping):
        value = price.get("value")
        # Boolean presence flags from catalogue.inspect are not price values; never display them
        # as a price.  Only a real numeric/textual price may leave the UNKNOWN state.
        if price.get("status") == "present" and value not in (None, True, False):
            return {"status": "present", "display": str(value), "value": value}
        reason = price.get("reason") or DEFAULT_PRICE_UNKNOWN_REASON
    else:
        reason = DEFAULT_PRICE_UNKNOWN_REASON
    return {
        "status": "UNKNOWN",
        "display": PRICE_UNKNOWN_DISPLAY,
        "reason": str(reason),
    }


def _attribute_items(
    selected_product: Mapping[str, Any] | None,
    product_evidence: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Verifiable present attributes only; each one is tied to a real evidence span if available."""
    attributes = (selected_product or {}).get("attributes") or {}
    if not isinstance(attributes, Mapping):
        return []
    evidence_by_attribute: dict[str, Mapping[str, Any]] = {}
    for item in product_evidence or ():
        attribute = item.get("attribute")
        if attribute and attribute not in evidence_by_attribute:
            evidence_by_attribute[str(attribute)] = item
    items: list[dict[str, Any]] = []
    for name, fact in attributes.items():
        if not isinstance(fact, Mapping) or fact.get("status") != "present":
            continue
        evidence = evidence_by_attribute.get(str(name))
        items.append(
            {
                "attribute": str(name),
                "status": "present",
                # Presence facts from the catalogue are boolean text-evidence facts.
                "value": "true",
                "evidence_id": evidence.get("evidence_id") if evidence else None,
                "raw_field": evidence.get("raw_field") if evidence else None,
                "span_text": evidence.get("span_text") if evidence else None,
            }
        )
    return items


def fact_items(product_info: Mapping[str, Any] | None,
               attribute_order: Iterable[str] = DETAIL_FACT_ORDER) -> list[dict[str, Any]]:
    facts = (product_info or {}).get("facts") or {}
    items: list[dict[str, Any]] = []
    for attribute in attribute_order:
        fact = facts.get(attribute) if isinstance(facts, Mapping) else None
        if not isinstance(fact, Mapping):
            fact = {"status": "unsupported", "value": None}
        items.append(
            {
                "attribute": attribute,
                "status": fact.get("status") or "unknown",
                "value": fact.get("value"),
                "reason": fact.get("reason"),
                "source": fact.get("source"),
                "verification": fact.get("verification"),
            }
        )
    return items


def search_retrieval_source(search_payload: Mapping[str, Any] | None) -> str:
    payload = search_payload or {}
    score_kind = str(payload.get("score_kind") or "").lower()
    mode = str(payload.get("search_mode") or "").lower()
    if "bge" in score_kind or (not score_kind and mode == "bge"):
        return RETRIEVAL_BGE
    return RETRIEVAL_BM25


# ---------------------------------------------------------------------------
# Product cards / products payload / details
# ---------------------------------------------------------------------------


def product_card(
    info_payload: Mapping[str, Any] | None,
    *,
    source: str,
    retrieval_source: str,
    match_status: str,
    recommended: bool,
    product_role: str | None = None,
) -> dict[str, Any]:
    payload = info_payload or {}
    selected = payload.get("selected_product") or {}
    product_info = payload.get("product_info") or {}
    product_evidence = payload.get("product_evidence") or []
    product_id = payload.get("product_id") or selected.get("item_id")
    namespaced_id = selected.get("namespaced_id") or (
        f"amazon_electronics_c4:{product_id}" if product_id else None
    )
    return {
        "product_id": product_id,
        "namespaced_id": namespaced_id,
        "title": selected.get("title") or product_info.get("title"),
        "product_type": selected.get("product_type") or product_info.get("product_type"),
        "form_factor": selected.get("form_factor") or product_info.get("form_factor"),
        "product_role": product_role,
        "source": source,
        "retrieval_source": retrieval_source,
        "match_status": match_status,
        "recommended": bool(recommended),
        "price": _price_view(product_info),
        "attributes": _attribute_items(selected, product_evidence),
        "unknown_fields": unknown_field_items(payload.get("unknown_fields")),
        "evidence_available": bool(product_evidence),
        "image_url": None,
        "source_record": _source_record(selected, product_evidence),
    }


def gate_view(gate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(gate, Mapping):
        return None
    return {
        "action": gate.get("action"),
        "reason_code": gate.get("reason_code"),
        "reason": gate.get("reason"),
        "selected_candidate_id": gate.get("selected_candidate_id"),
        "unknown_fields": list(gate.get("unknown_fields") or []),
        "product_role": gate.get("product_role"),
        "candidate_presentable_ids": list(gate.get("candidate_presentable_ids") or []),
    }


def products_payload(
    *,
    query: str,
    search_mode: str | None,
    score_kind: str | None,
    catalogue_size: int | None,
    items: list[dict[str, Any]],
    recommended_product_id: str | None,
    gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "query": query,
        "search_mode": search_mode,
        "score_kind": score_kind,
        "catalogue_size": catalogue_size,
        "items": items,
        "recommended_product_id": recommended_product_id,
        "gate": gate_view(gate),
    }


def evidence_payload(info_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``evidence`` SSE event payload for one selected/detailed product."""
    payload = info_payload or {}
    selected = payload.get("selected_product") or {}
    product_evidence = payload.get("product_evidence") or []
    return {
        "product_id": payload.get("product_id") or selected.get("item_id"),
        "evidence": evidence_items(product_evidence),
        "unknown_fields": unknown_field_items(payload.get("unknown_fields")),
        "source_record": _source_record(selected, product_evidence),
    }


def product_detail_payload(
    info_payload: Mapping[str, Any] | None,
    *,
    product_role: str | None = None,
) -> dict[str, Any]:
    payload = info_payload or {}
    selected = payload.get("selected_product") or {}
    product_info = payload.get("product_info") or {}
    product_evidence = payload.get("product_evidence") or []
    unknown = payload.get("unknown_fields") or []
    raw_fields: dict[str, list[str]] = {key: [] for key in RAW_FIELD_KEYS}
    for item in product_evidence:
        target = _RAW_FIELD_ALIASES.get(str(item.get("raw_field") or ""))
        if target and item.get("raw_excerpt"):
            excerpt = str(item["raw_excerpt"])
            if excerpt not in raw_fields[target]:
                raw_fields[target].append(excerpt)
    product_id = payload.get("product_id") or selected.get("item_id")
    return {
        "product_id": product_id,
        "namespaced_id": selected.get("namespaced_id") or (
            f"amazon_electronics_c4:{product_id}" if product_id else None
        ),
        "title": selected.get("title") or product_info.get("title"),
        "product_type": selected.get("product_type") or product_info.get("product_type"),
        "form_factor": selected.get("form_factor") or product_info.get("form_factor"),
        "product_role": product_role,
        "facts": fact_items(product_info),
        "evidence": evidence_items(product_evidence),
        "unknown_fields": unknown_field_items(unknown),
        "raw_fields": raw_fields,
        "source_record": _source_record(selected, product_evidence),
        "evidence_available": bool(product_evidence),
        "notes": [
            "catalogue facts only; UNKNOWN not filled from model knowledge",
            NO_HUMAN_REVIEW_NOTE,
            "raw_fields are exact catalogue excerpts from the evidence spans",
        ],
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _comparison_product(product: Mapping[str, Any]) -> dict[str, Any]:
    product_info = product.get("product_info") or {}
    bundle = product.get("evidence_bundle") or {}
    bundle_unknown = bundle.get("unknown_fields") or []
    product_unknown = product.get("unknown_fields") or []
    unknown = unknown_field_items(bundle_unknown) if bundle_unknown else unknown_field_items(product_unknown)
    return {
        "product_id": product.get("item_id") or (bundle.get("selected_product") or {}).get("item_id"),
        "title": product.get("title"),
        "product_type": product.get("product_type"),
        "role_status": product.get("role_status"),
        "facts": fact_items(product_info),
        "unknown_fields": unknown,
    }


def comparison_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_products = list(payload.get("products") or [])
    products = [_comparison_product(product) for product in raw_products]

    # Dimensions are a formatting view of real per-product facts; neither order nor status is
    # recomputed.  Attribute order follows the first product's requested fact order.
    attribute_order: list[str] = []
    for product in raw_products:
        for attribute in fact_items(product.get("product_info") or {}):
            if attribute["attribute"] not in attribute_order:
                attribute_order.append(attribute["attribute"])
    dimensions: list[dict[str, Any]] = []
    for attribute in attribute_order:
        cells: list[dict[str, Any]] = []
        for raw_product, product in zip(raw_products, products):
            fact = next(
                (item for item in product["facts"] if item["attribute"] == attribute),
                {"status": "unknown", "value": None},
            )
            evidence_ids: list[str] = []
            for claim in raw_product.get("claims") or ():
                if claim.get("claim_type") != "product_fact":
                    continue
                if claim.get("attribute") != attribute:
                    continue
                for reference in claim.get("evidence_refs") or ():
                    if reference not in evidence_ids:
                        evidence_ids.append(reference)
            cells.append(
                {
                    "product_id": product["product_id"],
                    "status": fact.get("status"),
                    "value": fact.get("value"),
                    "evidence_ids": evidence_ids,
                }
            )
        dimensions.append({"attribute": attribute, "cells": cells})

    # Preference relations: per P16 each product carries its own relations; expose the relation
    # per product alongside the shared (real) preference metadata.
    preference_relations: list[dict[str, Any]] = []
    seen_memory_ids: set[str] = set()
    for raw_product in raw_products:
        product_id = raw_product.get("item_id")
        for relation in raw_product.get("preference_relations") or ():
            memory_id = relation.get("memory_id")
            if not memory_id or memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            per_product: dict[str, str | None] = {}
            for other in raw_products:
                match = next(
                    (
                        item
                        for item in other.get("preference_relations") or ()
                        if item.get("memory_id") == memory_id
                    ),
                    None,
                )
                per_product[other.get("item_id")] = match.get("relation") if match else None
            preference_relations.append(
                {
                    "memory_id": memory_id,
                    "value": relation.get("value"),
                    "polarity": relation.get("polarity"),
                    "kind": relation.get("kind"),
                    "relation": relation.get("relation"),
                    "observed_in_catalogue_text": bool(relation.get("observed_in_catalogue_text")),
                    "source_text": relation.get("source_text"),
                    "per_product": per_product,
                }
            )

    conflicts: list[dict[str, Any]] = []
    for error in payload.get("errors") or ():
        if not isinstance(error, Mapping):
            continue
        conflict: dict[str, Any] = {"code": error.get("code")}
        for key in ("message", "detail", "ref", "item_id", "product_id"):
            if error.get(key) is not None:
                conflict[key] = error.get(key)
        conflicts.append(conflict)

    unknown_products: dict[str, list[str]] = {}
    for product in products:
        for item in product["unknown_fields"]:
            unknown_products.setdefault(item["attribute"], []).append(product["product_id"])
    unknown_fields = [
        {"attribute": attribute, "products": ids} for attribute, ids in unknown_products.items()
    ]

    final_answer = payload.get("final_answer") or ""
    conditional_text = None
    if payload.get("conditional_best_supported"):
        for line in str(final_answer).splitlines():
            if line.strip().startswith("Conditional choice:"):
                conditional_text = line.strip()
                break

    sources: list[str] = []
    for raw_product in raw_products:
        source = (raw_product.get("product_info") or {}).get("source")
        if source and source not in sources:
            sources.append(str(source))

    notes = list(payload.get("notes") or ())
    if CATALOGUE_ONLY_NOTE not in notes:
        notes.append(CATALOGUE_ONLY_NOTE)

    return {
        "status": payload.get("status"),
        "products": products,
        "dimensions": dimensions,
        "preference_relations": preference_relations,
        "conflicts": conflicts,
        "unknown_fields": unknown_fields,
        "claims_summary": {
            "valid": payload.get("claims_valid"),
            "rejected": payload.get("claims_rejected"),
            "unknown": payload.get("unknown_count"),
        },
        "conditional_recommendation": {
            "supported": payload.get("conditional_best_supported"),
            "text": conditional_text,
        },
        "final_answer": final_answer,
        "evidence_sources": sources,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Clarification / preferences / memory update
# ---------------------------------------------------------------------------


def clarification_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    question = payload.get("question") or payload.get("final_answer")
    return {
        "clarification_id": payload.get("clarification_id"),
        "question": question,
        "reason": payload.get("reason"),
        "expires_in_seconds": payload.get("expires_in_seconds"),
        "memory_write_enabled": bool(payload.get("memory_write_enabled")),
        # The current P16 clarification question is free-text; no fixed-choice options exist.
        "quick_options": [],
    }


def _preference_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": entry.get("memory_id"),
        "value": entry.get("value"),
        "polarity": entry.get("polarity"),
        "kind": entry.get("kind"),
        "status": entry.get("status"),
        "source_text": entry.get("source_text"),
        "source_turn_id": entry.get("source_turn_id"),
        "last_updated": iso_from_epoch(entry.get("created_at")),
    }


def preference_state(preferences_payload: Mapping[str, Any], *, profile_id: str) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "user_key": preferences_payload.get("authorized_user_key"),
        "history_status": preferences_payload.get("history_status"),
        "active": [_preference_entry(entry) for entry in preferences_payload.get("active_preferences") or ()],
        "inactive": [
            _preference_entry(entry) for entry in preferences_payload.get("inactive_preferences") or ()
        ],
        "audit_source": preferences_payload.get("audit_source"),
        "fetched_at": iso_now(),
    }


def memory_update_payload(
    reply_payload: Mapping[str, Any],
    preferences: Mapping[str, Any] | None,
) -> dict[str, Any]:
    update = reply_payload.get("update") or {}
    already_processed = update.get("already_processed")
    if already_processed is None:
        already_processed = reply_payload.get("already_processed")
    return {
        "added": int(update.get("added") or 0),
        "superseded": int(update.get("superseded") or 0),
        "removed": int(update.get("removed") or 0),
        "skipped_duplicates": int(update.get("skipped_duplicates") or 0),
        "already_processed": bool(already_processed),
        "confirmation_source": reply_payload.get("confirmation_source"),
        "preferences": dict(preferences) if preferences is not None else None,
    }


# ---------------------------------------------------------------------------
# Deterministic narratives (no LLM; only real payload values)
# ---------------------------------------------------------------------------


def narrative_search(
    *,
    search_payload: Mapping[str, Any],
    recommendation_payload: Mapping[str, Any] | None,
    selected_card: Mapping[str, Any] | None,
    gate: Mapping[str, Any] | None,
    catalogue_size: int | None,
) -> str:
    mode = search_payload.get("search_mode")
    score_kind = search_payload.get("score_kind")
    lines = [
        f"Search mode: {mode} ({score_kind}). Catalogue: "
        f"{int(catalogue_size or 0):,} products.",
    ]
    if recommendation_payload is not None:
        lines.append(f"Recommendation status: {recommendation_payload.get('status')}.")
    if gate:
        lines.append(f"Decision Gate: {gate.get('action')} ({gate.get('reason_code')}).")
    if selected_card:
        lines.append(
            f"Selected product: {selected_card.get('title')} ({selected_card.get('product_id')})."
        )
        unknown = [item["attribute"] for item in selected_card.get("unknown_fields") or ()]
        if unknown:
            lines.append("UNKNOWN fields: " + ", ".join(unknown) + ".")
    elif gate:
        lines.append("No product was finalized by the Gate.")
    lines.append(
        "Catalogue evidence only; this is not a claim about real-world quality, price, stock or ratings."
    )
    return "\n".join(lines)


def narrative_qwen(
    *,
    status: str,
    gate: Mapping[str, Any] | None,
    planner: Mapping[str, Any],
    selected_card: Mapping[str, Any] | None,
    clarification: Mapping[str, Any] | None,
    catalogue_size: int | None,
    final_answer: str | None = None,
    planner_steps: Mapping[str, Any] | None = None,
) -> str:
    """Deterministic narrative for a real Qwen turn; P15 final_answer wins when present."""
    if final_answer:
        # The P15 explanation already carries validated evidence claims; do not rewrite it.
        return str(final_answer)
    calls = planner.get("calls")
    lines = [
        f"Planner: {planner.get('model') or 'Qwen3.5-9B-Base'} "
        f"({calls or 0} real model calls). Status: {status}.",
    ]
    if gate:
        lines.append(f"Decision Gate: {gate.get('action')} ({gate.get('reason_code')}).")
    if clarification:
        lines.append(f"Question: {clarification.get('question')}")
        lines.append("No product recommendation has been finalized in this turn.")
    elif selected_card:
        lines.append(
            f"Selected product: {selected_card.get('title')} ({selected_card.get('product_id')})."
        )
        unknown = [item.get("attribute") for item in selected_card.get("unknown_fields") or ()]
        if unknown:
            lines.append("UNKNOWN fields: " + ", ".join(str(item) for item in unknown) + ".")
    else:
        lines.append("No product was finalized by the Decision Gate; nothing was invented.")
    if catalogue_size:
        lines.append(f"Catalogue: {int(catalogue_size):,} products; evidence is catalogue-only.")
    lines.append(
        "Catalogue evidence only; not a claim about real-world quality, price, stock or ratings."
    )
    return "\n".join(lines)


def narrative_clarification(clarification: Mapping[str, Any]) -> str:
    lines = [
        "Recommendation paused: the Decision Gate requires a user clarification before it can select a product.",
        f"Clarification reason: {clarification.get('reason')}.",
        f"Question: {clarification.get('question')}",
    ]
    if not clarification.get("memory_write_enabled"):
        lines.append(
            "Memory writes are disabled on this MCP server; a reply will be refused rather than faked."
        )
    lines.append("No product recommendation has been finalized in this turn.")
    return "\n".join(lines)


def narrative_comparison(comparison: Mapping[str, Any]) -> str:
    payload = comparison.get("comparison") or comparison
    return (
        f"Comparison status: {payload.get('status')}. "
        "Rendered from the real P16 comparison payload; results are catalogue evidence only and "
        "are not an objective quality ranking.\n\n"
        + str(payload.get("final_answer") or "")
    ).strip()


def narrative_memory(
    *,
    profile_id: str,
    memory_update: Mapping[str, Any],
    preference_payload: Mapping[str, Any],
    recommendation_payload: Mapping[str, Any] | None,
) -> str:
    lines = [
        f"Memory update ({profile_id}): added {memory_update.get('added')}, "
        f"superseded {memory_update.get('superseded')}, removed {memory_update.get('removed')}, "
        f"skipped duplicates {memory_update.get('skipped_duplicates')} "
        f"(already_processed={bool(memory_update.get('already_processed'))}, "
        f"confirmation_source={memory_update.get('confirmation_source')}).",
        f"Active preferences refreshed from M9: {len(preference_payload.get('active') or [])} active, "
        f"{len(preference_payload.get('inactive') or [])} inactive.",
    ]
    if recommendation_payload is not None:
        lines.append(f"Pending query resumed: recommend_products status "
                     f"{recommendation_payload.get('status')}.")
        selected = recommendation_payload.get("selected_product") or {}
        if selected:
            lines.append(
                f"Selected product: {selected.get('title')} ({selected.get('item_id')})."
            )
    lines.append(
        "Catalogue evidence only; this is not a claim about real-world quality, price, stock or ratings."
    )
    return "\n".join(lines)
