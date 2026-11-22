"""Versioned mapping from real MCP recommendation output to durable reply results.

The real ``recommend_products`` tool returns a single ``selected_product`` plus a Gate and
evidence bundle; it does **not** return ``products``, ``narrative`` or a recommendation id.
Those presentation fields are BFF-derived and must be supplied by the caller.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "CONTRACT_SCHEMA",
    "RecommendationResultError",
    "build_clarification_payload",
    "build_refusal_payload",
    "build_reply_payload",
    "turn_from_payload",
    "validate_recommendation",
]

CONTRACT_SCHEMA = "agentrecx.web.reply_turn.v1"

_FINALIZED = "finalized"
_NO_CANDIDATE = "no_candidates"
_CLARIFY = "awaiting_clarification"


class RecommendationResultError(ValueError):
    """The MCP recommendation response cannot be turned into a valid final reply."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecommendationResultError(message)


def validate_recommendation(recommendation: Any) -> str:
    """Return the business status, rejecting malformed/error responses.

    ``status="finalized"`` alone is not enough: the Gate must also allow finalization and
    the selected product must be internally consistent.
    """
    _require(isinstance(recommendation, Mapping), "recommendation must be a JSON object")
    status = str(recommendation.get("status") or "").strip()
    if recommendation.get("error") or status in {"error", "failed"}:
        raise RecommendationResultError(
            f"recommend_products failed: {recommendation.get('error') or status}"
        )
    if status not in {_FINALIZED, _NO_CANDIDATE, _CLARIFY}:
        raise RecommendationResultError(f"unexpected recommendation status {status!r}")
    if status == _FINALIZED:
        gate = recommendation.get("gate")
        _require(isinstance(gate, Mapping), "finalized recommendation requires a gate object")
        _require(bool(gate.get("allow_finalize")), "gate did not allow finalization")
        selected_id = str(recommendation.get("selected_product_id") or "").strip()
        _require(bool(selected_id), "finalized recommendation has no selected_product_id")
        selected = recommendation.get("selected_product")
        _require(isinstance(selected, Mapping), "finalized recommendation has no selected_product")
        _require(str(selected.get("item_id") or "") == selected_id,
                 "selected_product.item_id disagrees with selected_product_id")
    return status


def _selected_card(selected: Mapping[str, Any]) -> dict[str, Any]:
    item_id = selected.get("item_id")
    # ``item_id`` is the MCP selected-product identity; ``product_id`` is the BFF card
    # identity used by the behavior snapshot.  Both are carried so a restored reply can
    # rebuild its presentation without inventing either one.
    card = {"item_id": item_id, "product_id": item_id, "title": selected.get("title")}
    for key in ("product_type", "attributes", "namespaced_id", "source", "form_factor"):
        if key in selected:
            card[key] = selected[key]
    return {key: value for key, value in card.items() if value is not None}


def _item_identity(item: Mapping[str, Any]) -> str:
    return str(item.get("product_id") or item.get("item_id") or "").strip()


def _products_snapshot(products: Mapping[str, Any] | None, selected: Mapping[str, Any]) -> dict:
    if products is None:
        # Only real data: the selected product is a one-item snapshot.  Never fabricate a
        # larger catalogue list here; callers with the BFF search results pass them in.
        return {"items": [_selected_card(selected)], "source": "mcp_selected_product"}
    _require(isinstance(products, Mapping), "products must be a JSON object")
    items = products.get("items")
    _require(isinstance(items, list) and items, "products.items must be a non-empty list")
    for item in items:
        _require(
            isinstance(item, Mapping) and bool(_item_identity(item)),
            "every product item needs a non-empty product_id (or item_id)",
        )
    snapshot = dict(products)
    if str(selected.get("item_id") or "") not in {_item_identity(item) for item in items}:
        snapshot["items"] = [*items, _selected_card(selected)]
        snapshot["selected_appended"] = True
    return snapshot


def build_reply_payload(
    *,
    reply_turn_id: str,
    pending_id: str,
    user_message: str,
    recommendation: Mapping[str, Any],
    narrative: str,
    recommendation_id: str,
    ranking_source: str,
    products: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and build the single durable reply payload (no nested ``payload`` field)."""
    _require(bool(str(reply_turn_id or "").strip()), "reply_turn_id is required")
    _require(bool(str(pending_id or "").strip()), "pending_id is required")
    _require(bool(str(user_message or "").strip()), "user_message is required")
    status = validate_recommendation(recommendation)
    _require(bool(str(narrative or "").strip()), "assistant narrative is required")
    _require(bool(str(recommendation_id or "").strip()), "recommendation id is required")
    _require(bool(str(ranking_source or "").strip()), "ranking source is required")
    if status == _CLARIFY:
        raise RecommendationResultError(
            "reply requires another clarification; multi-round ASK is not supported yet"
        )

    selected = recommendation.get("selected_product") if status == _FINALIZED else {}
    selected = selected if isinstance(selected, Mapping) else {}
    gate = recommendation.get("gate") if isinstance(recommendation.get("gate"), Mapping) else {}
    bundle = (recommendation.get("evidence_bundle")
              if isinstance(recommendation.get("evidence_bundle"), Mapping) else {})
    evidence = bundle.get("preference_evidence")
    preference_evidence = list(evidence) if isinstance(evidence, list) else []

    return {
        "schema": CONTRACT_SCHEMA,
        "reply_turn_id": str(reply_turn_id),
        "pending_id": str(pending_id),
        "request_id": str(reply_turn_id),
        "user_text": str(user_message),
        "assistant_text": str(narrative),
        "products": _products_snapshot(products, selected),
        "recommendation": {
            "id": str(recommendation_id),
            "ranking_source": str(ranking_source),
            "status": status,
            "schema": recommendation.get("schema"),
            "search_mode": recommendation.get("search_mode"),
            "selected_product_id": recommendation.get("selected_product_id"),
            "gate": {
                key: gate.get(key)
                for key in ("action", "allow_finalize", "selected_candidate_id",
                            "reason_code", "preference_conflict_status",
                            "task_constraint_status")
                if key in gate
            },
        },
        "preference_evidence": preference_evidence,
    }


def turn_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a durable reply payload onto the existing ``web_turn`` TurnView shape."""
    _require(isinstance(payload, Mapping), "reply payload must be an object")
    _require(payload.get("schema") == CONTRACT_SCHEMA,
             f"unsupported reply payload schema {payload.get('schema')!r}")
    products = payload.get("products")
    _require(isinstance(products, Mapping), "reply payload products must be an object")
    return {
        "turn_id": str(payload["reply_turn_id"]),
        "user_text": str(payload["user_text"]),
        "assistant_text": str(payload["assistant_text"]),
        "products": dict(products),
    }


def build_clarification_payload(
    *,
    reply_turn_id: str,
    pending_id: str,
    user_message: str,
    question: str,
    reason: str,
    ranking_source: str,
    query: str,
    retrieval_query: str | None = None,
    task_requirements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable turn for a reply that produced a real next ASK.

    The reply itself succeeded: M9 was written and the MCP returned an explicit
    ``awaiting_clarification`` result with a new handle.  The turn therefore has no product
    and no recommendation id, but it must still be a real persisted turn so the transcript
    shows the question and a replay can rebuild it without re-running MCP.
    """
    _require(bool(str(reply_turn_id or "").strip()), "reply_turn_id is required")
    _require(bool(str(pending_id or "").strip()), "pending_id is required")
    _require(bool(str(user_message or "").strip()), "user_message is required")
    _require(bool(str(question or "").strip()), "the next clarification question is required")
    _require(bool(str(reason or "").strip()), "the next clarification reason is required")
    _require(bool(str(ranking_source or "").strip()), "ranking source is required")
    _require(bool(str(query or "").strip()), "query is required")
    requirements = (
        dict(task_requirements) if isinstance(task_requirements, Mapping) else None
    )
    gate = {
        "action": "NEED_USER_CLARIFICATION",
        "allow_finalize": False,
        "reason_code": str(reason),
        "reason": "one task requirement is still missing",
        "ask_question": str(question),
        "source": "task_clarification_policy",
    }
    products = {
        "items": [],
        "query": str(query),
        "retrieval_query": retrieval_query,
        "search_mode": None,
        "score_kind": None,
        "catalogue_size": None,
        "recommended_product_id": None,
        "gate": gate,
        "recommendation_id": None,
        "request_id": str(reply_turn_id),
        "ranker_version": None,
    }
    return {
        "schema": CONTRACT_SCHEMA,
        "reply_turn_id": str(reply_turn_id),
        "pending_id": str(pending_id),
        "request_id": str(reply_turn_id),
        "user_text": str(user_message),
        "assistant_text": str(question),
        "products": products,
        "recommendation": {
            "id": None,
            "ranking_source": str(ranking_source),
            "status": _CLARIFY,
            "gate": gate,
            "retrieval_query": retrieval_query,
        },
        "preference_evidence": [],
        "clarification": {
            "question": str(question),
            "reason": str(reason),
            "task_requirements": requirements,
        },
    }


def build_refusal_payload(
    *,
    reply_turn_id: str,
    pending_id: str,
    user_message: str,
    narrative: str,
    status: str,
    gate: Mapping[str, Any] | None,
    task_requirements: Mapping[str, Any] | None,
    ranking_source: str,
    budget: Mapping[str, Any] | None = None,
    query: str | None = None,
    retrieval_query: str | None = None,
) -> dict[str, Any]:
    """Build the durable payload for a *bounded refusal* after a real resume.

    A refusal is a known outcome, not an uncertain one: the recommendation ran and the Gate
    (or the task budget) explicitly declined to finalize a product.  It is therefore stored
    and replayable exactly like a successful reply, but with **no product** and an explicit
    machine-readable reason.  It must never be mistaken for a recommendation.
    """
    _require(bool(str(reply_turn_id or "").strip()), "reply_turn_id is required")
    _require(bool(str(pending_id or "").strip()), "pending_id is required")
    _require(bool(str(user_message or "").strip()), "user_message is required")
    _require(bool(str(narrative or "").strip()), "assistant narrative is required")
    _require(bool(str(status or "").strip()), "refusal status is required")
    _require(bool(str(ranking_source or "").strip()), "ranking source is required")
    gate_payload = dict(gate) if isinstance(gate, Mapping) else {}
    budget_payload = dict(budget) if isinstance(budget, Mapping) else None
    requirements_payload = (
        dict(task_requirements) if isinstance(task_requirements, Mapping) else None
    )
    refusal = {
        "code": str(status),
        "reason": str(gate_payload.get("reason_code") or status),
        "detail": str(gate_payload.get("reason") or ""),
        "budget": budget_payload,
        "task_requirements": requirements_payload,
        "retrieval_query": retrieval_query,
    }
    products = {
        "items": [],
        # Keep the original user text (including a superseded budget) as history; the
        # actual retrieval text is reported separately.
        "query": str(query or user_message),
        "retrieval_query": retrieval_query,
        "search_mode": None,
        "score_kind": None,
        "catalogue_size": None,
        "recommended_product_id": None,
        "gate": gate_payload,
        "recommendation_id": None,
        "request_id": str(reply_turn_id),
        "ranker_version": None,
        "refusal": refusal,
    }
    return {
        "schema": CONTRACT_SCHEMA,
        "reply_turn_id": str(reply_turn_id),
        "pending_id": str(pending_id),
        "request_id": str(reply_turn_id),
        "user_text": str(user_message),
        "assistant_text": str(narrative),
        "products": products,
        "recommendation": {
            "id": None,
            "ranking_source": str(ranking_source),
            "status": str(status),
            "gate": gate_payload,
            "retrieval_query": retrieval_query,
        },
        "preference_evidence": [],
        "refusal": refusal,
    }
