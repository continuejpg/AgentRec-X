"""Pure BFF presentation assembly for durable warm replies (M1-C Round B/C).

A persistent Warm Reply has to return exactly what the in-memory BFF path returns for the same
real MCP results: the same product cards, the same deterministic narrative, its own
``recommendation_id`` and the same ``ranking_source``.  This module is that single
implementation.  Both the HTTP warm path and the existing in-memory
``demo_flow.clarification_reply`` read their presentation from here, so the two modes cannot
drift apart.

Rules:

* it never writes Session state, never records behavior snapshots and never calls MCP;
* every field is derived from real tool payloads (``recommend_products``,
  ``get_product_info``, ``submit_clarification_reply``, ``get_active_preferences``);
* a missing or incomplete selected-product record is a hard error, never an empty
  recommendation;
* the recommendation id is supplied by the caller and issued exactly once (it is part of the
  durable result, so a replay reuses it instead of generating a new one).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from web_demo import viewmodels as vm
from web_demo.behavior import EXISTING_RANKER_VERSION
from web_demo.reply_result_contract import (
    RecommendationResultError,
    turn_from_payload,
    validate_recommendation,
)

__all__ = [
    "ReplyAssemblyError",
    "ReplyAssemblyRequest",
    "assemble_refusal_memory",
    "assemble_reply_presentation",
    "new_recommendation_id",
    "prepare_recommendation_products",
    "reply_response_from_result",
    "turn_from_result",
]

_FINALIZED = "finalized"


class ReplyAssemblyError(ValueError):
    """The real MCP results cannot be turned into a complete reply presentation."""


@dataclass(frozen=True)
class ReplyAssemblyRequest:
    """The real tool payloads one reply presentation is built from."""

    recommendation: Mapping[str, Any]
    product_infos: Mapping[str, Mapping[str, Any]]
    reply: Mapping[str, Any]
    preference_payload: Mapping[str, Any]
    query: str
    recommendation_id: str
    ranking_source: str
    profile_id: str
    turn_id: str


def new_recommendation_id() -> str:
    """The existing ``rec-{uuid12}`` identity rule, called exactly once per reply."""
    return f"rec-{uuid.uuid4().hex[:12]}"


def prepare_recommendation_products(
    products: Mapping[str, Any],
    *,
    recommendation_id: str,
    turn_id: str,
) -> dict[str, Any]:
    """Return a copy carrying recommendation metadata; never allocates ids or writes state."""
    prepared = dict(products)
    prepared["recommendation_id"] = recommendation_id
    prepared["request_id"] = turn_id
    prepared["ranker_version"] = EXISTING_RANKER_VERSION
    return prepared


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplyAssemblyError(message)


def assemble_reply_presentation(
    request: ReplyAssemblyRequest,
    *,
    catalogue_size: int | None = None,
) -> dict[str, Any]:
    """Build the reply presentation from real tool results (pure; inputs are not mutated).

    Raises :class:`ReplyAssemblyError` for anything that would otherwise become a fabricated
    or incomplete recommendation.  The caller decides how to fail closed.
    """
    recommendation = request.recommendation
    _require(isinstance(recommendation, Mapping), "recommendation must be a JSON object")
    try:
        status = validate_recommendation(recommendation)
    except RecommendationResultError as exc:
        raise ReplyAssemblyError(str(exc)) from exc
    _require(
        status == _FINALIZED,
        f"a durable warm reply requires a finalized recommendation, got {status!r}",
    )

    selected_id = str(recommendation.get("selected_product_id") or "").strip()
    _require(bool(selected_id), "finalized recommendation has no selected_product_id")
    info = request.product_infos.get(selected_id)
    _require(
        isinstance(info, Mapping) and bool(info),
        f"the selected product {selected_id!r} was not enriched with get_product_info",
    )
    selected = info.get("selected_product")
    _require(
        isinstance(selected, Mapping) and str(selected.get("item_id") or "") == selected_id,
        f"the get_product_info record for {selected_id!r} is missing selected_product",
    )
    _require(
        bool(str(selected.get("title") or "").strip()),
        f"the selected product {selected_id!r} has no title",
    )

    gate = recommendation.get("gate") or {}
    _require(isinstance(gate, Mapping), "recommendation gate must be an object")
    items = [
        vm.product_card(
            info,
            source="recommendation",
            retrieval_source=vm.RETRIEVAL_CATALOGUE,
            match_status="recommended",
            recommended=True,
            product_role=gate.get("product_role"),
        )
    ]
    products = vm.products_payload(
        query=str(request.query),
        search_mode=recommendation.get("search_mode"),
        score_kind=None,
        catalogue_size=catalogue_size,
        items=items,
        recommended_product_id=selected_id,
        gate=gate,
    )
    products = prepare_recommendation_products(
        products,
        recommendation_id=request.recommendation_id,
        turn_id=request.turn_id,
    )

    preference_payload = request.preference_payload
    _require(isinstance(preference_payload, Mapping),
             "get_active_preferences returned no preference object")
    preferences = vm.preference_state(
        dict(preference_payload), profile_id=request.profile_id
    )
    reply = request.reply
    _require(isinstance(reply, Mapping), "submit_clarification_reply returned no object")
    memory_update = vm.memory_update_payload(dict(reply), preferences)
    narrative = vm.narrative_memory(
        profile_id=request.profile_id,
        memory_update=memory_update,
        preference_payload=preferences,
        # The in-memory CPU resume path only has the status in scope at this point, so its
        # narrative carries the status line (not a second title line).  The durable reply
        # matches that text exactly; the full selected card is in ``products`` either way.
        recommendation_payload={"status": status},
    )
    _require(bool(narrative.strip()), "the reply narrative is empty")
    return {
        "products": products,
        "narrative": narrative,
        "recommendation_id": request.recommendation_id,
        "ranking_source": request.ranking_source,
        "preferences": preferences,
        "memory_update": memory_update,
    }


def assemble_refusal_memory(
    reply: Mapping[str, Any],
    preference_payload: Mapping[str, Any],
    *,
    profile_id: str,
) -> dict[str, Any]:
    """The M9 view of a bounded refusal: preferences and memory update, no product."""
    preferences = vm.preference_state(
        dict(preference_payload or {}), profile_id=profile_id
    )
    memory_update = vm.memory_update_payload(dict(reply or {}), preferences)
    return {"preferences": preferences, "memory_update": memory_update}


def turn_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a durable reply result onto the persisted TurnView shape.

    The durable result keeps the contract payload plus the BFF-derived extras; the reply turn
    therefore restores with the same ``memory_update`` and ``tool_activity`` the live response
    reported, without re-running any tool.
    """
    _require(isinstance(result, Mapping), "durable reply result must be an object")
    turn = turn_from_payload(result.get("reply") or {})
    turn["memory_update"] = result.get("memory_update")
    turn["tool_activity"] = list(result.get("steps") or [])
    return turn


def reply_response_from_result(
    result: Mapping[str, Any], *, clarification_id: str
) -> dict[str, Any]:
    """Rebuild the HTTP reply response from the durable result only (replay/restart safe)."""
    _require(isinstance(result, Mapping), "durable reply result must be an object")
    reply = result.get("reply")
    _require(isinstance(reply, Mapping), "durable reply result has no contract payload")
    narrative = str(reply.get("assistant_text") or "")
    _require(bool(narrative.strip()), "durable reply result has no assistant narrative")
    products = reply.get("products")
    has_products = (
        isinstance(products, Mapping) and bool(products.get("items"))
    )
    refusal = result.get("refusal")
    if refusal is None and isinstance(products, Mapping):
        refusal = products.get("refusal")
    return {
        "turn_id": reply.get("reply_turn_id"),
        "memory_update": result.get("memory_update"),
        "preferences": result.get("preferences"),
        # A bounded refusal has no product: never present an empty list as a recommendation.
        "recommendation": dict(products) if has_products else None,
        "refusal": dict(refusal) if isinstance(refusal, Mapping) else None,
        "task_requirements": result.get("task_requirements"),
        "steps": list(result.get("steps") or []),
        "clarification_id": clarification_id,
        "message": narrative,
        "clarification": None,
        "task_constraints": None,
        "candidate_reevaluation": None,
        "planner": None,
        "planner_model": None,
        "planner_calls": None,
        "planner_latency_ms": None,
        "planner_actions": None,
        "error": None,
    }
