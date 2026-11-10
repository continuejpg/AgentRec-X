"""Deterministic demo controllers for the Web Demo BFF.

No LLM is involved.  Every controller maps a real MCP tool payload into the view models from
``web_demo.viewmodels`` and yields SSE events in the contract order.  Tool activity is recorded
with the real per-call latency; failures become ``error`` events (never a fake success).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping

from web_demo import viewmodels as vm
from web_demo.config import COMPARE_MAX, COMPARE_MIN, SEARCH_TOP_K, Settings
from web_demo.mcp_manager import MCPManager, MCPManagerError
from web_demo.sessions import PendingClarification, Session, Turn

TOOL_LABELS = {
    "search_products": "Search",
    "get_product_info": "Product Info",
    "recommend_products": "Recommend",
    "compare_products": "Compare",
    "request_preference_clarification": "Clarification",
    "submit_clarification_reply": "Clarification",
    "get_active_preferences": "Memory",
    "run_agent_turn": "Qwen Agent",
    "qwen_planner": "Qwen Planner Request",
    "planner_action": "Planner Action",
    "decision_gate": "Decision Gate",
    "evidence_preparation": "Evidence Preparation",
}

# Explicit comparison intent.  The comparison path is only taken when the request also carries
# 2-5 selected real product ids (contract rule); otherwise the normal search flow runs.
COMPARE_PATTERN = re.compile(r"\b(compare|comparison|versus|vs\.?|difference between)\b", re.IGNORECASE)

CLARIFICATION_ACTIONS = {"NEED_USER_CLARIFICATION", "ASK_CLARIFICATION"}
COMPARISON_FAILURE_STATUSES = {
    "invalid_product_count",
    "invalid_product_reference",
    "role_mismatch",
}


def event(name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"event": name, "data": dict(data)}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


@dataclass
class TurnRecorder:
    """Records ToolStatus entries on the persisted turn (contract fields only)."""

    turn: Turn

    def start(self, tool: str, label: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "tool": tool,
            "label": label or TOOL_LABELS.get(tool, tool),
            "status": "running",
        }
        self.turn.tool_activity.append(entry)
        return entry

    @staticmethod
    def ok(entry: dict[str, Any], *, latency_ms: float, detail: str | None = None) -> None:
        entry["status"] = "ok"
        entry["latency_ms"] = latency_ms
        if detail:
            entry["detail"] = detail

    @staticmethod
    def error(entry: dict[str, Any], *, detail: str) -> None:
        entry["status"] = "error"
        entry["detail"] = detail


def _mark_running_tool_error(turn: Turn, message: str) -> dict[str, Any] | None:
    for entry in reversed(turn.tool_activity):
        if entry.get("status") == "running":
            entry["status"] = "error"
            entry["detail"] = message
            return entry
    return None


def is_comparison_request(message: str, selected_product_ids: Iterable[str]) -> bool:
    ids = [item for item in selected_product_ids if str(item or "").strip()]
    return COMPARE_MIN <= len(ids) <= COMPARE_MAX and bool(COMPARE_PATTERN.search(message or ""))


# ---------------------------------------------------------------------------
# Chat SSE controller
# ---------------------------------------------------------------------------


async def chat_events(
    *,
    manager: MCPManager,
    session: Session,
    settings: Settings,
    message: str,
    selected_product_ids: list[str] | None = None,
    mode: str = "cpu_live",
    qwen_health: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Full chat turn as contract-ordered SSE events (including ``done``)."""
    turn = Turn(turn_id=session.next_turn_id(), user_text=message)
    session.set_title_from(message)
    recorder = TurnRecorder(turn)
    state: dict[str, Any] = {"stop_reason": "finalized", "turn_id": turn.turn_id}
    started = time.perf_counter()
    try:
        ids = list(selected_product_ids or [])
        if mode == "qwen_live":
            if is_comparison_request(message, ids):
                # Comparison/history are not a Qwen planner decision; do not label a Skill call
                # as Qwen reasoning.  Keep the existing comparison skill path and record planner=none.
                turn.planner = "none"
                async for payload_event in _comparison_turn(
                    manager=manager, session=session, turn=turn, recorder=recorder,
                    product_refs=ids, state=state,
                ):
                    yield payload_event
            else:
                async for payload_event in _qwen_live_turn(
                    manager=manager, session=session, settings=settings, turn=turn,
                    recorder=recorder, message=message, state=state,
                    qwen_health=qwen_health,
                ):
                    yield payload_event
        elif is_comparison_request(message, ids):
            turn.planner = "rule"
            async for payload_event in _comparison_turn(
                manager=manager, session=session, turn=turn, recorder=recorder,
                product_refs=ids, state=state,
            ):
                yield payload_event
        else:
            turn.planner = "rule"
            async for payload_event in _search_and_recommend_turn(
                manager=manager, session=session, settings=settings, turn=turn,
                recorder=recorder, message=message, state=state,
            ):
                yield payload_event
    except MCPManagerError as exc:
        entry = _mark_running_tool_error(turn, exc.message)
        if entry is not None:
            yield event("tool_status", entry)
        error = exc.as_payload()
        turn.error = error
        turn.assistant_text = (
            f"Error from {exc.tool or 'MCP'}: {exc.message}"
        )
        session.add_turn(turn)
        yield event("error", error)
        yield event(
            "done",
            {
                "turn_id": turn.turn_id,
                "stop_reason": "error",
                "total_ms": _ms(started),
                "planner": turn.planner,
                "planner_calls": turn.planner_calls,
                "planner_latency_ms": turn.planner_latency_ms,
            },
        )
        return
    except Exception as exc:  # noqa: BLE001 - unexpected failure must be visible, not swallowed
        entry = _mark_running_tool_error(turn, f"{type(exc).__name__}: {exc}")
        if entry is not None:
            yield event("tool_status", entry)
        error = {
            "code": "internal_error",
            "message": f"{type(exc).__name__}: {exc}",
        }
        turn.error = error
        turn.assistant_text = f"Internal error: {error['message']}"
        session.add_turn(turn)
        yield event("error", error)
        yield event(
            "done",
            {
                "turn_id": turn.turn_id,
                "stop_reason": "error",
                "total_ms": _ms(started),
                "planner": turn.planner,
                "planner_calls": turn.planner_calls,
                "planner_latency_ms": turn.planner_latency_ms,
            },
        )
        return

    session.add_turn(turn)
    yield event(
        "done",
        {
            "turn_id": turn.turn_id,
            "stop_reason": state["stop_reason"],
            "total_ms": _ms(started),
            "planner": turn.planner,
            "planner_model": turn.planner_model,
            "planner_calls": turn.planner_calls,
            "planner_latency_ms": turn.planner_latency_ms,
        },
    )


async def _search_and_recommend_turn(
    *,
    manager: MCPManager,
    session: Session,
    settings: Settings,
    turn: Turn,
    recorder: TurnRecorder,
    message: str,
    state: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    profile_id = session.profile_id

    # 1. Real catalogue search.
    search_entry = recorder.start("search_products")
    yield event("tool_status", search_entry)
    started = time.perf_counter()
    search = await manager.call_tool(
        profile_id, "search_products", {"query": message, "top_k": SEARCH_TOP_K}
    )
    recorder.ok(search_entry, latency_ms=_ms(started))
    yield event("tool_status", search_entry)

    hits = [hit for hit in search.get("hits") or () if hit.get("item_id")]
    hit_ids = [str(hit["item_id"]) for hit in hits]
    retrieval_source = vm.search_retrieval_source(search)

    # 2. Enrich the search hits with real product info / evidence / UNKNOWN fields.
    enrich_entry = recorder.start("get_product_info", label="Product Info")
    yield event("tool_status", enrich_entry)
    started = time.perf_counter()
    infos: dict[str, dict[str, Any]] = {}
    for item_id in hit_ids:
        infos[item_id] = await manager.call_tool(
            profile_id, "get_product_info", {"product_ref": item_id}
        )
    recorder.ok(
        enrich_entry,
        latency_ms=_ms(started),
        detail=f"{len(infos)}/{len(hit_ids)} search hits enriched with real catalogue evidence",
    )
    yield event("tool_status", enrich_entry)

    # 3. The existing Rule Planner + Gate.
    recommend_entry = recorder.start("recommend_products")
    yield event("tool_status", recommend_entry)
    started = time.perf_counter()
    recommendation = await manager.call_tool(
        profile_id, "recommend_products", {"query": message}
    )
    recorder.ok(recommend_entry, latency_ms=_ms(started))
    yield event("tool_status", recommend_entry)
    _raise_for_recommendation_failure(recommendation)

    gate = recommendation.get("gate") or {}
    status = recommendation.get("status")
    selected_id = recommendation.get("selected_product_id")
    awaiting = status == "awaiting_clarification" or gate.get("action") in CLARIFICATION_ACTIONS

    search_cards: list[dict[str, Any]] = []
    for item_id in hit_ids:
        card = vm.product_card(
            infos[item_id],
            source="search",
            retrieval_source=retrieval_source,
            match_status="search_result",
            recommended=False,
        )
        search_cards.append(card)

    if awaiting:
        products = vm.products_payload(
            query=message,
            search_mode=search.get("search_mode"),
            score_kind=search.get("score_kind"),
            catalogue_size=search.get("catalogue_size"),
            items=search_cards,
            recommended_product_id=None,
            gate=gate,
        )
        turn.products = products
        yield event("products", products)

        clarification_entry = recorder.start(
            "request_preference_clarification", label="Clarification"
        )
        yield event("tool_status", clarification_entry)
        started = time.perf_counter()
        clarification_raw = await manager.call_tool(
            profile_id, "request_preference_clarification", {"query": message}
        )
        recorder.ok(clarification_entry, latency_ms=_ms(started))
        yield event("tool_status", clarification_entry)

        clarification = vm.clarification_payload(clarification_raw)
        turn.clarification = clarification
        session.pending_clarification = PendingClarification(
            payload=clarification,
            query=message,
            turn_id=turn.turn_id,
            reply_turn_id=session.next_reply_turn_id(),
        )
        yield event("clarification", clarification)
        narrative = vm.narrative_clarification(clarification)
        turn.assistant_text = narrative
        yield event("text", {"text": narrative, "final": True})
        state["stop_reason"] = "awaiting_clarification"
        return

    # 4. Build the products payload: real search cards + the real recommended card + Gate.
    items: list[dict[str, Any]] = []
    selected_card: dict[str, Any] | None = None
    selected_info = infos.get(str(selected_id)) if selected_id else None
    if selected_id and selected_info is None:
        # The Gate selected a real product that is not in the top search page; fetch its real
        # catalogue facts too (still through the MCP tool, never from a local import).
        detail_entry = recorder.start("get_product_info", label="Product Info")
        yield event("tool_status", detail_entry)
        started = time.perf_counter()
        selected_info = await manager.call_tool(
            profile_id, "get_product_info", {"product_ref": str(selected_id)}
        )
        infos[str(selected_id)] = selected_info
        recorder.ok(
            detail_entry,
            latency_ms=_ms(started),
            detail="selected product facts fetched from the real catalogue",
        )
        yield event("tool_status", detail_entry)

    for item_id in hit_ids:
        is_selected = bool(selected_id) and item_id == str(selected_id)
        card = vm.product_card(
            infos[item_id],
            source="search",
            retrieval_source=retrieval_source,
            match_status="search_result",
            recommended=is_selected,
            product_role=(gate.get("product_role") if is_selected else None),
        )
        items.append(card)
        if is_selected:
            selected_card = card
    if selected_id and str(selected_id) not in infos:
        # Defensive: a gate-selected id with no fetched info cannot become a painted card.
        selected_info = None
    if selected_id and str(selected_id) not in hit_ids and selected_info is not None:
        selected_card = vm.product_card(
            selected_info,
            source="recommendation",
            retrieval_source=vm.RETRIEVAL_CATALOGUE,
            match_status="recommended",
            recommended=True,
            product_role=gate.get("product_role"),
        )
        items.append(selected_card)

    products = vm.products_payload(
        query=message,
        search_mode=search.get("search_mode"),
        score_kind=search.get("score_kind"),
        catalogue_size=search.get("catalogue_size"),
        items=items,
        recommended_product_id=str(selected_id) if selected_id else None,
        gate=gate,
    )
    turn.products = products
    yield event("products", products)

    # 5. Evidence for the selected product only (real P15 spans + UNKNOWN fields).
    if selected_id and selected_info is not None:
        evidence = vm.evidence_payload(selected_info)
        yield event("evidence", evidence)

    narrative = vm.narrative_search(
        search_payload=search,
        recommendation_payload=recommendation,
        selected_card=selected_card,
        gate=gate,
        catalogue_size=search.get("catalogue_size"),
    )
    turn.assistant_text = narrative
    yield event("text", {"text": narrative, "final": True})
    state["stop_reason"] = "finalized"


async def _qwen_live_turn(
    *,
    manager: MCPManager,
    session: Session,
    settings: Settings,
    turn: Turn,
    recorder: TurnRecorder,
    message: str,
    state: dict[str, Any],
    qwen_health: Any | None,
) -> AsyncIterator[dict[str, Any]]:
    """Real Qwen Base Live: MCP run_agent_turn + MCP progress notifications + P15 products.

    No Rule fallback: any planner/transport failure becomes a visible error.  The BFF only calls
    this path after the background health check has validated a real PlannerAction.
    """
    profile_id = session.profile_id
    entry = recorder.start("run_agent_turn", label="Qwen Agent")
    yield event("tool_status", entry)

    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def progress_callback(progress: float, total: float | None, text: str | None) -> None:
        del progress, total
        try:
            payload = json.loads(text or "")
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        tool = str(payload.get("tool") or "qwen_progress")
        item = {
            "tool": tool,
            "label": str(payload.get("label") or TOOL_LABELS.get(tool, tool)),
            "status": str(payload.get("status") or "ok"),
            "detail": payload.get("detail"),
            "latency_ms": payload.get("latency_ms"),
        }
        _record_progress(turn, item)
        await progress_queue.put(event("tool_status", item))

    started = time.perf_counter()
    task = asyncio.create_task(
        manager.call_tool(
            profile_id, "run_agent_turn", {"query": message},
            timeout_s=settings.qwen_turn_timeout_s,
            progress_callback=progress_callback,
        ),
        name="qwen-live-run-agent-turn",
    )
    try:
        while True:
            if task.done():
                while not progress_queue.empty():
                    yield progress_queue.get_nowait()
                break
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            yield item
        result = await task
    except MCPManagerError as exc:
        recorder.error(entry, detail=exc.message)
        if qwen_health is not None:
            qwen_health.mark_unavailable(exc.message)
        raise
    except Exception:
        if qwen_health is not None:
            qwen_health.mark_unavailable("Qwen agent turn failed")
        raise
    latency_ms = _ms(started)
    recorder.ok(entry, latency_ms=latency_ms)
    yield event("tool_status", entry)

    planner = result.get("planner") or {}
    turn.planner = "qwen_base"
    turn.planner_model = planner.get("model")
    turn.planner_calls = planner.get("calls")
    planner_latencies = [float(value) for value in (planner.get("latency_ms") or []) if value is not None]
    turn.planner_latency_ms = round(sum(planner_latencies), 2) if planner_latencies else None

    status = str(result.get("status") or "error")
    gate = result.get("gate") or {}
    selected_id = result.get("selected_product_id")
    candidate_ids = [str(item) for item in (result.get("candidate_ids") or []) if str(item).strip()][:10]
    search = result.get("search") or {}
    retrieval_source = vm.search_retrieval_source(search)

    info_entry = recorder.start("get_product_info", label="Product Info")
    yield event("tool_status", info_entry)
    infos: dict[str, dict[str, Any]] = {}
    fetch_started = time.perf_counter()
    fetch_ids = list(candidate_ids)
    if selected_id and str(selected_id) not in fetch_ids:
        fetch_ids.append(str(selected_id))
    for item_id in fetch_ids:
        infos[item_id] = await manager.call_tool(
            profile_id, "get_product_info", {"product_ref": item_id}
        )
    recorder.ok(
        info_entry,
        latency_ms=_ms(fetch_started),
        detail=f"{len(infos)} real catalogue records for Qwen candidates",
    )
    yield event("tool_status", info_entry)

    items: list[dict[str, Any]] = []
    selected_card: dict[str, Any] | None = None
    for item_id in candidate_ids:
        card = vm.product_card(
            infos[item_id], source="search", retrieval_source=retrieval_source,
            match_status="search_result",
            recommended=bool(selected_id) and item_id == str(selected_id),
            product_role=(gate.get("product_role") if selected_id and item_id == str(selected_id) else None),
        )
        items.append(card)
        if selected_id and item_id == str(selected_id):
            selected_card = card
    if selected_id and str(selected_id) not in candidate_ids and str(selected_id) in infos:
        selected_card = vm.product_card(
            infos[str(selected_id)], source="recommendation",
            retrieval_source=vm.RETRIEVAL_CATALOGUE, match_status="recommended",
            recommended=True, product_role=gate.get("product_role"),
        )
        items.append(selected_card)

    products = vm.products_payload(
        query=message,
        search_mode=search.get("search_mode") or settings.search_mode,
        score_kind=search.get("score_kind"),
        catalogue_size=search.get("catalogue_size") or manager.catalogue_size(),
        items=items,
        recommended_product_id=str(selected_id) if selected_id else None,
        gate=gate,
    )
    turn.products = products
    yield event("products", products)

    if status == "awaiting_clarification":
        clarification_raw = result.get("clarification") or {}
        clarification = vm.clarification_payload(clarification_raw)
        clarification["memory_write_enabled"] = bool(clarification_raw.get("memory_write_enabled"))
        turn.clarification = clarification
        session.pending_clarification = PendingClarification(
            payload=clarification,
            query=message,
            turn_id=turn.turn_id,
            reply_turn_id=session.next_reply_turn_id(),
            mode="qwen_live",
        )
        yield event("clarification", clarification)
        narrative = vm.narrative_qwen(
            status=status, gate=gate, planner=planner, selected_card=None,
            clarification=clarification, catalogue_size=products.get("catalogue_size"),
        )
        turn.assistant_text = narrative
        yield event("text", {"text": narrative, "final": True})
        state["stop_reason"] = "awaiting_clarification"
        return

    if selected_id and selected_card is not None:
        evidence = vm.evidence_payload(infos.get(str(selected_id)))
        yield event("evidence", evidence)

    rag = result.get("rag") or {}
    final_answer = rag.get("final_answer") if isinstance(rag, Mapping) else None
    narrative = vm.narrative_qwen(
        status=status, gate=gate, planner=planner, selected_card=selected_card,
        clarification=None, catalogue_size=products.get("catalogue_size"),
        final_answer=str(final_answer) if final_answer else None,
        planner_steps=(result.get("planner_steps") or {}),
    )
    turn.assistant_text = narrative
    yield event("text", {"text": narrative, "final": True})
    state["stop_reason"] = status


def _record_progress(turn: Turn, item: dict[str, Any]) -> None:
    """Persist real progress rows on the turn (same view model as live SSE)."""
    incoming = dict(item)
    if incoming.get("status") == "running":
        turn.tool_activity.append(incoming)
        return
    for index in range(len(turn.tool_activity) - 1, -1, -1):
        row = turn.tool_activity[index]
        if row.get("tool") == incoming.get("tool") and row.get("status") == "running":
            turn.tool_activity[index] = {**row, **incoming}
            return
    turn.tool_activity.append(incoming)


async def _comparison_turn(
    *,
    manager: MCPManager,
    session: Session,
    turn: Turn,
    recorder: TurnRecorder,
    product_refs: list[str],
    state: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    compare_entry = recorder.start("compare_products")
    yield event("tool_status", compare_entry)
    started = time.perf_counter()
    comparison_raw = await manager.call_tool(
        session.profile_id, "compare_products", {"product_refs": list(product_refs)}
    )
    recorder.ok(compare_entry, latency_ms=_ms(started))
    yield event("tool_status", compare_entry)
    _raise_for_comparison_failure(comparison_raw)

    comparison = vm.comparison_payload(comparison_raw)
    turn.comparison = comparison
    yield event("comparison", comparison)
    narrative = vm.narrative_comparison(comparison)
    turn.assistant_text = narrative
    yield event("text", {"text": narrative, "final": True})
    state["stop_reason"] = "comparison"


def _raise_for_recommendation_failure(recommendation: Mapping[str, Any]) -> None:
    if recommendation.get("status") != "error":
        return
    raise MCPManagerError(
        str(recommendation.get("error") or "recommend_products returned status=error"),
        code="recommendation_failed",
        tool="recommend_products",
        detail={"status": "error"},
    )


def _raise_for_comparison_failure(comparison: Mapping[str, Any]) -> None:
    status = str(comparison.get("status") or "")
    if status not in COMPARISON_FAILURE_STATUSES:
        return
    raise MCPManagerError(
        str(comparison.get("final_answer") or f"comparison status={status}"),
        code=status,
        tool="compare_products",
        detail={"status": status, "errors": list(comparison.get("errors") or ())},
    )


# ---------------------------------------------------------------------------
# JSON endpoint helpers (details / compare / clarification reply)
# ---------------------------------------------------------------------------


async def product_details(
    *, manager: MCPManager, session: Session, product_ref: str
) -> dict[str, Any]:
    info = await manager.call_tool(
        session.profile_id, "get_product_info", {"product_ref": product_ref}
    )
    return vm.product_detail_payload(info)


async def compare_products(
    *, manager: MCPManager, session: Session, product_refs: list[str]
) -> dict[str, Any]:
    raw = await manager.call_tool(
        session.profile_id, "compare_products", {"product_refs": list(product_refs)}
    )
    _raise_for_comparison_failure(raw)
    return vm.comparison_payload(raw)


def _step(tool: str, started: float) -> dict[str, Any]:
    return {"tool": tool, "status": "ok", "latency_ms": _ms(started)}


async def clarification_reply(
    *,
    manager: MCPManager,
    session: Session,
    clarification_id: str,
    message: str,
    qwen_health: Any | None = None,
) -> dict[str, Any]:
    """Persist a real user reply through M9, refresh preferences and resume the pending query.

    The browser never supplies ``confirmed_by_user``; this controller sets it only because a real
    HTTP POST containing user-authored text arrived, and labels the channel as
    ``web_ui_explicit_user_reply``.
    """
    pending = session.pending_clarification
    if pending is None:
        raise MCPManagerError(
            "no pending clarification for this session; start a new clarification turn first",
            code="clarification_not_pending",
            http_status=409,
        )
    if pending.payload.get("clarification_id") != clarification_id:
        raise MCPManagerError(
            f"clarification_id {clarification_id!r} does not match this session's pending clarification",
            code="clarification_not_pending",
            http_status=409,
            detail={"pending_clarification_id": pending.payload.get("clarification_id")},
        )
    profile_id = session.profile_id
    if not manager.ensure_ready(profile_id).writes_enabled:
        raise MCPManagerError(
            "memory writes are disabled for this MCP server",
            code="memory_writes_disabled",
            http_status=409,
            detail={"profile_id": profile_id},
        )

    turn = Turn(turn_id=pending.reply_turn_id, user_text=message)
    recorder = TurnRecorder(turn)
    steps: list[dict[str, Any]] = []
    pending.reply_attempts += 1

    # 1. Real M9 write through the MCP tool (idempotent on the stable reply turn id).
    reply_entry = recorder.start("submit_clarification_reply", label="Clarification")
    started = time.perf_counter()
    try:
        reply = await manager.call_tool(
            profile_id,
            "submit_clarification_reply",
            {
                "clarification_id": clarification_id,
                "user_message": message,
                "turn_id": pending.reply_turn_id,
                "confirmed_by_user": True,
                "confirmation_source": "web_ui_explicit_user_reply",
            },
        )
    except MCPManagerError as exc:
        recorder.error(reply_entry, detail=exc.message)
        turn.tool_activity = [reply_entry]
        turn.error = exc.as_payload()
        turn.assistant_text = f"Clarification reply failed: {exc.message}"
        session.add_turn(turn)
        raise
    recorder.ok(reply_entry, latency_ms=_ms(started))
    steps.append(_step("submit_clarification_reply", started))

    # 2. Refresh the real active/inactive preferences from M9.
    prefs_entry = recorder.start("get_active_preferences", label="Memory")
    started = time.perf_counter()
    pref_payload = await manager.call_tool(profile_id, "get_active_preferences", {})
    recorder.ok(prefs_entry, latency_ms=_ms(started))
    steps.append(_step("get_active_preferences", started))
    preferences = vm.preference_state(pref_payload, profile_id=profile_id)

    # 3. Resume the pending query through the real recommendation pipeline.
    recommendation: dict[str, Any] | None = None
    planner_meta: dict[str, Any] | None = None
    followup_clarification: dict[str, Any] | None = None
    resume_task_constraints: dict[str, Any] | None = None
    candidate_reevaluation: dict[str, Any] | None = None
    if pending.query:
        if pending.mode == "qwen_live":
            # Real Qwen Base resume after the M9 write: no silent Rule fallback.
            qwen_entry = recorder.start("run_agent_turn", label="Qwen Agent")
            started = time.perf_counter()
            timeout_s = float(
                getattr(getattr(manager, "settings", None), "qwen_turn_timeout_s", 600.0)
            )
            try:
                rec_payload = await manager.call_tool(
                    profile_id, "run_agent_turn",
                    {"query": pending.query, "clarification_id": clarification_id},
                    timeout_s=timeout_s,
                )
            except MCPManagerError as exc:
                recorder.error(qwen_entry, detail=exc.message)
                if qwen_health is not None:
                    qwen_health.mark_unavailable(exc.message)
                raise
            recorder.ok(qwen_entry, latency_ms=_ms(started))
            steps.append(_step("run_agent_turn", started))
            planner_meta = dict(rec_payload.get("planner") or {})
            resume_task_constraints = rec_payload.get("task_constraints")
            candidate_reevaluation = rec_payload.get("candidate_reevaluation")
            turn.planner = "qwen_base"
            turn.planner_model = planner_meta.get("model")
            turn.planner_calls = planner_meta.get("calls")
            latencies = [float(v) for v in (planner_meta.get("latency_ms") or []) if v is not None]
            turn.planner_latency_ms = round(sum(latencies), 2) if latencies else None
            selected_id = rec_payload.get("selected_product_id")
            products_items: list[dict[str, Any]] = []
            selected_info: dict[str, Any] | None = None
            if selected_id:
                info_entry = recorder.start("get_product_info", label="Product Info")
                info_started = time.perf_counter()
                selected_info = await manager.call_tool(
                    profile_id, "get_product_info", {"product_ref": str(selected_id)}
                )
                recorder.ok(info_entry, latency_ms=_ms(info_started))
                steps.append(_step("get_product_info", info_started))
                gate = rec_payload.get("gate") or {}
                products_items.append(
                    vm.product_card(
                        selected_info, source="recommendation",
                        retrieval_source=vm.RETRIEVAL_CATALOGUE,
                        match_status="recommended", recommended=True,
                        product_role=gate.get("product_role"),
                    )
                )
            recommendation = vm.products_payload(
                query=pending.query,
                search_mode=(rec_payload.get("search") or {}).get("search_mode"),
                score_kind=(rec_payload.get("search") or {}).get("score_kind"),
                catalogue_size=(rec_payload.get("search") or {}).get("catalogue_size")
                or manager.catalogue_size(),
                items=products_items,
                recommended_product_id=str(selected_id) if selected_id else None,
                gate=rec_payload.get("gate"),
            )
            if rec_payload.get("status") == "awaiting_clarification" and rec_payload.get("clarification"):
                # The real model asked a follow-up because M9 still contains an unresolved conflict
                # (e.g. "avoid wired" is an AVOID entry, not an explicit retraction).  Expose the
                # server-issued handle honestly; do not force FINALIZE or fake a resolved memory.
                followup_clarification = vm.clarification_payload(rec_payload["clarification"])
                followup_clarification["memory_write_enabled"] = bool(
                    (rec_payload.get("clarification") or {}).get("memory_write_enabled")
                )
                turn.clarification = followup_clarification
                session.pending_clarification = PendingClarification(
                    payload=followup_clarification,
                    query=pending.query,
                    turn_id=turn.turn_id,
                    reply_turn_id=session.next_reply_turn_id(),
                    mode="qwen_live",
                )
        else:
            turn.planner = "rule"
            rec_entry = recorder.start("recommend_products", label="Recommend")
            started = time.perf_counter()
            rec_payload = await manager.call_tool(
                profile_id, "recommend_products", {"query": pending.query}
            )
            recorder.ok(rec_entry, latency_ms=_ms(started))
            steps.append(_step("recommend_products", started))
            _raise_for_recommendation_failure(rec_payload)
            recommendation = await _recommendation_view(
                manager=manager,
                session=session,
                query=pending.query,
                recommendation=rec_payload,
                recorder=recorder,
                steps=steps,
            )
    else:
        rec_payload = None

    memory_update = vm.memory_update_payload(reply, preferences)
    selected_product = None
    selected_info = locals().get("selected_info")
    if selected_info:
        selected_product = selected_info.get("selected_product") or {}
    narrative = vm.narrative_memory(
        profile_id=profile_id,
        memory_update=memory_update,
        preference_payload=preferences,
        recommendation_payload=(
            {
                "status": rec_payload.get("status"),
                "selected_product": selected_product,
            }
            if rec_payload is not None else None
        ),
    )
    if turn.planner == "qwen_base":
        narrative = (
            f"Planner: {turn.planner_model or 'Qwen3.5-9B-Base'} "
            f"({turn.planner_calls or 0} real model calls).\n" + narrative
        )
    turn.tool_activity = list(turn.tool_activity)
    turn.memory_update = memory_update
    turn.products = recommendation
    turn.assistant_text = narrative
    if followup_clarification is None:
        session.pending_clarification = None
    else:
        narrative = (
            narrative
            + "\n\nFollow-up needed from the real Qwen planner: "
            + str(followup_clarification.get("question") or "")
        )
        turn.assistant_text = narrative
    session.add_turn(turn)
    return {
        "turn_id": turn.turn_id,
        "memory_update": memory_update,
        "preferences": preferences,
        "recommendation": recommendation,
        "steps": steps,
        "clarification_id": clarification_id,
        "message": narrative,
        "clarification": followup_clarification,
        "task_constraints": resume_task_constraints,
        "candidate_reevaluation": candidate_reevaluation,
        "planner": turn.planner,
        "planner_model": turn.planner_model,
        "planner_calls": turn.planner_calls,
        "planner_latency_ms": turn.planner_latency_ms,
        "error": None,
    }


async def _recommendation_view(
    *,
    manager: MCPManager,
    session: Session,
    query: str,
    recommendation: Mapping[str, Any],
    recorder: TurnRecorder,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = recommendation.get("gate") or {}
    selected_id = recommendation.get("selected_product_id")
    items: list[dict[str, Any]] = []
    if selected_id:
        info_entry = recorder.start("get_product_info", label="Product Info")
        started = time.perf_counter()
        info = await manager.call_tool(
            session.profile_id, "get_product_info", {"product_ref": str(selected_id)}
        )
        recorder.ok(info_entry, latency_ms=_ms(started))
        steps.append(_step("get_product_info", started))
        items.append(
            vm.product_card(
                info,
                source="recommendation",
                retrieval_source=vm.RETRIEVAL_CATALOGUE,
                match_status="recommended",
                recommended=True,
                product_role=gate.get("product_role"),
            )
        )
    return vm.products_payload(
        query=query,
        search_mode=recommendation.get("search_mode"),
        score_kind=None,
        catalogue_size=manager.catalogue_size(),
        items=items,
        recommended_product_id=str(selected_id) if selected_id else None,
        gate=gate,
    )
