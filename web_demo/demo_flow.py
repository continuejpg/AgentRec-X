"""Deterministic demo controllers for the Web Demo BFF.

No LLM is involved.  Every controller maps a real MCP tool payload into the view models from
``web_demo.viewmodels`` and yields SSE events in the contract order.  Tool activity is recorded
with the real per-call latency; failures become ``error`` events (never a fake success).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping

from observability import OPENINFERENCE_AGENT, Tracing, annotate_current_span, get_tracing
from web_demo import viewmodels as vm
from web_demo.behavior import EXISTING_RANKER_VERSION  # re-exported: demo metadata helpers
from web_demo.config import (
    COMPARE_MAX,
    COMPARE_MIN,
    PROFILE_BY_ID,
    SEARCH_TOP_K,
    Settings,
)
from web_demo.continuation import (
    CONTINUATION_MALFORMED,
    CONTINUATION_SESSION_MISMATCH,
    build_web_continuation,
    validate_continuation,
)
from web_demo.handle_lifecycle import HandleCheck, HandleRef
from web_demo.mcp_manager import MCPManager, MCPManagerError
from web_demo.reply_assembly import (
    assemble_reply_presentation,
    prepare_recommendation_products,
    reply_response_from_result,
    turn_from_result,
)
from web_demo.reply_executor import (
    AMBIGUOUS,
    BUSY,
    CHECKPOINT_FAILED,
    COLD_RESUME_FAILED,
    CONFLICT,
    FENCED,
    REPLAYED,
    RESULT_REUSED,
    SUCCESS,
    WARM_UNAVAILABLE,
    ResolvedClarificationHandle,
    WarmReplyExecutor,
)
from web_demo.sessions import PendingClarification, Session, Turn

LOGGER = logging.getLogger("agentrecx.web_demo.demo_flow")

#: A cold restore is a process-local state rebuild, never a catalogue/LLM call.
COLD_RESUME_TIMEOUT_S = 30.0

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


def _attach_recommendation(
    session: Session,
    products: dict[str, Any],
    *,
    turn_id: str,
    ranking_source: str,
) -> dict[str, Any]:
    """Give this displayed list its own opaque recommendation id and freeze its snapshot.

    A new id is generated for every order the browser can see, so an impression from an
    earlier list can never be attributed to a later one.  The production order is copied
    unchanged; only the behavior shadow (a separate preview) may reorder it.
    """
    recommendation_id = f"rec-{uuid.uuid4().hex[:12]}"
    prepared = prepare_recommendation_products(
        products, recommendation_id=recommendation_id, turn_id=turn_id
    )
    products.update(prepared)
    session.record_recommendation(
        products, turn_id=turn_id, ranking_source=ranking_source
    )
    return products


def _persist_finished_turn(
    session: Session, task: Any | None, turn: Turn, stop_reason: str
) -> None:
    """Persist a completed chat turn (persistent mode only).  No-op in memory mode."""
    store = session._store
    if store is None:
        return
    store.persist_finished_turn(session, task, turn, stop_reason=stop_reason)


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
    tracing: Tracing | None = None,
    behavior_snapshot: Mapping[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Open one Agent span around the real SSE turn, then delegate unchanged."""
    runtime = tracing or get_tracing()
    with runtime.span(
        "agent.chat",
        openinference_kind=OPENINFERENCE_AGENT,
        attributes={
            "agent.name": "AgentRec-X CPU/BM25 shopping agent",
            "agentrecx.path": "chat",
            "agentrecx.mode": mode,
            "agentrecx.session_id": session.session_id,
            "agentrecx.profile_id": session.profile_id,
        },
    ):
        async for payload_event in _chat_events_impl(
            manager=manager,
            session=session,
            settings=settings,
            message=message,
            selected_product_ids=selected_product_ids,
            mode=mode,
            qwen_health=qwen_health,
            behavior_snapshot=behavior_snapshot,
        ):
            yield payload_event


async def _chat_events_impl(
    *,
    manager: MCPManager,
    session: Session,
    settings: Settings,
    message: str,
    selected_product_ids: list[str] | None = None,
    mode: str = "cpu_live",
    qwen_health: Any | None = None,
    behavior_snapshot: Mapping[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Full chat turn as contract-ordered SSE events (including ``done``)."""
    # Persistent mode: bind this chat to a durable task before any identity is issued.
    task = session._store.begin_chat_task(session, message) if session._store is not None else None
    turn = Turn(turn_id=session.next_turn_id(), user_text=message)
    annotate_current_span({
        "agentrecx.task_id": getattr(task, "task_id", None),
        "agentrecx.turn_id": turn.turn_id,
    })
    session.last_query = message
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
                    qwen_health=qwen_health, task=task,
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
                recorder=recorder, message=message, state=state, task=task,
                behavior_snapshot=behavior_snapshot,
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
        _persist_finished_turn(session, task, turn, "error")
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
                "planner_actions": turn.planner_actions,
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
        _persist_finished_turn(session, task, turn, "error")
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
                "planner_actions": turn.planner_actions,
            },
        )
        return

    session.add_turn(turn)
    _persist_finished_turn(session, task, turn, state["stop_reason"])
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
            "planner_actions": turn.planner_actions,
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
    task: Any | None = None,
    behavior_snapshot: Mapping[str, Any] | None = None,
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
    recommend_arguments: dict[str, Any] = {
        "query": message, "session_id": session.session_id,
    }
    if behavior_snapshot is not None:
        recommend_arguments["behavior_snapshot"] = dict(behavior_snapshot)
    recommendation = await manager.call_tool(
        profile_id, "recommend_products", recommend_arguments
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
        _attach_recommendation(
            session, products, turn_id=turn.turn_id,
            ranking_source="cpu_live:search_products+recommend_products+gate",
        )
        turn.products = products
        yield event("products", products)

        existing_clarification = recommendation.get("clarification")
        if (
            isinstance(existing_clarification, Mapping)
            and existing_clarification.get("clarification_id")
            and isinstance(existing_clarification.get("continuation"), Mapping)
        ):
            # recommend_products already issued the real ASK (task-slot policy or Gate).
            # Reuse that exact handle instead of asking the same MCP question twice.
            clarification_raw = dict(existing_clarification)
        else:
            clarification_entry = recorder.start(
                "request_preference_clarification", label="Clarification"
            )
            yield event("tool_status", clarification_entry)
            started = time.perf_counter()
            clarification_raw = await manager.call_tool(
                profile_id, "request_preference_clarification",
                {"query": message, "session_id": session.session_id},
            )
            recorder.ok(clarification_entry, latency_ms=_ms(started))
            yield event("tool_status", clarification_entry)

        clarification = vm.clarification_payload(clarification_raw)
        turn.clarification = clarification
        narrative = vm.narrative_clarification(clarification)
        turn.assistant_text = narrative
        # Versioned Cold Resume snapshot: the child's exact task/question/deadline plus the
        # Web binding.  It is committed in the same transaction as the pending row.
        continuation = build_web_continuation(
            mcp_continuation=clarification_raw.get("continuation"),
            query=message,
            profile_id=profile_id,
        )
        expires_at = (
            float(continuation["expires_at"]) if continuation is not None
            else time.time() + float(settings.clarification_ttl_s)
        )
        # The ASK is committed to SQLite *before* the browser is told it exists.  In
        # persistent mode commit_ask allocates the single reply identity itself.
        if session._store is not None:
            if task is None:  # pragma: no cover - persistent mode always binds a task
                raise RuntimeError("persistent ASK requires a task")
            session.pending_clarification = session._store.persist_ask(
                session, task, turn, clarification,
                query=message,
                clarification_id=str(clarification.get("clarification_id") or ""),
                mode="cpu_live",
                expires_at=expires_at,
                ranking_source="cpu_live:search_products+recommend_products+gate",
                continuation=continuation,
            )
        else:
            session.pending_clarification = PendingClarification(
                payload=clarification,
                query=message,
                turn_id=turn.turn_id,
                reply_turn_id=session.next_reply_turn_id(),
            )
        yield event("clarification", clarification)
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
    for key in ("memory_retrieval", "behavior_rerank"):
        if key in recommendation:
            products[key] = recommendation[key]
    _attach_recommendation(
        session, products, turn_id=turn.turn_id,
        ranking_source="cpu_live:search_products+recommend_products+gate",
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
    task: Any | None = None,
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
            profile_id, "run_agent_turn", {"query": message, "session_id": session.session_id},
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
    turn.planner_actions = list(planner.get("actions") or [])
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
    _attach_recommendation(
        session, products, turn_id=turn.turn_id,
        ranking_source="qwen_live:run_agent_turn+gate",
    )
    turn.products = products
    yield event("products", products)

    if status == "awaiting_clarification":
        clarification_raw = result.get("clarification") or {}
        clarification = vm.clarification_payload(clarification_raw)
        clarification["memory_write_enabled"] = bool(clarification_raw.get("memory_write_enabled"))
        turn.clarification = clarification
        narrative = vm.narrative_qwen(
            status=status, gate=gate, planner=planner, selected_card=None,
            clarification=clarification, catalogue_size=products.get("catalogue_size"),
        )
        turn.assistant_text = narrative
        if session._store is not None:
            if task is None:  # pragma: no cover - persistent mode always binds a task
                raise RuntimeError("persistent ASK requires a task")
            session.pending_clarification = session._store.persist_ask(
                session, task, turn, clarification,
                query=message,
                clarification_id=str(clarification.get("clarification_id") or ""),
                mode="qwen_live",
                expires_at=time.time() + float(settings.clarification_ttl_s),
                ranking_source="qwen_live:run_agent_turn+gate",
            )
        else:
            session.pending_clarification = PendingClarification(
                payload=clarification,
                query=message,
                turn_id=turn.turn_id,
                reply_turn_id=session.next_reply_turn_id(),
                mode="qwen_live",
            )
        yield event("clarification", clarification)
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
        session.profile_id, "compare_products",
        {
            "product_refs": list(product_refs),
            "session_id": session.session_id,
            "query": session.last_query or None,
        },
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
    *, manager: MCPManager, session: Session, product_refs: list[str], query: str | None = None
) -> dict[str, Any]:
    raw = await manager.call_tool(
        session.profile_id, "compare_products",
        {
            "product_refs": list(product_refs),
            "session_id": session.session_id,
            "query": query or session.last_query or None,
        },
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
    tracing: Tracing | None = None,
    behavior_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a real user reply through M9, refresh preferences and resume the pending query.

    The browser never supplies ``confirmed_by_user``; this controller sets it only because a real
    HTTP POST containing user-authored text arrived, and labels the channel as
    ``web_ui_explicit_user_reply``.

    Persistent sessions take the durable Warm Reply path: the coordinator proves the live MCP
    handle, claims the durable request, performs the real MCP reply + resume, checkpoints the
    complete result and only returns success after the Finalize transaction commits.  The
    in-process (memory) path below is unchanged.
    """
    if session._store is not None:
        return await _persistent_clarification_reply(
            manager=manager,
            session=session,
            clarification_id=clarification_id,
            message=message,
            tracing=tracing,
            behavior_snapshot=behavior_snapshot,
        )
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

    session.last_query = pending.query

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
                "session_id": session.session_id,
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
    pref_payload = await manager.call_tool(profile_id, "get_active_preferences",
                                           {"session_id": session.session_id})
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
                    {"query": pending.query, "clarification_id": clarification_id,
                     "session_id": session.session_id},
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
            _attach_recommendation(
                session, recommendation, turn_id=turn.turn_id,
                ranking_source="qwen_live:clarification_resume+gate",
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
            rec_arguments: dict[str, Any] = {
                "query": pending.query, "session_id": session.session_id,
            }
            if behavior_snapshot is not None:
                rec_arguments["behavior_snapshot"] = dict(behavior_snapshot)
            rec_payload = await manager.call_tool(
                profile_id, "recommend_products", rec_arguments,
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
                turn_id=turn.turn_id,
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
        "planner_actions": turn.planner_actions,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Persistent Warm Reply (durable coordinator)
# ---------------------------------------------------------------------------

#: How each coordinator outcome leaves the HTTP layer.  Nothing here is ever a 200.
_REPLY_OUTCOME_ERRORS: dict[str, tuple[str, int, str]] = {
    WARM_UNAVAILABLE: (
        "clarification_resume_not_ready", 409,
        "no live MCP handle proves this clarification is still resumable",
    ),
    BUSY: (
        "clarification_reply_in_progress", 409,
        "another executor is already processing this reply",
    ),
    CONFLICT: (
        "clarification_reply_conflict", 409,
        "this request was already used with different reply content",
    ),
    FENCED: (
        "clarification_reply_fenced", 409,
        "the session, task or pending changed after this reply was claimed",
    ),
    AMBIGUOUS: (
        "clarification_reply_ambiguous", 503,
        "the M9 write may have happened but no durable result was produced; "
        "automatic retry is refused",
    ),
    CHECKPOINT_FAILED: (
        "clarification_result_not_durable", 503,
        "the reply result could not be persisted; nothing was committed",
    ),
}


def _reply_outcome_error(outcome: Any) -> MCPManagerError:
    code, status, message = _REPLY_OUTCOME_ERRORS.get(
        outcome.outcome,
        ("clarification_reply_failed", 503, "the reply could not be completed"),
    )
    detail: dict[str, Any] = {"outcome": outcome.outcome, "reason": outcome.detail}
    if outcome.handle is not None:
        detail["handle"] = outcome.handle.as_payload()
    return MCPManagerError(
        f"{message}: {outcome.detail}" if outcome.detail else message,
        code=code,
        http_status=status,
        detail=detail,
    )


def _mirror_committed_reply(
    session: Session,
    pending: Any,
    result: Mapping[str, Any],
    *,
    next_pending: Any | None = None,
) -> None:
    """Reflect a committed reply in the cached Session (the database is already authoritative).

    The Finalize transaction wrote the reply turn, consumed the pending and either completed
    the task or opened the next ASK.  This only mirrors that committed state for subsequent
    in-process reads and does not perform any durable write.
    """
    turn_view = turn_from_result(result)
    known = {row.turn_id for row in session.turns}
    if turn_view["turn_id"] not in known:
        session.add_turn(Turn(
            turn_id=str(turn_view["turn_id"]),
            user_text=str(turn_view["user_text"]),
            assistant_text=str(turn_view["assistant_text"]),
            products=dict(turn_view.get("products") or {}),
            memory_update=turn_view.get("memory_update"),
            tool_activity=list(turn_view.get("tool_activity") or []),
        ))
    session.last_query = str(pending.query)
    clarification = result.get("clarification")
    if isinstance(clarification, Mapping):
        # Multi-round ASK: the same transaction also opened the next pending.  On a replay
        # the caller has no new PendingRecord, so only attach it if it is still the current
        # OPEN question; never clear or replace a newer one.
        record = next_pending
        if record is None:
            store = getattr(session._store, "state_store", None)
            candidate = (
                store.get_open_pending_for_session(session.session_id)
                if store is not None else None
            )
            if (
                candidate is not None
                and candidate.clarification_id
                == str(clarification.get("clarification_id") or "")
            ):
                record = candidate
        if record is not None:
            session.pending_clarification = PendingClarification(
                payload=dict(record.payload),
                query=record.query,
                turn_id=record.turn_id,
                reply_turn_id=record.reply_turn_id,
                mode=record.mode,
                created_at=vm.iso_from_epoch(record.created_at),
            )
        return
    session.pending_clarification = None
    reply_payload = (result.get("reply") or {}) if isinstance(result, Mapping) else {}
    recommendation_meta = reply_payload.get("recommendation") or {}
    ranking_source = str(recommendation_meta.get("ranking_source") or "")
    products = turn_view.get("products") or {}
    if not (isinstance(products, Mapping) and products.get("items")):
        # A bounded refusal has no product; there is no behavior snapshot to build.
        return
    try:
        session.record_recommendation(
            dict(products), turn_id=str(turn_view["turn_id"]),
            ranking_source=ranking_source or "durable_reply",
        )
    except Exception as exc:  # noqa: BLE001 - behavior context is not the durable authority
        LOGGER.warning(
            "committed reply turn %s could not rebuild its behavior snapshot: %s",
            turn_view.get("turn_id"), exc,
        )


async def _cold_resume_handle(
    *,
    manager: MCPManager,
    store: Any,
    session: Session,
    pending: Any,
) -> ResolvedClarificationHandle:
    tracing = getattr(manager, "tracing", None) or get_tracing()
    with tracing.span(
        "resume.cold_restore",
        attributes={
            "agentrecx.resume.mode": "cold",
            "agentrecx.session_id": session.session_id,
            "agentrecx.pending_id": pending.pending_id,
            "agentrecx.clarification_id": pending.clarification_id,
            "agentrecx.continuation.present": pending.continuation is not None,
        },
    ) as span:
        resolved = await _cold_resume_handle_impl(
            manager=manager, store=store, session=session, pending=pending,
        )
        span.set_attributes({
            "agentrecx.handle.ok": resolved.handle.ok,
            "agentrecx.handle.code": resolved.handle.code,
            "agentrecx.resume.restored": resolved.restored,
        })
        return resolved


async def _cold_resume_handle_impl(
    *,
    manager: MCPManager,
    store: Any,
    session: Session,
    pending: Any,
) -> ResolvedClarificationHandle:
    """Rebuild one pending's handle in the CURRENT MCP process from its durable snapshot.

    This runs before the request claim and performs no business effect: the child only
    installs an in-process handle.  The real M9 write / recommendation still happen after
    ``EXTERNAL_STARTED`` through the normal coordinator path.
    """
    record = store.get_session(session.session_id)
    if record is None:  # pragma: no cover - the HTTP route already resolved the session
        return ResolvedClarificationHandle(
            HandleCheck(False, CONTINUATION_SESSION_MISMATCH,
                        "the session row no longer exists"),
            pending.clarification_id,
        )
    check = validate_continuation(
        pending.continuation,
        session_id=session.session_id,
        profile_id=session.profile_id,
        task_id=pending.task_id,
        pending_id=pending.pending_id,
        clarification_id=pending.clarification_id,
        reset_generation=record.reset_generation,
        expires_at=pending.expires_at,
        query=pending.query,
        expected_user_key=PROFILE_BY_ID[session.profile_id].user_key,
    )
    if not check.ok:
        return ResolvedClarificationHandle(
            HandleCheck(False, check.code, check.detail), pending.clarification_id
        )
    try:
        restored = await manager.call_tool(
            session.profile_id,
            "restore_clarification_handle",
            {"continuation": check.payload, "session_id": session.session_id},
            timeout_s=COLD_RESUME_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - no business effect happened yet
        return ResolvedClarificationHandle(
            HandleCheck(False, COLD_RESUME_FAILED,
                        f"restore_clarification_handle failed: {type(exc).__name__}"),
            pending.clarification_id,
        )
    if not isinstance(restored, Mapping):
        return ResolvedClarificationHandle(
            HandleCheck(False, CONTINUATION_MALFORMED,
                        "the child returned no restore object"),
            pending.clarification_id,
        )
    new_handle = str(restored.get("clarification_id") or "")
    if not new_handle or new_handle == pending.clarification_id:
        return ResolvedClarificationHandle(
            HandleCheck(False, CONTINUATION_MALFORMED,
                        "the child did not issue a new handle"),
            pending.clarification_id,
        )
    if str(restored.get("session_id") or "") != session.session_id:
        return ResolvedClarificationHandle(
            HandleCheck(False, CONTINUATION_SESSION_MISMATCH,
                        "the restored handle belongs to another session"),
            pending.clarification_id,
        )
    handle = manager.check_handle(HandleRef(
        profile_id=session.profile_id,
        session_id=session.session_id,
        clarification_id=new_handle,
    ))
    if handle.ok:
        LOGGER.info(
            "cold resume restored session=%s pending=%s old_handle=%s new_handle=%s",
            session.session_id, pending.pending_id,
            pending.clarification_id, new_handle,
        )
    return ResolvedClarificationHandle(handle, new_handle, restored=True)


async def _persistent_clarification_reply(
    *,
    manager: MCPManager,
    session: Session,
    clarification_id: str,
    message: str,
    tracing: Tracing | None = None,
    behavior_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Durable Warm/Cold Reply: resolve a handle, claim, execute, checkpoint, finalize."""
    store = getattr(session._store, "state_store", None)
    if store is None:  # pragma: no cover - persistent sessions always expose the state store
        raise MCPManagerError(
            "durable clarification reply requires a persistent session store",
            code="clarification_resume_not_ready",
            http_status=409,
        )
    pending = store.get_pending_for_clarification(session.session_id, clarification_id)
    if pending is None:
        raise MCPManagerError(
            "no clarification in this session matches this reply",
            code="clarification_not_pending",
            http_status=409,
            detail={"clarification_id": clarification_id},
        )
    if pending.status in {"CANCELLED", "EXPIRED"}:
        # Reset/supersede/expiry already closed this question; it is not a warm handle issue.
        raise MCPManagerError(
            f"clarification {clarification_id!r} is {pending.status}, not pending",
            code="clarification_not_pending",
            http_status=409,
            detail={"clarification_id": clarification_id, "status": pending.status},
        )
    if pending.mode != "cpu_live":
        # Cold Resume is implemented for the CPU rule planner only.  A Qwen-issued handle
        # carries in-child planner/executor continuation that is not JSON-serialisable, so it
        # stays a 409 instead of resuming with a different task.
        raise MCPManagerError(
            "durable clarification replies are implemented for the CPU rule planner only; "
            f"this pending was issued by mode={pending.mode!r}",
            code="clarification_resume_not_ready",
            http_status=409,
            detail={"clarification_id": clarification_id, "mode": pending.mode},
        )
    runtime = tracing or get_tracing()
    executor = WarmReplyExecutor(
        store,
        call_tool=manager.call_tool,
        assemble=lambda request: assemble_reply_presentation(
            request, catalogue_size=manager.catalogue_size()
        ),
        profile_id=session.profile_id,
        handle_check=manager.check_handle,
        cold_resume=lambda pending_row: _cold_resume_handle(
            manager=manager, store=store, session=session, pending=pending_row,
        ),
        tracing=runtime,
    )
    # A pure handle check tells us which *observed* path this attempt takes.  It performs no
    # MCP call and no state write; the executor still proves the handle itself.
    warm_before = (
        manager.check_handle(HandleRef(
            profile_id=session.profile_id,
            session_id=session.session_id,
            clarification_id=clarification_id,
        )).ok
        if runtime.enabled
        else None
    )
    with runtime.span(
        "agent.reply",
        openinference_kind=OPENINFERENCE_AGENT,
        attributes={
            "agent.name": "AgentRec-X durable clarification reply",
            "agentrecx.path": "reply",
            "agentrecx.mode": pending.mode,
            "agentrecx.session_id": session.session_id,
            "agentrecx.profile_id": session.profile_id,
            "agentrecx.task_id": pending.task_id,
            "agentrecx.pending_id": pending.pending_id,
            "agentrecx.reply_turn_id": pending.reply_turn_id,
            "agentrecx.request_id": pending.reply_turn_id,
            "agentrecx.clarification_id": clarification_id,
        },
    ) as reply_span:
        outcome = await executor.execute(
            session, pending, message=message, clarification_id=clarification_id,
            behavior_snapshot=behavior_snapshot,
        )
        if runtime.enabled:
            if outcome.outcome == REPLAYED:
                reply_path = "replay"
            elif outcome.outcome == RESULT_REUSED:
                reply_path = "checkpoint_finalize"
            elif outcome.outcome == SUCCESS:
                reply_path = "warm" if warm_before else "cold_resume"
            else:
                reply_path = "warm_attempt" if warm_before else "cold_resume_attempt"
            reply_span.set_attributes({
                "agentrecx.reply.outcome": outcome.outcome,
                "agentrecx.reply.path": reply_path,
                "agentrecx.reply.succeeded": outcome.outcome in {
                    SUCCESS, REPLAYED, RESULT_REUSED,
                },
                "agentrecx.request_status": (
                    outcome.request.status if outcome.request is not None else None
                ),
                "agentrecx.execution_stage": (
                    outcome.request.execution_stage if outcome.request is not None else None
                ),
            })
    if outcome.outcome not in {SUCCESS, REPLAYED, RESULT_REUSED} or outcome.result is None:
        raise _reply_outcome_error(outcome)
    result = outcome.result
    response = reply_response_from_result(
        result, clarification_id=pending.clarification_id
    )
    _mirror_committed_reply(
        session, pending, result, next_pending=outcome.next_pending
    )
    return response


async def _recommendation_view(
    *,
    manager: MCPManager,
    session: Session,
    query: str,
    recommendation: Mapping[str, Any],
    recorder: TurnRecorder,
    steps: list[dict[str, Any]],
    turn_id: str,
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
    products = vm.products_payload(
        query=query,
        search_mode=recommendation.get("search_mode"),
        score_kind=None,
        catalogue_size=manager.catalogue_size(),
        items=items,
        recommended_product_id=str(selected_id) if selected_id else None,
        gate=gate,
    )
    return _attach_recommendation(
        session, products, turn_id=turn_id,
        ranking_source="cpu_live:clarification_resume+recommend_products+gate",
    )
