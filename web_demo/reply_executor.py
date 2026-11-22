"""Fenced Warm Reply execution coordinator (S5b-P2b-B + M1-C Round A/B/C).

The coordinator owns the order of operations for a *warm* clarification reply:

    replay? -> prove live handle -> claim -> heartbeat -> EXTERNAL_STARTED
    -> submit_clarification_reply -> get_active_preferences -> recommend_products
    -> get_product_info -> assembly -> contract -> checkpoint -> finalize -> committed result

Security properties kept from the prototype:

* the first step is a real handle proof, not a caller boolean
  (:mod:`web_demo.handle_lifecycle`); a missing, expired, foreign-session, stale-process or
  not-ready handle ends the run *before* any claim, any MCP call and any M9 write;
* a completed request is replayed from its durable result and never re-executes MCP;
* an ``EXTERNAL_STARTED``/``AMBIGUOUS`` request is never retried automatically;
* a ``RESULT_DURABLE`` checkpoint is finalized from persistence alone (no MCP);
* nothing is reported as success before ``finalize_reply_request`` commits the reply turn,
  the task, the pending and the request in one transaction;
* the Web SQLite and the M9 SQLite are separate databases: the reply text is written to M9
  by the MCP child before the Web state commits, and a failure between the two is reported as
  ambiguous, never as success.

Presentation (products, narrative, recommendation id, ranking source) comes from the injected
assembler (:mod:`web_demo.reply_assembly`), so both the warm and the in-memory HTTP paths use
the same real business logic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from observability import OPENINFERENCE_CHAIN, Tracing, get_tracing
from web_demo import viewmodels as vm
from web_demo.continuation import build_web_continuation
from web_demo.handle_lifecycle import (
    HANDLE_EXPIRED,
    HANDLE_MISSING,
    HANDLE_STALE_PROCESS,
    HandleCheck,
    HandleRef,
)
from web_demo.reply_assembly import (
    ReplyAssemblyError,
    ReplyAssemblyRequest,
    assemble_refusal_memory,
    new_recommendation_id,
    turn_from_result,
)
from web_demo.reply_result_contract import (
    RecommendationResultError,
    build_clarification_payload,
    build_refusal_payload,
    build_reply_payload,
)
from web_demo.sessions import PendingClarification, Session
from web_demo.state_store import (
    PendingRecord,
    RequestAmbiguous,
    RequestBusy,
    RequestConflict,
    RequestFenced,
    RequestRecord,
    StaleTaskVersion,
    StateStoreError,
    TaskRecord,
    TaskStateConflict,
    PendingStateConflict,
    SQLiteWebStateStore,
)

__all__ = [
    "COLD_RESUME_FAILED",
    "CPU_RESUME_RANKING_SOURCE",
    "ColdResumer",
    "HandleChecker",
    "ReplyAssembler",
    "ReplyExecutionOutcome",
    "ResolvedClarificationHandle",
    "WarmReplyExecutor",
]

CallTool = Callable[[str, str, Mapping[str, Any]], Awaitable[Any]]
#: The production implementation is ``MCPManager.check_handle``; tests inject a fake.
HandleChecker = Callable[[HandleRef], HandleCheck]
#: The production implementation is :func:`web_demo.reply_assembly.assemble_reply_presentation`.
ReplyAssembler = Callable[[ReplyAssemblyRequest], Mapping[str, Any]]
#: Cold Resume: rebuild a handle in the current MCP process from the durable snapshot.
ColdResumer = Callable[[PendingRecord], Awaitable["ResolvedClarificationHandle"]]

#: Warm-handle outcomes that may be answered by rebuilding the handle from a durable
#: continuation.  A foreign-session ticket or a not-ready child is never "cold resumed".
COLD_RESUMABLE_CODES = frozenset({HANDLE_MISSING, HANDLE_STALE_PROCESS, HANDLE_EXPIRED})

#: Exactly the ranking source the in-memory rule-planner resume path records.
CPU_RESUME_RANKING_SOURCE = "cpu_live:clarification_resume+recommend_products+gate"

EXECUTION_STAGE_RESULT = "RESULT_DURABLE"
SUCCESS = "completed"
REPLAYED = "replayed"
RESULT_REUSED = "result_reused"
WARM_UNAVAILABLE = "warm_unavailable"
BUSY = "busy"
CONFLICT = "conflict"
AMBIGUOUS = "ambiguous"
FENCED = "fenced"
CHECKPOINT_FAILED = "checkpoint_failed"

#: A cold resume failed before any external execution began (validation/MCP error).
COLD_RESUME_FAILED = "cold_resume_failed"

#: Legitimate "no product can be finalized" outcomes caused by the task budget.  The real
#: recommendation ran and explicitly declined; they are durable, replayable refusals, never
#: AMBIGUOUS.  Generic gate outcomes keep the existing 503 semantics.
REFUSAL_STATUSES = frozenset({
    "budget_unverifiable",
    "budget_exceeded",
    # A verified hard current-task constraint can make every candidate in the bounded pool
    # ineligible.  That is a known, durable refusal, not an uncertain external outcome.
    "no_verified_candidate_in_scope",
})


@dataclass(frozen=True)
class ResolvedClarificationHandle:
    """The handle a reply may use, and whether it was rebuilt in this process."""

    handle: HandleCheck
    clarification_id: str
    restored: bool = False


@dataclass(frozen=True)
class ReplyExecutionOutcome:
    """Explicit, diagnostic result of one coordinator run."""

    outcome: str
    request: RequestRecord | None = None
    task: TaskRecord | None = None
    pending: PendingRecord | None = None
    turn: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    #: The newly committed OPEN pending when this reply produced a next ASK.
    next_pending: PendingRecord | None = None
    detail: str = ""
    #: The handle decision that gated this run (``None`` for replay/no pending).
    handle: HandleCheck | None = None


def _default_recommend_arguments(session: Session, pending: PendingRecord) -> dict[str, Any]:
    return {"query": pending.query, "session_id": session.session_id}


def _reject_all_handles(ref: HandleRef) -> HandleCheck:  # noqa: ARG001 - fail-closed default
    return HandleCheck(
        False, HANDLE_MISSING,
        "no live-handle checker is wired; a warm reply is unavailable",
    )


def _latency_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _step(tool: str, started: float) -> dict[str, Any]:
    return {"tool": tool, "status": "ok", "latency_ms": _latency_ms(started)}


def _payload_of(request: RequestRecord | None) -> dict[str, Any] | None:
    """The durable result payload of a completed/checkpointed request, if any."""
    body = (request.result if request is not None else None) or {}
    payload = body.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else None


def _reply_task_constraints(
    reply: Any, pending: PendingRecord
) -> dict[str, Any] | None:
    """Return the merged task constraints from the reply, with a Cold Resume fallback.

    The MCP reply is authoritative; the durable continuation is the fallback for an older
    child that does not echo the field.  This keeps the CPU recommendation and the restored
    task on the same constraint snapshot without inventing a second store.
    """
    if isinstance(reply, Mapping) and isinstance(reply.get("task_constraints"), Mapping):
        return dict(reply["task_constraints"])
    continuation = pending.continuation
    if isinstance(continuation, Mapping):
        mcp = continuation.get("mcp")
        if isinstance(mcp, Mapping) and isinstance(mcp.get("task_constraints"), Mapping):
            return dict(mcp["task_constraints"])
    return None


def _validated_presentation(
    presentation: Any, *, recommendation_id: str
) -> Mapping[str, Any]:
    """Reject a broken assembler result instead of persisting a half-empty reply."""
    if not isinstance(presentation, Mapping):
        raise ReplyAssemblyError("the reply assembler returned no presentation object")
    for key in ("products", "narrative", "recommendation_id", "ranking_source"):
        if key not in presentation:
            raise ReplyAssemblyError(f"the reply presentation has no {key!r}")
    if str(presentation.get("recommendation_id")) != recommendation_id:
        raise ReplyAssemblyError(
            "the reply presentation re-issued the recommendation id instead of reusing it"
        )
    return presentation


def _budget_label(budget: Mapping[str, Any] | None) -> str | None:
    if not isinstance(budget, Mapping):
        return None
    currency = str(budget.get("currency") or "currency unknown")
    if isinstance(budget.get("max_amount"), (int, float)):
        return f"<= {float(budget['max_amount']):g} {currency}"
    if isinstance(budget.get("min_amount"), (int, float)):
        return f">= {float(budget['min_amount']):g} {currency}"
    return None


def _refusal_narrative(
    *,
    status: str,
    recommendation: Mapping[str, Any],
    task_requirements: Mapping[str, Any] | None,
) -> str:
    """Deterministic, data-backed explanation of a bounded refusal (no fabricated product)."""
    gate = recommendation.get("gate") if isinstance(recommendation, Mapping) else {}
    budget = (
        recommendation.get("budget_verification")
        if isinstance(recommendation, Mapping) else None
    )
    budget = budget if isinstance(budget, Mapping) else {}
    requirements = task_requirements if isinstance(task_requirements, Mapping) else {}
    task_budget = requirements.get("budget")
    task_budget = task_budget if isinstance(task_budget, Mapping) else {}
    lines: list[str] = ["No product recommendation can be finalized for this task."]
    label = str(budget.get("label") or "") or _budget_label(task_budget)
    if label:
        lines.append(f"Task budget: {label}.")
        source = str(task_budget.get("source_text") or "").strip()
        if source:
            lines.append(f"Budget source: {source!r}.")
    reason = str(
        budget.get("selected_reason")
        or (gate or {}).get("reason")
        or (gate or {}).get("reason_code")
        or status
    )
    lines.append(f"Constraint check: {reason}.")
    counts: list[str] = []
    if isinstance(budget.get("over"), int) and budget.get("over"):
        counts.append(f"{budget['over']} above the budget")
    if isinstance(budget.get("unknown"), int) and budget.get("unknown"):
        counts.append(f"{budget['unknown']} with no verifiable catalogue price (price UNKNOWN)")
    if isinstance(budget.get("currency_mismatch"), int) and budget.get("currency_mismatch"):
        counts.append(
            f"{budget['currency_mismatch']} priced in another currency (no exchange rate is available)"
        )
    if counts:
        lines.append("Candidates not verifiable under the budget: " + ", ".join(counts) + ".")
    category = requirements.get("category")
    use_cases = requirements.get("use_cases")
    task_bits: list[str] = []
    if isinstance(category, str) and category:
        task_bits.append(f"category={category}")
    if isinstance(use_cases, list) and use_cases:
        task_bits.append("use_case=" + ", ".join(str(item) for item in use_cases))
    if task_bits:
        lines.append("Merged task requirements: " + "; ".join(task_bits) + ".")
    lines.append(
        "No product is shown rather than presenting an unverified match."
    )
    lines.append(
        "Catalogue evidence only; this is not a recommendation."
    )
    return "\n".join(lines)


class WarmReplyExecutor:
    """Coordinates one warm reply using the durable request ledger.

    ``handle_check`` must be bound to the *current* MCP lifecycle (``MCPManager.check_handle``
    in production); the executor proves the handle itself, so a caller cannot assert warmth
    with a boolean.  ``cold_resume`` (optional) rebuilds a handle in the current MCP process
    from the pending's durable continuation before the claim; it performs no external business
    effect.  ``assemble`` is the pure BFF presentation builder.  ``call_tool`` is the real
    ``MCPManager.call_tool`` in production and a controlled fake in tests.
    """

    def __init__(
        self,
        store: SQLiteWebStateStore,
        *,
        call_tool: CallTool,
        assemble: ReplyAssembler,
        profile_id: str,
        handle_check: HandleChecker | None = None,
        cold_resume: ColdResumer | None = None,
        ranking_source: str = CPU_RESUME_RANKING_SOURCE,
        heartbeat_interval: float = 1.0,
        lease_duration: float = 60.0,
        tool_timeout: float = 60.0,
        tracing: Tracing | None = None,
    ) -> None:
        if not callable(assemble):
            raise TypeError("WarmReplyExecutor requires a reply assembler")
        self._store = store
        self._call_tool = call_tool
        self._assemble = assemble
        self._profile_id = profile_id
        self._handle_check = handle_check or _reject_all_handles
        self._cold_resume = cold_resume
        self._ranking_source = ranking_source
        self._heartbeat_interval = heartbeat_interval
        self._lease_duration = lease_duration
        self._tool_timeout = tool_timeout
        self._tracing = tracing or get_tracing()
        self._heartbeat: asyncio.Task[None] | None = None
        self._heartbeat_failed = False
        self._request_id: str | None = None
        self._claim_token: str | None = None
        self._session_id: str | None = None

    # -- public ------------------------------------------------------------- #

    async def execute(
        self,
        session: Session,
        pending: PendingRecord | None,
        *,
        message: str,
        clarification_id: str,
        recommend_arguments: Mapping[str, Any] | None = None,
    ) -> ReplyExecutionOutcome:
        if pending is None:
            return ReplyExecutionOutcome(WARM_UNAVAILABLE, detail="no pending clarification")

        # Ledger states first: a completed request replays, a durable checkpoint finalizes,
        # and an already-started ambiguous execution stays refused — none of them needs a
        # handle, an MCP process or a Cold Resume snapshot (the result is in SQLite).
        existing = self._store.get_request(session.session_id, pending.reply_turn_id)
        if existing is not None:
            if pending.clarification_id != clarification_id:
                return ReplyExecutionOutcome(
                    CONFLICT, pending=pending, detail="clarification id mismatch"
                )
            lease_live = (
                existing.lease_expires_at is not None
                and float(existing.lease_expires_at) > time.time()
            )
            # Only a claim that died before external execution (CLAIMED with an expired
            # lease) may be reclaimed and executed again; that path still needs a handle.
            needs_handle = (
                existing.status == "PROCESSING"
                and existing.execution_stage == "CLAIMED"
                and not lease_live
            )
            if not needs_handle:
                return await self._claim(session, pending, message)

        if pending.status != "OPEN":
            return ReplyExecutionOutcome(
                WARM_UNAVAILABLE, pending=pending,
                detail=f"pending is {pending.status}, not OPEN",
            )
        # Resolve the handle before anything durable happens: no claim, no MCP business call
        # and no M9 write can occur for a handle this process cannot prove.  When the old
        # handle died with a previous MCP process, the durable continuation may rebuild one.
        resolved = await self._resolve_handle(session, pending)
        if not resolved.handle.ok:
            return ReplyExecutionOutcome(
                WARM_UNAVAILABLE, pending=pending,
                detail=f"{resolved.handle.code}: {resolved.handle.detail}",
                handle=resolved.handle,
            )
        if pending.clarification_id != clarification_id:
            return ReplyExecutionOutcome(
                CONFLICT, pending=pending, handle=resolved.handle,
                detail="clarification id mismatch",
            )
        return await self._claim(
            session, pending, message, handle=resolved.handle,
            handle_clarification_id=resolved.clarification_id,
            recommend_arguments=recommend_arguments,
        )

    async def _resolve_handle(
        self, session: Session, pending: PendingRecord
    ) -> ResolvedClarificationHandle:
        with self._tracing.span(
            "reply.handle_check",
            openinference_kind=OPENINFERENCE_CHAIN,
            attributes={
                "agentrecx.session_id": session.session_id,
                "agentrecx.pending_id": pending.pending_id,
                "agentrecx.clarification_id": pending.clarification_id,
            },
        ) as span:
            resolved = await self._resolve_handle_impl(session, pending)
            span.set_attributes({
                "agentrecx.handle.ok": resolved.handle.ok,
                "agentrecx.handle.code": resolved.handle.code,
                "agentrecx.resume.restored": resolved.restored,
                "agentrecx.resume.mode": "cold" if resolved.restored else "warm",
            })
            return resolved

    async def _resolve_handle_impl(
        self, session: Session, pending: PendingRecord
    ) -> ResolvedClarificationHandle:
        """Return the handle a reply may use, rebuilding it only when that is provable."""
        ref = HandleRef(
            profile_id=self._profile_id,
            session_id=session.session_id,
            clarification_id=pending.clarification_id,
        )
        check = self._handle_check(ref)
        if check.ok:
            return ResolvedClarificationHandle(check, pending.clarification_id)
        if self._cold_resume is None or check.code not in COLD_RESUMABLE_CODES:
            return ResolvedClarificationHandle(check, pending.clarification_id)
        try:
            restored = await self._cold_resume(pending)
        except Exception as exc:  # noqa: BLE001 - a failed restore is never a success
            return ResolvedClarificationHandle(
                HandleCheck(False, COLD_RESUME_FAILED,
                            f"{type(exc).__name__}: {exc}"),
                pending.clarification_id,
            )
        if not isinstance(restored, ResolvedClarificationHandle):
            return ResolvedClarificationHandle(
                HandleCheck(False, COLD_RESUME_FAILED,
                            "the cold resumer returned no resolved handle"),
                pending.clarification_id,
            )
        return restored

    # -- claim -------------------------------------------------------------- #

    def _claim_request(
        self, session: Session, pending: PendingRecord, payload: Mapping[str, Any]
    ) -> Any:
        with self._tracing.span(
            "reply.claim",
            attributes={
                "agentrecx.session_id": session.session_id,
                "agentrecx.pending_id": pending.pending_id,
                "agentrecx.reply_turn_id": pending.reply_turn_id,
                "agentrecx.request_id": pending.reply_turn_id,
            },
        ) as span:
            try:
                claim = self._store.claim_reply_request(
                    session.session_id, pending.pending_id, payload=payload,
                    lease_duration=self._lease_duration,
                )
            except Exception as exc:
                span.set_attribute(
                    "agentrecx.claim.result",
                    {
                        RequestBusy: "busy",
                        RequestConflict: "conflict",
                        RequestAmbiguous: "ambiguous",
                        PendingStateConflict: "fenced",
                        RequestFenced: "fenced",
                        StateStoreError: "state_error",
                    }.get(type(exc), "error"),
                )
                raise
            span.set_attributes({
                "agentrecx.claim.result": claim.outcome,
                "agentrecx.request_id": claim.request.request_id,
                "agentrecx.request_status": claim.request.status,
                "agentrecx.execution_stage": claim.request.execution_stage,
            })
            return claim

    async def _claim(
        self,
        session: Session,
        pending: PendingRecord,
        message: str,
        *,
        handle: HandleCheck | None = None,
        handle_clarification_id: str | None = None,
        recommend_arguments: Mapping[str, Any] | None = None,
    ) -> ReplyExecutionOutcome:
        payload = {
            "message": message,
            "confirmed_by_user": True,
            "confirmation_source": "web_ui_explicit_user_reply",
        }
        try:
            claim = self._claim_request(session, pending, payload)
        except RequestBusy as exc:
            return ReplyExecutionOutcome(BUSY, pending=pending, detail=str(exc))
        except RequestConflict as exc:
            return ReplyExecutionOutcome(CONFLICT, pending=pending, detail=str(exc))
        except RequestAmbiguous as exc:
            return ReplyExecutionOutcome(AMBIGUOUS, pending=pending, detail=str(exc))
        except PendingStateConflict as exc:
            return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc))
        except (RequestFenced, TaskStateConflict, StaleTaskVersion) as exc:
            return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc))
        except StateStoreError as exc:
            return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc))

        self._request_id = claim.request.request_id
        self._claim_token = claim.claim_token
        self._session_id = session.session_id

        if claim.outcome == "replay":
            result = _payload_of(claim.request)
            if result is None:
                return ReplyExecutionOutcome(
                    CHECKPOINT_FAILED, pending=pending,
                    detail="the completed request has no readable durable result",
                )
            try:
                turn_from_result(result)
            except Exception as exc:  # noqa: BLE001 - a corrupt result is never success
                return ReplyExecutionOutcome(
                    CHECKPOINT_FAILED, pending=pending,
                    detail=f"the durable reply result is invalid: {exc}",
                )
            self._tracing.annotate_current({
                "agentrecx.replay": True,
                "agentrecx.execution_stage": claim.request.execution_stage,
            })
            return ReplyExecutionOutcome(
                REPLAYED, request=claim.request, pending=pending,
                result=result, turn=turn_from_result(result), handle=handle,
            )
        if claim.outcome == "result_ready":
            # A durable checkpoint exists; finalize it without touching MCP again.
            return self._finalize_from_checkpoint(session, pending, claim, handle=handle)

        return await self._execute_external(
            session, pending, message, recommend_arguments, handle=handle,
            handle_clarification_id=handle_clarification_id,
        )

    # -- external execution ------------------------------------------------- #

    async def _execute_external(
        self,
        session: Session,
        pending: PendingRecord,
        message: str,
        recommend_arguments: Mapping[str, Any] | None,
        *,
        handle: HandleCheck | None = None,
        handle_clarification_id: str | None = None,
    ) -> ReplyExecutionOutcome:
        with self._tracing.span(
            "reply.external_execution",
            openinference_kind=OPENINFERENCE_CHAIN,
            attributes={
                "agentrecx.session_id": session.session_id,
                "agentrecx.pending_id": pending.pending_id,
                "agentrecx.request_id": self._request_id,
                "agentrecx.reply_turn_id": pending.reply_turn_id,
            },
        ) as span:
            outcome = await self._execute_external_impl(
                session, pending, message, recommend_arguments,
                handle=handle, handle_clarification_id=handle_clarification_id,
            )
            span.set_attributes({
                "agentrecx.reply.outcome": outcome.outcome,
                "agentrecx.request_status": (
                    outcome.request.status if outcome.request is not None else None
                ),
                "agentrecx.execution_stage": (
                    outcome.request.execution_stage if outcome.request is not None else None
                ),
            })
            return outcome

    async def _execute_external_impl(
        self,
        session: Session,
        pending: PendingRecord,
        message: str,
        recommend_arguments: Mapping[str, Any] | None,
        *,
        handle: HandleCheck | None = None,
        handle_clarification_id: str | None = None,
    ) -> ReplyExecutionOutcome:
        assert self._claim_token is not None and self._request_id is not None
        self._start_heartbeat(session.session_id, self._request_id, self._claim_token)
        # A cold resume installs a NEW handle in the current process; the old id is never
        # sent back to the child.  The durable pending/request identities stay unchanged.
        reply_handle_id = str(handle_clarification_id or pending.clarification_id)
        starting_arguments = {
            "clarification_id": reply_handle_id,
            "user_message": message,
            "turn_id": pending.reply_turn_id,
            "confirmed_by_user": True,
            "confirmation_source": "web_ui_explicit_user_reply",
            "session_id": session.session_id,
        }
        steps: list[dict[str, Any]] = []
        try:
            try:
                with self._tracing.span(
                    "reply.external_started",
                    attributes={
                        "agentrecx.session_id": session.session_id,
                        "agentrecx.request_id": self._request_id,
                    },
                ) as stage_span:
                    self._store.mark_request_external_started(
                        session.session_id, self._request_id,
                        claim_token=self._claim_token,
                    )
                    stage_span.set_attribute("agentrecx.execution_stage", "EXTERNAL_STARTED")
            except (RequestFenced, StateStoreError) as exc:
                return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc),
                                             handle=handle)

            started = time.perf_counter()
            try:
                reply = await asyncio.wait_for(
                    self._call_tool(
                        self._profile_id, "submit_clarification_reply", starting_arguments
                    ),
                    timeout=self._tool_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - external outcome unknown
                self._mark_ambiguous(f"submit_clarification_reply failed: {exc}")
                return ReplyExecutionOutcome(
                    AMBIGUOUS, pending=pending, handle=handle,
                    detail=f"external execution outcome unknown: {type(exc).__name__}",
                )
            steps.append(_step("submit_clarification_reply", started))
            # The reply carries the merged task requirements (original query + real reply).
            # Passing them into the resumed recommendation makes the budget/use case
            # explicit even for a cold-resumed handle.
            reply_requirements: dict[str, Any] | None = None
            if (isinstance(reply, Mapping)
                    and isinstance(reply.get("task_requirements"), Mapping)):
                reply_requirements = dict(reply["task_requirements"])
            reply_constraints = _reply_task_constraints(reply, pending)

            try:
                started = time.perf_counter()
                preferences = await asyncio.wait_for(
                    self._call_tool(
                        self._profile_id, "get_active_preferences",
                        {"session_id": session.session_id},
                    ),
                    timeout=self._tool_timeout,
                )
                steps.append(_step("get_active_preferences", started))

                recommend_payload = dict(
                    recommend_arguments or _default_recommend_arguments(session, pending)
                )
                if reply_requirements is not None:
                    recommend_payload["task_requirements"] = reply_requirements
                if reply_constraints is not None:
                    recommend_payload["task_constraints"] = reply_constraints
                started = time.perf_counter()
                recommendation = await asyncio.wait_for(
                    self._call_tool(
                        self._profile_id, "recommend_products", recommend_payload
                    ),
                    timeout=self._tool_timeout,
                )
                steps.append(_step("recommend_products", started))

                product_infos: dict[str, Mapping[str, Any]] = {}
                selected_id = (
                    str(recommendation.get("selected_product_id") or "")
                    if isinstance(recommendation, Mapping) else ""
                )
                if selected_id:
                    started = time.perf_counter()
                    product_infos[selected_id] = await asyncio.wait_for(
                        self._call_tool(
                            self._profile_id, "get_product_info",
                            {"product_ref": selected_id},
                        ),
                        timeout=self._tool_timeout,
                    )
                    steps.append(_step("get_product_info", started))
            except Exception as exc:  # noqa: BLE001 - result incomplete
                self._mark_ambiguous(f"post-reply read failed: {exc}")
                return ReplyExecutionOutcome(
                    AMBIGUOUS, pending=pending, handle=handle,
                    detail=f"reply executed but result incomplete: {type(exc).__name__}",
                )

            if self._heartbeat_failed:
                return ReplyExecutionOutcome(
                    FENCED, pending=pending, handle=handle,
                    detail="lease heartbeat failed",
                )

            recommendation_status = (
                str(recommendation.get("status") or "")
                if isinstance(recommendation, Mapping) else ""
            )
            if recommendation_status == "awaiting_clarification":
                # A complete, newly issued next ASK is a known outcome of this reply: the M9
                # write happened and the MCP explicitly asked one more question.  Persist it
                # atomically instead of collapsing it into the old 503 AMBIGUOUS.
                return self._clarification_and_finalize(
                    session, pending,
                    message=message,
                    reply=reply,
                    preferences=preferences,
                    recommendation=(
                        recommendation if isinstance(recommendation, Mapping) else {}
                    ),
                    reply_requirements=reply_requirements,
                    reply_constraints=reply_constraints,
                    steps=steps,
                    handle=handle,
                )
            if recommendation_status in REFUSAL_STATUSES:
                # The real recommendation ran and explicitly declined to finalize a product
                # (for example the task budget cannot be verified against the catalogue
                # price).  This is a known, durable refusal -- not AMBIGUOUS, and never a
                # fabricated recommendation.
                gate = recommendation.get("gate")
                budget = recommendation.get("budget_verification")
                contract = build_refusal_payload(
                    reply_turn_id=pending.reply_turn_id,
                    pending_id=pending.pending_id,
                    user_message=message,
                    narrative=_refusal_narrative(
                        status=recommendation_status, recommendation=recommendation,
                        task_requirements=reply_requirements,
                    ),
                    status=recommendation_status,
                    gate=gate if isinstance(gate, Mapping) else None,
                    task_requirements=reply_requirements,
                    ranking_source=self._ranking_source,
                    budget=budget if isinstance(budget, Mapping) else None,
                    query=pending.query,
                    retrieval_query=(
                        str(recommendation.get("effective_query"))
                        if isinstance(recommendation, Mapping)
                        and recommendation.get("effective_query") else None
                    ),
                )
                memory = assemble_refusal_memory(
                    reply if isinstance(reply, Mapping) else {},
                    preferences if isinstance(preferences, Mapping) else {},
                    profile_id=self._profile_id,
                )
                result = {
                    "reply": contract,
                    "preferences": memory["preferences"],
                    "memory_update": memory["memory_update"],
                    "steps": steps,
                    "refusal": dict(contract["refusal"]),
                    "task_requirements": reply_requirements,
                    "budget_verification": (
                        dict(budget) if isinstance(budget, Mapping) else None
                    ),
                    "retrieval_query": (
                        str(recommendation.get("effective_query"))
                        if isinstance(recommendation, Mapping)
                        and recommendation.get("effective_query") else None
                    ),
                }
                if isinstance(reply_constraints, Mapping):
                    result["task_constraints"] = dict(reply_constraints)
                turn = turn_from_result(result)
                return self._checkpoint_and_finalize(
                    session, pending, turn=turn, result=result, handle=handle
                )

            # Presentation is pure BFF logic over the real tool payloads.  A non-final or
            # unenrichable result is never turned into a fabricated recommendation.
            recommendation_id = new_recommendation_id()
            try:
                presentation = _validated_presentation(
                    self._assemble(ReplyAssemblyRequest(
                        recommendation=recommendation,
                        product_infos=product_infos,
                        reply=reply,
                        preference_payload=preferences,
                        query=pending.query,
                        recommendation_id=recommendation_id,
                        ranking_source=self._ranking_source,
                        profile_id=self._profile_id,
                        turn_id=pending.reply_turn_id,
                    )),
                    recommendation_id=recommendation_id,
                )
                contract = build_reply_payload(
                    reply_turn_id=pending.reply_turn_id,
                    pending_id=pending.pending_id,
                    user_message=message,
                    recommendation=recommendation,
                    narrative=str(presentation["narrative"]),
                    recommendation_id=str(presentation["recommendation_id"]),
                    ranking_source=str(presentation["ranking_source"]),
                    products=presentation["products"],
                )
            except (ReplyAssemblyError, RecommendationResultError) as exc:
                self._mark_ambiguous(f"reply presentation rejected: {exc}")
                return ReplyExecutionOutcome(
                    AMBIGUOUS, pending=pending, handle=handle,
                    detail=f"reply result cannot be presented: {exc}",
                )
            except Exception as exc:  # noqa: BLE001 - never persist half a reply
                self._mark_ambiguous(f"reply presentation failed: {exc}")
                return ReplyExecutionOutcome(
                    AMBIGUOUS, pending=pending, handle=handle,
                    detail=f"reply presentation failed: {type(exc).__name__}",
                )

            result = {
                "reply": contract,
                "preferences": presentation.get("preferences"),
                "memory_update": presentation.get("memory_update"),
                "steps": steps,
                "task_requirements": reply_requirements,
                "budget_verification": (
                    dict(recommendation.get("budget_verification"))
                    if isinstance(recommendation, Mapping)
                    and isinstance(recommendation.get("budget_verification"), Mapping)
                    else None
                ),
                "retrieval_query": (
                    str(recommendation.get("effective_query"))
                    if isinstance(recommendation, Mapping)
                    and recommendation.get("effective_query") else None
                ),
            }
            if isinstance(reply_constraints, Mapping):
                result["task_constraints"] = dict(reply_constraints)
            # The durable turn carries the same BFF extras as the live response so a restored
            # transcript shows the memory update and tool activity without re-running MCP.
            turn = turn_from_result(result)
            return self._checkpoint_and_finalize(
                session, pending, turn=turn, result=result, handle=handle
            )
        finally:
            await self._stop_heartbeat()

    def _clarification_and_finalize(
        self,
        session: Session,
        pending: PendingRecord,
        *,
        message: str,
        reply: Any,
        preferences: Any,
        recommendation: Mapping[str, Any],
        reply_requirements: Mapping[str, Any] | None,
        reply_constraints: Mapping[str, Any] | None,
        steps: list[dict[str, Any]],
        handle: HandleCheck | None,
    ) -> ReplyExecutionOutcome:
        """Persist a real next ASK as the reply's committed outcome.

        The MCP has already written M9 and returned an explicit ``awaiting_clarification``
        result with a newly issued handle.  This method validates that result, builds the
        durable turn and checkpoint, and lets the store commit the answered pending, the reply
        turn, the new pending and the completed request in one transaction.
        """
        raw = recommendation.get("clarification")
        if not isinstance(raw, Mapping):
            self._mark_ambiguous("recommendation asked for clarification but issued no handle")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail="the recommendation requested a next ASK without a clarification handle",
            )
        next_clarification_id = str(raw.get("clarification_id") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not next_clarification_id or not question:
            self._mark_ambiguous("the next ASK is missing its id or question")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail="the next ASK is incomplete",
            )
        if next_clarification_id == pending.clarification_id:
            self._mark_ambiguous("the next ASK reused the answered clarification id")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail="the next ASK reused the answered handle",
            )
        continuation = build_web_continuation(
            mcp_continuation=raw.get("continuation"),
            query=pending.query,
            profile_id=self._profile_id,
        )
        if continuation is None:
            self._mark_ambiguous("the next ASK has no restorable continuation")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail="the next ASK has no restorable continuation",
            )
        expires_at = float(continuation["expires_at"])
        if expires_at <= time.time():
            self._mark_ambiguous("the next ASK continuation is already expired")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail="the next ASK continuation is already expired",
            )

        clarification = vm.clarification_payload(raw)
        clarification["memory_write_enabled"] = bool(raw.get("memory_write_enabled"))
        narrative = (
            "One more task detail is required before a product can be recommended.\n"
            f"Question: {question}\n"
            "No product recommendation has been finalized in this reply."
        )
        try:
            contract = build_clarification_payload(
                reply_turn_id=pending.reply_turn_id,
                pending_id=pending.pending_id,
                user_message=message,
                question=question,
                reason=str(raw.get("reason") or "task_clarification"),
                ranking_source=self._ranking_source,
                query=pending.query,
                retrieval_query=(
                    str(recommendation.get("effective_query"))
                    if recommendation.get("effective_query") else None
                ),
                task_requirements=reply_requirements,
            )
        except Exception as exc:  # noqa: BLE001 - never persist half a reply
            self._mark_ambiguous(f"next-ASK reply contract failed: {exc}")
            return ReplyExecutionOutcome(
                AMBIGUOUS, pending=pending, handle=handle,
                detail=f"next-ASK reply contract failed: {type(exc).__name__}",
            )
        # The durable turn asks the question; the response's narrative is the same text.
        contract["assistant_text"] = narrative
        preference_view = (
            vm.preference_state(dict(preferences), profile_id=self._profile_id)
            if isinstance(preferences, Mapping) else None
        )
        memory_update = vm.memory_update_payload(
            dict(reply) if isinstance(reply, Mapping) else {},
            preference_view,
        )
        result = {
            "reply": contract,
            "preferences": preference_view,
            "memory_update": memory_update,
            "steps": steps,
            "task_requirements": reply_requirements,
            "clarification": clarification,
            "clarification_continuation": continuation,
            "clarification_expires_at": expires_at,
            "clarification_mode": "cpu_live",
            "retrieval_query": (
                str(recommendation.get("effective_query"))
                if recommendation.get("effective_query") else None
            ),
        }
        if isinstance(reply_constraints, Mapping):
            result["task_constraints"] = dict(reply_constraints)
        turn = turn_from_result(result)
        return self._checkpoint_and_finalize(
            session, pending, turn=turn, result=result, handle=handle
        )

    # -- commit ------------------------------------------------------------- #

    def _checkpoint_and_finalize(
        self,
        session: Session,
        pending: PendingRecord,
        *,
        turn: Mapping[str, Any],
        result: Mapping[str, Any],
        handle: HandleCheck | None,
    ) -> ReplyExecutionOutcome:
        assert self._claim_token is not None and self._request_id is not None
        try:
            with self._tracing.span(
                "reply.checkpoint",
                attributes={
                    "agentrecx.session_id": session.session_id,
                    "agentrecx.request_id": self._request_id,
                    "agentrecx.reply_turn_id": pending.reply_turn_id,
                },
            ) as span:
                self._store.checkpoint_reply_result(
                    session.session_id, self._request_id,
                    claim_token=self._claim_token, result=result,
                )
                span.set_attribute("agentrecx.execution_stage", "RESULT_DURABLE")
        except RequestFenced as exc:
            return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc),
                                         handle=handle)
        except RequestConflict as exc:
            return ReplyExecutionOutcome(CONFLICT, pending=pending, detail=str(exc),
                                         handle=handle)
        except Exception as exc:  # noqa: BLE001 - checkpoint never durable
            self._mark_ambiguous(f"checkpoint failed: {exc}")
            return ReplyExecutionOutcome(
                CHECKPOINT_FAILED, pending=pending, handle=handle,
                detail=f"checkpoint not durable: {type(exc).__name__}",
            )
        return self._finalize(
            session, pending, turn=dict(turn), result=result, handle=handle,
            source="execute",
        )

    # -- finalize ----------------------------------------------------------- #

    def _finalize_from_checkpoint(
        self,
        session: Session,
        pending: PendingRecord,
        claim: Any,
        *,
        handle: HandleCheck | None = None,
    ) -> ReplyExecutionOutcome:
        assert self._claim_token is not None and self._request_id is not None
        result = _payload_of(claim.request)
        if result is None:
            return ReplyExecutionOutcome(
                CHECKPOINT_FAILED, pending=pending, handle=handle,
                detail="the durable checkpoint has no readable result payload",
            )
        try:
            turn = turn_from_result(result)
        except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint is never finalized
            return ReplyExecutionOutcome(
                CHECKPOINT_FAILED, pending=pending, handle=handle,
                detail=f"the durable checkpoint is invalid: {exc}",
            )
        self._tracing.annotate_current({
            "agentrecx.used_checkpoint": True,
            "agentrecx.execution_stage": "RESULT_DURABLE",
        })
        outcome = self._finalize(
            session, pending, turn=turn, result=result, handle=handle,
            source="checkpoint",
        )
        if outcome.outcome == SUCCESS:
            return ReplyExecutionOutcome(
                RESULT_REUSED, request=outcome.request, task=outcome.task,
                pending=outcome.pending, next_pending=outcome.next_pending,
                turn=outcome.turn, result=outcome.result, handle=handle,
            )
        return outcome

    def _finalize(
        self,
        session: Session,
        pending: PendingRecord,
        *,
        turn: dict[str, Any],
        result: Mapping[str, Any],
        handle: HandleCheck | None = None,
        source: str = "execute",
    ) -> ReplyExecutionOutcome:
        assert self._claim_token is not None and self._request_id is not None
        clarification = result.get("clarification")
        continuation = result.get("clarification_continuation")
        next_pending: PendingRecord | None = None
        try:
            with self._tracing.span(
                "reply.finalize",
                attributes={
                    "agentrecx.session_id": session.session_id,
                    "agentrecx.request_id": self._request_id,
                    "agentrecx.reply_turn_id": pending.reply_turn_id,
                    "agentrecx.finalize.source": source,
                    "agentrecx.finalize.next_ask": isinstance(clarification, Mapping),
                },
            ) as span:
                if isinstance(clarification, Mapping) and isinstance(continuation, Mapping):
                    task, consumed, next_pending, request = (
                        self._store.finalize_reply_with_clarification(
                            session.session_id, self._request_id,
                            claim_token=self._claim_token,
                            requirements=(
                                result.get("task_requirements")
                                if isinstance(result.get("task_requirements"), Mapping)
                                else None
                            ),
                            turn=turn,
                            # No product was ranked in this turn; the reply only asks again.
                            ranking_source=None,
                            result=result,
                            next_clarification_id=str(
                                clarification.get("clarification_id") or ""
                            ),
                            next_payload=clarification,
                            next_continuation=continuation,
                            next_expires_at=float(
                                result.get("clarification_expires_at") or 0.0
                            ),
                            mode=str(result.get("clarification_mode") or "cpu_live"),
                        )
                    )
                else:
                    task, consumed, request = self._store.finalize_reply_request(
                        session.session_id, self._request_id,
                        claim_token=self._claim_token, outcome="completed",
                        turn=turn, result=result,
                    )
                span.set_attributes({
                    "agentrecx.execution_stage": request.execution_stage,
                    "agentrecx.request_status": request.status,
                    "agentrecx.turn_id": pending.reply_turn_id,
                    "agentrecx.next_pending_id": (
                        None if next_pending is None else next_pending.pending_id
                    ),
                })
        except (RequestFenced, StaleTaskVersion, TaskStateConflict,
                PendingStateConflict, RequestConflict) as exc:
            return ReplyExecutionOutcome(FENCED, pending=pending, detail=str(exc),
                                         handle=handle)
        except (ValueError, TypeError, StateStoreError) as exc:
            # The checkpoint is already durable; never invent a replacement ASK or clear it.
            return ReplyExecutionOutcome(
                CHECKPOINT_FAILED, pending=pending, handle=handle,
                detail=f"the next ASK could not be committed: {exc}",
            )
        return ReplyExecutionOutcome(
            SUCCESS, request=request, task=task, pending=consumed,
            next_pending=next_pending, turn=turn, result=dict(result), handle=handle,
        )

    # -- heartbeat ---------------------------------------------------------- #

    def _start_heartbeat(self, session_id: str, request_id: str, token: str) -> None:
        self._heartbeat_failed = False

        async def beat() -> None:
            try:
                while True:
                    await asyncio.sleep(self._heartbeat_interval)
                    try:
                        self._store.renew_request_lease(
                            session_id, request_id, claim_token=token,
                            lease_duration=self._lease_duration,
                        )
                    except Exception:  # noqa: BLE001 - lost submission rights
                        self._heartbeat_failed = True
                        return
            except asyncio.CancelledError:
                raise

        self._heartbeat = asyncio.create_task(beat(), name="warm-reply-heartbeat")

    async def _stop_heartbeat(self) -> None:
        task = self._heartbeat
        self._heartbeat = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _mark_ambiguous(self, reason: str) -> None:
        if (self._request_id is None or self._claim_token is None
                or self._session_id is None):
            return
        try:
            self._store.mark_request_ambiguous(
                self._session_id, self._request_id,
                claim_token=self._claim_token, reason=reason,
            )
        except Exception:  # noqa: BLE001 - diagnostic only; state stays EXTERNAL_STARTED
            pass
