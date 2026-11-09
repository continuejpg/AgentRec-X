"""Four controlled policy adapters for the multi-turn shopping benchmark.

A and B use the same stateless Agent recommendation path and the same trusted sources.
C uses the existing stateful shopping service (M5 + R1 + M7). D extends C with an
LLM preference proposal step.  No adapter receives task expectations or writes them into
the conversation.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from recommendation.control.model_client import ModelCallError, ModelRequest
from recommendation.demo.agent_schemas import AgentRecommendRequest

from .schemas import BenchTask, PolicyId, PolicyStatus, PolicyTask, TaskRun, TurnTrace

SOURCES = ("history", "catalog_search")
_K = 5


class PolicyUnavailable(RuntimeError):
    """A policy requires a provider that is not configured."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class QueryRewriter(Protocol):
    def rewrite(self, *, messages: Sequence[str], current: str, mode: str) -> tuple[str, int, int]: ...


class PreferenceProposer(Protocol):
    def propose(self, *, messages: Sequence[str], current: str) -> tuple[str, list[dict[str, Any]], int, int]: ...


@dataclass
class AdapterContext:
    """Objects shared by adapters; no task expectations cross this boundary."""

    service: Any
    rewrite_client: Any = None
    proposal_client: Any = None
    state_db_path: str | None = None


def _tokens(response: Any) -> tuple[int | None, int | None, int | None]:
    return (
        getattr(response, "input_tokens", None),
        getattr(response, "output_tokens", None),
        getattr(response, "total_tokens", None),
    )


def _extract_json(text: str) -> Any:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("rewriter did not return JSON")
        return json.loads(stripped[start : end + 1])


class LLMQueryRewriter:
    """Minimal structured query rewrite over the existing provider seam."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "complete", None)):
            raise TypeError("rewrite client must expose complete(ModelRequest)")
        self.client = client

    def rewrite(self, *, messages: Sequence[str], current: str, mode: str) -> tuple[str, int, int]:
        system = (
            "You rewrite a shopping request into one concise catalogue search query. "
            "Use only facts stated by the user. In mode A use only the current message. "
            "In mode B use the full user history and the current message, resolving the "
            "currently effective requirements; do not invent products, prices or attributes. "
            'Return JSON exactly as {"query":"..."} and nothing else.'
        )
        context = {"mode": mode, "user_messages": list(messages), "current": current}
        response = self.client.complete(
            ModelRequest(system_prompt=system, context_payload=context)
        )
        payload = _extract_json(response.text)
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("rewriter returned an empty query")
        input_tokens, output_tokens, _ = _tokens(response)
        return query, int(input_tokens or 0), int(output_tokens or 0)


class LLMPreferenceProposer:
    """Minimal structured preference proposal over the existing provider seam."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "complete", None)):
            raise TypeError("proposal client must expose complete(ModelRequest)")
        self.client = client

    def propose(self, *, messages: Sequence[str], current: str) -> tuple[str, list[dict[str, Any]], int, int]:
        system = (
            "Extract only explicitly stated shopping preferences from the current user "
            "message. Do not invent values. Return JSON with keys: rewritten_message and "
            "preferences (list of {attribute, action, value, evidence}). Attributes may be "
            "price, weight, color, material, brand, feature. Actions are add, replace, remove. "
            "The rewritten_message must remain a faithful paraphrase of the current message."
        )
        response = self.client.complete(
            ModelRequest(
                system_prompt=system,
                context_payload={"user_messages": list(messages), "current": current},
            )
        )
        payload = _extract_json(response.text)
        rewritten = str(payload.get("rewritten_message", "")).strip()
        if not rewritten:
            raise ValueError("proposal returned an empty rewritten message")
        proposals: list[dict[str, Any]] = []
        for item in payload.get("preferences", []) or []:
            if not isinstance(item, Mapping):
                continue
            attribute = str(item.get("attribute", "")).strip().lower()
            action = str(item.get("action", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            evidence = " ".join(str(item.get("evidence", "")).split())
            if attribute not in {"price", "weight", "color", "material", "brand", "feature"}:
                continue
            if action not in {"add", "replace", "remove"} or not value:
                continue
            if not evidence or evidence.lower() not in current.lower():
                continue
            if re.search(r"\bB[0-9A-Z]{9}\b", value):
                continue
            proposals.append(
                {"attribute": attribute, "action": action, "value": value[:120], "evidence": evidence[:240]}
            )
        input_tokens, output_tokens, _ = _tokens(response)
        return rewritten, proposals, int(input_tokens or 0), int(output_tokens or 0)


def _resolve_option(response: Any, user_input: str) -> str | None:
    """Choose a structured option from the pending question using the user text.

    This is a simple benchmark user simulator, not an Agent component.  It never sees
    expected answers; it only reads the options the Agent actually displayed.
    """
    clarification = getattr(response, "clarification", None)
    if clarification is None:
        return None
    preference = getattr(clarification, "preference", None)
    options = list(getattr(preference, "options", ()) or ()) if preference is not None else []
    if not options:
        research = list(getattr(clarification, "research_options", ()) or ())
        for option in research:
            if str(getattr(option, "action", "")) == "stop_research":
                return str(getattr(option, "option_id", "")) or None
        return None
    text = " ".join(str(user_input).lower().split())
    for option in options:
        action = str(getattr(option, "action", "")).lower()
        value = str(getattr(option, "value", "") or "").lower()
        label = str(getattr(option, "label", "") or "").lower()
        if action == "lower" and any(word in text for word in ("lighter", "lower", "cheaper", "less expensive")):
            return str(option.option_id)
        if action == "higher" and any(word in text for word in ("heavier", "higher", "more expensive")):
            return str(option.option_id)
        if action == "value" and value and value in text:
            return str(option.option_id)
        if action in {"none", "skip"} and any(
            word in text for word in ("no additional", "no preference", "skip", "anything", "any color")
        ):
            return str(option.option_id)
        if label and label in text:
            return str(option.option_id)
    # Never answer a pending question with a neutral option merely because the user's
    # utterance did not match a label.  The caller treats None as "this is not an answer"
    # and can handle a new goal/retraction as a fresh turn.
    return None


def _action_from_diagnostics(diagnostics: Mapping[str, Any] | None) -> str | None:
    if not diagnostics:
        return None
    action = diagnostics.get("action")
    return str(action) if action else None


def _tool_call_labels(response: Any) -> tuple[str, ...]:
    """Bounded labels for sources/tools the run actually consulted."""
    observation = getattr(response, "observation", None)
    if observation is None:
        return ()
    labels: list[str] = []
    source_calls = getattr(observation, "source_calls", {}) or {}
    if isinstance(source_calls, Mapping):
        labels.extend(f"source:{key}:{value}" for key, value in sorted(source_calls.items()))
    used = getattr(observation, "tool_calls_used", None)
    if used is not None:
        labels.append(f"tool_calls_used:{used}")
    for action in tuple(getattr(response, "research", None).actions if getattr(response, "research", None) else ()):
        labels.append(f"research:{getattr(action, 'tool', 'unknown')}")
    return tuple(labels)


def _state_events(path: str | None, task_id: str, *, after_sequence: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read new M5 event rows for one task; returns events and last sequence."""
    if not path:
        return [], after_sequence
    try:
        connection = sqlite3.connect(path)
    except sqlite3.Error:
        return [], after_sequence
    try:
        rows = connection.execute(
            "select sequence, event_type, payload from events "
            "where task_id = ? and sequence > ? order by sequence",
            (str(task_id), int(after_sequence)),
        ).fetchall()
    except sqlite3.Error:
        return [], after_sequence
    finally:
        connection.close()
    events: list[dict[str, Any]] = []
    last = int(after_sequence)
    for sequence, event_type, payload in rows:
        last = max(last, int(sequence))
        try:
            parsed = json.loads(payload) if payload else {}
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        events.append({"sequence": int(sequence), "event_type": str(event_type), "payload": parsed})
    return events, last


def _active_attributes(service: Any, task_id: str, session_id: str) -> tuple[str, ...]:
    """Read trusted active preference kinds from the task and memory projections."""
    values: set[str] = set()
    for pref in tuple(getattr(service, "_task_preferences", {}).get(task_id, ())):
        attribute = str(getattr(pref, "attribute", "") or "").strip().lower()
        if attribute:
            values.add(attribute)
    memory_service = getattr(service, "_memory_service", None)
    getter = getattr(memory_service, "get_active_preferences", None)
    if callable(getter):
        try:
            snapshot = getter(session_id)
            for entry in tuple(getattr(snapshot, "active_entries", ()) or ()):
                kind = getattr(entry, "kind", "")
                kind = getattr(kind, "value", kind)
                if kind:
                    values.add(str(kind).lower())
        except Exception:  # noqa: BLE001 - a missing projection is not a benchmark fact
            pass
    return tuple(sorted(values))


class _BaseAdapter:
    policy_id: PolicyId
    requires_llm = False

    def __init__(self, context: AdapterContext) -> None:
        self.context = context

    def availability(self) -> str | None:
        return None

    def run(self, task: PolicyTask, *, state_dir: str | None = None) -> TaskRun:
        raise NotImplementedError


class QueryRewriteAdapter(_BaseAdapter):
    """A/B: rewrite the query and use the same stateless candidate path."""

    def __init__(self, context: AdapterContext, *, policy: PolicyId, rewriter: QueryRewriter | None = None) -> None:
        super().__init__(context)
        self.policy_id = policy
        self.requires_llm = True
        self.rewriter = rewriter or (LLMQueryRewriter(context.rewrite_client) if context.rewrite_client else None)

    def availability(self) -> str | None:
        if self.rewriter is None:
            return "llm_not_configured"
        return None

    def run(self, task: PolicyTask, *, state_dir: str | None = None) -> TaskRun:
        reason = self.availability()
        if reason:
            return TaskRun(task.task_id, self.policy_id, False, PolicyStatus.UNAVAILABLE.value, unavailable_reason=reason)
        assert self.rewriter is not None
        start = time.perf_counter()
        trace: list[TurnTrace] = []
        messages: list[str] = []
        final_ids: tuple[str, ...] = ()
        total_in = total_out = 0
        for index, turn in enumerate(task.turns):
            messages.append(turn.user_input)
            context = messages if self.policy_id is PolicyId.B else [turn.user_input]
            turn_start = time.perf_counter()
            try:
                rewritten, in_tokens, out_tokens = self.rewriter.rewrite(
                    messages=context, current=turn.user_input, mode=self.policy_id.value
                )
                total_in += in_tokens
                total_out += out_tokens
                payload = AgentRecommendRequest(
                    message=rewritten,
                    history=list(task.history),
                    k=_K,
                    sources=SOURCES,
                    preference_discovery=False,
                )
                response = self.context.service.run(payload)
                # Reuse the same trusted type/role post-check as the stateful shopping
                # path so the query-rewrite arms are not compared on an unfiltered list.
                try:
                    from recommendation.demo.shopping_intent import promote_explicit_type_matches

                    promoted = promote_explicit_type_matches(
                        rewritten,
                        response.recommendations,
                        getattr(getattr(self.context.service, "_metadata", None), "records", {}) or {},
                    )
                    if promoted != response.recommendations:
                        response = response.model_copy(
                            update={
                                "recommendations": tuple(
                                    item.model_copy(update={"rank": position})
                                    for position, item in enumerate(promoted, start=1)
                                )
                            }
                        )
                except Exception:  # noqa: BLE001 - post-check failure must remain visible
                    pass
                final_ids = tuple(item.parent_asin for item in response.recommendations)
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=rewritten,
                        response_status=response.terminal.status,
                        final_ids=final_ids,
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=in_tokens + out_tokens,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - benchmark records failures
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=turn.user_input,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                    )
                )
        return TaskRun(
            task_id=task.task_id,
            policy=self.policy_id,
            runnable=True,
            status="completed" if trace and trace[-1].error is None else "error",
            turns=trace,
            final_ids=final_ids,
            wall_ms=(time.perf_counter() - start) * 1000,
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_in + total_out,
        )


class RuleMemoryPlannerAdapter(_BaseAdapter):
    """C: existing M5 Memory + R1 Retrieval Planner + M7 grounded interaction."""

    policy_id = PolicyId.C

    def run(self, task: PolicyTask, *, state_dir: str | None = None) -> TaskRun:
        start = time.perf_counter()
        session_id = f"bench:{self.policy_id.value}:{task.task_id}:{uuid.uuid4().hex[:10]}"
        task_id = session_id
        trace: list[TurnTrace] = []
        final_ids: tuple[str, ...] = ()
        active_attributes: tuple[str, ...] = ()
        actions: list[str] = []
        pending: Any = None
        last_event_sequence = 0
        for index, turn in enumerate(task.turns):
            turn_start = time.perf_counter()
            option_id = _resolve_option(pending, turn.user_input) if pending is not None else None
            pending_clarification = (
                getattr(pending.clarification, "clarification_id", None)
                if pending is not None and getattr(pending, "clarification", None) is not None
                else None
            )
            # If the user ignores an Agent question and starts a new goal, treat the new
            # message as a fresh turn rather than feeding it as an invalid answer.  A
            # matching structured option is still used when the utterance is an answer.
            if option_id is None:
                pending_clarification = None
            try:
                payload = AgentRecommendRequest(
                    message=turn.user_input,
                    history=list(task.history),
                    k=_K,
                    sources=SOURCES,
                    preference_discovery=bool(task.preference_discovery),
                    clarification_id=pending_clarification,
                    clarification_option=option_id,
                )
                response = self.context.service.run_shopping_turn(
                    payload,
                    session_id=session_id,
                    task_id=task_id,
                    turn_id=f"{session_id}:{index:06d}",
                )
                pending = response
                final_ids = tuple(item.parent_asin for item in response.recommendations)
                grounded = getattr(response, "grounded", None)
                candidate_pool_ids = tuple(
                    str(getattr(item, "parent_asin", ""))
                    for item in (getattr(grounded, "candidates", ()) or ())
                    if getattr(item, "parent_asin", None)
                )
                diagnostic = self.context.service.r1_diagnostics(task_id) or {}
                action = _action_from_diagnostics(diagnostic)
                if action and action not in actions:
                    actions.append(action)
                preferences = tuple(getattr(self.context.service, "_task_preferences", {}).get(task_id, ()))
                active_attributes = _active_attributes(self.context.service, task_id, session_id)
                state = self.context.service.state_diagnostics(task_id) or {}
                events, last_event_sequence = _state_events(
                    self.context.state_db_path, task_id, after_sequence=last_event_sequence
                )
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=turn.user_input,
                        response_status=response.terminal.status,
                        clarification_id=(
                            getattr(response.clarification, "clarification_id", None)
                            if response.clarification is not None
                            else None
                        ),
                        clarification_question=(
                            getattr(response.clarification, "question", None)
                            if response.clarification is not None
                            else None
                        ),
                        clarification_options=tuple(
                            str(getattr(option, "label", ""))
                            for option in (
                                getattr(
                                    getattr(response.clarification, "preference", None),
                                    "options",
                                    (),
                                )
                                if response.clarification is not None
                                else ()
                            )
                            or ()
                        ),
                        selected_option_id=option_id,
                        candidate_pool_ids=candidate_pool_ids,
                        final_ids=final_ids,
                        retrieval_actions=(action,) if action else (),
                        memory_events=(
                            {
                                "state_digest": state.get("state_digest"),
                                "preference_count": state.get("preference_count"),
                                "soft_hypothesis_count": state.get("soft_hypothesis_count"),
                                "hard_constraint_count": state.get("hard_constraint_count"),
                                "active_attributes": list(active_attributes),
                                "r1_reason": diagnostic.get("reason"),
                                "r1_query_terms": diagnostic.get("query_terms"),
                                "r1_sources": diagnostic.get("sources"),
                                "r1_constraints": diagnostic.get("constraints"),
                                "m5_soft_preferences": diagnostic.get("m5_soft_preferences"),
                                "r1_before": diagnostic.get("before"),
                                "r1_after": diagnostic.get("after"),
                                "role_diagnostics": self.context.service.role_diagnostics(task_id),
                                "soft_rerank": self.context.service.soft_m5_diagnostics(task_id),
                                "events": events,
                            },
                        ),
                        tool_calls=_tool_call_labels(response),
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=turn.user_input,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                    )
                )
                break
        return TaskRun(
            task_id=task.task_id,
            policy=self.policy_id,
            runnable=True,
            status="completed" if trace and trace[-1].error is None else "error",
            turns=trace,
            final_ids=final_ids,
            active_preference_attributes=active_attributes,
            retrieval_actions=tuple(actions),
            wall_ms=(time.perf_counter() - start) * 1000,
        )


class LLMPreferenceAdapter(RuleMemoryPlannerAdapter):
    """D: C plus a validated LLM structured preference proposal on each user turn."""

    policy_id = PolicyId.D
    requires_llm = True

    def __init__(self, context: AdapterContext, *, proposer: PreferenceProposer | None = None) -> None:
        super().__init__(context)
        self.proposer = proposer or (
            context.proposal_client
            if callable(getattr(context.proposal_client, "propose", None))
            else (LLMPreferenceProposer(context.proposal_client) if context.proposal_client else None)
        )

    def availability(self) -> str | None:
        if self.proposer is None:
            return "llm_not_configured"
        return None

    def run(self, task: PolicyTask, *, state_dir: str | None = None) -> TaskRun:
        reason = self.availability()
        if reason:
            return TaskRun(task.task_id, self.policy_id, False, PolicyStatus.UNAVAILABLE.value, unavailable_reason=reason)
        # First version: the validated LLM proposal is used only as a faithful rewrite
        # input to the same C pipeline.  It cannot inject products or evaluation facts.
        assert self.proposer is not None
        start = time.perf_counter()
        session_id = f"bench:{self.policy_id.value}:{task.task_id}:{uuid.uuid4().hex[:10]}"
        task_id = session_id
        trace: list[TurnTrace] = []
        final_ids: tuple[str, ...] = ()
        active_attributes: tuple[str, ...] = ()
        actions: list[str] = []
        pending: Any = None
        messages: list[str] = []
        total_in = total_out = 0
        last_event_sequence = 0
        for index, turn in enumerate(task.turns):
            messages.append(turn.user_input)
            turn_start = time.perf_counter()
            option_id = _resolve_option(pending, turn.user_input) if pending is not None else None
            pending_clarification = (
                getattr(pending.clarification, "clarification_id", None)
                if pending is not None and getattr(pending, "clarification", None) is not None
                else None
            )
            if option_id is None:
                pending_clarification = None
            try:
                rewritten, proposals, in_tokens, out_tokens = self.proposer.propose(
                    messages=messages, current=turn.user_input
                )
                proposal_payload = getattr(self.proposer, "last_payload", None)
                total_in += in_tokens
                total_out += out_tokens
                proposal_rows = tuple(dict(item) for item in proposals)
                payload = AgentRecommendRequest(
                    message=rewritten,
                    history=list(task.history),
                    k=_K,
                    sources=SOURCES,
                    preference_discovery=bool(task.preference_discovery),
                    clarification_id=pending_clarification,
                    clarification_option=option_id,
                )
                response = self.context.service.run_shopping_turn(
                    payload,
                    session_id=session_id,
                    task_id=task_id,
                    turn_id=f"{session_id}:{index:06d}",
                    proposal_operations=proposal_payload,
                )
                pending = response
                final_ids = tuple(item.parent_asin for item in response.recommendations)
                grounded = getattr(response, "grounded", None)
                candidate_pool_ids = tuple(
                    str(getattr(item, "parent_asin", ""))
                    for item in (getattr(grounded, "candidates", ()) or ())
                    if getattr(item, "parent_asin", None)
                )
                diagnostic = self.context.service.r1_diagnostics(task_id) or {}
                action = _action_from_diagnostics(diagnostic)
                if action and action not in actions:
                    actions.append(action)
                preferences = tuple(getattr(self.context.service, "_task_preferences", {}).get(task_id, ()))
                active_attributes = _active_attributes(self.context.service, task_id, session_id)
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=rewritten,
                        response_status=response.terminal.status,
                        clarification_id=(
                            getattr(response.clarification, "clarification_id", None)
                            if response.clarification is not None
                            else None
                        ),
                        clarification_question=(
                            getattr(response.clarification, "question", None)
                            if response.clarification is not None
                            else None
                        ),
                        clarification_options=tuple(
                            str(getattr(option, "label", ""))
                            for option in (
                                getattr(
                                    getattr(response.clarification, "preference", None),
                                    "options",
                                    (),
                                )
                                if response.clarification is not None
                                else ()
                            )
                            or ()
                        ),
                        selected_option_id=option_id,
                        candidate_pool_ids=candidate_pool_ids,
                        final_ids=final_ids,
                        retrieval_actions=(action,) if action else (),
                        llm_proposals=proposal_rows,
                        memory_events=(
                            {"proposal_bridge": self.context.service.proposal_diagnostics(task_id)},
                        ) if proposal_payload is not None else (),
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=in_tokens + out_tokens,
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                trace.append(
                    TurnTrace(
                        turn_index=index,
                        user_input=turn.user_input,
                        effective_input=turn.user_input,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - turn_start) * 1000,
                    )
                )
                break
        return TaskRun(
            task_id=task.task_id,
            policy=self.policy_id,
            runnable=True,
            status="completed" if trace and trace[-1].error is None else "error",
            turns=trace,
            final_ids=final_ids,
            active_preference_attributes=active_attributes,
            retrieval_actions=tuple(actions),
            wall_ms=(time.perf_counter() - start) * 1000,
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_in + total_out,
        )


def build_adapter(
    policy: PolicyId,
    *,
    service: Any,
    rewrite_client: Any = None,
    proposal_client: Any = None,
    state_db_path: str | None = None,
) -> _BaseAdapter:
    """Build one adapter without calling any provider."""
    context = AdapterContext(
        service=service,
        rewrite_client=rewrite_client,
        proposal_client=proposal_client,
        state_db_path=state_db_path,
    )
    if policy is PolicyId.A:
        return QueryRewriteAdapter(context, policy=PolicyId.A)
    if policy is PolicyId.B:
        return QueryRewriteAdapter(context, policy=PolicyId.B)
    if policy is PolicyId.C:
        return RuleMemoryPlannerAdapter(context)
    if policy is PolicyId.D:
        return LLMPreferenceAdapter(context)
    raise ValueError(f"unknown policy {policy!r}")
