"""P16 preference-clarification skill: ask one answerable question, then persist the real reply.

Uses the existing Planner/Gate question when available, and the existing M9
PreferenceMemoryService.process_turn for updates/retractions.  It never invents a user reply and
never writes a synthetic answer from evaluator data.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from experiments.p14b1.diagnose import conflicts
from recommendation.preference.history import HistoryStateReader

CLARIFICATION_SKILL = "agentrecx.p16.preference_clarification.v1"


def _question_from_gate(gate: Mapping[str, Any] | None) -> str | None:
    if not gate:
        return None
    question = gate.get("ask_question")
    if gate.get("action") in {"NEED_USER_CLARIFICATION", "ASK"} and question:
        return str(question)
    return None


def ask_clarification(
    *,
    task: Mapping[str, Any],
    preferences: list[dict[str, Any]],
    gate: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return an awaiting-clarification decision; never a product recommendation."""
    query = str(task.get("current_query") or "")
    question = _question_from_gate(gate)
    reason = "gate_need_user_clarification"
    if question is None:
        conflict_list = conflicts(preferences)
        if conflict_list:
            reason = "unresolved_preference_conflict"
            pairs = [f"{item.get('left')} vs {item.get('right')}" for item in conflict_list]
            question = ("Your history contains conflicting preferences: " + "; ".join(pairs)
                        + ". Which one should take priority for this purchase?")
        elif not preferences:
            reason = "missing_key_need"
            question = ("What matters most for this purchase - for example the use case, a budget, "
                        "or a feature you need?")
        else:
            reason = "missing_decision_criteria"
            question = ("Which of your stated preferences matters most here, or is there a hard budget/"
                        "feature requirement I should use?")
    return {
        "schema": CLARIFICATION_SKILL,
        "skill": "preference-clarification",
        "status": "awaiting_clarification",
        "current_query": query,
        "user_key": str(task.get("task_id") or ""),
        "question": question,
        "reason": reason,
        "selected_product_id": None,
        "finalize_allowed": False,
        "final_answer": question,
        "requires_user_reply": True,
        "notes": ["no product recommendation is produced while clarification is pending",
                  "the reply must be a real user message; no evaluator answer is read"],
    }


def resume_with_user_reply(
    *,
    memory: Any,
    task: Mapping[str, Any],
    user_message: str,
    turn_id: str,
    now: float | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Persist a real user reply through the accepted M9 user-event writer and re-read state."""
    if not str(user_message or "").strip():
        raise ValueError("a real non-empty user reply is required; the skill never fabricates one")
    if not str(turn_id or "").strip():
        raise ValueError("turn_id is required for duplicate-reply idempotency")
    user_key = str(task.get("task_id") or "")
    started = time.perf_counter()
    result = memory.preference_service.process_turn(
        user_key=user_key, user_message=str(user_message), turn_id=str(turn_id),
        now=now,
    )
    history = HistoryStateReader(memory.preference_service).read(
        user_key, conversation_id=user_key,
        current_query=str(task.get("current_query") or ""),
    )
    update = result.update
    return {
        "schema": CLARIFICATION_SKILL,
        "skill": "preference-clarification",
        "status": "memory_updated",
        "user_key": user_key,
        "user_message": str(user_message),
        "turn_id": str(turn_id),
        "already_processed": bool(update.already_processed),
        "update": {
            "added": len(update.added),
            "superseded": len(update.superseded),
            "removed": len(update.removed),
            "skipped_duplicates": int(update.skipped_duplicates),
            "removal_directives": int(update.removal_directives),
            "already_processed": bool(update.already_processed),
        },
        "active_preferences": history.get("active_preferences") or [],
        "inactive_preferences": history.get("inactive_preferences") or [],
        "history_status": history.get("status"),
        "audit_source": "M9 PreferenceMemoryService.process_turn (same accepted writer); "
                        "inactive/removed entries retained in get_memory_history",
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "notes": ["this is an actual user-authored reply submitted by the caller; no reply is synthesized"],
    }


def read_authorized_history(*, memory: Any, user_key: str, conversation_id: str,
                            current_query: str = "", **_: Any) -> dict[str, Any]:
    """Read only the authorised user's history; cross-user conversation is rejected."""
    return HistoryStateReader(memory.preference_service).read(
        user_key, conversation_id=conversation_id, current_query=current_query,
    )
