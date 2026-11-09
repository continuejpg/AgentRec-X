"""P14-B0 legal history-state read path.

The accepted M9 query adapter is lexical: it only returns preferences whose value
tokens overlap the query.  That is correct for a *targeted* lookup, but it makes a
current shopping request lose historical preferences that the user already told the
agent.  This module reads the **already-applied M9 state** for the authorised user
(query-independent), keeps the audit trail, and only then applies a deterministic,
budget-bounded relevance ordering.

It is not a second memory store: reads go through the accepted
``PreferenceMemoryService``; writes remain the service's job.  It never accepts
evaluator-only data and never infers preferences from the current query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from recommendation.memory.schemas import PreferenceMemoryEntry, PreferenceStatus

__all__ = [
    "HistoryReadConfig",
    "HistoryStateReader",
    "HISTORY_READ_SCHEMA",
]

HISTORY_READ_SCHEMA = "agentrecx.p14b0.history_state_read.v1"
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(str(text or "").lower()))


def _entry_payload(entry: PreferenceMemoryEntry) -> dict[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "kind": entry.kind.value,
        "value": entry.value,
        "polarity": entry.polarity.value,
        "status": entry.status.value,
        "logical_seq": int(entry.logical_seq),
        "source_turn_id": entry.source_turn_id,
        "source_text": entry.source_text,
        "created_at": float(entry.created_at),
    }


@dataclass(frozen=True)
class HistoryReadConfig:
    """Bounds and authorisation rules for one read."""

    max_preferences: int = 10
    require_same_user: bool = True
    include_inactive: bool = True

    def __post_init__(self) -> None:
        if self.max_preferences < 1:
            raise ValueError("max_preferences must be >= 1")


class HistoryStateReader:
    """Read the current user's M9 state with provenance (query-independent)."""

    def __init__(self, preference_service: Any, config: HistoryReadConfig | None = None) -> None:
        if preference_service is None or not hasattr(preference_service, "get_active_preferences"):
            raise TypeError("history reader needs the accepted M9 preference service")
        self._service = preference_service
        self._config = config or HistoryReadConfig()

    @property
    def service(self) -> Any:
        return self._service

    def read(
        self,
        user_key: str,
        *,
        conversation_id: str | None = None,
        current_query: str = "",
    ) -> dict[str, Any]:
        """Return active preferences + audit trail for ``user_key``.

        ``user_key`` is the only identity this method will read.  When
        ``conversation_id`` is supplied it must be identical: this is what prevents a
        benchmark run from reading another persona's history.
        """
        user = str(user_key or "").strip()
        if not user:
            raise ValueError("user_key must be non-empty")
        if conversation_id is not None and str(conversation_id).strip() != user:
            raise PermissionError("history read refused: conversation_id is not the authorised user")
        if self._config.require_same_user and conversation_id is None:
            # still safe: user_key is the authorised key; no other identity is read
            pass

        active_snapshot = self._service.get_active_preferences(user)
        active = [
            entry for entry in active_snapshot.entries
            if entry.status is PreferenceStatus.ACTIVE
        ]
        history_entries = self._service.get_memory_history(user).entries
        inactive = [
            entry for entry in history_entries
            if entry.status in (PreferenceStatus.REMOVED, PreferenceStatus.SUPERSEDED)
        ] if self._config.include_inactive else []

        query_tokens = _tokens(current_query)
        ranked: list[tuple[int, int, PreferenceMemoryEntry]] = []
        for entry in active:
            overlap = len(query_tokens & _tokens(entry.value))
            # Higher overlap first, newer logical_seq first as the deterministic fallback.
            ranked.append((-overlap, -int(entry.logical_seq), entry))
        ranked.sort(key=lambda item: (item[0], item[1], item[2].memory_id))
        selected = [entry for _overlap, _seq, entry in ranked[: self._config.max_preferences]]

        payload = [_entry_payload(entry) for entry in selected]
        for item, (_overlap, _seq, entry) in zip(payload, ranked[: len(payload)]):
            item["query_token_overlap"] = -_overlap
        inactive_payload = [_entry_payload(entry) for entry in inactive]
        relevant = any(item["query_token_overlap"] > 0 for item in payload)

        return {
            "schema": HISTORY_READ_SCHEMA,
            "user_key": user,
            "available": bool(payload or inactive_payload),
            "status": "active_preferences" if payload else (
                "history_without_active_preferences" if inactive_payload else "no_history"
            ),
            "active_preferences": payload,
            "inactive_preferences": inactive_payload,
            "query_relevant_preferences": sum(1 for item in payload if item["query_token_overlap"] > 0),
            "query_filter_used": bool(query_tokens),
            "relevant": relevant,
            "read_policy": {
                "max_preferences": self._config.max_preferences,
                "ordering": "query_token_overlap DESC, logical_seq DESC, memory_id ASC",
                "source": "accepted M9 PreferenceMemoryService.get_active_preferences/get_memory_history",
                "evaluator_labels_read": False,
            },
            "provenance": [
                {
                    "memory_id": item["memory_id"],
                    "source_turn_id": item["source_turn_id"],
                    "source_text": item["source_text"],
                    "logical_seq": item["logical_seq"],
                }
                for item in payload
            ],
        }
