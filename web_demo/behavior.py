"""Behavior Feedback V1: first-party event tracking, short-term profile and shadow ranking.

Design boundaries
-----------------
* The browser posts **structured, whitelisted** events to this local BFF.  There is no
  third-party analytics SDK and no free text in the event payload.
* Only ``LIVE_USER_INTERACTION`` events enter the behavior profile.  Recorded replay and
  synthetic-test events are stored with their provenance but are excluded from every
  profile signal and every shadow score.
* Impressions are defined by *browser visibility* (IntersectionObserver in the Web Demo),
  never by the backend's Top-K list.  The BFF validates that the product and displayed
  position belong to the recommendation it issued to that same session.
* An explicit item Like/Dislike is **not** an M9 preference.  It is a weak short-term
  signal; only a separate user confirmation through the existing M9 writer can make a
  long-term preference.
* Behavior feedback is an opt-in shadow path.  It never changes the production default
  ``PreferenceAwareScorer`` order or the Decision Gate result.

Storage is a small SQLite database (stdlib only) under ``runs/web_demo/behavior`` by
default.  It is deliberately not Kafka/Flink/Redis: the demo is local and single-host.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Versions and fixed aggregation parameters (declared before any evaluation).
# ---------------------------------------------------------------------------

BEHAVIOR_EVENT_SCHEMA = "agentrecx.behavior.event.v1"
BEHAVIOR_PROFILE_VERSION = "behavior_feedback_profile_v1"
BEHAVIOR_SHADOW_VERSION = "behavior_shadow_v1"
EXISTING_RANKER_VERSION = "existing_ranking_v1"

EVENT_TYPES: tuple[str, ...] = (
    "product_impression",
    "product_click",
    "product_compare",
    "explicit_item_feedback",
)
EVENT_SOURCES: tuple[str, ...] = (
    "LIVE_USER_INTERACTION",
    "RECORDED_REPLAY",
    "SYNTHETIC_TEST",
)
LIVE_SOURCE = "LIVE_USER_INTERACTION"

#: Browser visibility contract: the card must stay at least 60% visible for 1000 ms.
IMPRESSION_MIN_VISIBLE_MS = 1000
IMPRESSION_VISIBILITY_THRESHOLD = 0.6

DEFAULT_WINDOW_S = 6 * 60 * 60
DEFAULT_HALF_LIFE_S = 30 * 60
DEFAULT_RETENTION_DAYS = 7
MAX_RECOMMENDATION_CONTEXTS = 30
MAX_BATCH_EVENTS = 50

#: Fixed event weights.  These are intentionally simple and are not tuned on P20/P100.
EVENT_WEIGHTS: dict[str, float] = {
    "product_impression": 0.15,
    "product_click": 1.0,
    "product_compare": 1.0,
    "explicit_like": 2.0,
    "explicit_dislike": -2.0,
}
#: Repeated events for one product are capped so a click loop cannot dominate a profile.
EVENT_COUNT_CAPS: dict[str, int] = {
    "product_impression": 1,
    "product_click": 2,
    "product_compare": 2,
    "product_feedback": 1,
}
PRODUCT_SCORE_CAP = 3.0
ATTRIBUTE_SCORE_CAP = 3.0

#: Shadow policy: the behavior component is bounded to this fraction of the base score.
SHADOW_BEHAVIOR_WEIGHT = 0.15
SHADOW_EXPLICIT_PREFERENCE_SCALE = 0.75
SHADOW_PRODUCT_SIGNAL_WEIGHT = 0.75
SHADOW_ATTRIBUTE_SIGNAL_WEIGHT = 0.25
#: An attribute present on most of the fixed pool says almost nothing about *this* item.
#: Attributes covering more than this share of the candidate pool are ignored by the shadow.
SHADOW_MAX_ATTRIBUTE_COVERAGE = 0.6

MAX_ATTRIBUTE_INTERESTS = 8
MAX_RECENT_ITEMS = 5
MAX_RECENT_EVENTS = 20


class BehaviorError(Exception):
    """A behavior API failure with a stable, browser-safe payload."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        http_status: int = 400,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status
        self.detail = detail

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def utc_iso(ts: float | None = None) -> str:
    from datetime import datetime, timezone

    value = time.time() if ts is None else ts
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())


def _humanize_attribute(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _decay(age_seconds: float, half_life_s: float) -> float:
    if half_life_s <= 0:
        return 1.0
    return float(0.5 ** (max(0.0, age_seconds) / half_life_s))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


# ---------------------------------------------------------------------------
# Recommendation snapshot: server-owned candidate/position/attribute truth
# ---------------------------------------------------------------------------


def build_recommendation_snapshot(
    payload: Mapping[str, Any],
    *,
    request_id: str | None = None,
    turn_id: str | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Freeze the exact list the browser is about to see.

    Only real catalogue facts already present in the BFF view model are copied.  The
    snapshot is the validation source for later events and the fixed candidate pool for
    the shadow ranking; the browser cannot supply candidates of its own.
    """
    created_ts = time.time() if now_ts is None else float(now_ts)
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("items") or (), start=1):
        if not isinstance(raw, Mapping):
            continue
        product_id = str(raw.get("product_id") or "").strip()
        if not product_id:
            continue
        verified: list[dict[str, Any]] = []
        for attribute in raw.get("attributes") or ():
            if not isinstance(attribute, Mapping):
                continue
            if attribute.get("status") != "present":
                continue
            # Verified means the P15 evidence span is attached.  A present catalogue flag
            # without an evidence id is not enough to create an attribute-level interest.
            evidence_id = str(attribute.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            verified.append(
                {
                    "attribute": str(attribute.get("attribute") or "").strip(),
                    "value": attribute.get("value"),
                    "evidence_id": evidence_id,
                    "span_text": attribute.get("span_text"),
                }
            )
        items.append(
            {
                "product_id": product_id,
                "position": int(raw.get("displayed_position") or index),
                "title": str(raw.get("title") or "") or None,
                "product_type": raw.get("product_type"),
                "form_factor": raw.get("form_factor"),
                "source": raw.get("source"),
                "match_status": raw.get("match_status"),
                "recommended": bool(raw.get("recommended")),
                "verified_attributes": verified,
            }
        )
    gate = payload.get("gate") or {}
    return {
        "recommendation_id": str(payload.get("recommendation_id") or ""),
        "request_id": request_id or payload.get("request_id"),
        "turn_id": turn_id or payload.get("turn_id"),
        "query": str(payload.get("query") or ""),
        "created_at": utc_iso(created_ts),
        "created_ts": created_ts,
        "ranker_version": str(payload.get("ranker_version") or EXISTING_RANKER_VERSION),
        "gate": {
            "action": gate.get("action") if isinstance(gate, Mapping) else None,
            "reason_code": gate.get("reason_code") if isinstance(gate, Mapping) else None,
            "selected_candidate_id": (
                gate.get("selected_candidate_id") if isinstance(gate, Mapping) else None
            ),
            "candidate_presentable_ids": (
                list(gate.get("candidate_presentable_ids") or ())
                if isinstance(gate, Mapping) else []
            ),
        },
        "items": items,
    }


# ---------------------------------------------------------------------------
# SQLite event store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS behavior_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    recommendation_id TEXT NOT NULL,
    request_id TEXT,
    turn_id TEXT,
    product_id TEXT NOT NULL,
    displayed_position INTEGER NOT NULL,
    ranker_version TEXT,
    occurred_at TEXT NOT NULL,
    occurred_ts REAL NOT NULL,
    received_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_behavior_session_time
    ON behavior_events(session_id, occurred_ts);
CREATE UNIQUE INDEX IF NOT EXISTS ux_behavior_impression
    ON behavior_events(session_id, recommendation_id, product_id)
    WHERE event_type = 'product_impression';
CREATE TABLE IF NOT EXISTS behavior_settings (
    session_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class BehaviorStore:
    """Process-local SQLite store for first-party behavior events.

    The BFF uses one store for all sessions; every query is scoped by the opaque
    server-issued session id, never by a browser-supplied user id.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        retention_days: float = DEFAULT_RETENTION_DAYS,
        clock: Any = None,
    ) -> None:
        self._path = str(path)
        self._retention_days = float(retention_days)
        self._clock = clock or time.time
        self._lock = threading.RLock()
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        if self._path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self.cleanup()

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def cleanup(self) -> int:
        """Delete events older than the retention window; returns deleted row count."""
        cutoff = float(self._clock()) - max(0.0, self._retention_days) * 86400.0
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM behavior_events WHERE occurred_ts < ?", (cutoff,)
            )
        return int(cursor.rowcount or 0)

    # -- settings -----------------------------------------------------------
    def get_enabled(self, session_id: str, default: bool = True) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT enabled FROM behavior_settings WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return bool(default)
        return bool(row["enabled"])

    def set_enabled(self, session_id: str, enabled: bool) -> None:
        with self._lock:
            self._connection.execute(
                """INSERT INTO behavior_settings(session_id, enabled, updated_at)
                   VALUES(?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET enabled=excluded.enabled,
                                                         updated_at=excluded.updated_at""",
                (session_id, 1 if enabled else 0, utc_iso(float(self._clock()))),
            )

    # -- ingest -------------------------------------------------------------
    def record(
        self,
        session_id: str,
        events: Sequence[Mapping[str, Any]],
        *,
        contexts: Mapping[str, Mapping[str, Any]],
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Validate and store one batch.  Per-event rejections do not abort the batch."""
        if not enabled:
            raise BehaviorError(
                "behavior personalization is disabled for this session",
                code="behavior_disabled",
                http_status=409,
            )
        accepted: list[dict[str, Any]] = []
        duplicates: list[str] = []
        rejected: list[dict[str, Any]] = []
        now = float(self._clock())
        with self._lock:
            for raw in events:
                event = dict(raw)
                event_id = str(event.get("event_id") or "").strip()
                try:
                    stored = self._validate_and_build(
                        session_id=session_id,
                        event=event,
                        contexts=contexts,
                        now=now,
                    )
                    if self._event_exists(event_id):
                        duplicates.append(event_id)
                        continue
                    if stored["event_type"] == "product_impression" and self._impression_exists(
                        session_id, stored["recommendation_id"], stored["product_id"]
                    ):
                        duplicates.append(event_id)
                        continue
                    self._insert(stored)
                except BehaviorError as exc:
                    rejected.append(
                        {
                            "event_id": event_id or None,
                            "code": exc.code,
                            "message": exc.message,
                            "detail": exc.detail,
                        }
                    )
                    continue
                accepted.append(
                    _row_view(
                        {
                            **stored,
                            "metadata": json.loads(stored["metadata_json"]),
                            "snapshot": json.loads(stored["snapshot_json"]),
                        }
                    )
                )
        self.cleanup()
        return {
            "session_id": session_id,
            "accepted": accepted,
            "duplicates": duplicates,
            "rejected": rejected,
            "accepted_count": len(accepted),
            "duplicate_count": len(duplicates),
            "rejected_count": len(rejected),
        }

    def _validate_and_build(
        self,
        *,
        session_id: str,
        event: Mapping[str, Any],
        contexts: Mapping[str, Mapping[str, Any]],
        now: float,
    ) -> dict[str, Any]:
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise BehaviorError("event_id is required", code="invalid_event_id")
        event_type = str(event.get("event_type") or "").strip()
        if event_type not in EVENT_TYPES:
            raise BehaviorError(
                f"unsupported event_type {event_type!r}", code="invalid_event_type"
            )
        event_source = str(event.get("event_source") or "").strip()
        if event_source not in EVENT_SOURCES:
            raise BehaviorError(
                f"unsupported event_source {event_source!r}", code="invalid_event_source"
            )
        recommendation_id = str(event.get("recommendation_id") or "").strip()
        context = contexts.get(recommendation_id)
        if not recommendation_id or context is None:
            raise BehaviorError(
                "recommendation_id is not owned by this session",
                code="unknown_recommendation",
            )
        product_id = str(event.get("product_id") or "").strip()
        items = {
            str(item.get("product_id")): item
            for item in context.get("items") or ()
            if isinstance(item, Mapping) and item.get("product_id")
        }
        item = items.get(product_id)
        if item is None:
            raise BehaviorError(
                "product_id is not present in the issued recommendation",
                code="unknown_product",
            )
        try:
            position = int(event.get("displayed_position"))
        except (TypeError, ValueError):
            raise BehaviorError(
                "displayed_position must be an integer", code="invalid_position"
            ) from None
        if position != int(item.get("position") or 0):
            raise BehaviorError(
                "displayed_position does not match the issued display position",
                code="invalid_position",
            )
        metadata = dict(event.get("metadata") or {})
        extra_keys = set(metadata) - {"feedback", "compared_product_ids"}
        if extra_keys:
            raise BehaviorError(
                "event metadata contains fields that are not whitelisted",
                code="invalid_metadata",
                detail=sorted(extra_keys),
            )
        stored_metadata: dict[str, Any] = {}
        if event_type == "product_compare":
            related = _dedupe(metadata.get("compared_product_ids") or ())
            if not 2 <= len(related) <= 5:
                raise BehaviorError(
                    "product_compare needs 2-5 compared_product_ids",
                    code="invalid_compare",
                )
            if product_id not in related:
                raise BehaviorError(
                    "the primary product must appear in compared_product_ids",
                    code="invalid_compare",
                )
            compared_items: list[dict[str, Any]] = []
            for related_id in related:
                related_item = items.get(related_id)
                if related_item is None:
                    raise BehaviorError(
                        f"compared product {related_id!r} is not in the issued recommendation",
                        code="unknown_product",
                    )
                compared_items.append(dict(related_item))
            stored_metadata = {
                "compared_product_ids": related,
                "compared_items": compared_items,
            }
        elif event_type == "explicit_item_feedback":
            feedback = str(metadata.get("feedback") or "").strip().lower()
            if feedback not in {"like", "dislike"}:
                raise BehaviorError(
                    "explicit_item_feedback needs feedback=like|dislike",
                    code="invalid_feedback",
                )
            stored_metadata = {"feedback": feedback}
        else:
            if metadata:
                raise BehaviorError(
                    f"{event_type} does not accept metadata", code="invalid_metadata"
                )

        occurred_ts = float(now)
        return {
            "event_id": event_id,
            "session_id": session_id,
            "event_type": event_type,
            "event_source": event_source,
            "recommendation_id": recommendation_id,
            "request_id": context.get("request_id"),
            "turn_id": context.get("turn_id"),
            "product_id": product_id,
            "displayed_position": position,
            "ranker_version": context.get("ranker_version") or EXISTING_RANKER_VERSION,
            "occurred_at": utc_iso(occurred_ts),
            "occurred_ts": occurred_ts,
            "received_at": utc_iso(occurred_ts),
            "metadata_json": json.dumps(stored_metadata, ensure_ascii=False, sort_keys=True),
            "snapshot_json": json.dumps(dict(item), ensure_ascii=False, sort_keys=True),
        }

    def _event_exists(self, event_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM behavior_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return row is not None

    def _impression_exists(
        self, session_id: str, recommendation_id: str, product_id: str
    ) -> bool:
        row = self._connection.execute(
            """SELECT 1 FROM behavior_events
               WHERE session_id = ? AND recommendation_id = ? AND product_id = ?
                 AND event_type = 'product_impression'""",
            (session_id, recommendation_id, product_id),
        ).fetchone()
        return row is not None

    def _insert(self, stored: Mapping[str, Any]) -> None:
        self._connection.execute(
            """INSERT INTO behavior_events(
                   event_id, session_id, event_type, event_source, recommendation_id,
                   request_id, turn_id, product_id, displayed_position, ranker_version,
                   occurred_at, occurred_ts, received_at, metadata_json, snapshot_json
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stored["event_id"],
                stored["session_id"],
                stored["event_type"],
                stored["event_source"],
                stored["recommendation_id"],
                stored["request_id"],
                stored["turn_id"],
                stored["product_id"],
                stored["displayed_position"],
                stored["ranker_version"],
                stored["occurred_at"],
                stored["occurred_ts"],
                stored["received_at"],
                stored["metadata_json"],
                stored["snapshot_json"],
            ),
        )

    # -- read ---------------------------------------------------------------
    def fetch_events(
        self,
        session_id: str,
        *,
        limit: int = 200,
        as_of: float | None = None,
        live_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if as_of is not None:
            clauses.append("occurred_ts <= ?")
            params.append(float(as_of))
        if live_only:
            clauses.append("event_source = ?")
            params.append(LIVE_SOURCE)
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT * FROM behavior_events
                    WHERE {' AND '.join(clauses)}
                    ORDER BY occurred_ts DESC, rowid DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def profile(
        self,
        session_id: str,
        *,
        enabled: bool = True,
        now: float | None = None,
        as_of: float | None = None,
        window_s: float = DEFAULT_WINDOW_S,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
    ) -> dict[str, Any]:
        now_ts = float(self._clock() if now is None else now)
        reference_ts = float(as_of) if as_of is not None else now_ts
        rows = self.fetch_events(
            session_id, limit=2000, as_of=reference_ts, live_only=False
        )
        return build_behavior_profile(
            rows,
            session_id=session_id,
            enabled=enabled,
            now_ts=now_ts,
            reference_ts=reference_ts,
            window_s=window_s,
            half_life_s=half_life_s,
        )

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM behavior_events WHERE session_id = ?", (session_id,)
            )
            self._connection.execute(
                "DELETE FROM behavior_settings WHERE session_id = ?", (session_id,)
            )
        return int(cursor.rowcount or 0)


# ---------------------------------------------------------------------------
# Profile aggregation (pure function over stored event rows)
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    payload["snapshot"] = json.loads(payload.pop("snapshot_json") or "{}")
    return payload


def _row_view(row: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = row.get("snapshot") or {}
    return {
        "event_id": row.get("event_id"),
        "event_type": row.get("event_type"),
        "event_source": row.get("event_source"),
        "recommendation_id": row.get("recommendation_id"),
        "request_id": row.get("request_id"),
        "turn_id": row.get("turn_id"),
        "product_id": row.get("product_id"),
        "product_title": snapshot.get("title"),
        "displayed_position": row.get("displayed_position"),
        "ranker_version": row.get("ranker_version"),
        "occurred_at": row.get("occurred_at"),
        "metadata": _public_metadata(row.get("metadata") or {}),
        "enters_profile": row.get("event_source") == LIVE_SOURCE,
    }


def _public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if metadata.get("feedback"):
        output["feedback"] = metadata["feedback"]
    if metadata.get("compared_product_ids"):
        output["compared_product_ids"] = list(metadata["compared_product_ids"])
    return output


def _event_weight(event_type: str, metadata: Mapping[str, Any]) -> float:
    if event_type == "explicit_item_feedback":
        return EVENT_WEIGHTS[
            "explicit_like" if metadata.get("feedback") == "like" else "explicit_dislike"
        ]
    return EVENT_WEIGHTS.get(event_type, 0.0)


def _product_event_cap_key(event_type: str, metadata: Mapping[str, Any]) -> str:
    if event_type == "explicit_item_feedback":
        return "product_feedback"
    return event_type


def _affected_items(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    snapshot = dict(event.get("snapshot") or {})
    if event.get("event_type") != "product_compare":
        return [snapshot] if snapshot.get("product_id") else []
    compared = (event.get("metadata") or {}).get("compared_items") or []
    if compared:
        return [dict(item) for item in compared if isinstance(item, Mapping)]
    return [snapshot] if snapshot.get("product_id") else []


def _verified_attributes(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        attribute
        for attribute in item.get("verified_attributes") or ()
        if isinstance(attribute, Mapping)
        and attribute.get("attribute")
        and attribute.get("evidence_id")
    ]


def build_behavior_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    enabled: bool,
    now_ts: float,
    reference_ts: float,
    window_s: float = DEFAULT_WINDOW_S,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
) -> dict[str, Any]:
    """Aggregate live events into a transparent, evidence-bearing short-term profile."""
    window_start = reference_ts - max(0.0, float(window_s))
    live = [
        row for row in rows
        if row.get("event_source") == LIVE_SOURCE
        and window_start <= float(row.get("occurred_ts") or 0.0) <= reference_ts
    ]
    source_counts = {
        "stored_total": len(rows),
        "live_user_interaction": 0,
        "recorded_replay": 0,
        "synthetic_test": 0,
    }
    for row in rows:
        source = str(row.get("event_source") or "")
        if source == LIVE_SOURCE:
            source_counts["live_user_interaction"] += 1
        elif source == "RECORDED_REPLAY":
            source_counts["recorded_replay"] += 1
        elif source == "SYNTHETIC_TEST":
            source_counts["synthetic_test"] += 1

    counts = {
        "live_events": len(live),
        "product_impression": 0,
        "product_click": 0,
        "product_compare": 0,
        "explicit_feedback": 0,
        "explicit_like": 0,
        "explicit_dislike": 0,
        "stored_total": len(rows),
        "recorded_replay": source_counts["recorded_replay"],
        "synthetic_test": source_counts["synthetic_test"],
    }
    for row in live:
        event_type = str(row.get("event_type") or "")
        if event_type in {"product_impression", "product_click", "product_compare"}:
            counts[event_type] += 1
        elif event_type == "explicit_item_feedback":
            counts["explicit_feedback"] += 1
            feedback = (row.get("metadata") or {}).get("feedback")
            if feedback == "like":
                counts["explicit_like"] += 1
            elif feedback == "dislike":
                counts["explicit_dislike"] += 1

    if not enabled:
        return {
            "schema": BEHAVIOR_PROFILE_VERSION,
            "session_id": session_id,
            "enabled": False,
            "generated_at": utc_iso(now_ts),
            "window_seconds": float(window_s),
            "half_life_seconds": float(half_life_s),
            "counts": {key: 0 for key in counts},
            "source_counts": source_counts,
            "recent_viewed": [],
            "recent_compared": [],
            "recent_explicit_feedback": [],
            "recent_categories": [],
            "attribute_interests": [],
            "product_interests": [],
            "recent_events": [],
            "notes": ["behavior personalization is switched off; no new events are collected"],
        }

    live.sort(key=lambda row: (float(row.get("occurred_ts") or 0.0), str(row.get("event_id"))))
    counted_per_product: dict[str, dict[str, int]] = {}
    product_scores: dict[str, float] = {}
    product_positive: dict[str, float] = {}
    product_negative: dict[str, float] = {}
    product_last: dict[str, str] = {}
    product_titles: dict[str, str] = {}
    product_evidence_count: dict[str, int] = {}
    attribute_scores: dict[str, float] = {}
    attribute_evidence: dict[str, dict[str, Any]] = {}
    category_scores: dict[str, float] = {}
    recent_viewed: list[dict[str, Any]] = []
    recent_compared: list[dict[str, Any]] = []
    recent_feedback: list[dict[str, Any]] = []
    seen_viewed: set[str] = set()

    for row in live:
        event_type = str(row.get("event_type") or "")
        metadata = dict(row.get("metadata") or {})
        occurred_ts = float(row.get("occurred_ts") or 0.0)
        age = max(0.0, reference_ts - occurred_ts)
        decay = _decay(age, half_life_s)
        weight = _event_weight(event_type, metadata) * decay
        affected = _affected_items(row)
        if not affected:
            continue

        if event_type == "product_compare":
            primary = str(row.get("product_id") or "")
            recent_compared.append(
                {
                    "event_id": row.get("event_id"),
                    "product_id": primary,
                    "product_title": (row.get("snapshot") or {}).get("title"),
                    "compared_product_ids": list(metadata.get("compared_product_ids") or ()),
                    "occurred_at": row.get("occurred_at"),
                }
            )
        if event_type == "explicit_item_feedback":
            recent_feedback.append(
                {
                    "event_id": row.get("event_id"),
                    "product_id": row.get("product_id"),
                    "product_title": (row.get("snapshot") or {}).get("title"),
                    "feedback": metadata.get("feedback"),
                    "occurred_at": row.get("occurred_at"),
                }
            )

        for item in affected:
            product_id = str(item.get("product_id") or "")
            if not product_id:
                continue
            counts_for_product = counted_per_product.setdefault(product_id, {})
            cap_key = _product_event_cap_key(event_type, metadata)
            if counts_for_product.get(cap_key, 0) >= EVENT_COUNT_CAPS.get(cap_key, 999):
                continue
            counts_for_product[cap_key] = counts_for_product.get(cap_key, 0) + 1
            product_titles[product_id] = str(item.get("title") or product_titles.get(product_id, ""))
            product_last[product_id] = str(row.get("occurred_at") or product_last.get(product_id, ""))
            product_evidence_count[product_id] = product_evidence_count.get(product_id, 0) + 1
            product_scores[product_id] = product_scores.get(product_id, 0.0) + weight
            if weight >= 0:
                product_positive[product_id] = product_positive.get(product_id, 0.0) + weight
            else:
                product_negative[product_id] = product_negative.get(product_id, 0.0) + weight

            if event_type == "product_click" and product_id not in seen_viewed:
                seen_viewed.add(product_id)
                recent_viewed.append(
                    {
                        "product_id": product_id,
                        "product_title": item.get("title"),
                        "occurred_at": row.get("occurred_at"),
                        "event_id": row.get("event_id"),
                    }
                )

            for attribute in _verified_attributes(item):
                name = str(attribute["attribute"])
                evidence_id = str(attribute["evidence_id"])
                attribute_scores[name] = attribute_scores.get(name, 0.0) + weight
                bucket = attribute_evidence.setdefault(
                    name,
                    {
                        "event_ids": [],
                        "product_ids": [],
                        "evidence_ids": [],
                        "occurred_at": [],
                        "labels": [],
                    },
                )
                bucket["event_ids"].append(str(row.get("event_id")))
                bucket["product_ids"].append(product_id)
                bucket["evidence_ids"].append(evidence_id)
                bucket["occurred_at"].append(str(row.get("occurred_at") or ""))
                bucket["labels"].append(_humanize_attribute(name))

            category = str(item.get("product_type") or "").strip()
            if category and weight > 0:
                category_scores[category] = category_scores.get(category, 0.0) + weight

    # Apply caps and produce explainable product rows.
    product_interests: list[dict[str, Any]] = []
    for product_id, raw_score in product_scores.items():
        bounded = _clamp(raw_score, -PRODUCT_SCORE_CAP, PRODUCT_SCORE_CAP)
        product_interests.append(
            {
                "product_id": product_id,
                "product_title": product_titles.get(product_id),
                "score": round(bounded, 4),
                "positive_score": round(min(PRODUCT_SCORE_CAP, product_positive.get(product_id, 0.0)), 4),
                "negative_score": round(max(-PRODUCT_SCORE_CAP, product_negative.get(product_id, 0.0)), 4),
                "counted_events": product_evidence_count.get(product_id, 0),
                "last_event_at": product_last.get(product_id),
            }
        )
    product_interests.sort(key=lambda item: (-abs(item["score"]), item["product_id"]))
    product_interests = product_interests[:MAX_RECENT_ITEMS]

    attribute_interests: list[dict[str, Any]] = []
    for name, raw_score in attribute_scores.items():
        bounded = _clamp(raw_score, -ATTRIBUTE_SCORE_CAP, ATTRIBUTE_SCORE_CAP)
        if bounded <= 0:
            continue
        bucket = attribute_evidence.get(name, {})
        evidence = [
            {
                "event_id": event_id,
                "product_id": product_id,
                "evidence_id": evidence_id,
                "occurred_at": occurred_at,
            }
            for event_id, product_id, evidence_id, occurred_at in zip(
                bucket.get("event_ids", []),
                bucket.get("product_ids", []),
                bucket.get("evidence_ids", []),
                bucket.get("occurred_at", []),
            )
        ]
        attribute_interests.append(
            {
                "attribute": name,
                "label": _humanize_attribute(name),
                "score": round(bounded, 4),
                "distinct_products": len(set(bucket.get("product_ids", []))),
                "evidence_count": len(set(bucket.get("event_ids", []))),
                "evidence": evidence[:MAX_RECENT_ITEMS],
            }
        )
    attribute_interests.sort(key=lambda item: (-item["score"], item["attribute"]))
    attribute_interests = attribute_interests[:MAX_ATTRIBUTE_INTERESTS]

    recent_categories = [
        {"category": category, "score": round(score, 4)}
        for category, score in sorted(category_scores.items(), key=lambda pair: (-pair[1], pair[0]))[:MAX_RECENT_ITEMS]
    ]
    recent_viewed = recent_viewed[:MAX_RECENT_ITEMS]
    recent_compared = recent_compared[:MAX_RECENT_ITEMS]
    recent_feedback = recent_feedback[:MAX_RECENT_ITEMS]

    recent_events = [
        _row_view(row)
        for row in rows
        if window_start <= float(row.get("occurred_ts") or 0.0) <= reference_ts
    ][:MAX_RECENT_EVENTS]

    return {
        "schema": BEHAVIOR_PROFILE_VERSION,
        "session_id": session_id,
        "enabled": True,
        "generated_at": utc_iso(now_ts),
        "as_of": utc_iso(reference_ts),
        "window_seconds": float(window_s),
        "half_life_seconds": float(half_life_s),
        "counts": counts,
        "source_counts": source_counts,
        "recent_viewed": recent_viewed,
        "recent_compared": recent_compared,
        "recent_explicit_feedback": recent_feedback,
        "recent_categories": recent_categories,
        "attribute_interests": attribute_interests,
        "product_interests": product_interests,
        "recent_events": recent_events,
        "notes": [
            "only LIVE_USER_INTERACTION events contribute to interests",
            "click/compare/like are weak short-term signals, never M9 preferences",
            "attribute interests require a verified P15 evidence span",
            "impressions are browser-visible cards, not the backend Top-K list",
        ],
    }


def build_suggestion(
    profile: Mapping[str, Any],
    *,
    active_preferences: Sequence[Mapping[str, Any]] = (),
    inactive_preferences: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Propose at most one long-term preference for *explicit user confirmation*.

    This function never writes memory.  The Web Demo shows the exact message and calls the
    existing M9 writer only after the user clicks "Remember".
    """
    if not profile.get("enabled"):
        return None
    interests = list(profile.get("attribute_interests") or ())
    active = [_norm(item.get("value")) for item in active_preferences if item.get("value")]
    inactive = [_norm(item.get("value")) for item in inactive_preferences if item.get("value")]
    for interest in interests:
        if int(interest.get("distinct_products") or 0) < 2:
            continue
        if int(interest.get("evidence_count") or 0) < 2:
            continue
        attribute = _norm(interest.get("attribute"))
        if not attribute:
            continue
        if any(attribute in value or value in attribute for value in active):
            return None
        if any(attribute in value or value in attribute for value in inactive):
            # A withdrawn preference must not be silently restored from old clicks.
            return None
        label = str(interest.get("label") or interest.get("attribute"))
        message = f"I prefer products with this verified attribute: {label}."
        suggestion_id = hashlib.sha1(
            f"{profile.get('session_id')}|{attribute}|{'|'.join(sorted(str(e) for e in (interest.get('evidence') or [])[:3]))}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        return {
            "suggestion_id": suggestion_id,
            "attribute": interest.get("attribute"),
            "label": label,
            "score": interest.get("score"),
            "distinct_products": interest.get("distinct_products"),
            "evidence_count": interest.get("evidence_count"),
            "evidence": list(interest.get("evidence") or [])[:5],
            "message": message,
            "inferred_from_behavior": True,
            "requires_confirmation": True,
            "note": (
                "This is a behavior inference, not an explicit preference. Nothing is written "
                "to M9 unless you confirm; a confirm uses the existing preference writer."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Behavior-aware shadow ranking (fixed candidate pool, fixed default ranking)
# ---------------------------------------------------------------------------


def _preference_polarity(value: Any) -> str:
    polarity = str(value or "positive").casefold()
    return "negative" if polarity in {"avoid", "negative", "dislike"} else "positive"


def build_shadow(
    *,
    snapshot: Mapping[str, Any],
    profile: Mapping[str, Any],
    active_preferences: Sequence[Mapping[str, Any]] = (),
    inactive_preferences: Sequence[Mapping[str, Any]] = (),
    enabled: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute OFF/ON rankings over the exact frozen recommendation snapshot.

    The function performs no retrieval and no catalogue access.  Both modes receive the
    same candidate identity, position, verified attributes, Gate and query.  Only the
    bounded behavior contribution can differ.
    """
    items = sorted(
        (dict(item) for item in snapshot.get("items") or () if isinstance(item, Mapping)),
        key=lambda item: int(item.get("position") or 0),
    )
    n = len(items)
    gate = dict(snapshot.get("gate") or {})
    presentable = {str(value) for value in gate.get("candidate_presentable_ids") or ()}
    product_signals = {
        str(item.get("product_id")): float(item.get("score") or 0.0)
        for item in profile.get("product_interests") or ()
        if item.get("product_id")
    }
    attribute_signals = {
        str(item.get("attribute")): float(item.get("score") or 0.0)
        for item in profile.get("attribute_interests") or ()
        if item.get("attribute")
    }
    active = [
        {
            "value": _norm(item.get("value")),
            "polarity": _preference_polarity(item.get("polarity")),
            "kind": item.get("kind"),
            "memory_id": item.get("memory_id"),
        }
        for item in active_preferences
        if item.get("value")
    ]
    inactive = [
        {
            "value": _norm(item.get("value")),
            "polarity": _preference_polarity(item.get("polarity")),
            "kind": item.get("kind"),
            "memory_id": item.get("memory_id"),
        }
        for item in inactive_preferences
        if item.get("value")
    ]
    negative_active = [entry["value"] for entry in active if entry["polarity"] == "negative"]
    positive_active = [entry["value"] for entry in active if entry["polarity"] == "positive"]
    inactive_values = [entry["value"] for entry in inactive]
    explicit_scale = SHADOW_EXPLICIT_PREFERENCE_SCALE if active else 1.0
    attribute_coverage: dict[str, int] = {}
    for item in items:
        for attribute in item.get("verified_attributes") or ():
            if isinstance(attribute, Mapping) and attribute.get("attribute"):
                name = str(attribute["attribute"])
                attribute_coverage[name] = attribute_coverage.get(name, 0) + 1
    pool_size = max(1, len(items))

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        product_id = str(item.get("product_id") or "")
        base_score = 1.0 if n <= 1 else round((n - index - 1) / (n - 1), 6)
        attributes = [
            str(attribute.get("attribute") or "")
            for attribute in item.get("verified_attributes") or ()
            if isinstance(attribute, Mapping) and attribute.get("attribute")
        ]
        attribute_norm = {_norm(attribute) for attribute in attributes}
        common_ignored = [
            {
                "attribute": name,
                "score": attribute_signals[name],
                "coverage": round(attribute_coverage.get(name, 0) / pool_size, 4),
            }
            for name in attributes
            if name in attribute_signals
            and attribute_coverage.get(name, 0) / pool_size > SHADOW_MAX_ATTRIBUTE_COVERAGE
        ]
        matched = [
            {
                "attribute": name,
                "score": attribute_signals[name],
                "evidence": [
                    dict(evidence)
                    for interest in profile.get("attribute_interests") or ()
                    if str(interest.get("attribute")) == name
                    for evidence in (interest.get("evidence") or ())[:3]
                ],
            }
            for name in attributes
            if name in attribute_signals
            and attribute_coverage.get(name, 0) / pool_size <= SHADOW_MAX_ATTRIBUTE_COVERAGE
        ]
        product_signal = _clamp(product_signals.get(product_id, 0.0) / PRODUCT_SCORE_CAP, -1.0, 1.0)
        attribute_signal = _clamp(
            sum(entry["score"] for entry in matched) / ATTRIBUTE_SCORE_CAP, 0.0, 1.0
        )
        raw_signal = (
            SHADOW_PRODUCT_SIGNAL_WEIGHT * product_signal
            + SHADOW_ATTRIBUTE_SIGNAL_WEIGHT * attribute_signal
        )
        suppression_reason: str | None = None
        if not enabled:
            suppression_reason = "behavior_disabled"
        elif not profile.get("enabled") or not profile.get("product_interests") and not profile.get("attribute_interests"):
            suppression_reason = "no_behavior_data"
        elif any(value in attribute_norm for value in negative_active):
            suppression_reason = "explicit_negative_preference"
        elif any(value in attribute_norm for value in inactive_values):
            suppression_reason = "withdrawn_preference_not_restored"
        elif any(value in attribute_norm for value in positive_active):
            # The existing explicit preference already owns this signal; weak behavior must
            # not double-count it.
            suppression_reason = "explicit_positive_preference_already_active"
        contribution = 0.0
        if suppression_reason is None and raw_signal:
            contribution = round(
                SHADOW_BEHAVIOR_WEIGHT * explicit_scale * _clamp(raw_signal, -1.0, 1.0), 6
            )
        final_score = round(base_score + contribution, 6)
        rows.append(
            {
                "product_id": product_id,
                "product_title": item.get("title"),
                "base_rank": index + 1,
                "base_score": base_score,
                "behavior_contribution": contribution,
                "final_shadow_score": final_score,
                "product_interest_signal": round(product_signal, 4),
                "attribute_interest_signal": round(attribute_signal, 4),
                "matched_attribute_interests": matched,
                "ignored_common_attributes": common_ignored,
                "explicit_preference_conflict": suppression_reason in {
                    "explicit_negative_preference",
                    "withdrawn_preference_not_restored",
                },
                "suppressed_reason": suppression_reason,
                "recommended": bool(item.get("recommended")),
                "gate_eligible": (not presentable) or product_id in presentable,
                "gate_selected": product_id == str(gate.get("selected_candidate_id") or ""),
            }
        )

    ordered = sorted(
        rows,
        key=lambda row: (-row["final_shadow_score"], row["base_rank"], row["product_id"]),
    )
    for rank, row in enumerate(ordered, start=1):
        row["shadow_rank"] = rank
    by_id = {row["product_id"]: row for row in rows}
    moved = [
        row for row in rows if row["base_rank"] != row["shadow_rank"]
    ]
    moved.sort(
        key=lambda row: (
            -abs(row["shadow_rank"] - row["base_rank"]),
            -abs(row["behavior_contribution"]),
            row["shadow_rank"],
        )
    )
    affected = [
        row for row in rows
        if row["base_rank"] == row["shadow_rank"] and row["behavior_contribution"] != 0
    ]
    affected.sort(key=lambda row: (-abs(row["behavior_contribution"]), row["base_rank"]))
    changes = [
        {
            "product_id": row["product_id"],
            "product_title": row["product_title"],
            "base_rank": row["base_rank"],
            "shadow_rank": row["shadow_rank"],
            "behavior_contribution": row["behavior_contribution"],
            "matched_attribute_interests": row["matched_attribute_interests"],
            "suppressed_reason": row["suppressed_reason"],
            "gate_eligible": row["gate_eligible"],
        }
        for row in moved + affected
    ]
    applied = any(row["behavior_contribution"] != 0 for row in rows)
    return {
        "schema": BEHAVIOR_SHADOW_VERSION,
        "recommendation_id": snapshot.get("recommendation_id"),
        "request_id": snapshot.get("request_id"),
        "query": snapshot.get("query"),
        "created_at": snapshot.get("created_at"),
        "generated_at": utc_iso(),
        "enabled": bool(enabled),
        "applied": bool(applied),
        "top_k": int(top_k),
        "base_ranker_version": snapshot.get("ranker_version"),
        "shadow_version": BEHAVIOR_SHADOW_VERSION,
        "behavior_weight": SHADOW_BEHAVIOR_WEIGHT,
        "explicit_preference_scale": explicit_scale,
        "gate": gate,
        "hard_constraint_violations": 0,
        "eligibility_unchanged": True,
        "base_ranking": [
            {
                "rank": item["base_rank"],
                "product_id": item["product_id"],
                "product_title": item["product_title"],
                "base_score": item["base_score"],
            }
            for item in sorted(rows, key=lambda row: row["base_rank"])[: max(1, int(top_k))]
        ],
        "behavior_ranking": [
            {
                "rank": row["shadow_rank"],
                "product_id": row["product_id"],
                "product_title": row["product_title"],
                "final_shadow_score": row["final_shadow_score"],
                "behavior_contribution": row["behavior_contribution"],
            }
            for row in ordered[: max(1, int(top_k))]
        ],
        "rows": rows,
        "changes": changes,
        "order_changed": any(row["base_rank"] != row["shadow_rank"] for row in rows),
        "active_preferences": [
            {"value": entry["value"], "polarity": entry["polarity"], "kind": entry["kind"]}
            for entry in active
        ],
        "inactive_preference_count": len(inactive),
        "notes": [
            "fixed query, catalog candidates, displayed positions, eligibility and Gate",
            "no retrieval, no scoring-model call and no M9 write",
            "behavior contribution is capped at the declared behavior_weight",
            "behavior never changes the production default ranking or the Gate action; the "
            "Gate-selected product and eligibility flags are copied verbatim",
            "the shadow preview re-orders the already-displayed result pool only; it cannot add "
            "a candidate or change what the Gate selected",
        ],
    }


__all__ = [
    "BEHAVIOR_EVENT_SCHEMA",
    "BEHAVIOR_PROFILE_VERSION",
    "BEHAVIOR_SHADOW_VERSION",
    "BehaviorError",
    "BehaviorStore",
    "DEFAULT_HALF_LIFE_S",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_WINDOW_S",
    "EVENT_SOURCES",
    "EVENT_TYPES",
    "EXISTING_RANKER_VERSION",
    "IMPRESSION_MIN_VISIBLE_MS",
    "IMPRESSION_VISIBILITY_THRESHOLD",
    "LIVE_SOURCE",
    "MAX_BATCH_EVENTS",
    "build_behavior_profile",
    "build_recommendation_snapshot",
    "build_shadow",
    "build_suggestion",
    "utc_iso",
]
