"""M8.1 — minimal persistent decision-choice store (offline demo experiment).

Only a user interaction through the demo choice route may be persisted as a *real user
choice*; the store has no target-label column and never derives a correctness label from
the selection.  Ground-truth targets remain an offline evaluator-only input.

The store is a single SQLite table keyed by ``(session_id, task_id)`` so a repeated submit
returns the existing record instead of appending a duplicate.  A different payload for the
same task is a conflict rather than a silent overwrite.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ENV_M8_CHOICE_COLLECTION",
    "ENV_M8_CHOICE_DB",
    "ChoiceCollectionDisabled",
    "CollectedChoice",
    "DecisionChoiceConflict",
    "DecisionChoiceStore",
    "DecisionChoiceStoreError",
    "choice_collection_enabled",
    "choice_database_path",
    "load_collected_trajectories",
]

ENV_M8_CHOICE_COLLECTION = "AGENTRECX_M8_CHOICE_COLLECTION"
ENV_M8_CHOICE_DB = "AGENTRECX_M8_CHOICE_DB"

_CREATE = """
CREATE TABLE IF NOT EXISTS decision_choices (
    choice_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    user_key TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    trajectory_digest TEXT NOT NULL,
    selected_parent_asin TEXT,
    no_suitable_item INTEGER NOT NULL DEFAULT 0,
    satisfaction INTEGER,
    confidence INTEGER,
    comment TEXT,
    idempotency_key TEXT,
    interaction_source TEXT NOT NULL,
    final_top_k_json TEXT NOT NULL,
    trajectory_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (session_id, task_id)
)
"""


class DecisionChoiceStoreError(RuntimeError):
    """The persistent decision-choice store failed or was misconfigured."""


class ChoiceCollectionDisabled(DecisionChoiceStoreError):
    """Choice collection is not enabled in this process."""


class DecisionChoiceConflict(DecisionChoiceStoreError):
    """The same session/task already has a different recorded choice."""


def choice_collection_enabled(value: str | None = None) -> bool:
    raw = value if value is not None else os.environ.get(ENV_M8_CHOICE_COLLECTION, "")
    return str(raw).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def choice_database_path() -> Path:
    configured = str(os.environ.get(ENV_M8_CHOICE_DB, "")).strip()
    if not configured:
        raise DecisionChoiceStoreError(
            f"{ENV_M8_CHOICE_DB} must name a SQLite file when choice collection is enabled"
        )
    return Path(configured)


@dataclass(frozen=True)
class CollectedChoice:
    """One persisted user decision.  It carries no target/correctness field."""

    choice_id: str
    session_id: str
    task_id: str
    user_key: str
    trajectory_id: str
    trajectory_digest: str
    final_top_k: tuple[str, ...]
    selected_parent_asin: str | None
    no_suitable_item: bool
    satisfaction: int | None
    confidence: int | None
    comment: str | None
    idempotency_key: str | None
    interaction_source: str
    created_at: float
    duplicate: bool = False

    @property
    def is_real_user_interaction(self) -> bool:
        """Only the demo HTTP route may claim a real user interaction."""
        return self.interaction_source == "demo_http"

    def as_dict(self) -> dict[str, Any]:
        return {
            "choice_id": self.choice_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "user_key": self.user_key,
            "trajectory_id": self.trajectory_id,
            "trajectory_digest": self.trajectory_digest,
            "final_top_k": list(self.final_top_k),
            "selected_parent_asin": self.selected_parent_asin,
            "no_suitable_item": self.no_suitable_item,
            "satisfaction": self.satisfaction,
            "confidence": self.confidence,
            "comment": self.comment,
            "idempotency_key": self.idempotency_key,
            "interaction_source": self.interaction_source,
            "created_at": self.created_at,
            "duplicate": self.duplicate,
        }


def _row_to_choice(row: sqlite3.Row, *, duplicate: bool = False) -> CollectedChoice:
    return CollectedChoice(
        choice_id=str(row["choice_id"]),
        session_id=str(row["session_id"]),
        task_id=str(row["task_id"]),
        user_key=str(row["user_key"]),
        trajectory_id=str(row["trajectory_id"]),
        trajectory_digest=str(row["trajectory_digest"]),
        final_top_k=tuple(json.loads(row["final_top_k_json"] or "[]")),
        selected_parent_asin=row["selected_parent_asin"],
        no_suitable_item=bool(row["no_suitable_item"]),
        satisfaction=row["satisfaction"],
        confidence=row["confidence"],
        comment=row["comment"],
        idempotency_key=row["idempotency_key"],
        interaction_source=str(row["interaction_source"]),
        created_at=float(row["created_at"]),
        duplicate=duplicate,
    )


class DecisionChoiceStore:
    """Small SQLite store for real user choices; no target labels, no scoring."""

    def __init__(self, database: str | Path) -> None:
        self._database = Path(database)
        self._database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(_CREATE)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._database), timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def record(
        self,
        *,
        session_id: str,
        task_id: str,
        user_key: str,
        trajectory: Mapping[str, Any],
        selected_parent_asin: str | None,
        no_suitable_item: bool = False,
        satisfaction: int | None = None,
        confidence: int | None = None,
        comment: str | None = None,
        idempotency_key: str | None = None,
        interaction_source: str = "demo_http",
    ) -> CollectedChoice:
        session_id = str(session_id)
        task_id = str(task_id)
        user_key = str(user_key)
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        trajectory_digest = str(trajectory.get("digest") or "")
        if not trajectory_digest:
            trajectory_digest = hashlib.sha256(
                json.dumps(
                    trajectory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()
        final_top_k = tuple(str(item) for item in trajectory.get("final_top_k") or ())
        if selected_parent_asin is not None and str(selected_parent_asin) not in final_top_k:
            raise DecisionChoiceStoreError(
                "the selected candidate is not in the recorded final top-K"
            )
        if no_suitable_item and selected_parent_asin is not None:
            raise DecisionChoiceStoreError(
                "no-suitable-item and a selected candidate are mutually exclusive"
            )
        if not no_suitable_item and selected_parent_asin is None:
            raise DecisionChoiceStoreError(
                "a selected candidate or no_suitable_item=True is required"
            )
        payload = {
            "session": session_id,
            "task": task_id,
            "user": user_key,
            "selected": selected_parent_asin,
            "no_suitable": bool(no_suitable_item),
            "satisfaction": satisfaction,
            "confidence": confidence,
            "comment": comment,
            "source": interaction_source,
        }
        choice_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:32]
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM decision_choices WHERE session_id=? AND task_id=?",
                (session_id, task_id),
            ).fetchone()
            if existing is not None:
                record = _row_to_choice(existing, duplicate=True)
                if (
                    record.selected_parent_asin == selected_parent_asin
                    and record.no_suitable_item == bool(no_suitable_item)
                    and record.satisfaction == satisfaction
                    and record.confidence == confidence
                    and record.comment == comment
                    and record.interaction_source == interaction_source
                ):
                    return record
                raise DecisionChoiceConflict(
                    "this task already has a different recorded choice"
                )
            connection.execute(
                "INSERT INTO decision_choices ("
                "choice_id, session_id, task_id, user_key, trajectory_id, "
                "trajectory_digest, selected_parent_asin, no_suitable_item, satisfaction, "
                "confidence, comment, idempotency_key, interaction_source, "
                "final_top_k_json, trajectory_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    choice_id,
                    session_id,
                    task_id,
                    user_key,
                    trajectory_id,
                    trajectory_digest,
                    selected_parent_asin,
                    int(bool(no_suitable_item)),
                    satisfaction,
                    confidence,
                    comment,
                    idempotency_key,
                    interaction_source,
                    json.dumps(list(final_top_k)),
                    json.dumps(trajectory, sort_keys=True),
                    time.time(),
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM decision_choices WHERE choice_id=?", (choice_id,)
            ).fetchone()
            assert row is not None
            return _row_to_choice(row)

    def choices_for_session(self, session_id: str) -> tuple[CollectedChoice, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM decision_choices WHERE session_id=? ORDER BY created_at",
                (str(session_id),),
            ).fetchall()
        return tuple(_row_to_choice(row) for row in rows)

    def real_choices(self) -> tuple[CollectedChoice, ...]:
        """Return persisted choices that came from the actual demo HTTP interaction."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM decision_choices WHERE interaction_source='demo_http' "
                "ORDER BY created_at"
            ).fetchall()
        return tuple(_row_to_choice(row) for row in rows)

    def export_jsonl(self, destination: str | Path) -> int:
        """Export every choice to JSONL for offline evaluation; returns row count."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM decision_choices ORDER BY created_at"
            ).fetchall()
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    payload = _row_to_choice(row).as_dict()
                    payload["trajectory"] = json.loads(row["trajectory_json"])
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return len(rows)

    def close(self) -> None:
        """Idempotent; connections are per-operation, so no connection is retained."""
        return None


def load_collected_trajectories(
    database: str | Path,
) -> tuple[Any, ...]:
    """Load persisted real choices as M8 decision trajectories.

    The trajectory is the recorded one; its ``choice`` is replaced by the persisted real
    user choice.  Target labels are *not* stored or implied here; an evaluator must supply
    them separately or Team Accuracy remains NOT_EXECUTED.
    """
    from recommendation.evaluation.decision import DecisionTrajectory, UserChoice
    from dataclasses import replace

    store = DecisionChoiceStore(database)
    trajectories: list[Any] = []
    try:
        with store._lock, store._connect() as connection:  # noqa: SLF001 - same module contract
            rows = connection.execute(
                "SELECT * FROM decision_choices WHERE interaction_source='demo_http' "
                "ORDER BY created_at"
            ).fetchall()
        for row in rows:
            raw = json.loads(row["trajectory_json"])
            trajectory = DecisionTrajectory.from_dict(raw)
            trajectory = replace(
                trajectory,
                choice=UserChoice(
                    selected_candidate_id=row["selected_parent_asin"],
                    source="user",
                    no_suitable_item=bool(row["no_suitable_item"]),
                ),
            )
            trajectories.append(trajectory)
    finally:
        store.close()
    return tuple(trajectories)
