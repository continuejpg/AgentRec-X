"""Shared deterministic fixtures for the M5 unified decision-state tests.

Every helper builds a fully validated :class:`~recommendation.state.events.StateEvent`
with fixed defaults, so a test only states the field it is actually testing.  The event
ids are process-local and monotonic; a test that needs to re-submit an id passes
``event_id=`` explicitly.

Why :func:`temporary_database` instead of ``tmp_path``
-----------------------------------------------------
A temporary SQLite database is exactly what these tests need, but ``pytest``'s
``tmp_path`` (and ``tempfile.mkdtemp``) create directories with mode ``0o700``.  On the
confined Windows host that runs this worktree, a 0700 directory cannot be listed or
opened for SQLite by the test process, so every database fixture would error before the
test starts.  ``temporary_database`` creates an ephemeral directory under the system
temporary root with the platform default mode and removes it afterwards; it is portable
(the same helper works on Linux) and keeps the database strictly temporary, which is
what the milestone asks for.  Tests must not write repository files.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from recommendation.state.events import StateEvent, new_event

__all__ = [
    "DEFAULT_SESSION",
    "DEFAULT_TASK",
    "DEFAULT_TS",
    "DEFAULT_USER",
    "candidate_pool_event",
    "comparison_event",
    "confirmed_event",
    "constraint_event",
    "context_event",
    "explanation_requested_event",
    "explanation_shown_event",
    "hypothesis_event",
    "next_event_id",
    "preference_stated_event",
    "priority_event",
    "ranking_snapshot_event",
    "retracted_event",
    "temporary_database",
    "understanding_event",
]

#: Fixed event time: replay determinism must not depend on a wall clock.
DEFAULT_TS = 1_700_000_000.0
DEFAULT_USER = "user-a"
DEFAULT_TASK = "task-1"
DEFAULT_SESSION = "session-1"

_counter = itertools.count(1)
_database_counter = itertools.count(1)


def next_event_id(prefix: str = "evt") -> str:
    """Return a deterministic, process-local event id."""
    return f"{prefix}-{next(_counter):06d}"


def _base(
    event_type: str,
    *,
    event_id: str | None,
    user_key: str,
    task_id: str,
    session_id: str | None,
    actor: str,
    source: str,
    timestamp: float,
    **payload_fields: Any,
) -> StateEvent:
    return new_event(
        event_type=event_type,
        event_id=event_id or next_event_id(),
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        **payload_fields,
    )


def preference_stated_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    kind: str = "feature",
    value: str = "lightweight",
    polarity: str = "prefer",
    scope: str = "task",
    source_text: str = "I want something lightweight.",
    direction: str | None = None,
    replaces_value: str | None = None,
) -> StateEvent:
    """An explicit user preference statement."""
    return _base(
        "user_preference_stated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        source_text=source_text,
        direction=direction,
        replaces_value=replaces_value,
    )


def hypothesis_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "agent",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    kind: str = "feature",
    value: str = "lightweight",
    polarity: str = "prefer",
    scope: str = "task",
    strength: str = "weak",
    basis: str = "browsing_history",
    reason: str = "recent browsing skewed lightweight",
    evidence_refs: tuple[str, ...] = ("history-snapshot-1",),
) -> StateEvent:
    """A system-inferred soft preference hypothesis."""
    return _base(
        "preference_hypothesis_proposed",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        strength=strength,
        basis=basis,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def confirmed_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    kind: str = "feature",
    value: str = "lightweight",
    polarity: str = "prefer",
    scope: str = "task",
    source_text: str = "Yes, lightweight is what I want.",
) -> StateEvent:
    """The user confirmed an inferred hypothesis."""
    return _base(
        "preference_confirmed",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        source_text=source_text,
    )


def retracted_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    kind: str = "feature",
    value: str | None = "lightweight",
    polarity: str | None = None,
    scope: str = "task",
    source_text: str = "Actually, forget the lightweight requirement.",
) -> StateEvent:
    """The user retracted a belief (kind-wide when ``value`` is ``None``)."""
    return _base(
        "preference_retracted",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        source_text=source_text,
    )


def constraint_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    kind: str = "price_max",
    value: str = "150",
    polarity: str = "prefer",
    scope: str = "task",
    source_text: str = "My maximum budget is $150.",
) -> StateEvent:
    """An explicit user constraint."""
    return _base(
        "user_constraint_stated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        kind=kind,
        value=value,
        polarity=polarity,
        scope=scope,
        source_text=source_text,
    )


def priority_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    higher: str = "comfort",
    lower: str = "weight",
    kind: str = "feature",
    scope: str = "task",
    source_text: str = "Comfort matters more than weight.",
) -> StateEvent:
    """A relative feature-importance statement."""
    return _base(
        "preference_priority_stated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        higher=higher,
        lower=lower,
        kind=kind,
        scope=scope,
        source_text=source_text,
    )


def comparison_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    compared_refs: tuple[str, ...] = ("pool:item-a", "pool:item-b"),
    selected_ref: str | None = "pool:item-b",
    feedback_text: str = "Item B is more comfortable.",
) -> StateEvent:
    """A comparison observation; it must never create a preference by itself."""
    return _base(
        "product_comparison_feedback",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        compared_refs=compared_refs,
        selected_ref=selected_ref,
        feedback_text=feedback_text,
    )


def explanation_requested_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    attribute: str = "waterproofing",
    question_text: str = "What does waterproofing mean?",
) -> StateEvent:
    """The user asked what an attribute means."""
    return _base(
        "feature_explanation_requested",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        attribute=attribute,
        question_text=question_text,
    )


def explanation_shown_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "agent",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    attribute: str = "waterproofing",
    explanation_ref: str = "explanation-1",
) -> StateEvent:
    """An explanation was displayed (not proof of understanding)."""
    return _base(
        "feature_explanation_shown",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        attribute=attribute,
        explanation_ref=explanation_ref,
    )


def understanding_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "user",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    attribute: str = "waterproofing",
    evidence_text: str = "Waterproofing means it keeps out rain, not that it is submersible.",
) -> StateEvent:
    """The user demonstrated understanding in their own words."""
    return _base(
        "user_understanding_demonstrated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        attribute=attribute,
        evidence_text=evidence_text,
    )


def context_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "agent",
    source: str = "test",
    timestamp: float = DEFAULT_TS,
    intent: str = "recommend",
    request_text: str = "I need a hiking backpack for a weekend trip.",
) -> StateEvent:
    """The controller supplied the typed task context."""
    return _base(
        "task_context_updated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=source,
        timestamp=timestamp,
        intent=intent,
        request_text=request_text,
    )


def candidate_pool_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "trusted_component",
    event_source: str = "test",
    timestamp: float = DEFAULT_TS,
    ref: str = "candidate-pool-1",
    pool_source: str = "recommendation_tool",
    candidate_count: int = 10,
    digest: str | None = None,
) -> StateEvent:
    """A read-only candidate-pool reference from a trusted component."""
    return new_event(
        event_type="candidate_pool_referenced",
        event_id=event_id or next_event_id(),
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=event_source,
        timestamp=timestamp,
        payload={
            "ref": ref,
            "source": pool_source,
            "candidate_count": candidate_count,
            "digest": digest,
        },
    )


def ranking_snapshot_event(
    *,
    event_id: str | None = None,
    user_key: str = DEFAULT_USER,
    task_id: str = DEFAULT_TASK,
    session_id: str | None = DEFAULT_SESSION,
    actor: str = "trusted_component",
    event_source: str = "test",
    timestamp: float = DEFAULT_TS,
    ref: str = "ranking-snapshot-1",
    snapshot_source: str = "serving_v1",
    method: str = "rrf_rank_fusion",
    candidate_count: int = 10,
    digest: str | None = None,
) -> StateEvent:
    """A read-only ranking-snapshot reference from the trusted engine."""
    return new_event(
        event_type="ranking_snapshot_referenced",
        event_id=event_id or next_event_id(),
        user_key=user_key,
        task_id=task_id,
        session_id=session_id,
        actor=actor,
        source=event_source,
        timestamp=timestamp,
        payload={
            "ref": ref,
            "source": snapshot_source,
            "method": method,
            "candidate_count": candidate_count,
            "digest": digest,
        },
    )


@contextlib.contextmanager
def temporary_database() -> Iterator[Path]:
    """Yield a temporary SQLite database path and remove it afterwards."""
    root = Path(tempfile.gettempdir()) / (
        f"agentrecx_state_{os.getpid()}_{next(_database_counter)}"
    )
    root.mkdir()
    try:
        yield root / "events.sqlite3"
    finally:
        shutil.rmtree(root, ignore_errors=True)
