"""M5.1/M5.2 -- unified decision state integration for the shopping Agent runtime.

What this module is
-------------------
One small, opt-in composition layer that records the **validated** facts of one
multi-turn shopping turn as M5 decision-state events, replays the resulting state and
projects it for inspection:

```text
DemoAgentService.run_shopping_turn (planning and clarification-answer branches)
  -> accepted M9 extractor        (explicit preferences, retractions)
  -> accepted shopping intent     (trusted hard constraints, intent label)
  -> trusted clarification option (a selected structured answer is an explicit fact)
  -> this module
       resolve_turn_identity(...) server turn id / client idempotency key identity
       build_turn_events(...)     TASK_CONTEXT_UPDATED / USER_PREFERENCE_STATED /
                                  USER_CONSTRAINT_STATED / PREFERENCE_RETRACTED
       record_candidate_snapshot  CANDIDATE_POOL_REFERENCED /
                                  RANKING_SNAPSHOT_REFERENCED (opaque trusted refs)
       StateManager.load_or_create / apply_event / replay_state
       for_agent / for_ranker / for_constraints
  -> bounded, non-sensitive diagnostics (last 16 tasks)
```

M5.2 additions (all still opt-in; ``off`` remains the default):

* an M9 ``REMOVE`` is resolved against the *current* task state and only a resolved
  selector becomes a ``PREFERENCE_RETRACTED`` event.  An unresolved (value-only,
  ambiguous or absent) removal is reported as a bounded ``retraction_match`` diagnostic
  instead of being guessed -- the documented legacy-memory/M5 disagreement;
* a selected structured clarification option becomes an explicit feedback preference
  with a trusted ``direction`` (``minimize``/``maximize``) when that is what the option
  means; ambiguous/declined answers (``none``/``skip``) create no preference fact;
* after a **presented** recommendation response, the trusted candidate-pool and ranking
  snapshot references are recorded (opaque refs, counts, digests and
  :data:`SNAPSHOT_SCHEMA_VERSION` only -- no product identity, rank or score);
* diagnostics additionally carry ``turn_identity_source``, ``feedback_kind``,
  ``retraction_match``, ``legacy_m5`` and a ``snapshot`` block.

Modes (:data:`ENV_STATE_MODE`)

``off`` (default)
    Nothing is done.  No SQLite database is opened, no event is written and no
    projection is built.  The response is exactly the pre-M5.1 response.

``shadow``
    Validated facts are recorded and projected, but the outcome can never change the
    answer: the caller keeps using the existing trusted constraints.  A state failure is
    recorded as a bounded diagnostic error code and the turn continues exactly as
    ``off``; the failure is *surfaced*, never silently swallowed, but it cannot change
    user-visible behaviour.

``apply_constraints`` (experimental)
    The state's authorized hard constraints (``for_constraints``: user-explicit only,
    task-over-long-term precedence applied) are merged into the existing trusted
    constraint inputs.  A state failure is **fail-closed**: :class:`StateIntegrationError`
    is raised so the service can refuse the turn instead of falling back to the legacy
    constraint interpretation.

Trust boundaries
----------------
* Only already validated explicit facts can be converted to events -- the adapter has no
  entry point for an inference, model text or an arbitrary string.  Inferred hypotheses
  can therefore never become a filter.
* All events are TASK-scoped: M9 preference memory remains the only long-term writer and
  no memory database is migrated.
* Diagnostics never contain the raw message, a support span or the user key.  They carry
  bounded counts, versions, a full state digest, kind labels and a kind-level comparison
  of the state's authorized constraints against the trusted interpretation.

Concurrency
-----------
One integration object owns one ``StateManager`` (one SQLite store) per process.  The
manager serialises validate -> append -> replay and the store serialises appends, so
concurrent requests in one process are safe.  Cross-process writers of the same state
database are **not** supported: a second process would pre-validate against its own
replay.  Deploy single-process (the demo runtime already is) or one database per process.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config
from recommendation.state import shopping_adapter
from recommendation.state.decision_state import UnifiedDecisionState
from recommendation.state.errors import (
    CrossUserStateError,
    EventConflictError,
    EventStoreError,
    StateEngineError,
    UnsupportedTransitionError,
)
from recommendation.state.event_store import STATE_DB_ENV_VAR, SQLiteEventStore
from recommendation.state.state_manager import StateManager
from recommendation.state.state_projector import for_agent, for_constraints, for_ranker

__all__ = [
    "DEFAULT_DEMO_STATE_DB",
    "DEFAULT_STATE_MODE",
    "ENV_STATE_MODE",
    "MAX_DIAGNOSTIC_TASKS",
    "SNAPSHOT_SCHEMA_VERSION",
    "STATE_MODE_VALUES",
    "ShoppingStateIntegration",
    "SnapshotOutcome",
    "StateIntegrationError",
    "StateMode",
    "StateTurnOutcome",
    "resolve_state_mode",
    "state_database_path",
]

#: Environment variable that selects the M5.1 integration mode.
ENV_STATE_MODE = "AGENTRECX_STATE_MODE"

#: The default mode: the integration is composed but inert.
DEFAULT_STATE_MODE = "off"

#: The closed set of accepted mode values.
STATE_MODE_VALUES: tuple[str, ...] = ("off", "shadow", "apply_constraints")

#: Default decision-event database for the demo.  Deliberately separate from the accepted
#: M9 preference-memory database and from the trusted purchase history SASRec consumes.
#: Git-ignored: it is runtime data, not an artifact.
DEFAULT_DEMO_STATE_DB = project_config.ARTIFACTS_DIR / "demo" / "decision_state.sqlite3"

#: Diagnostics are process-local and bounded: the last N tasks are kept.
MAX_DIAGNOSTIC_TASKS = 16

#: Version of the trusted candidate/ranking snapshot contract carried by the M5.2
#: reference events.  It versions the producer's snapshot shape, not the event envelope.
SNAPSHOT_SCHEMA_VERSION = 1

#: Maximum length of a bounded diagnostic label (feedback kind / identity source).
MAX_DIAGNOSTIC_LABEL = 40

#: Bounded failure codes.  ``client-safe``: they identify a failure category, never the
#: raw input that caused it.
_ERROR_INVALID_STATE_MODE = "invalid_state_mode"
_ERROR_STATE_STORE_UNAVAILABLE = "state_store_unavailable"
_ERROR_STATE_EVENT_INVALID = "state_event_invalid"
_ERROR_STATE_TRANSITION_REJECTED = "state_transition_rejected"

_BOUNDED_ERROR_CODES = frozenset(
    {
        _ERROR_INVALID_STATE_MODE,
        _ERROR_STATE_STORE_UNAVAILABLE,
        _ERROR_STATE_EVENT_INVALID,
        _ERROR_STATE_TRANSITION_REJECTED,
    }
)

#: Client-safe messages.  ``str(StateIntegrationError(...))`` returns one of these, never
#: the raw input, a user fragment or a database path.
_ERROR_MESSAGES: dict[str, str] = {
    _ERROR_INVALID_STATE_MODE: "the decision-state mode is not a supported value",
    _ERROR_STATE_STORE_UNAVAILABLE: "the decision-state store is unavailable",
    _ERROR_STATE_EVENT_INVALID: "a decision-state event was rejected as invalid",
    _ERROR_STATE_TRANSITION_REJECTED: "the decision-state transition was rejected",
}


class StateMode(str, Enum):
    """The M5.1 integration mode.  The default is :attr:`OFF`."""

    OFF = "off"
    SHADOW = "shadow"
    APPLY_CONSTRAINTS = "apply_constraints"


class StateIntegrationError(RuntimeError):
    """A decision-state integration failure with a bounded, client-safe code.

    ``code`` is one of the bounded categories below; the string form is a fixed
    category message and never contains user text, an event payload or a filesystem path.

    * ``invalid_state_mode`` -- the configured mode value is not accepted;
    * ``state_store_unavailable`` -- the event store could not be opened or read;
    * ``state_event_invalid`` -- a fact could not be converted into a valid M5 event;
    * ``state_transition_rejected`` -- the reducer refused an otherwise valid event.
    """

    def __init__(self, code: str, detail: str | None = None) -> None:
        if code not in _BOUNDED_ERROR_CODES:
            raise ValueError(f"unknown StateIntegrationError code: {code!r}")
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code
        #: Optional server-side diagnostic; never part of the client-visible string.
        self.detail = detail


def resolve_state_mode(value: str | None = None) -> StateMode:
    """Resolve the integration mode: explicit value, else environment, else ``off``.

    An empty or unset environment variable resolves to :attr:`StateMode.OFF` (the
    default).  An unknown value raises :class:`StateIntegrationError` with code
    ``invalid_state_mode``: there is no silent fallback to ``off``, because a deployment
    that asked for a mode it did not get would be misconfigured invisibly.
    """
    raw = value if value is not None else os.environ.get(ENV_STATE_MODE)
    if raw is None:
        return StateMode.OFF
    normalised = str(raw).strip().lower()
    if not normalised:
        return StateMode.OFF
    try:
        return StateMode(normalised)
    except ValueError as exc:
        raise StateIntegrationError(
            _ERROR_INVALID_STATE_MODE,
            detail=f"unsupported state mode value ({len(str(raw))} chars)",
        ) from exc


def state_database_path() -> Path:
    """Resolve the decision-event database path.

    Precedence: ``AGENTRECX_STATE_DB`` (the M5 state-store environment variable), then
    :data:`DEFAULT_DEMO_STATE_DB`.  The parent directory is created when the store is
    actually opened, not when this path is resolved.
    """
    configured = os.environ.get(STATE_DB_ENV_VAR, "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_DEMO_STATE_DB


@dataclass(frozen=True)
class StateTurnOutcome:
    """The derived result of integrating one shopping turn into the decision state.

    ``state`` is the replayed state after this turn's events; ``event_ids`` are the
    deterministic ids of the event batch; ``authorized_constraints`` is exactly
    ``for_constraints(state)`` as ``(kind, value)`` pairs (user-explicit only, with
    task-over-long-term precedence applied); ``diagnostics`` is the bounded record stored
    for the task.
    """

    state: UnifiedDecisionState
    event_ids: tuple[str, ...]
    authorized_constraints: tuple[tuple[str, str], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotOutcome:
    """The derived result of recording one trusted candidate/ranking snapshot (M5.2).

    ``candidate_pool_ref`` / ``ranking_ref`` are the opaque references the state now
    holds; ``candidate_count`` / ``ranking_count`` are the reference counts (never a
    product row); ``event_ids`` are the deterministic ids of the reference events;
    ``state_digest`` / ``event_count`` describe the replayed state **after** the snapshot.
    """

    candidate_pool_ref: str | None
    candidate_count: int
    candidate_digest: str | None
    ranking_ref: str | None
    ranking_count: int
    ranking_digest: str | None
    ranking_method: str | None
    event_ids: tuple[str, ...]
    state_digest: str
    event_count: int
    snapshot_schema_version: int = SNAPSHOT_SCHEMA_VERSION


def _normalised_value(value: object) -> str:
    """Whitespace-collapsed, case-folded value used for constraint comparison."""
    return " ".join(str(value or "").split()).casefold()


def _signature(kind: object, value: object) -> tuple[str, str]:
    """Comparison signature of one constraint: lower-cased kind + folded value."""
    return (str(kind or "").strip().lower(), _normalised_value(value))


def _turn_digest(turn_key: str) -> str:
    """Bounded, non-sensitive digest of the logical turn key (first 12 hex chars)."""
    return hashlib.sha256(str(turn_key).encode("utf-8")).hexdigest()[:12]


def _failure_code(exc: BaseException) -> str:
    """Map a state failure onto one of the four bounded error codes.

    The mapping is by *category*, not by text: the returned code never depends on the
    exception message, so no raw input can leak through it.
    """
    if isinstance(exc, UnsupportedTransitionError):
        return _ERROR_STATE_TRANSITION_REJECTED
    if isinstance(exc, (EventConflictError, CrossUserStateError)):
        # A reused event id with different content, or an event that would cross a user
        # boundary: both are invalid event *identities*, not store failures.
        return _ERROR_STATE_EVENT_INVALID
    if isinstance(exc, EventStoreError):
        return _ERROR_STATE_STORE_UNAVAILABLE
    if isinstance(exc, StateEngineError):
        return _ERROR_STATE_TRANSITION_REJECTED
    if isinstance(exc, (ValueError, TypeError)):
        # Pydantic's ValidationError is a ValueError subclass: an over-bound, blank,
        # secret-like or unknown-kind fact lands here.
        return _ERROR_STATE_EVENT_INVALID
    if isinstance(exc, (OSError, sqlite3.Error)):
        return _ERROR_STATE_STORE_UNAVAILABLE
    return _ERROR_STATE_STORE_UNAVAILABLE


def _bounded_label(value: object, *, default: str | None = None) -> str | None:
    """Whitespace-collapse and bound a diagnostic label; never returns user text."""
    if value is None:
        return default
    text = " ".join(str(value).split())
    if not text:
        return default
    return text[:MAX_DIAGNOSTIC_LABEL]


def _identity_list(values: Sequence[str]) -> tuple[str, ...]:
    """Clean a trusted snapshot identity list, refusing blank entries.

    The caller may only pass identities that came from the trusted response views; this
    function is the structural guard that a blank/placeholder identity can never become a
    state reference.  A malformed list raises ``ValueError`` (the mode policy maps it).
    """
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError("a trusted snapshot identity must not be blank")
        cleaned.append(text)
    return tuple(cleaned)


def _retraction_match_record(
    requested: int,
    *,
    resolved_count: int,
    unmatched: Sequence[shopping_adapter.ExplicitRetraction],
) -> dict[str, Any]:
    """Bounded kind-level report of how an M9 removal set resolved against the state.

    Only counts and kind labels: no value, no source span.  ``unmatched_value_present``
    says a value-only selector failed to resolve; it never says *which* value.
    """
    kinds = sorted(
        {
            str(getattr(retraction.kind, "value", retraction.kind))
            for retraction in unmatched
            if retraction.kind is not None
        }
    )
    value_present = any(
        _normalised_value(retraction.value) for retraction in unmatched
    )
    return {
        "requested": int(requested),
        "resolved": int(resolved_count),
        "unmatched_kinds": kinds,
        "unmatched_value_present": bool(value_present),
    }


def _snapshot_diagnostic_block(
    *,
    candidate_pool: shopping_adapter.SnapshotReference | None,
    ranking: shopping_adapter.SnapshotReference | None,
    ranking_method: str | None,
    snapshot_error: str | None,
    state: UnifiedDecisionState | None,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The bounded ``snapshot`` diagnostics block (opaque refs and counts only)."""
    prior = previous or {}
    if state is not None:
        state_digest = state.state_digest()
        event_count = int(state.interaction.event_count)
    else:
        state_digest = prior.get("state_digest")
        event_count = int(prior.get("event_count") or 0)
    return {
        "candidate_pool_ref": candidate_pool.ref if candidate_pool is not None else None,
        "candidate_count": candidate_pool.count if candidate_pool is not None else 0,
        "candidate_digest_present": bool(
            candidate_pool is not None and candidate_pool.digest
        ),
        "ranking_ref": ranking.ref if ranking is not None else None,
        "ranking_count": ranking.count if ranking is not None else 0,
        "ranking_method": ranking_method,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "state_digest": state_digest,
        "event_count": event_count,
        "snapshot_error": snapshot_error,
    }


class ShoppingStateIntegration:
    """Records validated shopping turns as M5 decision-state events (OFF/SHADOW/APPLY).

    Parameters
    ----------
    mode:
        The resolved :class:`StateMode`.  ``OFF`` makes every operation inert.
    database:
        Optional decision-event database path.  ``None`` resolves it from the
        environment on first use (see :func:`state_database_path`).  The store is opened
        **lazily**, so an ``OFF`` deployment (or a composing deployment that never serves
        a shopping turn) never creates a database file.
    manager:
        Optional injected :class:`StateManager` (tests / a future backend).  When one is
        injected this object does not own its store and will not close it.
    """

    def __init__(
        self,
        *,
        mode: StateMode,
        database: str | Path | None = None,
        manager: StateManager | None = None,
    ) -> None:
        if not isinstance(mode, StateMode):
            try:
                mode = StateMode(str(mode).strip().lower())
            except ValueError as exc:
                raise StateIntegrationError(_ERROR_INVALID_STATE_MODE) from exc
        self._mode = mode
        self._database = Path(database) if database is not None else None
        self._manager = manager
        self._store: SQLiteEventStore | None = None
        #: Whether the store behind ``_manager`` was created by this object.
        self._owns_store = manager is None
        self._init_lock = threading.RLock()
        self._diagnostics_lock = threading.RLock()
        #: Bounded per-task diagnostics, oldest evicted first (dict preserves insertion).
        self._diagnostics: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        #: How many integration attempts were recorded for each tracked task.
        self._turn_counts: dict[str, int] = {}

    # -- composition ------------------------------------------------------- #

    @classmethod
    def from_env(cls) -> "ShoppingStateIntegration":
        """Build an integration from ``AGENTRECX_STATE_MODE``.

        Only the **mode** is validated here; the database path is resolved and the store
        opened lazily on the first integrated turn.  That separation is what lets a
        per-request store failure follow the mode-specific policy instead of failing the
        whole service at composition time.  An unknown mode raises
        :class:`StateIntegrationError` (``invalid_state_mode``) so composition can refuse
        the deployment explicitly.
        """
        return cls(mode=resolve_state_mode())

    @property
    def mode(self) -> StateMode:
        """The resolved mode."""
        return self._mode

    @property
    def enabled(self) -> bool:
        """True when the integration is not ``OFF``."""
        return self._mode is not StateMode.OFF

    # -- the one operation ------------------------------------------------- #

    def integrate_turn(
        self,
        *,
        user_key: str,
        session_id: str,
        task_id: str,
        turn_key: str,
        message: str,
        intent_label: str,
        preferences: Sequence[shopping_adapter.ExplicitPreference],
        constraints: Sequence[shopping_adapter.ExplicitConstraint],
        trusted_constraints: Sequence[tuple[str, str]],
        timestamp: float | None = None,
        retractions: Sequence[shopping_adapter.ExplicitRetraction] = (),
        feedback_kind: str | None = None,
        turn_identity_source: str = "unspecified",
        legacy_preference_count: int | None = None,
    ) -> StateTurnOutcome | None:
        """Record one shopping turn's validated facts and replay its state.

        Returns ``None`` in ``OFF`` and -- in ``SHADOW`` only -- when the state operation
        failed (the failure is recorded in the diagnostics and the caller continues
        unchanged).  ``APPLY_CONSTRAINTS`` raises :class:`StateIntegrationError` on any
        state failure.  A successful ``SHADOW``/``APPLY_CONSTRAINTS`` call always returns
        a :class:`StateTurnOutcome`, even when only the task-context event is emitted.

        ``retractions`` are M9 ``REMOVE`` selectors.  Each is resolved against the
        **current** task state first; only a resolved selector becomes a
        ``PREFERENCE_RETRACTED`` event.  An unresolved one is reported in the bounded
        ``retraction_match`` diagnostic and is never guessed into an event.  The other
        keyword arguments are bounded diagnostic metadata only: ``feedback_kind`` (for
        example ``explicit_statement`` / ``correction`` / ``retraction`` /
        ``clarification_option``), ``turn_identity_source`` and the caller's legacy-M9
        active-entry count (a count, never memory text).

        The caller supplies only validated facts; this method has no parameter for an
        inference or free model text, and all events are TASK-scoped.
        """
        if self._mode is StateMode.OFF:
            return None
        task_key = str(task_id)
        turn_count = self._begin_turn(task_key)
        turn_digest = _turn_digest(turn_key)
        trusted_pairs = tuple(
            (str(kind), str(value)) for kind, value in trusted_constraints
        )
        retraction_batch = tuple(retractions)
        resolved_retractions: list[shopping_adapter.ExplicitRetraction] = []
        unmatched_retractions: list[shopping_adapter.ExplicitRetraction] = []
        # Only an actually attempted resolution may claim a disagreement; a store failure
        # before resolution is reported by ``error`` instead.
        retraction_match = _retraction_match_record(
            len(retraction_batch), resolved_count=0, unmatched=()
        )
        try:
            manager = self._ensure_manager()
            current_state = manager.load_or_create(
                str(user_key), session_id=task_key, task_id=task_key
            )
            # Resolution is a pure read of the current task state: it identifies *which*
            # kind a value-only removal refers to, or reports that it cannot be known.
            for retraction in retraction_batch:
                resolved = shopping_adapter.resolve_retraction(
                    current_state, retraction
                )
                if resolved is None:
                    unmatched_retractions.append(retraction)
                else:
                    resolved_retractions.append(resolved)
            retraction_match = _retraction_match_record(
                len(retraction_batch),
                resolved_count=len(resolved_retractions),
                unmatched=tuple(unmatched_retractions),
            )
            events = shopping_adapter.build_turn_events(
                identity=shopping_adapter.TurnIdentity(
                    user_key=str(user_key),
                    session_id=str(session_id),
                    task_id=task_key,
                    turn_key=str(turn_key),
                ),
                message=str(message),
                intent_label=str(intent_label),
                preferences=tuple(preferences),
                constraints=tuple(constraints),
                retractions=tuple(resolved_retractions),
                timestamp=float(timestamp) if timestamp is not None else time.time(),
            )
            event_ids: list[str] = []
            for event in events:
                manager.apply_event(event)
                event_ids.append(event.event_id)
            state = manager.replay_state(str(user_key), task_key)
            agent_projection = for_agent(state)
            ranker_projection = for_ranker(state)
            constraint_projection = for_constraints(state)
            authorized = shopping_adapter.authorized_constraint_pairs(state)
            diagnostics = self._diagnostics_record(
                task_id=task_key,
                turn_digest=turn_digest,
                turn_count=turn_count,
                events=len(events),
                state=state,
                agent_projection=agent_projection,
                ranker_projection=ranker_projection,
                constraint_projection=constraint_projection,
                authorized=authorized,
                trusted_pairs=trusted_pairs,
                retraction_match=retraction_match,
                feedback_kind=feedback_kind,
                turn_identity_source=turn_identity_source,
                legacy_preference_count=legacy_preference_count,
                error=None,
            )
            self._record(task_key, diagnostics)
            return StateTurnOutcome(
                state=state,
                event_ids=tuple(event_ids),
                authorized_constraints=authorized,
                diagnostics=diagnostics,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to a bounded category below
            code = _failure_code(exc)
            diagnostics = self._diagnostics_record(
                task_id=task_key,
                turn_digest=turn_digest,
                turn_count=turn_count,
                events=0,
                state=None,
                agent_projection=None,
                ranker_projection=None,
                constraint_projection=None,
                authorized=(),
                trusted_pairs=trusted_pairs,
                retraction_match=retraction_match,
                feedback_kind=feedback_kind,
                turn_identity_source=turn_identity_source,
                legacy_preference_count=legacy_preference_count,
                error=code,
            )
            self._record(task_key, diagnostics)
            if self._mode is StateMode.SHADOW:
                # Observational mode: the failure is visible in diagnostics and the turn
                # continues with OFF-identical user-visible behaviour.
                return None
            raise StateIntegrationError(code) from exc

    # -- trusted candidate / ranking snapshots (M5.2) ----------------------- #

    def record_candidate_snapshot(
        self,
        *,
        user_key: str,
        session_id: str,
        task_id: str,
        turn_key: str,
        candidate_identities: Sequence[str],
        ranking_identities: Sequence[str],
        ranking_method: str,
        timestamp: float | None = None,
    ) -> SnapshotOutcome | None:
        """Record opaque candidate-pool / ranking references for one presented turn.

        The caller may only pass identities taken from the **trusted response views**
        (the grounded candidate view and the presented recommendation order).  This method
        never stores an identity, a rank or a score: it computes an opaque reference,
        count and canonical digest through :mod:`recommendation.state.shopping_adapter`
        and appends two ``trusted_component``-actor reference events.  It is the only
        producer of ``CANDIDATE_POOL_REFERENCED`` / ``RANKING_SNAPSHOT_REFERENCED`` in
        this integration.

        Returns ``None`` in ``OFF``, when the candidate list is empty (no empty snapshot
        is fabricated), and -- in ``SHADOW`` only -- when the snapshot could not be
        recorded (the failure is recorded in the bounded ``snapshot`` diagnostics block,
        and the user-visible answer is untouched).  ``APPLY_CONSTRAINTS`` raises
        :class:`StateIntegrationError` on a snapshot failure (fail closed).

        A malformed provenance input (a blank identity, a repeated identity inside the
        presented ranking order or a blank method label) is refused with ``ValueError``,
        which the mode policy then maps exactly like any other snapshot failure.
        """
        if self._mode is StateMode.OFF:
            return None
        raw_candidates = tuple(candidate_identities)
        if not raw_candidates:
            # Do not fabricate an empty candidate snapshot: an empty presentation is not
            # a trusted snapshot of anything.
            return None
        task_key = str(task_id)
        candidate_pool: shopping_adapter.SnapshotReference | None = None
        ranking: shopping_adapter.SnapshotReference | None = None
        try:
            candidates = _identity_list(raw_candidates)
            ranking_order = _identity_list(tuple(ranking_identities))
            if len(set(ranking_order)) != len(ranking_order):
                raise ValueError("the presented ranking order must not repeat an identity")
            method = " ".join(str(ranking_method or "").split())
            if not method:
                raise ValueError("a ranking snapshot needs a method label")
            candidate_pool = shopping_adapter.candidate_pool_snapshot(candidates)
            ranking = (
                shopping_adapter.ranking_snapshot(ranking_order)
                if ranking_order
                else None
            )
            events = shopping_adapter.build_snapshot_events(
                identity=shopping_adapter.TurnIdentity(
                    user_key=str(user_key),
                    session_id=str(session_id),
                    task_id=task_key,
                    turn_key=str(turn_key),
                ),
                candidate_pool=candidate_pool,
                ranking=ranking,
                ranking_method=method,
                timestamp=float(timestamp) if timestamp is not None else time.time(),
            )
            manager = self._ensure_manager()
            manager.load_or_create(str(user_key), session_id=task_key, task_id=task_key)
            event_ids: list[str] = []
            for event in events:
                manager.apply_event(event)
                event_ids.append(event.event_id)
            state = manager.replay_state(str(user_key), task_key)
            previous = self.diagnostics(task_key)
            block = _snapshot_diagnostic_block(
                candidate_pool=candidate_pool,
                ranking=ranking,
                ranking_method=method,
                snapshot_error=None,
                state=state,
                previous=previous,
            )
            self._update_snapshot_diagnostics(
                task_key, block, state=state, events_added=len(events)
            )
            return SnapshotOutcome(
                candidate_pool_ref=candidate_pool.ref,
                candidate_count=candidate_pool.count,
                candidate_digest=candidate_pool.digest,
                ranking_ref=ranking.ref if ranking is not None else None,
                ranking_count=ranking.count if ranking is not None else 0,
                ranking_digest=ranking.digest if ranking is not None else None,
                ranking_method=method,
                event_ids=tuple(event_ids),
                state_digest=state.state_digest(),
                event_count=int(state.interaction.event_count),
            )
        except Exception as exc:  # noqa: BLE001 - mapped to a bounded category below
            code = _failure_code(exc)
            block = _snapshot_diagnostic_block(
                candidate_pool=candidate_pool,
                ranking=ranking,
                ranking_method=(
                    " ".join(str(ranking_method or "").split()) or None
                ),
                snapshot_error=code,
                state=None,
                previous=self.diagnostics(task_key),
            )
            self._update_snapshot_diagnostics(task_key, block)
            if self._mode is StateMode.SHADOW:
                # Observational mode: a snapshot failure is surfaced in diagnostics and
                # can never change the presented answer.
                return None
            raise StateIntegrationError(code) from exc

    # -- M7 grounded interaction events ------------------------------------ #

    def record_grounded_interaction(
        self,
        *,
        user_key: str,
        session_id: str,
        task_id: str,
        turn_key: str,
        compared_refs: Sequence[str] = (),
        comparison_feedback: str = "",
        explanation_requested: str | None = None,
        question_text: str = "",
        explanation_shown: str | None = None,
        explanation_ref: str | None = None,
        timestamp: float | None = None,
    ) -> tuple[str, ...] | None:
        """Record the user/agent-visible M7 interaction events for one turn.

        Only real events are accepted: comparison feedback is emitted only when the user
        actually answered a comparison, an explanation request only when the user's own
        message asked for it, and an explanation-shown event when the agent displayed
        grounded facts.  The event ids are deterministic from the frozen turn identity,
        so a retry is idempotent and a replay is identical.

        Returns the event ids, or ``None`` in OFF and (in SHADOW) on failure.  APPLY
        fails closed with :class:`StateIntegrationError`.
        """
        if self._mode is StateMode.OFF:
            return None
        task_key = str(task_id)
        try:
            manager = self._ensure_manager()
            manager.load_or_create(str(user_key), session_id=str(session_id), task_id=task_key)
            events = shopping_adapter.build_grounded_interaction_events(
                identity=shopping_adapter.TurnIdentity(
                    user_key=str(user_key),
                    session_id=str(session_id),
                    task_id=task_key,
                    turn_key=str(turn_key),
                ),
                timestamp=float(timestamp) if timestamp is not None else time.time(),
                compared_refs=tuple(compared_refs),
                comparison_feedback=str(comparison_feedback or ""),
                explanation_requested=explanation_requested,
                question_text=str(question_text or ""),
                explanation_shown=explanation_shown,
                explanation_ref=explanation_ref,
            )
            event_ids: list[str] = []
            for event in events:
                manager.apply_event(event)
                event_ids.append(event.event_id)
            return tuple(event_ids)
        except Exception as exc:  # noqa: BLE001 - bounded category below
            code = _failure_code(exc)
            if self._mode is StateMode.SHADOW:
                return None
            raise StateIntegrationError(code) from exc

    # -- diagnostics ------------------------------------------------------- #

    def diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Return a copy of one task's bounded diagnostics, or ``None`` when untracked.

        The record never contains the raw message, a support span or the user key.
        """
        with self._diagnostics_lock:
            record = self._diagnostics.get(str(task_id))
            return copy.deepcopy(record) if record is not None else None

    def forget_diagnostics(self, task_id: str) -> None:
        """Drop one task's diagnostics (used when a demo session is reset)."""
        with self._diagnostics_lock:
            self._diagnostics.pop(str(task_id), None)
            self._turn_counts.pop(str(task_id), None)

    def close(self) -> None:
        """Close the decision-event store, if this object owns one.  Idempotent."""
        with self._init_lock:
            store, self._store = self._store, None
            self._manager = None
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    # -- internals --------------------------------------------------------- #

    def _ensure_manager(self) -> StateManager:
        """Return the state manager, opening the SQLite store on first use.

        Opening is deliberately lazy: composition must validate configuration only, so a
        per-turn store failure can follow the mode-specific policy.  The parent directory
        is created here, when the path is actually used.
        """
        manager = self._manager
        if manager is not None:
            return manager
        with self._init_lock:
            if self._manager is not None:
                return self._manager
            database = self._database
            if database is None:
                database = state_database_path()
            if str(database) != ":memory:":
                Path(database).parent.mkdir(parents=True, exist_ok=True)
            store = SQLiteEventStore(database)
            try:
                manager = StateManager(store)
            except BaseException:
                store.close()
                raise
            self._store = store
            self._manager = manager
            return manager

    def _begin_turn(self, task_id: str) -> int:
        """Count this integration attempt for the task (bounded diagnostic metadata)."""
        with self._diagnostics_lock:
            count = int(self._turn_counts.get(task_id, 0)) + 1
            self._turn_counts[task_id] = count
            return count

    def _record(self, task_id: str, record: dict[str, Any]) -> None:
        """Store the latest diagnostics record for one task, keeping the bound.

        The stored copy is detached from the value returned to the caller, so a consumer
        cannot mutate the retained diagnostic by touching the returned outcome/dict.
        """
        with self._diagnostics_lock:
            self._diagnostics.pop(task_id, None)
            self._diagnostics[task_id] = copy.deepcopy(record)
            while len(self._diagnostics) > MAX_DIAGNOSTIC_TASKS:
                oldest, _ = self._diagnostics.popitem(last=False)
                self._turn_counts.pop(oldest, None)

    def _update_snapshot_diagnostics(
        self,
        task_id: str,
        block: dict[str, Any],
        *,
        state: UnifiedDecisionState | None = None,
        events_added: int = 0,
    ) -> None:
        """Attach the bounded snapshot block to the task's stored diagnostics.

        The refresh is only ever a widening of the same bounded record: the opaque
        candidate/ranking references replace the previous ones (latest trusted snapshot
        wins) and, after a successful append, the record's ``state_digest`` /
        ``event_count`` / ``events`` are refreshed to the state replay produced by the
        snapshot, so a diagnostics digest always describes the state that is on disk.
        """
        with self._diagnostics_lock:
            record = self._diagnostics.get(str(task_id))
            if record is None:
                # Only reachable when a caller records a snapshot without a turn: keep a
                # bounded, honestly-partial record rather than claiming a turn happened.
                record = self._diagnostics_record(
                    task_id=str(task_id),
                    turn_digest="",
                    turn_count=0,
                    events=0,
                    state=None,
                    agent_projection=None,
                    ranker_projection=None,
                    constraint_projection=None,
                    authorized=(),
                    trusted_pairs=(),
                    retraction_match=_retraction_match_record(
                        0, resolved_count=0, unmatched=()
                    ),
                    feedback_kind=None,
                    turn_identity_source="unspecified",
                    legacy_preference_count=None,
                    error=None,
                )
                self._diagnostics[str(task_id)] = record
            record["snapshot"] = copy.deepcopy(block)
            if state is not None:
                record["state_digest"] = state.state_digest()
                record["event_count"] = int(state.interaction.event_count)
                record["events"] = int(record.get("events") or 0) + int(events_added)

    def _diagnostics_record(
        self,
        *,
        task_id: str,
        turn_digest: str,
        turn_count: int,
        events: int,
        state: UnifiedDecisionState | None,
        agent_projection: Any,
        ranker_projection: Any,
        constraint_projection: Any,
        authorized: Sequence[tuple[str, str]],
        trusted_pairs: Sequence[tuple[str, str]],
        retraction_match: Mapping[str, Any],
        feedback_kind: str | None,
        turn_identity_source: str,
        legacy_preference_count: int | None,
        error: str | None,
    ) -> dict[str, Any]:
        """Build the bounded, non-sensitive diagnostics record for one turn.

        Bounded by construction: counts, versions, sorted kind labels and one full state
        digest.  No raw message, support span, user key or constraint value is stored.
        """
        trusted_signatures = {_signature(kind, value) for kind, value in trusted_pairs}
        authorized_signatures = {_signature(kind, value) for kind, value in authorized}
        matched = len(trusted_signatures & authorized_signatures)
        missing = sorted(
            {kind for kind, _ in trusted_signatures - authorized_signatures}
        )
        extra = sorted({kind for kind, _ in authorized_signatures - trusted_signatures})
        constraint_match = {
            "matched": matched,
            "missing_from_state": missing,
            "extra_in_state": extra,
        }
        memory_active_count = (
            int(legacy_preference_count)
            if legacy_preference_count is not None
            else None
        )
        state_explicit_count = (
            len(state.explicit_or_confirmed_beliefs) if state is not None else 0
        )
        legacy_m5 = {
            "memory_active_count": memory_active_count,
            "state_explicit_count": int(state_explicit_count),
            "delta": (
                int(state_explicit_count) - memory_active_count
                if memory_active_count is not None and state is not None
                else None
            ),
        }
        record: dict[str, Any] = {
            "mode": self._mode.value,
            "task_id": task_id,
            "turn_key_digest": turn_digest,
            "events": int(events),
            "state_version": None,
            "state_digest": None,
            "event_count": 0,
            "user_turn_count": 0,
            "preference_count": 0,
            "soft_hypothesis_count": 0,
            "hard_constraint_count": 0,
            "constraint_kinds": [],
            "suppressed_long_term_count": 0,
            "agent_projection_version": None,
            "ranker_signal_version": None,
            "ranker_feature_count": 0,
            "ranker_priority_count": 0,
            "constraints_version": None,
            "excluded_hypothesis_count": 0,
            "trusted_constraint_count": len(trusted_pairs),
            "constraint_match": constraint_match,
            "turn_identity_source": _bounded_label(
                turn_identity_source, default="unspecified"
            ),
            "feedback_kind": _bounded_label(feedback_kind),
            "retraction_match": dict(retraction_match),
            "legacy_m5": legacy_m5,
            "snapshot": None,
            "error": error,
            "turn_count": int(turn_count),
        }
        if state is None:
            return record
        record.update(
            {
                "state_version": int(state.state_version),
                "state_digest": state.state_digest(),
                "event_count": int(state.interaction.event_count),
                "user_turn_count": int(state.interaction.user_turn_count),
                "preference_count": len(agent_projection.preferences),
                "soft_hypothesis_count": int(agent_projection.inferred_preference_count),
                # The constraints actually projected for this task (precedence applied),
                # i.e. exactly the authorized pairs the APPLY hook merges.
                "hard_constraint_count": len(authorized),
                "constraint_kinds": sorted({kind for kind, _ in authorized}),
                "suppressed_long_term_count": len(
                    constraint_projection.suppressed_long_term_ids
                ),
                "agent_projection_version": int(agent_projection.projection_version),
                "ranker_signal_version": int(ranker_projection.ranker_signal_version),
                "ranker_feature_count": len(ranker_projection.features),
                "ranker_priority_count": len(ranker_projection.priorities),
                "constraints_version": int(constraint_projection.constraints_version),
                "excluded_hypothesis_count": int(
                    constraint_projection.excluded_hypothesis_count
                ),
            }
        )
        return record
