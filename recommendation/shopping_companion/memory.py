"""P14-A Shopping Companion memory adapter (M9 preference memory + M5 state replay).

What this module is, and what it deliberately is not
-----------------------------------------------------
This is the **adapter** between the released Shopping Companion benchmark and the two
accepted memory components of AgentRec-X:

* :mod:`recommendation.memory` -- Milestone 9 *preference memory*: the deterministic
  rule extractor, the validated entry schema, the lifecycle service and the store;
* :mod:`recommendation.state` -- M5-MVP *unified decision state*: validated
  ``StateEvent`` construction (``build_turn_events``), ``StateManager`` replay,
  ``BeliefUpdater`` and the read-only ``state_projector.for_agent``.

It adds no second memory store, no second preference ontology, no second event schema
and no second Agent runtime.  It contains:

* a session normaliser for the ``{session_id?, date, turns:[{role, content}]}`` shape
  that the (unreleased) official conversation-history JSONL would have had;
* a per-turn **evidence ledger** that keeps the exact dialogue span every stored
  preference came from, tagged with its ``source_kind``;
* a lexical ``search`` read model over the M9 ACTIVE entries (M9 ships no preference
  search; only a catalogue-query adapter), plus conflict reporting and honest
  availability semantics;
* an M5 write/read path that replays the ingested *user* turns as real
  ``StateEvent`` batches and exposes them through ``for_agent``.

The single most important fact about this dataset
-------------------------------------------------
The released Shopping Companion dataset on Hugging Face has only ``train.parquet``,
``test.parquet`` and ``products.jsonl``.  The conversation-history JSONL that the
official memory service needs (haystack sessions) is **not released**; the only real
dialogue text in the release is the *oracle memory block* inside every stage_2 prompt.
Consequently a normal P14-A run has **no legal long-term history**:

* a fresh :class:`ShoppingCompanionMemory` returns ``available=False``,
  ``status="history_unavailable"``, zero preferences and a reason that names the
  unreleased history corpus.  It never invents a preference, never reads the
  evaluator-only ground truth and has no code path to target product ids.
* the oracle memory block may be ingested **only** with the explicit
  ``source_kind="released_oracle_diagnostic"`` tag.  Every evidence item and every
  returned preference is tagged with that source kind, and such evidence is returned
  **only** by an instance constructed with ``oracle_diagnostic=True`` -- i.e. a run the
  caller explicitly labels as an oracle diagnostic.  Oracle evidence can therefore
  never silently become a normal Agent success; the runtime bridge
  (``recommendation.shopping_companion.planner_bridge``) independently refuses leaked
  oracle evidence in non-diagnostic runs.
* ``source_kind="history"`` is refused by name: no fabricated real history is accepted.

Reuse: what is existing code, and what is new
---------------------------------------------
Reused unchanged (no copy, no fork):

* :class:`recommendation.memory.service.PreferenceMemoryService` -- the only writer of
  preference memory.  ``process_turn`` validates every extracted candidate, assigns
  ``logical_seq``/ids, deduplicates and applies M9 add/replace/remove semantics.
* :class:`recommendation.memory.store.InMemoryPreferenceStore` -- durable (in-process)
  preference storage; injectable so the runtime can choose another accepted store.
* :class:`recommendation.memory.extraction.RuleBasedPreferenceExtractor` -- the M9
  extractor.  It is *already* deterministic, offline, dependency-free and model-free,
  so the task's "if the extractor needs an LLM, write a deterministic local one"
  escape hatch does not apply: this adapter reuses the accepted extractor instead of
  writing a second one.
* :class:`recommendation.memory.schemas.PreferenceKind/PreferencePolarity/PreferenceStatus`
  and ``PreferenceMemoryEntry``/``PreferenceMemorySnapshot`` -- the accepted ontology and
  provenance record.
* :func:`recommendation.state.shopping_adapter.build_turn_events` -- validated M5 event
  construction (the only event producer used).
* :class:`recommendation.state.state_manager.StateManager` over an ephemeral
  ``SQLiteEventStore(":memory:")`` -- append/replay of the ingested turns.
* :class:`recommendation.state.belief_updater.BeliefUpdater` -- used as a pure dry run
  *before* any write, so invalid evidence cannot half-write.
* :func:`recommendation.state.state_projector.for_agent` -- the M5 read projection;
* :func:`recommendation.state.legacy_adapter.project_legacy_memory` and
  ``merge_legacy_memory`` -- the accepted read-only M9 -> M5 long-term bridge (used by
  :meth:`ShoppingCompanionMemory.m5_long_term_projection`).

Newly written here (and why no accepted module already provides it):

* session normalisation for the Shopping Companion dialogue shape;
* the evidence ledger (per session/turn provenance, ``source_kind`` tagging), because
  M9 entries only carry ``source_turn_id``/``source_text`` and the dataset needs the
  date/session/turn triple retained as audit evidence;
* a deterministic lexical preference search over M9 ACTIVE entries (M9 has a
  catalogue-query adapter but no memory search read model), and the
  available/status/reason/conflict result contract;
* the oracle-diagnostic gating mode and the M5 ingestion glue.

M5 integration status (real, with one documented gap)
-----------------------------------------------------
The integration is **real**, not a documentation gap.  Two accepted M5 read paths are
exposed, each of them a projection of a different authority, never merged into one
state:

* :meth:`ShoppingCompanionMemory.m5_agent_projection` -- **event replay**: ingestion
  builds real ``StateEvent`` batches with :func:`build_turn_events` (one
  ``task_context_updated`` per non-blank user turn, then its
  ``user_preference_stated`` / ``preference_retracted`` events), replays a
  ``BeliefUpdater`` dry run, applies the events to an instance-local
  :class:`StateManager` and returns ``state_projector.for_agent``.  A test asserts the
  ingested preferences are returned here.  These are ``task``-scope beliefs because
  that is exactly what the accepted ``build_turn_events`` emits.
* :meth:`ShoppingCompanionMemory.m5_long_term_projection` -- **long-term bridge**: the
  accepted read-only :func:`recommendation.state.legacy_adapter.project_legacy_memory`
  + ``merge_legacy_memory`` project the ACTIVE M9 entries as
  ``LONG_TERM``/``EXPLICIT``/``LEGACY_MEMORY`` beliefs and return
  ``state_projector.for_agent``.  It is a pure read: no event is appended and nothing is
  written back to M9.
* The M5 task namespace is derived from the source kind
  (``p14a_memory_<source_kind>``), so an M5 consumer can always tell that a state came
  from tagged P14-A evidence rather than from a live conversational task.
* Documented gap: an M9 **value-only retraction** (``removal.kind is None``, e.g.
  "forget lightweight") is applied to the M9 store by the accepted service, but it is
  not replayed as an M5 ``PREFERENCE_RETRACTED`` event because the accepted M5
  retraction payload needs a resolved preference kind (``resolve_retraction`` needs a
  live M5 state).  The number of such unmapped retractions is reported in the ingest
  summary instead of being guessed at.
* Conflict semantics differ by design: M9 keeps two independent constraints
  ("I prefer red" + "I don't want red" stays two entries and is reported by this
  adapter as a conflict), whereas the accepted M5 reducer supersedes the
  same-kind/value opposite-polarity belief.  The adapter reports the M9 view (the
  long-term preference read path) and does not normalise M5 to it.

Data-safety rules enforced here
-------------------------------
* no file/network/parquet/arrow access at all -- sessions are passed in, nothing is
  read from disk; the module imports only the standard library plus M9/M5;
* extraction runs only on **user-authored** turns (``role`` ``user``/``human``);
  assistant turns are evidence only and never become preferences;
* every evidence item and every returned preference carries ``source_kind``;
* oracle evidence is excluded unless the instance was constructed with
  ``oracle_diagnostic=True``; ``source_kind="history"`` is rejected at ingestion;
* no evaluation ground truth, no target product ids, no model, no GPU.

Chronology: sessions are ingested and replayed in the caller's order (expected
chronological, as the project's temporal rules require); the adapter neither sorts nor
reorders sessions, and each M5 event timestamp is derived deterministically from the
session date plus the turn index.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from recommendation.memory import (
    InMemoryPreferenceStore,
    PreferenceExtraction,
    PreferenceExtractionError,
    PreferenceExtractor,
    PreferenceMemoryService,
    PreferenceStatus,
    RuleBasedPreferenceExtractor,
)
from recommendation.state import (
    AgentStateProjection,
    BeliefUpdater,
    ExplicitPreference,
    ExplicitRetraction,
    SQLiteEventStore,
    StateEvent,
    StateManager,
    TurnIdentity,
    UnifiedDecisionState,
    build_turn_events,
    empty_state,
    for_agent,
    merge_legacy_memory,
    project_legacy_memory,
)
from recommendation.state.schemas import (
    MAX_SESSION_ID_LENGTH,
    MAX_USER_KEY_LENGTH,
    bounded_identifier,
)

__all__ = [
    "HISTORY_STATUSES",
    "MEMORY_ADAPTER_SCHEMA",
    "M5_INTENT_LABEL",
    "M5_TASK_PREFIX",
    "ORACLE_REFUSED_REASON",
    "RELEASED_HISTORY_REASON",
    "SCOPES",
    "SCOPE_LONG_TERM",
    "SOURCE_KINDS",
    "SOURCE_KIND_ORACLE_DIAGNOSTIC",
    "SOURCE_KIND_SYNTHETIC_FIXTURE",
    "STATUSES",
    "STATUS_AVAILABLE",
    "STATUS_HISTORY_UNAVAILABLE",
    "STATUS_ORACLE_DIAGNOSTIC",
    "STATUS_SYNTHETIC_FIXTURE",
    "MemoryIngestError",
    "OracleDiagnosticRefused",
    "ShoppingCompanionMemory",
]

#: Schema label carried by :meth:`ShoppingCompanionMemory.snapshot` and
#: :meth:`ShoppingCompanionMemory.history_status`.
MEMORY_ADAPTER_SCHEMA = "agentrecx.p14a.shopping_memory.v1"

#: The only accepted ingestion source kinds.  ``"history"`` is intentionally absent:
#: the released dataset does not contain real conversation history and this adapter
#: must never accept a fabricated ``history`` label.
SOURCE_KIND_ORACLE_DIAGNOSTIC = "released_oracle_diagnostic"
SOURCE_KIND_SYNTHETIC_FIXTURE = "synthetic_fixture"
SOURCE_KINDS: tuple[str, ...] = (SOURCE_KIND_ORACLE_DIAGNOSTIC, SOURCE_KIND_SYNTHETIC_FIXTURE)

#: The ``source_kind`` values that make a preference usable by the normal runtime.
#: Synthetic fixtures are contract-test data; they are usable by a test/contract-only
#: run but are never a benchmark result.
NORMAL_MODE_SOURCE_KINDS: tuple[str, ...] = (SOURCE_KIND_SYNTHETIC_FIXTURE,)

#: Search statuses.  ``STATUS_AVAILABLE`` is reserved for a future **released** real
#: history corpus; no code path can produce it while the conversation histories are
#: unreleased, and it is deliberately unreachable rather than being faked.
STATUS_HISTORY_UNAVAILABLE = "history_unavailable"
STATUS_AVAILABLE = "available"
STATUS_ORACLE_DIAGNOSTIC = "oracle_diagnostic"
STATUS_SYNTHETIC_FIXTURE = "synthetic_fixture"
STATUSES: tuple[str, ...] = (
    STATUS_HISTORY_UNAVAILABLE,
    STATUS_AVAILABLE,
    STATUS_ORACLE_DIAGNOSTIC,
    STATUS_SYNTHETIC_FIXTURE,
)

#: The statuses :meth:`ShoppingCompanionMemory.history_status` may report (a stricter
#: subset of :data:`STATUSES`: ``available`` is reserved for a future released corpus).
HISTORY_STATUSES: tuple[str, ...] = (
    STATUS_HISTORY_UNAVAILABLE,
    STATUS_ORACLE_DIAGNOSTIC,
    STATUS_SYNTHETIC_FIXTURE,
)

#: The only memory scope P14-A can honestly serve.  ``task``/``session`` are accepted
#: as *inputs* (the tool contract names them) but return an honest unavailable payload,
#: because a P14-A run has no live task/session conversation state.
SCOPE_LONG_TERM = "long_term"
SCOPES: tuple[str, ...] = ("long_term", "task", "session")

#: M5 task namespace derived from the source kind, so M5 state can never be mistaken
#: for a live conversational task.
M5_TASK_PREFIX = "p14a_memory_"

#: Intent label recorded on the M5 ``task_context_updated`` events (the accepted M5
#: label grammar is ``^[a-z][a-z0-9_]{0,39}$``).
M5_INTENT_LABEL = "p14a_memory_ingest"

#: Why normal memory is unavailable.  Kept as one exported constant so the runtime, the
#: trajectory and the tests quote exactly the same fact.
RELEASED_HISTORY_REASON = (
    "the released Shopping Companion dataset ships train.parquet, test.parquet and "
    "products.jsonl only; the cross-session conversation-history corpus required by the "
    "official memory service is not released, so normal long-term history is unavailable "
    "and no history or preference is fabricated"
)

#: Why oracle-diagnostic evidence is not served by a strict instance.
ORACLE_REFUSED_REASON = (
    "oracle-diagnostic dialogue evidence is present but this instance is not in explicit "
    "oracle_diagnostic mode, so the evidence is refused; construct "
    "ShoppingCompanionMemory(oracle_diagnostic=True) only for an explicitly labelled "
    "oracle diagnostic run, and never count that evidence as recovered user history"
)

#: Role values treated as user-authored text (the only text M9 may extract from).
#: ``human`` is accepted because the normaliser for the official haystack-style history
#: may use ``Human``/``Assistant`` role labels.
_USER_ROLES: frozenset[str] = frozenset({"user", "human"})

#: A deterministic tokenisation: lowercase alphanumeric runs.  No stemming, no model.
#: (The RAG package has a tokenizer, but importing the RAG/catalogue stack into the
#: memory adapter would couple memory to a subsystem it does not use; this three-line
#: rule is not a second retrieval system.)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: English function words that carry no preference meaning.  Used only so a natural
#: question ("what is my budget?") does not match every stored preference through "my".
_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "i", "me", "my", "mine", "myself", "we", "our", "ours",
        "you", "your", "yours", "he", "she", "it", "its", "they", "them", "their",
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
        "done", "have", "has", "had", "will", "would", "shall", "should", "can",
        "could", "may", "might", "must", "to", "of", "for", "and", "or", "but", "if",
        "in", "on", "at", "by", "with", "about", "as", "from", "into", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "now", "please", "tell", "show", "what", "which",
        "who", "whom", "this", "that", "these", "those", "prefer", "preferred",
        "preference", "preferences", "like", "liked", "want", "wanted", "need",
        "needed", "remember", "memory", "memories", "know", "knows", "recall",
    }
)

#: Extra search vocabulary for kinds whose stored value is a number rather than text.
#: This is a small fixed vocabulary (not inference): price bounds are stored as
#: ``kind="price_max"/"price_min"`` with a numeric value, so a user question about the
#: budget must still be able to match them lexically.
_KIND_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "price_max": ("price", "budget", "max", "maximum", "ceiling", "cost"),
    "price_min": ("price", "budget", "min", "minimum", "floor", "cost"),
}


class MemoryIngestError(ValueError):
    """Ingestion was refused: malformed input or an illegal source kind.

    Subclasses :class:`ValueError` so callers that validate untrusted tool input keep
    working; the message always states the exact refusal reason.
    """


class OracleDiagnosticRefused(RuntimeError):
    """Oracle evidence was requested from an instance not in oracle-diagnostic mode."""


# --------------------------------------------------------------------------- #
# Internal records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TurnEvidence:
    """One stored dialogue turn plus the preferences it supports.

    This is the adapter's provenance record: it is what makes a returned preference
    traceable to an exact ``(conversation_id, session, turn, date, role)`` after the M9
    store has normalised away everything except ``source_turn_id``/``source_text``.
    """

    conversation_id: str
    session_index: int
    session_id: str
    turn_index: int
    turn_id: str
    date: str
    role: str
    text: str
    source_kind: str
    source_ref: str
    extracted: tuple[dict[str, Any], ...] = ()
    removals: tuple[dict[str, Any], ...] = ()

    @property
    def is_user_turn(self) -> bool:
        """True when the turn is user-authored (and therefore extractable)."""
        return self.role in _USER_ROLES

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable evidence view (always tagged with ``source_kind``)."""
        return {
            "conversation_id": self.conversation_id,
            "session_index": self.session_index,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "turn_id": self.turn_id,
            "date": self.date,
            "role": self.role,
            "text": self.text,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "extracted_preferences": [dict(item) for item in self.extracted],
            "removals": [dict(item) for item in self.removals],
        }


@dataclass(frozen=True)
class _PendingTurn:
    """One non-blank turn produced by normalising the caller's sessions."""

    session_index: int
    session_id: str
    turn_index: int
    turn_id: str
    date: str
    timestamp: float
    role: str
    text: str

    @property
    def is_user_turn(self) -> bool:
        """True when the turn is user-authored."""
        return self.role in _USER_ROLES


@dataclass(frozen=True)
class _NormalisedSessions:
    """The validated, normalised view of one ``ingest_sessions`` call."""

    sessions: tuple[dict[str, Any], ...]
    turns: tuple[_PendingTurn, ...]
    user_turns: tuple[_PendingTurn, ...]
    digest: str
    turns_seen: int
    turns_skipped_blank: int


@dataclass
class _ConversationRecord:
    """Everything the adapter retains about one ingested conversation."""

    conversation_id: str
    source_kind: str
    source_ref: str
    digest: str
    sessions: tuple[dict[str, Any], ...]
    evidence: tuple[_TurnEvidence, ...]
    by_turn_id: dict[str, _TurnEvidence]
    m5_task_id: str
    m5_session_ids: tuple[str, ...]
    m5_event_count: int
    m5_unmapped_retractions: int
    ingested_at: float
    turns_seen: int
    turns_skipped_blank: int
    preferences_extracted: int
    preferences_added: int
    preferences_skipped_duplicates: int
    removal_directives: int
    state_version: int

    @property
    def user_turns(self) -> int:
        """Number of user-authored turns in this conversation."""
        return sum(1 for item in self.evidence if item.is_user_turn)


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #


def _tokenize(text: object) -> tuple[str, ...]:
    """Return lowercase alphanumeric tokens; a non-string yields no tokens."""
    if not isinstance(text, str) or not text:
        return ()
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


def _query_tokens(query: str) -> tuple[str, ...]:
    """Return the meaningful, de-duplicated tokens of a free-text query."""
    seen: list[str] = []
    for token in _tokenize(query):
        if token in _QUERY_STOPWORDS or token in seen:
            continue
        seen.append(token)
    return tuple(seen)


def _canonical_json(value: Any) -> str:
    """Deterministic JSON used only for the ingest fingerprint (not for hashing state)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _m5_task_id(source_kind: str) -> str:
    """The M5 task namespace for one source kind."""
    return (M5_TASK_PREFIX + source_kind)[:128]


def _parse_session_date(raw: object, session_index: int) -> tuple[str, float]:
    """Validate a session date and return ``(date_text, unix_timestamp)``.

    Accepts an ISO-8601 date or datetime (the dataset's date form).  A missing or
    unparseable date is refused rather than invented: the evidence record must be able
    to state *when* a preference was expressed.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise MemoryIngestError(
            f"sessions[{session_index}] must carry a non-blank ISO-8601 'date'"
        )
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MemoryIngestError(
            f"sessions[{session_index}] date {text!r} is not an ISO-8601 date/datetime"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return text, parsed.timestamp()


def _normalise_session_id(raw: object, session_index: int) -> str:
    """Validate an optional session id, defaulting to a deterministic placeholder."""
    if raw is None:
        return f"session_{session_index}"
    try:
        return bounded_identifier(
            raw, field=f"sessions[{session_index}].session_id", max_length=MAX_SESSION_ID_LENGTH
        )
    except ValueError as exc:
        raise MemoryIngestError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class ShoppingCompanionMemory:
    """Honest, M5/M9-backed long-term memory for the P14-A Shopping Companion Agent.

    The class satisfies the runtime's memory protocol
    (``search(query, *, scope, k)`` + ``history_status()``) while adding the
    diagnostic-only ingestion API.  A fresh instance is exactly as honest as
    ``BlockedMemoryTool``: it reports the unreleased-history state and never fabricates a
    preference.

    Parameters
    ----------
    store:
        Any accepted :class:`~recommendation.memory.store.PreferenceStore`.  Defaults to
        a process-local :class:`~recommendation.memory.store.InMemoryPreferenceStore`;
        inject :class:`~recommendation.memory.store.SQLitePreferenceStore` for a durable
        runtime.
    extractor:
        Any accepted deterministic extractor; defaults to the M9
        :class:`~recommendation.memory.extraction.RuleBasedPreferenceExtractor`.  It is
        used unchanged by the accepted service.
    oracle_diagnostic:
        ``True`` only for a run the caller explicitly labels as an oracle diagnostic.
        It is the single gate that lets ``search``/``m5_*`` return
        ``released_oracle_diagnostic`` evidence.  Default ``False`` = strict normal mode.
    clock:
        Callable returning Unix seconds; used only for ingest/snapshot timestamps so
        tests can be deterministic.
    """

    def __init__(
        self,
        *,
        store: Any | None = None,
        extractor: PreferenceExtractor | None = None,
        oracle_diagnostic: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store if store is not None else InMemoryPreferenceStore()
        self._extractor = extractor if extractor is not None else RuleBasedPreferenceExtractor()
        # The accepted M9 service is the only writer; the adapter never builds entries
        # itself, so validation/ids/lifecycle stay exactly the accepted ones.
        self._service = PreferenceMemoryService(self._store, self._extractor)
        self._oracle_diagnostic = bool(oracle_diagnostic)
        self._clock = clock
        self._conversations: dict[str, _ConversationRecord] = {}
        self._m5_store: SQLiteEventStore | None = None
        self._m5_manager: StateManager | None = None
        self._closed = False

    # -- introspection ----------------------------------------------------- #

    @property
    def preference_service(self) -> PreferenceMemoryService:
        """The accepted M9 service this adapter writes through (inspection/tests)."""
        return self._service

    @property
    def store(self) -> Any:
        """The accepted M9 store this adapter writes through."""
        return self._store

    @property
    def oracle_diagnostic(self) -> bool:
        """True when this instance may serve oracle-diagnostic evidence."""
        return self._oracle_diagnostic

    @property
    def conversation_ids(self) -> tuple[str, ...]:
        """Every ingested conversation id, in ingest order."""
        return tuple(self._conversations)

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        return self._closed

    # -- lifecycle --------------------------------------------------------- #

    def __enter__(self) -> "ShoppingCompanionMemory":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the instance-local M5 event store (the M9 store owns its own lifecycle)."""
        if self._m5_store is not None:
            self._m5_store.close()
        self._m5_store = None
        self._m5_manager = None
        self._closed = True

    # -- ingestion --------------------------------------------------------- #

    def ingest_sessions(
        self,
        conversation_id: str,
        sessions: Sequence[Mapping[str, Any]],
        *,
        source_kind: str,
        source_ref: str,
    ) -> dict[str, Any]:
        """Ingest the sessions of one conversation as tagged dialogue evidence.

        ``sessions`` is a list of ``{session_id?, date, turns:[{role, content}]}`` (the
        shape the unreleased official history JSONL would have had, and the shape the
        oracle memory block is normalised into).  Every non-blank turn is stored as
        evidence; only **user-authored** turns are passed to the accepted M9 extractor
        and therefore can produce a preference.

        ``source_kind`` must be one of
        :data:`SOURCE_KIND_ORACLE_DIAGNOSTIC` / :data:`SOURCE_KIND_SYNTHETIC_FIXTURE`.
        ``"history"`` and every other label are refused: the released dataset has no
        real conversation history, so a fabricated history label is never accepted.
        ``source_ref`` names where the sessions came from (e.g. an evaluator-only row
        id or the synthetic fixture path); it is stored verbatim on every evidence item
        for auditability and is not resolved on disk.

        Re-ingesting the same conversation with identical content is an idempotent
        no-op; re-ingesting it with different content raises
        :class:`MemoryIngestError` rather than silently merging evidence.

        Returns a JSON-serialisable summary of what was actually ingested.
        """
        self._require_open()
        conversation_id = self._normalise_conversation_id(conversation_id)
        source_kind = self._validate_source_kind(source_kind)
        source_ref = self._normalise_source_ref(source_ref)
        normalised = self._normalise_sessions(sessions)

        existing = self._conversations.get(conversation_id)
        if existing is not None:
            same = (
                existing.digest == normalised.digest
                and existing.source_kind == source_kind
                and existing.source_ref == source_ref
            )
            if not same:
                raise MemoryIngestError(
                    f"conversation_id {conversation_id!r} was already ingested with different "
                    "content/source; refusing to merge evidence - use a fresh memory instance"
                )
            return self._summary(existing, idempotent=True, dry_state=None)

        # 1. Dry extraction for every user turn with the accepted extractor.  This
        #    happens before any write so a malformed extractor cannot half-ingest.
        extractions: dict[str, PreferenceExtraction] = {}
        for turn in normalised.user_turns:
            extractions[turn.turn_id] = self._extract(turn.text)

        # 2. Build the M5 event batches (pure) with the accepted event producer.
        task_id = _m5_task_id(source_kind)
        events: list[StateEvent] = []
        unmapped_retractions = 0
        for turn in normalised.user_turns:
            try:
                turn_events, turn_unmapped = self._build_m5_events(
                    conversation_id=conversation_id,
                    task_id=task_id,
                    turn=turn,
                    extraction=extractions[turn.turn_id],
                )
            except MemoryIngestError:
                raise
            except ValueError as exc:
                raise MemoryIngestError(
                    f"M5 event construction refused turn {turn.turn_id!r} of "
                    f"{conversation_id!r}: {exc}"
                ) from exc
            events.extend(turn_events)
            unmapped_retractions += turn_unmapped

        # 3. Pure M5 dry run: the accepted reducer must accept the whole stream before
        #    anything is written anywhere.  A rejected stream leaves memory untouched.
        try:
            dry_state = BeliefUpdater().apply_all(
                empty_state(user_key=conversation_id, task_id=task_id), events
            )
        except Exception as exc:  # noqa: BLE001 - normalise reducer failure
            raise MemoryIngestError(
                f"M5 replay rejected the ingested evidence for {conversation_id!r} before "
                f"any write: {exc}"
            ) from exc

        # 4. M9 write path: the accepted service, once per user turn, with a
        #    deterministic turn id and timestamp.
        evidence: list[_TurnEvidence] = []
        added = skipped_duplicates = removal_directives = 0
        for turn in normalised.turns:
            extraction = extractions.get(turn.turn_id)
            if extraction is not None:
                result = self._service.process_turn(
                    user_key=conversation_id,
                    user_message=turn.text,
                    turn_id=turn.turn_id,
                    now=turn.timestamp,
                )
                added += len(result.update.added)
                skipped_duplicates += result.update.skipped_duplicates
                removal_directives += result.update.removal_directives
                extracted = tuple(
                    {
                        "kind": candidate.kind.value,
                        "polarity": candidate.polarity.value,
                        "value": candidate.value,
                    }
                    for candidate in result.extraction.preferences
                )
                removals = tuple(
                    {
                        "kind": removal.kind.value if removal.kind is not None else None,
                        "value": removal.value,
                    }
                    for removal in result.extraction.removals
                )
            else:
                extracted = ()
                removals = ()
            evidence.append(
                _TurnEvidence(
                    conversation_id=conversation_id,
                    session_index=turn.session_index,
                    session_id=turn.session_id,
                    turn_index=turn.turn_index,
                    turn_id=turn.turn_id,
                    date=turn.date,
                    role=turn.role,
                    text=turn.text,
                    source_kind=source_kind,
                    source_ref=source_ref,
                    extracted=extracted,
                    removals=removals,
                )
            )

        # 5. M5 write path: append the validated stream to the instance-local state.
        manager = self._ensure_m5()
        try:
            for event in events:
                manager.apply_event(event)
        except Exception as exc:  # noqa: BLE001 - should be impossible after the dry run
            raise MemoryIngestError(
                f"M5 state rejected the ingested evidence for {conversation_id!r} after the "
                f"dry run accepted it: {exc}"
            ) from exc

        record = _ConversationRecord(
            conversation_id=conversation_id,
            source_kind=source_kind,
            source_ref=source_ref,
            digest=normalised.digest,
            sessions=normalised.sessions,
            evidence=tuple(evidence),
            by_turn_id={item.turn_id: item for item in evidence},
            m5_task_id=task_id,
            m5_session_ids=tuple(dict.fromkeys(item.session_id for item in evidence)),
            m5_event_count=len(events),
            m5_unmapped_retractions=unmapped_retractions,
            ingested_at=float(self._clock()),
            turns_seen=normalised.turns_seen,
            turns_skipped_blank=normalised.turns_skipped_blank,
            preferences_extracted=sum(len(item.extracted) for item in evidence),
            preferences_added=added,
            preferences_skipped_duplicates=skipped_duplicates,
            removal_directives=removal_directives,
            state_version=int(dry_state.state_version),
        )
        self._conversations[conversation_id] = record
        return self._summary(record, idempotent=False, dry_state=dry_state)

    # -- runtime read path ------------------------------------------------- #

    def search(
        self,
        query: str,
        *,
        scope: str = SCOPE_LONG_TERM,
        k: int = 10,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Return long-term preferences that match ``query``, with full provenance.

        The result is the runtime contract::

            {"available": bool, "status": str, "preferences": [...],
             "evidence": [...], "conflicts": [...], "reason": str, ...}

        Honest semantics:

        * a fresh instance returns ``available=False``,
          ``status="history_unavailable"`` and empty lists, because the dataset's
          conversation histories are unreleased;
        * oracle-diagnostic evidence is excluded (and therefore not leaked) unless this
          instance was constructed with ``oracle_diagnostic=True``;
        * matching is deterministic and lexical over the structured preference fields
          (kind/value plus a small fixed price vocabulary); the evidence text is
          provenance, not a query surface.  A query that matches nothing returns an
          empty preference list -- never a fabricated one;
        * every returned preference and evidence item is tagged with ``source_kind``.

        ``scope`` other than ``"long_term"`` (``task``/``session``) returns an honest
        unavailable payload: P14-A has no live task/session history.  An unknown scope
        or a non-positive ``k`` is a programming error and raises :class:`ValueError`.
        ``conversation_id`` optionally restricts the search to one ingested
        conversation (callers that ingest several conversations should use it, or one
        instance per conversation, to avoid cross-conversation recall).
        """
        self._require_open()
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES!r}, got {scope!r}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if conversation_id is not None:
            conversation_id = self._normalise_conversation_id(conversation_id)

        if scope != SCOPE_LONG_TERM:
            return self._unavailable_payload(
                query=query,
                scope=scope,
                k=k,
                reason=(
                    f"scope {scope!r} is not backed by P14-A memory: a P14-A run has no live "
                    "task/session conversation state, and the released long-term history "
                    "corpus is unavailable"
                ),
            )

        if conversation_id is not None and conversation_id not in self._conversations:
            return self._unavailable_payload(
                query=query,
                scope=scope,
                k=k,
                reason=(
                    f"conversation_id {conversation_id!r} was not ingested into this memory "
                    "instance; no evidence is guessed at"
                ),
            )

        records = self._usable_records(conversation_id)
        if not records:
            return self._unavailable_payload(query=query, scope=scope, k=k, reason=None)

        query_tokens = _query_tokens(query)
        candidates: list[tuple[int, int, _ConversationRecord, Any, _TurnEvidence]] = []
        total_active = 0
        unprovenanced = 0
        for order, record in enumerate(records):
            active = self._service.get_active_preferences(record.conversation_id)
            for entry in active.entries:
                if entry.status is not PreferenceStatus.ACTIVE:
                    continue
                total_active += 1
                evidence_item = record.by_turn_id.get(entry.source_turn_id)
                if evidence_item is None:
                    # A preference without traceable evidence is never returned: this
                    # adapter's contract is evidence-first.
                    unprovenanced += 1
                    continue
                score = self._match_score(entry.kind.value, entry.value, query_tokens)
                if score <= 0:
                    continue
                candidates.append((score, order, record, entry, evidence_item))

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[3].logical_seq,
                item[3].memory_id,
            )
        )
        selected = candidates[:k]

        status = self._status_for_records(records)
        preferences = [
            self._preference_payload(record, entry, evidence_item)
            for _score, _order, record, entry, evidence_item in selected
        ]
        evidence_payloads: list[dict[str, Any]] = []
        seen_evidence: set[tuple[str, str]] = set()
        for _score, _order, record, _entry, evidence_item in selected:
            key = (evidence_item.conversation_id, evidence_item.turn_id)
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence_payloads.append(evidence_item.as_dict())
        conflicts = self._conflicts(preferences)

        if preferences:
            reason = (
                f"{len(preferences)} active long-term preference(s) matched the query from "
                f"{status} evidence; every returned preference cites its source turn and "
                "source_kind"
            )
        else:
            reason = (
                f"no active long-term preference matched the query; {total_active} active "
                f"preference(s) are stored from {status} evidence and none is invented"
            )
        if unprovenanced:
            reason += (
                f"; {unprovenanced} stored preference(s) had no traceable evidence and were "
                "excluded"
            )

        payload = {
            "available": True,
            "status": status,
            "preferences": preferences,
            "evidence": evidence_payloads,
            "conflicts": conflicts,
            "reason": reason,
            "scope": scope,
            "query": query,
            "k": k,
            "oracle_diagnostic": self._oracle_diagnostic,
            "source_kinds": sorted({record.source_kind for record in records}),
            "conversation_ids": [record.conversation_id for record in records],
            "active_preference_count": total_active,
            "returned_preference_count": len(preferences),
            "unprovenanced_preference_count": unprovenanced,
        }
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        return payload

    def history_status(self) -> dict[str, Any]:
        """Report whether usable long-term history exists, and why (or why not).

        This is the runtime protocol's second method and is stored verbatim in the
        P14-A trajectory.  A normal run gets ``available=False``,
        ``status="history_unavailable"`` and a reason naming the unreleased corpus; an
        explicitly labelled oracle diagnostic / contract fixture gets its own status so
        the run can never be reported as a normal success.
        """
        self._require_open()
        records = list(self._conversations.values())
        usable = self._usable_records(None)
        oracle_present = any(
            record.source_kind == SOURCE_KIND_ORACLE_DIAGNOSTIC for record in records
        )
        synthetic_present = any(
            record.source_kind == SOURCE_KIND_SYNTHETIC_FIXTURE for record in records
        )
        available = bool(usable)
        if available:
            status = self._status_for_records(usable)
            if status == STATUS_ORACLE_DIAGNOSTIC:
                reason = (
                    "oracle-diagnostic dialogue evidence is ingested and served because this "
                    "instance is in explicit oracle_diagnostic mode; it is diagnostic evidence "
                    "from the released oracle memory block, not recovered user history, and is "
                    "never counted as a normal Agent success"
                )
            else:
                reason = (
                    "synthetic_fixture dialogue evidence is ingested; it is contract-test data "
                    "only, never released conversation history, and never a benchmark result"
                )
        elif oracle_present:
            status = STATUS_HISTORY_UNAVAILABLE
            reason = f"{RELEASED_HISTORY_REASON}. {ORACLE_REFUSED_REASON}"
        else:
            status = STATUS_HISTORY_UNAVAILABLE
            reason = RELEASED_HISTORY_REASON

        return {
            "schema": MEMORY_ADAPTER_SCHEMA,
            "available": available,
            "history_available": available,
            "status": status,
            "reason": reason,
            "released_history_present": False,
            "released_history_reason": RELEASED_HISTORY_REASON,
            "oracle_diagnostic_mode": self._oracle_diagnostic,
            "conversations": len(records),
            "sessions": sum(len(record.sessions) for record in records),
            "evidence_turns": sum(len(record.evidence) for record in records),
            "user_turns": sum(record.user_turns for record in records),
            "active_preferences": self._active_preference_count(usable),
            "oracle_diagnostic_evidence_present": oracle_present,
            "synthetic_evidence_present": synthetic_present,
            "source_kinds": sorted({record.source_kind for record in records}),
        }

    # -- diagnostics / audit ----------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """Return the complete adapter state as a JSON-serialisable audit snapshot.

        This is a local audit view (it includes every stored evidence item, still
        tagged with ``source_kind``).  The runtime read paths are :meth:`search` and
        :meth:`history_status`; a strict instance never serves oracle evidence through
        them, and an M5 read of an oracle conversation is refused by
        :meth:`m5_state`.
        """
        self._require_open()
        records = list(self._conversations.values())
        preferences: list[dict[str, Any]] = []
        for record in records:
            active = self._service.get_active_preferences(record.conversation_id)
            for entry in active.entries:
                evidence_item = record.by_turn_id.get(entry.source_turn_id)
                preferences.append(
                    self._preference_payload(record, entry, evidence_item, include_evidence=True)
                )
        return {
            "schema": MEMORY_ADAPTER_SCHEMA,
            "generated_at": float(self._clock()),
            "oracle_diagnostic_mode": self._oracle_diagnostic,
            "released_history_present": False,
            "history": self.history_status(),
            "conversations": [
                {
                    "conversation_id": record.conversation_id,
                    "source_kind": record.source_kind,
                    "source_ref": record.source_ref,
                    "digest": record.digest,
                    "sessions": [dict(session) for session in record.sessions],
                    "turns_seen": record.turns_seen,
                    "turns_skipped_blank": record.turns_skipped_blank,
                    "user_turns": record.user_turns,
                    "evidence_turns": len(record.evidence),
                    "preferences_extracted": record.preferences_extracted,
                    "preferences_added": record.preferences_added,
                    "preferences_skipped_duplicates": record.preferences_skipped_duplicates,
                    "removal_directives": record.removal_directives,
                    "active_preferences": len(
                        self._service.get_active_preferences(record.conversation_id).active_entries
                    ),
                    "ingested_at": record.ingested_at,
                    "m5_task_id": record.m5_task_id,
                    "m5_session_ids": list(record.m5_session_ids),
                    "m5_event_count": record.m5_event_count,
                    "m5_unmapped_retractions": record.m5_unmapped_retractions,
                }
                for record in records
            ],
            "preferences": preferences,
            "evidence": [
                item.as_dict() for record in records for item in record.evidence
            ],
            "m5": {
                "task_ids": [record.m5_task_id for record in records],
                "event_count": sum(record.m5_event_count for record in records),
                "unmapped_retractions": sum(record.m5_unmapped_retractions for record in records),
            },
        }

    # -- M5 read path ------------------------------------------------------ #

    def m5_task_id(self, conversation_id: str) -> str:
        """Return the M5 task namespace of an ingested conversation."""
        record = self._record(conversation_id)
        return record.m5_task_id

    def m5_state(self, conversation_id: str) -> UnifiedDecisionState:
        """Return the M5 state replayed from the conversation's ingested user turns.

        Oracle-diagnostic conversations are refused unless the instance is in explicit
        ``oracle_diagnostic`` mode, mirroring the strict normal-mode gate of
        :meth:`search`.
        """
        record = self._record(conversation_id)
        self._require_oracle_allowed(record)
        manager = self._ensure_m5()
        return manager.replay_state(record.conversation_id, record.m5_task_id)

    def m5_agent_projection(self, conversation_id: str) -> AgentStateProjection:
        """Return ``state_projector.for_agent`` of the conversation's replayed M5 state.

        This is the M5 event-replay read path: the projection is the accepted one, so
        the loaded preferences are M5 ``task``-scope explicit beliefs (see the module
        docstring for why task scope is the accepted M5.1 behaviour and where the
        long-term read lives).
        """
        return for_agent(self.m5_state(conversation_id))

    def m5_long_term_projection(self, conversation_id: str) -> AgentStateProjection:
        """Return ``for_agent`` of the accepted M9 -> M5 **long-term** belief bridge.

        This is the second, pure-read M5 path: the accepted
        :func:`recommendation.state.legacy_adapter.project_legacy_memory` projects the
        conversation's ACTIVE M9 entries as ``LONG_TERM``/``EXPLICIT``/``LEGACY_MEMORY``
        beliefs and :func:`recommendation.state.legacy_adapter.merge_legacy_memory`
        merges them into an empty task state.  Nothing is written to the M9 store and no
        M5 events are appended, so the event-replay state and this projection stay
        independent views of the same accepted memory rather than one state pretending
        to be both.
        """
        record = self._record(conversation_id)
        self._require_oracle_allowed(record)
        snapshot = self._service.get_active_preferences(record.conversation_id)
        legacy = project_legacy_memory(snapshot, task_id=record.m5_task_id)
        state = merge_legacy_memory(
            empty_state(user_key=record.conversation_id, task_id=record.m5_task_id),
            legacy,
        )
        return for_agent(state)

    # -- internal: modes and gating ---------------------------------------- #

    def _usable_records(self, conversation_id: str | None) -> tuple[_ConversationRecord, ...]:
        """Conversations whose evidence the current instance mode may serve."""
        allowed = SOURCE_KINDS if self._oracle_diagnostic else NORMAL_MODE_SOURCE_KINDS
        records: list[_ConversationRecord] = []
        for record in self._conversations.values():
            if conversation_id is not None and record.conversation_id != conversation_id:
                continue
            if record.source_kind not in allowed:
                continue
            records.append(record)
        return tuple(records)

    @staticmethod
    def _status_for_records(records: Sequence[_ConversationRecord]) -> str:
        """The single status of a non-empty set of usable conversations."""
        if any(record.source_kind == SOURCE_KIND_ORACLE_DIAGNOSTIC for record in records):
            return STATUS_ORACLE_DIAGNOSTIC
        return STATUS_SYNTHETIC_FIXTURE

    def _active_preference_count(self, records: Sequence[_ConversationRecord]) -> int:
        """Total ACTIVE M9 entries across the given conversations."""
        return sum(
            self._service.get_active_preferences(record.conversation_id).active_count
            for record in records
        )

    def _unavailable_payload(
        self,
        *,
        query: str,
        scope: str,
        k: int,
        reason: str | None,
    ) -> dict[str, Any]:
        """The honest ``available=False`` search payload."""
        records = list(self._conversations.values())
        oracle_present = any(
            record.source_kind == SOURCE_KIND_ORACLE_DIAGNOSTIC for record in records
        )
        if reason is None:
            reason = RELEASED_HISTORY_REASON
            if oracle_present:
                reason = f"{RELEASED_HISTORY_REASON}. {ORACLE_REFUSED_REASON}"
        return {
            "available": False,
            "status": STATUS_HISTORY_UNAVAILABLE,
            "preferences": [],
            "evidence": [],
            "conflicts": [],
            "reason": reason,
            "scope": scope,
            "query": query,
            "k": k,
            "oracle_diagnostic": self._oracle_diagnostic,
            "source_kinds": sorted({record.source_kind for record in records}),
            "conversation_ids": [record.conversation_id for record in records],
            "active_preference_count": 0,
            "returned_preference_count": 0,
            "oracle_diagnostic_evidence_present": oracle_present,
        }

    def _record(self, conversation_id: str) -> _ConversationRecord:
        """Return the record for one conversation, or raise ``KeyError``."""
        self._require_open()
        key = self._normalise_conversation_id(conversation_id)
        record = self._conversations.get(key)
        if record is None:
            raise KeyError(f"conversation_id {key!r} was not ingested into this memory instance")
        return record

    def _require_oracle_allowed(self, record: _ConversationRecord) -> None:
        """Refuse oracle evidence on a strict instance (the M5 read path gate)."""
        if record.source_kind == SOURCE_KIND_ORACLE_DIAGNOSTIC and not self._oracle_diagnostic:
            raise OracleDiagnosticRefused(
                f"conversation {record.conversation_id!r} carries oracle-diagnostic evidence; "
                + ORACLE_REFUSED_REASON
            )

    def _require_open(self) -> None:
        """Refuse use after :meth:`close`."""
        if self._closed:
            raise RuntimeError("this ShoppingCompanionMemory instance is closed")

    # -- internal: validation/normalisation -------------------------------- #

    @staticmethod
    def _normalise_conversation_id(value: object) -> str:
        """Validate the conversation (M9 user key / evidence namespace) id."""
        try:
            return bounded_identifier(
                value, field="conversation_id", max_length=MAX_USER_KEY_LENGTH
            )
        except ValueError as exc:
            raise MemoryIngestError(str(exc)) from exc

    @staticmethod
    def _validate_source_kind(source_kind: object) -> str:
        """Accept only the two legal tags; refuse ``history`` explicitly by name."""
        if not isinstance(source_kind, str) or not source_kind.strip():
            raise MemoryIngestError(
                "source_kind is required and must be one of "
                f"{SOURCE_KINDS!r} (an explicit tag is the oracle-diagnostic gate)"
            )
        resolved = source_kind.strip()
        if resolved == "history":
            raise MemoryIngestError(
                "source_kind 'history' is refused: the released Shopping Companion dataset "
                "does not include the conversation-history corpus, so no real history may be "
                "claimed; use 'released_oracle_diagnostic' only for an explicit oracle "
                "diagnostic, or 'synthetic_fixture' only for a clearly synthetic contract test"
            )
        if resolved not in SOURCE_KINDS:
            raise MemoryIngestError(
                f"unsupported source_kind {resolved!r}; allowed values are {SOURCE_KINDS!r}"
            )
        return resolved

    @staticmethod
    def _normalise_source_ref(source_ref: object) -> str:
        """Require a non-blank provenance reference (never resolved on disk)."""
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise MemoryIngestError(
                "source_ref is required and must be a non-blank provenance reference "
                "(e.g. a row id or fixture path); it is stored on every evidence item"
            )
        return " ".join(source_ref.split())

    @staticmethod
    def _normalise_sessions(sessions: object) -> _NormalisedSessions:
        """Validate and normalise the caller's sessions; skip only blank turn text.

        Structural problems (not a list, a session without ``turns``, a turn that is not
        a mapping, a non-string role/content, a missing/invalid date) are hard errors:
        the adapter refuses garbage rather than turning it into evidence.  A turn whose
        text is blank is skipped and counted -- it is empty evidence, not a preference.
        """
        if isinstance(sessions, (str, bytes)) or not isinstance(sessions, Sequence):
            raise MemoryIngestError(
                "sessions must be a list of {session_id?, date, turns:[{role, content}]}"
            )
        if not sessions:
            raise MemoryIngestError(
                "sessions must not be empty: there is no evidence to ingest and none is invented"
            )

        summaries: list[dict[str, Any]] = []
        turns: list[_PendingTurn] = []
        turns_seen = 0
        skipped_blank = 0
        for session_index, session in enumerate(sessions):
            if not isinstance(session, Mapping):
                raise MemoryIngestError(f"sessions[{session_index}] must be a mapping")
            raw_turns = session.get("turns")
            if (
                isinstance(raw_turns, (str, bytes))
                or not isinstance(raw_turns, Sequence)
                or not raw_turns
            ):
                raise MemoryIngestError(
                    f"sessions[{session_index}] must carry a non-empty 'turns' list"
                )
            date_text, timestamp = _parse_session_date(session.get("date"), session_index)
            session_id = _normalise_session_id(session.get("session_id"), session_index)
            session_turn_count = 0
            for turn_index, turn in enumerate(raw_turns):
                if not isinstance(turn, Mapping):
                    raise MemoryIngestError(
                        f"sessions[{session_index}].turns[{turn_index}] must be a mapping"
                    )
                role = turn.get("role")
                content = turn.get("content")
                if not isinstance(role, str) or not role.strip():
                    raise MemoryIngestError(
                        f"sessions[{session_index}].turns[{turn_index}] must carry a non-blank "
                        "string 'role'"
                    )
                normalised_role = role.strip().lower()
                if len(normalised_role) > 64:
                    raise MemoryIngestError(
                        f"sessions[{session_index}].turns[{turn_index}] role is too long"
                    )
                if not isinstance(content, str):
                    raise MemoryIngestError(
                        f"sessions[{session_index}].turns[{turn_index}] must carry a string "
                        "'content' (missing/blank evidence is handled separately)"
                    )
                text = content.strip()
                if not text:
                    skipped_blank += 1
                    turns_seen += 1
                    continue
                turns_seen += 1
                session_turn_count += 1
                turns.append(
                    _PendingTurn(
                        session_index=session_index,
                        session_id=session_id,
                        turn_index=turn_index,
                        turn_id=f"s{session_index}t{turn_index}",
                        date=date_text,
                        timestamp=timestamp + float(turn_index),
                        role=normalised_role,
                        text=text,
                    )
                )
            summaries.append(
                {
                    "session_id": session_id,
                    "date": date_text,
                    "turn_count": session_turn_count,
                }
            )
        if not turns:
            raise MemoryIngestError(
                "sessions contained no non-blank turns; refusing to ingest empty evidence "
                "(the adapter never fabricates a dialogue)"
            )
        user_turns = tuple(turn for turn in turns if turn.is_user_turn)
        digest_payload = {
            "sessions": [
                {
                    "session_id": summary["session_id"],
                    "date": summary["date"],
                    "turns": [
                        {"role": turn.role, "content": turn.text}
                        for turn in turns
                        if turn.session_index == session_index
                    ],
                }
                for session_index, summary in enumerate(summaries)
            ]
        }
        digest = hashlib.sha256(_canonical_json(digest_payload).encode("utf-8")).hexdigest()
        return _NormalisedSessions(
            sessions=tuple(summaries),
            turns=tuple(turns),
            user_turns=user_turns,
            digest=digest,
            turns_seen=turns_seen,
            turns_skipped_blank=skipped_blank,
        )

    # -- internal: extraction / M5 event construction ---------------------- #

    def _extract(self, text: str) -> PreferenceExtraction:
        """Call the accepted extractor with the M9 extraction contract enforced."""
        try:
            extraction = self._extractor.extract(text)
        except PreferenceExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise collaborator failure
            raise PreferenceExtractionError(
                f"preference extractor failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(extraction, PreferenceExtraction):
            raise PreferenceExtractionError(
                "extractor must return a PreferenceExtraction, got "
                f"{type(extraction).__name__}"
            )
        return extraction

    def _build_m5_events(
        self,
        *,
        conversation_id: str,
        task_id: str,
        turn: _PendingTurn,
        extraction: PreferenceExtraction,
    ) -> tuple[tuple[StateEvent, ...], int]:
        """Build the accepted M5 events for one user turn.

        Returns the events and the number of value-only M9 retractions that could not be
        represented as M5 retractions (the documented M5 gap).
        """
        preferences = tuple(
            ExplicitPreference(
                kind=candidate.kind.value,
                value=candidate.value,
                polarity=candidate.polarity.value,
                source_text=candidate.source_text,
            )
            for candidate in extraction.preferences
        )
        retractions: list[ExplicitRetraction] = []
        unmapped = 0
        for removal in extraction.removals:
            if removal.kind is None:
                # Value-only retraction: the accepted M5 payload needs a resolved kind,
                # and resolving it needs a live M5 state (resolve_retraction).  The M9
                # store has already applied it; M5 omits it and the summary reports it.
                unmapped += 1
                continue
            retractions.append(
                ExplicitRetraction(
                    kind=removal.kind.value,
                    value=removal.value,
                    source_text=removal.source_text,
                )
            )
        identity = TurnIdentity(
            user_key=conversation_id,
            session_id=turn.session_id,
            task_id=task_id,
            turn_key=turn.turn_id,
        )
        events = build_turn_events(
            identity=identity,
            message=turn.text,
            intent_label=M5_INTENT_LABEL,
            preferences=preferences,
            retractions=tuple(retractions),
            timestamp=turn.timestamp,
        )
        return events, unmapped

    def _ensure_m5(self) -> StateManager:
        """Create the instance-local ephemeral M5 event store/manager on first use."""
        if self._m5_manager is None:
            self._m5_store = SQLiteEventStore(":memory:")
            self._m5_manager = StateManager(self._m5_store)
        return self._m5_manager

    # -- internal: search helpers ------------------------------------------ #

    @staticmethod
    def _match_score(kind: str, value: str, query_tokens: Sequence[str]) -> int:
        """Deterministic lexical match score between query tokens and one preference.

        Exact token overlap over the preference kind/value plus a small fixed price
        vocabulary; a ``len >= 4`` prefix match adds tolerance for plurals
        ("colors" -> "color").  Evidence text is deliberately not a query surface: a
        turn may mention several things, and only the structured preference names what
        the user actually asked to remember.
        """
        if not query_tokens:
            return 0
        haystack = set(_tokenize(kind))
        haystack.update(_tokenize(value))
        haystack.update(_KIND_SEARCH_TERMS.get(kind, ()))
        score = sum(1 for token in query_tokens if token in haystack)
        if score:
            return score
        for token in query_tokens:
            if len(token) < 4:
                continue
            if any(
                len(candidate) >= 4 and (candidate.startswith(token) or token.startswith(candidate))
                for candidate in haystack
            ):
                return 1
        return 0

    def _preference_payload(
        self,
        record: _ConversationRecord,
        entry: Any,
        evidence_item: _TurnEvidence | None,
        *,
        include_evidence: bool = False,
    ) -> dict[str, Any]:
        """Build one ``preferences`` result item (always carries its provenance tags)."""
        payload: dict[str, Any] = {
            "kind": entry.kind.value,
            "polarity": entry.polarity.value,
            "value": entry.value,
            "status": entry.status.value,
            "source": record.source_kind,
            "source_kind": record.source_kind,
            "conversation_id": record.conversation_id,
            "memory_id": entry.memory_id,
            "logical_seq": int(entry.logical_seq),
            "source_turn_id": entry.source_turn_id,
            "source_text": entry.source_text,
        }
        if evidence_item is not None:
            payload["evidence_turn_id"] = evidence_item.turn_id
            payload["evidence"] = evidence_item.as_dict()
        if include_evidence and evidence_item is not None:
            payload["date"] = evidence_item.date
            payload["role"] = evidence_item.role
        return payload

    @staticmethod
    def _conflicts(preferences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Report genuine contradictions among the returned preferences.

        A conflict is one ``(kind, value)`` stated with **both** prefer and avoid
        polarity -- the one case that is unambiguously contradictory.  Two independent
        values of the same kind are *not* a conflict (the accepted M9 semantics keeps
        them as independent constraints), so they are deliberately not reported.
        """
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for preference in preferences:
            key = (str(preference.get("kind")), str(preference.get("value")).casefold())
            groups.setdefault(key, []).append(preference)
        conflicts: list[dict[str, Any]] = []
        for (kind, value_key), items in sorted(groups.items(), key=lambda item: item[0]):
            polarities = {str(item.get("polarity")) for item in items}
            if polarities != {"prefer", "avoid"}:
                continue
            conflicts.append(
                {
                    "kind": kind,
                    "value": next(
                        str(item.get("value"))
                        for item in items
                        if str(item.get("value")).casefold() == value_key
                    ),
                    "polarities": sorted(polarities),
                    "memory_ids": sorted(str(item.get("memory_id")) for item in items),
                    "conversation_ids": sorted(
                        {str(item.get("conversation_id")) for item in items}
                    ),
                    "reason": (
                        "the same kind and value are stated with both prefer and avoid "
                        "polarity; both remain ACTIVE in the accepted M9 memory"
                    ),
                }
            )
        return conflicts

    # -- internal: summaries ----------------------------------------------- #

    def _summary(
        self,
        record: _ConversationRecord,
        *,
        idempotent: bool,
        dry_state: UnifiedDecisionState | None,
    ) -> dict[str, Any]:
        """Build the JSON-serialisable ingest summary for one conversation."""
        note = (
            "synthetic fixture evidence is contract-test data only; it is not released "
            "conversation history and never a benchmark result"
            if record.source_kind == SOURCE_KIND_SYNTHETIC_FIXTURE
            else "oracle-diagnostic evidence is for an explicitly labelled oracle diagnostic "
            "only; it is not recovered user history and is never a normal Agent success"
        )
        if dry_state is None:
            # Idempotent re-ingest: report the already-replayed state rather than a guess.
            replayed = self._ensure_m5().replay_state(record.conversation_id, record.m5_task_id)
            active_beliefs = len(replayed.active_preference_beliefs)
        else:
            active_beliefs = len(dry_state.active_preference_beliefs)
        payload: dict[str, Any] = {
            "schema": MEMORY_ADAPTER_SCHEMA,
            "conversation_id": record.conversation_id,
            "source_kind": record.source_kind,
            "source_ref": record.source_ref,
            "status": (
                STATUS_SYNTHETIC_FIXTURE
                if record.source_kind == SOURCE_KIND_SYNTHETIC_FIXTURE
                else STATUS_ORACLE_DIAGNOSTIC
            ),
            "oracle_diagnostic": record.source_kind == SOURCE_KIND_ORACLE_DIAGNOSTIC,
            "usable_evidence": True,
            "released_history_present": False,
            "idempotent": idempotent,
            "digest": record.digest,
            "sessions_ingested": len(record.sessions),
            "turns_seen": record.turns_seen,
            "turns_skipped_blank": record.turns_skipped_blank,
            "evidence_turns": len(record.evidence),
            "user_turns": record.user_turns,
            "preferences_extracted": record.preferences_extracted,
            "preferences_added": 0 if idempotent else record.preferences_added,
            "preferences_skipped_duplicates": record.preferences_skipped_duplicates,
            "removal_directives": record.removal_directives,
            "active_preferences": len(
                self._service.get_active_preferences(record.conversation_id).active_entries
            ),
            "m5": {
                "task_id": record.m5_task_id,
                "event_count": record.m5_event_count,
                "state_version": record.state_version,
                "active_preference_count": active_beliefs,
                "unmapped_value_only_retractions": record.m5_unmapped_retractions,
            },
            "ingested_at": record.ingested_at,
            "note": note,
        }
        return payload
