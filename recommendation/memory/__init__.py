"""Preference memory (Milestone 9).

Adds a trustworthy conversational memory layer **without touching the recommendation
trust boundary**::

    trusted interaction history        (application-owned, unchanged)
            -> RecommendationTool -> SASRec          [M7A/M6, untouched]

    explicit user statement
            -> injected extractor
            -> validated PreferenceCandidate
            -> PreferenceMemoryService -> store      [this package]

The two domains are separate types in separate packages:

* **trusted interaction memory** is the chronological ``parent_asin`` history SASRec
  consumes.  It stays in application/graph input state and reaches the Tool through
  the accepted :class:`~recommendation.tools.schemas.RecommendationContext`.  This
  package has no field, method or table for it, so conversational text can never
  become a behavioural event and the Agent can never mutate the history.
* **preference memory** is explicit, typed, user-scoped, attributable and mutable
  conversational state.

Guarantees:

* preferences are stored only from explicit user-authored statements -- never inferred
  from SASRec history, candidates, catalogue metadata, RAG evidence or model scores;
* every entry keeps provenance (source turn + exact source span);
* lifecycle is deterministic and **explicit**: a statement ADDs an independent
  constraint, an explicitly corrective statement REPLACEs the value it corrects, and a
  retraction REMOVEs matching constraints.  Replacement is never inferred from the
  preference kind, so "I don't want red" followed by "I don't want blue" keeps both;
  the audit trail is always preserved;
* users are isolated by an explicit key; there is no process-global store;
* extraction and storage are offline and injected, so no provider API is required;
* **preference memory does not modify recommendation candidate order** -- identity,
  count, rank and raw SASRec score are untouched.  Preference-aware reranking is
  Milestone 10.

Persistence uses the Python standard library: an in-memory store for tests and a
versioned SQLite store for runtime.
"""

from __future__ import annotations

from .extraction import (
    EXTRACTOR_NAME,
    RuleBasedPreferenceExtractor,
    ScriptedPreferenceExtractor,
)
from .schemas import (
    MEMORY_SCHEMA_VERSION,
    PreferenceCandidate,
    PreferenceExtraction,
    PreferenceExtractionError,
    PreferenceExtractor,
    PreferenceKind,
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
    PreferenceMode,
    PreferencePolarity,
    PreferenceRemoval,
    PreferenceStatus,
    contains_secret_like_text,
)
from .service import (
    MemoryUpdateSummary,
    PreferenceMemoryService,
    TurnMemoryResult,
)
from .store import (
    MEMORY_DB_ENV_VAR,
    InMemoryPreferenceStore,
    PreferenceStore,
    SQLitePreferenceStore,
    deterministic_memory_id,
)

__all__ = [
    "EXTRACTOR_NAME",
    "MEMORY_DB_ENV_VAR",
    "MEMORY_SCHEMA_VERSION",
    "InMemoryPreferenceStore",
    "MemoryUpdateSummary",
    "PreferenceCandidate",
    "PreferenceExtraction",
    "PreferenceExtractionError",
    "PreferenceExtractor",
    "PreferenceKind",
    "PreferenceMemoryEntry",
    "PreferenceMemoryService",
    "PreferenceMemorySnapshot",
    "PreferenceMode",
    "PreferencePolarity",
    "PreferenceRemoval",
    "PreferenceStatus",
    "PreferenceStore",
    "RuleBasedPreferenceExtractor",
    "SQLitePreferenceStore",
    "ScriptedPreferenceExtractor",
    "TurnMemoryResult",
    "contains_secret_like_text",
    "deterministic_memory_id",
]

__version__ = "0.1.0"
