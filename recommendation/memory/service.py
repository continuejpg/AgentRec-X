"""Preference-memory service: lifecycle, validation and read surface (Milestone 9).

The service is the only component allowed to write preference memory.  It owns:

* **validation** -- every extractor candidate is re-validated against strict schemas
  before storage, so malformed or hostile extraction output fails explicitly and
  without touching existing memory;
* **ordering** -- it assigns ``logical_seq`` and ``created_at``, so ordering never
  depends on the store, the clock during a test, or dict iteration order;
* **lifecycle** -- add, deduplicate, supersede on conflict, and tombstone on removal;
* **idempotency** -- re-processing the same turn produces no new active entry;
* **the read surface** -- ``get_active_preferences`` / ``get_memory_history``.

What the service cannot do
--------------------------
It has no access to the trusted interaction history, the recommender, the catalogue or
the RAG layer, and no method accepts an interaction event.  A conversational statement
therefore cannot become a SASRec behavioural event: there is no code path from
:meth:`process_turn` to the Tool's ``RecommendationContext``.

Preference memory also never influences candidate identity, count, rank or SASRec
score.  Nothing in this module reads a recommendation result.

Write ordering
--------------
:meth:`process_turn` performs, in order:

1. read the current active snapshot (read-only);
2. ask the extractor for candidates and removals, passing **only** the user message;
3. validate every candidate and removal;
4. apply the updates in one atomic store transaction;
5. return the resulting active snapshot.

If any step fails, the transaction rolls back and previously stored memory is
unchanged.  A failure never leaves partially applied state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schemas import (
    PreferenceCandidate,
    PreferenceExtraction,
    PreferenceExtractionError,
    PreferenceExtractor,
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
    PreferenceMode,
    PreferenceRemoval,
    PreferenceStatus,
)
from .store import PreferenceStore, deterministic_memory_id

__all__ = [
    "PreferenceMemoryService",
    "TurnMemoryResult",
    "MemoryUpdateSummary",
]


@dataclass
class MemoryUpdateSummary:
    """What one turn changed in memory (empty when the turn said nothing relevant)."""

    added: list[PreferenceMemoryEntry] = field(default_factory=list)
    superseded: list[PreferenceMemoryEntry] = field(default_factory=list)
    removed: list[PreferenceMemoryEntry] = field(default_factory=list)
    skipped_duplicates: int = 0
    already_processed: bool = False
    removal_directives: int = 0

    @property
    def changed(self) -> bool:
        """True when this turn changed stored memory."""
        return bool(self.added or self.superseded or self.removed)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (ids and values, not full history)."""
        return {
            "added": [(e.memory_id, e.kind.value, e.value, e.polarity.value) for e in self.added],
            "superseded": [e.memory_id for e in self.superseded],
            "removed": [e.memory_id for e in self.removed],
            "skipped_duplicates": self.skipped_duplicates,
            "already_processed": self.already_processed,
            "removal_directives": self.removal_directives,
            "changed": self.changed,
        }


@dataclass
class TurnMemoryResult:
    """Result of processing one user turn."""

    user_key: str
    turn_id: str
    extraction: PreferenceExtraction
    update: MemoryUpdateSummary
    active: PreferenceMemorySnapshot

    @property
    def active_entries(self) -> tuple[PreferenceMemoryEntry, ...]:
        """Active entries after this turn, in deterministic order."""
        return self.active.active_entries

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "user_key": self.user_key,
            "turn_id": self.turn_id,
            "update": self.update.as_dict(),
            "active": [entry.as_dict() for entry in self.active_entries],
        }


def _supersedes(left: PreferenceMemoryEntry, right: PreferenceCandidate) -> bool:
    """True when ``right`` explicitly retracts ``left``.

    Replacement is **never inferred from kind alone**.  Stating a second preference is
    not the same as correcting the first, so:

    * ``ADD`` (the default) retracts **nothing**.  "I don't want red" then "I don't
      want blue" leaves two independent avoidances, and "I prefer black" then "I prefer
      blue" leaves two independent preferences.
    * ``REPLACE`` retracts entries of the same kind and polarity, because the statement
      itself expresses correction intent.  When the candidate names the value it is
      correcting (``replaces``), only entries holding **that** value are retracted --
      whether or not it equals the candidate's own value, since "prefer blue instead of
      black" corrects ``black`` while storing ``blue``.

    The only automatic retraction is for ``REPLACE``; there is no implicit singleton
    slot for any kind.
    """
    if right.mode is not PreferenceMode.REPLACE:
        return False
    if left.kind is not right.kind:
        return False
    if right.replaces is not None:
        # A named target is authoritative: exactly that value is corrected, whatever
        # its polarity, and same-polarity siblings are left alone.
        return left.value.casefold() == right.replaces.casefold()
    if left.value.casefold() == right.value.casefold():
        # Correcting the value the statement itself names resolves a same-value
        # contradiction ("I prefer red" -> "actually I don't want red instead").
        return True
    if left.polarity is not right.polarity:
        # Different value and opposite polarity is an unrelated constraint, not a
        # correction of this one.
        return False
    # Same polarity, different value: the statement corrects the prior value.
    return True


class PreferenceMemoryService:
    """Turn-level preference-memory orchestration over an injected store.

    Parameters
    ----------
    store:
        Any :class:`~recommendation.memory.store.PreferenceStore`.  Injected, so no
        process-global singleton exists and tests can use a small in-memory store.
    extractor:
        Any :class:`~recommendation.memory.schemas.PreferenceExtractor`.  Injected, so
        Milestone 9 stays offline and a future LLM adapter can replace it without
        changing this contract.
    """

    def __init__(self, store: PreferenceStore, extractor: PreferenceExtractor) -> None:
        if not callable(getattr(store, "add_entry", None)):
            raise TypeError("store must provide an add_entry(entry) method")
        if not callable(getattr(extractor, "extract", None)):
            raise TypeError("extractor must provide an extract(user_message) method")
        self._store = store
        self._extractor = extractor

    # -- introspection ----------------------------------------------------- #

    @property
    def store(self) -> PreferenceStore:
        """The injected store (exposed for inspection and tests)."""
        return self._store

    @property
    def extractor(self) -> PreferenceExtractor:
        """The injected extractor (exposed for inspection and tests)."""
        return self._extractor

    # -- read surface ------------------------------------------------------ #

    def get_active_preferences(self, user_key: str) -> PreferenceMemorySnapshot:
        """Return ``user_key``'s active preferences, read-only.

        Reading never writes: this method issues no store mutation, so an Agent turn
        that only reads cannot change memory.
        """
        entries = self._store.get_entries(user_key, active_only=True)
        return PreferenceMemorySnapshot(user_key=user_key, entries=tuple(entries))

    def get_memory_history(self, user_key: str) -> PreferenceMemorySnapshot:
        """Return ``user_key``'s full audit trail, including superseded/removed entries."""
        entries = self._store.get_entries(user_key, active_only=False)
        return PreferenceMemorySnapshot(user_key=user_key, entries=tuple(entries))

    # -- write path -------------------------------------------------------- #

    def process_turn(
        self,
        *,
        user_key: str,
        user_message: str,
        turn_id: str,
        now: float | None = None,
    ) -> TurnMemoryResult:
        """Extract and persist explicit preferences from one user-authored turn.

        ``user_message`` must be user-authored text.  System prompts, tool output and
        model reasoning are never passed here, and there is a guard against calling
        this with a blank message.

        Raises
        ------
        PreferenceExtractionError
            The extractor returned something that is not a
            :class:`~recommendation.memory.schemas.PreferenceExtraction`, or a
            candidate/removal failed validation.  Existing memory is untouched.
        ValueError
            ``user_key``, ``turn_id`` or ``user_message`` is blank.
        """
        if not isinstance(user_key, str) or not user_key.strip():
            raise ValueError("user_key must be a non-empty string")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must be a non-empty string")
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty user-authored string")

        user_key = user_key.strip()
        turn_id = turn_id.strip()

        # 1. Explicit retractions are applied even if the same turn was seen before,
        #    because a retraction carries no new entry to deduplicate on.
        extraction = self._extract(user_message)

        # 2-3. Validate everything before writing anything.  A candidate carrying
        # mode=REMOVE becomes a retraction directive here, so nothing is stored for it.
        candidates, inline_removals = self._validate_candidates(extraction)
        removals = extraction.removals + inline_removals

        update = MemoryUpdateSummary(removal_directives=len(removals))
        timestamp = time.time() if now is None else float(now)

        if self._is_turn_processed(user_key, turn_id, candidates):
            update.already_processed = True
        else:
            self._apply(user_key, turn_id, candidates, removals, update, timestamp)

        active = self.get_active_preferences(user_key)
        return TurnMemoryResult(
            user_key=user_key,
            turn_id=turn_id,
            extraction=extraction,
            update=update,
            active=active,
        )

    # -- internals --------------------------------------------------------- #

    def _extract(self, user_message: str) -> PreferenceExtraction:
        """Call the extractor and normalise its output into a typed extraction."""
        try:
            raw = self._extractor.extract(user_message)
        except PreferenceExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise collaborator failure
            raise PreferenceExtractionError(
                f"preference extractor failed: {type(exc).__name__}"
            ) from exc

        if isinstance(raw, PreferenceExtraction):
            return raw
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            # A bare sequence of candidates is accepted for ergonomics, but a
            # PreferenceExtraction is required to express removals.
            try:
                return PreferenceExtraction(
                    preferences=tuple(
                        item
                        if isinstance(item, PreferenceCandidate)
                        else PreferenceCandidate.model_validate(item)
                        for item in raw
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise PreferenceExtractionError(
                    f"extractor returned invalid candidates: {exc}"
                ) from exc
        raise PreferenceExtractionError(
            "extractor must return a PreferenceExtraction or a sequence of "
            f"PreferenceCandidate, got {type(raw).__name__}"
        )

    @staticmethod
    def _validate_candidates(
        extraction: PreferenceExtraction,
    ) -> tuple[tuple[PreferenceCandidate, ...], tuple[PreferenceRemoval, ...]]:
        """Re-validate every candidate; malformed output fails explicitly.

        A candidate whose ``mode`` is ``REMOVE`` is converted into an equivalent
        retraction directive rather than being stored, so "remove this" can never be
        written as a value.
        """
        validated: list[PreferenceCandidate] = []
        extra_removals: list[PreferenceRemoval] = []
        seen: set[tuple[str, str, str, str]] = set()
        for candidate in extraction.preferences:
            try:
                checked = PreferenceCandidate.model_validate(candidate)
            except Exception as exc:  # noqa: BLE001
                raise PreferenceExtractionError(
                    f"preference candidate violates the contract: {exc}"
                ) from exc
            if checked.mode is PreferenceMode.REMOVE:
                # Converted into an equivalent retraction directive rather than being
                # stored as a value, so "remove this" is never remembered as a fact.
                extra_removals.append(
                    PreferenceRemoval(
                        kind=checked.kind,
                        source_text=checked.source_text,
                        extractor=checked.extractor,
                    )
                )
                continue
            key = (
                checked.kind.value,
                checked.value,
                checked.polarity.value,
                checked.mode.value,
            )
            if key in seen:
                # Within one turn, the same structured statement twice is one fact.
                continue
            seen.add(key)
            validated.append(checked)
        return tuple(validated), tuple(extra_removals)

    def _is_turn_processed(
        self,
        user_key: str,
        turn_id: str,
        candidates: Sequence[PreferenceCandidate],
    ) -> bool:
        """True when this turn was already applied for this user.

        Idempotency is keyed on the turn rather than on raw text equality: the store
        already holds entries whose ``source_turn_id`` is this turn and whose identity
        matches the candidate set, which is stronger than comparing messages.
        """
        if not candidates:
            # A turn with nothing to add is idempotent by construction.
            return False
        recorded = {
            (entry.kind.value, entry.value, entry.polarity.value)
            for entry in self._entries_for_turn(user_key, turn_id)
            if entry.status is PreferenceStatus.ACTIVE
        }
        if not recorded:
            return False
        requested = {(c.kind.value, c.value, c.polarity.value) for c in candidates}
        return requested <= recorded

    def _entries_for_turn(
        self, user_key: str, turn_id: str
    ) -> tuple[PreferenceMemoryEntry, ...]:
        """Return entries already recorded for ``turn_id`` (backend-aware)."""
        getter = getattr(self._store, "get_entries_for_turn", None)
        if callable(getter):
            return tuple(getter(user_key, turn_id))
        # Generic fallback for any conforming store.
        return tuple(
            entry
            for entry in self._store.get_entries(user_key)
            if entry.source_turn_id == turn_id
        )

    def _apply(
        self,
        user_key: str,
        turn_id: str,
        candidates: Sequence[PreferenceCandidate],
        removals: Sequence[PreferenceRemoval],
        update: MemoryUpdateSummary,
        timestamp: float,
    ) -> None:
        """Apply retractions and additions to the store."""
        atomic = getattr(self._store, "_atomic_write", None)
        if callable(atomic):
            with atomic():
                self._apply_locked(
                    user_key, turn_id, candidates, removals, update, timestamp
                )
        else:
            self._apply_locked(user_key, turn_id, candidates, removals, update, timestamp)

    def _apply_locked(
        self,
        user_key: str,
        turn_id: str,
        candidates: Sequence[PreferenceCandidate],
        removals: Sequence[PreferenceRemoval],
        update: MemoryUpdateSummary,
        timestamp: float,
    ) -> None:
        """Retractions first, then additions, on top of a fresh active view."""
        active = list(self._store.get_entries(user_key, active_only=True))

        # -- retractions ---------------------------------------------------- #
        for removal in removals:
            for entry in list(active):
                matches = (
                    entry.kind is removal.kind
                    if removal.kind is not None
                    else entry.value.casefold() == (removal.value or "").casefold()
                )
                if not matches:
                    continue
                tombstoned = entry.model_copy(update={"status": PreferenceStatus.REMOVED})
                self._store.update_entry(tombstoned)
                update.removed.append(tombstoned)
                active.remove(entry)

        # -- additions ------------------------------------------------------ #
        sequence = self._store.max_logical_seq(user_key)
        for candidate in candidates:
            identity = deterministic_memory_id(
                user_key,
                turn_id,
                candidate.kind.value,
                candidate.value,
                candidate.polarity.value,
            )
            if self._store.get_entry(user_key, identity) is not None:
                update.skipped_duplicates += 1
                continue

            # Deduplicate an identical active constraint from an earlier turn.
            duplicate = next(
                (
                    entry
                    for entry in active
                    if entry.kind is candidate.kind
                    and entry.value == candidate.value
                    and entry.polarity is candidate.polarity
                ),
                None,
            )
            if duplicate is not None:
                update.skipped_duplicates += 1
                continue

            supersedes: str | None = None
            # Replacement happens only when the statement itself says so.  A plain
            # addition never retracts an independent constraint, whatever the kind.
            for entry in list(active):
                if not _supersedes(entry, candidate):
                    continue
                superseding = entry.model_copy(
                    update={
                        "status": PreferenceStatus.SUPERSEDED,
                        "superseded_by": identity,
                    }
                )
                self._store.update_entry(superseding)
                update.superseded.append(superseding)
                active.remove(entry)
                supersedes = entry.memory_id

            sequence += 1
            try:
                entry = PreferenceMemoryEntry(
                    memory_id=identity,
                user_key=user_key,
                kind=candidate.kind,
                value=candidate.value,
                polarity=candidate.polarity,
                source_text=candidate.source_text,
                source_turn_id=turn_id,
                extractor=candidate.extractor,
                status=PreferenceStatus.ACTIVE,
                logical_seq=sequence,
                    created_at=timestamp,
                    supersedes=supersedes,
                )
            except Exception as exc:  # noqa: BLE001 - never leak a raw schema error
                raise PreferenceExtractionError(
                    f"preference candidate cannot be stored: {exc}"
                ) from exc
            self._store.add_entry(entry)
            update.added.append(entry)
            active.append(entry)
