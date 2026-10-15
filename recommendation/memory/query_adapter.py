"""Deterministic Memory -> Catalog Search query adapter (V2 M-02A).

One pure function turns the *trusted, already-read* M9 preference snapshot plus the caller's own
query terms into the terms the lexical search tool will run.  Design rules, all frozen here:

* only this user's ``ACTIVE`` entries participate - ``REMOVED``/``SUPERSEDED`` never do, and the
  adapter re-checks status even though the store already filters, so the guarantee is local;
* only ``PREFER`` entries contribute, and only for kinds whose text the catalogue actually
  carries (CATEGORY / BRAND / FEATURE / COLOR / MATERIAL).  ``PRICE_MIN``/``PRICE_MAX`` are
  numbers and ``FREE_FORM_CONSTRAINT`` is unverified free text: neither is appended;
* ``AVOID`` memory is never appended as a positive keyword.  (A user message may itself contain
  a negation; that is the message's own text and is left exactly as it was.)
* the caller's own terms always come first and are never removed or rewritten: history is a
  supplement to the current request, never a replacement;
* ordering is total and reproducible (``logical_seq`` then ``memory_id``), duplicates are dropped
  case-insensitively, whitespace is collapsed, and both the count and the length of appended
  terms are bounded.

The adaptation is reported, not hidden: :class:`QueryAdaptation` carries the ``memory_id`` values
that actually contributed, so a downstream layer can cite them.

What crosses into a policy, and what does not
---------------------------------------------
The controller never hands a policy the M9 snapshot.  It hands
:class:`ActivePreferenceProjection`: a value-level, ACTIVE-only view built by
:func:`project_active_preferences`, whose entries carry the six values this module reads and
nothing else.  The snapshot's ``user_key``, its entries' verbatim ``source_text``, their
``source_turn_id``/``extractor``/``created_at``, and every removed or superseded entry stay on the
trusted side of the boundary - the policy's view is the minimum a query decision needs, which is
also what keeps a model-backed policy from being able to quote the user's raw words or their key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schemas import PreferenceKind, PreferencePolarity, PreferenceStatus

#: Kinds whose values are text a lexical catalogue index can match.
TEXT_KINDS: frozenset[PreferenceKind] = frozenset(
    {
        PreferenceKind.CATEGORY,
        PreferenceKind.BRAND,
        PreferenceKind.FEATURE,
        PreferenceKind.COLOR,
        PreferenceKind.MATERIAL,
    }
)

#: At most this many terms are appended, and each is truncated to this many characters.
MAX_ADDED_TERMS = 3
MAX_TERM_CHARS = 40


@dataclass(frozen=True)
class QueryAdaptation:
    """The query terms to run, plus which memory entries contributed."""

    terms: tuple[str, ...]
    added: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "terms": list(self.terms),
            "added": list(self.added),
            "memory_ids": list(self.memory_ids),
            "changed": self.changed,
        }


@dataclass(frozen=True)
class ActivePreference:
    """One ACTIVE preference, projected to the minimum a *retrieval query* may be shaped from.

    The minimality is the security property, so it is stated rather than implied.  The trusted M9
    entry this comes from also carries the owning ``user_key``, the verbatim ``source_text`` span of
    the user's message, the ``source_turn_id``, the extractor name, ``created_at`` and the
    supersede/removed lifecycle.  None of that is a query input, so none of it crosses this
    boundary: what crosses is exactly the six values :func:`adapt_terms` reads.
    """

    #: The entry's opaque internal id: a deterministic sort tie-break and the identifier
    #: :class:`QueryAdaptation` reports for the entries that contributed.  It is not a user
    #: identifier and carries no user data.
    memory_id: str
    kind: PreferenceKind
    polarity: PreferencePolarity
    value: str
    #: Always :data:`PreferenceStatus.ACTIVE`: this projection is built from active entries only, so
    #: the adapter's own re-check stays meaningful instead of becoming an unchecked assumption.
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    #: Per-user monotonic sequence, the stable sort key.
    logical_seq: int = 1


@dataclass(frozen=True)
class ActivePreferenceProjection:
    """The read-only preference view a policy is handed: the user's ACTIVE entries and nothing else.

    ``entries`` is named to match the shape :func:`adapt_terms` already reads, so the accepted
    adapter needs no change and the M-02A behaviour is preserved value for value.  Removed and
    superseded entries are excluded **here**, not merely filtered later: a preference the user
    withdrew must not reach a policy, a prompt or a retrieval query at all.
    """

    entries: tuple[ActivePreference, ...] = ()

    @property
    def count(self) -> int:
        """How many ACTIVE preferences this projection carries."""
        return len(self.entries)


def project_active_preferences(snapshot: Any | None) -> ActivePreferenceProjection:
    """Project a trusted M9 snapshot down to what a policy may see.

    ``None`` (no memory configured) and an empty snapshot both project to an empty projection, so a
    deployment without memory - and every run whose ``user_key`` is unset - behaves exactly as it
    did before.  Only ``entries`` is read from the container; only the six fields of
    :class:`ActivePreference` are copied out of each entry; only ``ACTIVE`` entries are copied at
    all.
    """
    if snapshot is None:
        return ActivePreferenceProjection()
    return ActivePreferenceProjection(
        entries=tuple(
            ActivePreference(
                memory_id=str(entry.memory_id),
                kind=entry.kind,
                polarity=entry.polarity,
                value=str(entry.value),
                status=entry.status,
                logical_seq=int(entry.logical_seq),
            )
            for entry in (getattr(snapshot, "entries", ()) or ())
            if getattr(entry, "status", None) == PreferenceStatus.ACTIVE
        )
    )


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())[:MAX_TERM_CHARS].strip()


def adapt_terms(
    base_terms: Sequence[str] | Iterable[str],
    snapshot: Any | None = None,
    *,
    max_added: int = MAX_ADDED_TERMS,
) -> QueryAdaptation:
    """Return the terms the search tool should run.  Pure and deterministic.

    ``snapshot is None`` (or an empty/unsuitable snapshot) reproduces ``base_terms`` exactly, so a
    deployment without memory behaves as it did before this module existed.
    """
    base: list[str] = []
    seen: set[str] = set()
    for raw in base_terms or ():
        value = _normalise(raw)
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        base.append(value)

    entries = tuple(getattr(snapshot, "entries", ()) or ()) if snapshot is not None else ()
    candidates = [
        entry
        for entry in entries
        if getattr(entry, "status", None) == PreferenceStatus.ACTIVE
        and getattr(entry, "polarity", None) == PreferencePolarity.PREFER
        and getattr(entry, "kind", None) in TEXT_KINDS
    ]
    candidates.sort(key=lambda entry: (int(getattr(entry, "logical_seq", 0) or 0),
                                       str(getattr(entry, "memory_id", ""))))

    added: list[str] = []
    memory_ids: list[str] = []
    limit = max(0, int(max_added))
    for entry in candidates:
        if len(added) >= limit:
            break
        value = _normalise(getattr(entry, "value", ""))
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        added.append(value)
        memory_ids.append(str(getattr(entry, "memory_id", "")))

    return QueryAdaptation(
        terms=tuple(base) + tuple(added),
        added=tuple(added),
        memory_ids=tuple(memory_ids),
        changed=bool(added),
    )


__all__ = [
    "ActivePreference",
    "ActivePreferenceProjection",
    "MAX_ADDED_TERMS",
    "MAX_TERM_CHARS",
    "QueryAdaptation",
    "TEXT_KINDS",
    "adapt_terms",
    "project_active_preferences",
]
