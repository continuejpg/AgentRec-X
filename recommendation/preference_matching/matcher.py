"""Preference-candidate matcher (Milestone 10A).

Produces structured, attributable evidence describing whether each candidate's
available metadata **satisfies**, **violates** or is **inconclusive** about each
ACTIVE preference.

What it does not do
-------------------
* it does not rerank, filter, add, drop or replace candidates;
* it does not compute a preference score, weight or combined number;
* it does not retrieve anything: it reads only the metadata already attached to the
  candidates by the accepted M8 enricher;
* it does not open a memory store or query users; active preferences are passed in.

Matching rules
--------------
Every rule is a deterministic read over real ``ProductMetadata`` fields:

===========================  ==============================================  ==================
Preference kind              Evidence source                                 Match / violation
===========================  ==============================================  ==================
``color``                    ``details.Color`` / ``details.Colour``          MATCH / VIOLATION
``material``                 ``details.Material``                            MATCH / VIOLATION
``brand``                    ``store``, then ``details["Brand Name"]``       MATCH / VIOLATION
``price_max``                ``price_text`` (numeric)                        MATCH / VIOLATION
``price_min``                ``price_text`` (numeric)                        MATCH / VIOLATION
``feature``                  ``features``, ``title``, ``description``        MATCH only
``category``                 ``categories``, ``main_category``, ``title``    MATCH only
``free_form_constraint``     none                                            always UNKNOWN
===========================  ==============================================  ==================

Categorical matching compares **whole normalised tokens**, never substrings, so
``"red"`` does not match ``"hundred"`` and ``"blue"`` does not match ``"blueberry"``.

Conservative asymmetry: a *negative* constraint is violated by finding the forbidden
value, while a *positive* preference is only satisfied by finding the wanted value —
its absence is ``UNKNOWN``, because M8 metadata is not an exhaustive product
specification and a missing token proves nothing.  Positive preferences never produce
a violation.

Complexity is O(candidates x active preferences); each pair reads a bounded set of
metadata fields and needs no indexing.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from recommendation.catalog.schemas import MissingMetadata, ProductMetadata
from recommendation.memory.schemas import (
    PreferenceKind,
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
    PreferencePolarity,
)

from .schemas import (
    METADATA_FIELD_BRAND,
    METADATA_FIELD_CATEGORIES,
    METADATA_FIELD_COLOR,
    METADATA_FIELD_FEATURES,
    METADATA_FIELD_MATERIAL,
    METADATA_FIELD_PRICE,
    METADATA_FIELD_TITLE,
    CandidateEvidenceCounts,
    CandidatePreferenceEvidence,
    EvidenceStatus,
    MatchingSupport,
    PreferenceEvidence,
    PreferenceEvidenceReport,
    ReasonCode,
    matching_support_for,
)

__all__ = [
    "PreferenceCandidateMatcher",
    "active_preferences",
    "match_candidates",
]

#: Tokenisation mirrors the accepted M8 RAG tokeniser: lowercase alphanumeric runs.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Text-bearing fields used for the partially-supported kinds, in evaluation order.
_PARTIAL_FEATURE_FIELDS = ("features", "title", "description")
_PARTIAL_CATEGORY_FIELDS = ("categories", "main_category", "title")


def tokenize(text: str) -> set[str]:
    """Return the set of lowercase alphanumeric tokens in ``text``."""
    if not isinstance(text, str) or not text:
        return set()
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def _covers(metadata_value: str, preference_value: str) -> bool:
    """True when every token of ``preference_value`` appears as a token of the metadata.

    Whole-token containment, so ``"red"`` does not match ``"hundred"``.
    """
    wanted = tokenize(preference_value)
    if not wanted:
        return False
    return wanted <= tokenize(metadata_value)


def _lookup_detail(record: ProductMetadata, keys: Sequence[str]) -> tuple[str, str] | None:
    """Return ``(field_label, value)`` for the first matching detail key.

    ``details`` keeps the source's own attribute names, so lookups try the preferred
    spellings first and then fall back to a case-insensitive scan in source order.
    """
    details = dict(record.details)
    for key in keys:
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return f"details.{key}", value
    for key, value in details.items():
        if key.strip().casefold() in {candidate.casefold() for candidate in keys}:
            if value.strip():
                return f"details.{key}", value
    return None


def _brand_value(record: ProductMetadata) -> tuple[str, str] | None:
    """Return the brand-like value, preferring the structured ``store`` field."""
    if record.store:
        return METADATA_FIELD_BRAND, record.store
    for key in ("Brand Name", "Brand"):
        value = dict(record.details).get(key)
        if isinstance(value, str) and value.strip():
            return f"details.{key}", value
    return None


def _parse_price(price_text: str | None) -> float | None:
    """Parse a price token into a float, or ``None`` when it is not numeric.

    M8 preserves the source price as canonical text and performs no currency
    normalisation, so this performs no conversion either: a non-numeric token (the
    source uses an em-dash placeholder) is simply unreadable.
    """
    if not isinstance(price_text, str):
        return None
    cleaned = price_text.strip().lstrip("$").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _format_number(value: float) -> str:
    """Render a bound the way the preference stores it (no trailing ``.0``)."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _evaluate_text_fields(
    record: ProductMetadata,
    fields: Sequence[str],
    preference_value: str,
) -> tuple[bool, str | None, str | None]:
    """Search text-bearing fields; return ``(found, field_label, value)``."""
    for field in fields:
        raw = getattr(record, field, None)
        if raw is None:
            continue
        values = raw if isinstance(raw, tuple) else (raw,)
        for value in values:
            if isinstance(value, str) and _covers(value, preference_value):
                return True, field, value
    return False, None, None


def _evidence(
    *,
    entry: PreferenceMemoryEntry,
    status: EvidenceStatus,
    reason_code: ReasonCode,
    metadata_present: bool,
    metadata_field: str | None = None,
    metadata_value: str | None = None,
    detail: str | None = None,
) -> PreferenceEvidence:
    """Assemble one evidence record from an active preference entry."""
    return PreferenceEvidence(
        preference_id=entry.memory_id,
        preference_kind=entry.kind,
        preference_polarity=entry.polarity,
        preference_value=entry.value,
        preference_source_text=entry.source_text,
        preference_source_turn_id=entry.source_turn_id,
        preference_logical_seq=entry.logical_seq,
        status=status,
        reason_code=reason_code,
        support=matching_support_for(entry.kind),
        metadata_field=metadata_field,
        metadata_value=metadata_value,
        metadata_present=metadata_present,
        detail=detail,
    )


def _unknown(
    entry: PreferenceMemoryEntry,
    *,
    metadata_present: bool,
    reason_code: ReasonCode,
    metadata_field: str | None = None,
    metadata_value: str | None = None,
    detail: str | None = None,
) -> PreferenceEvidence:
    """Assemble an UNKNOWN record."""
    return _evidence(
        entry=entry,
        status=EvidenceStatus.UNKNOWN,
        reason_code=reason_code,
        metadata_present=metadata_present,
        metadata_field=metadata_field,
        metadata_value=metadata_value,
        detail=detail,
    )


def _match_categorical(
    entry: PreferenceMemoryEntry,
    record: ProductMetadata,
    located: tuple[str, str] | None,
) -> PreferenceEvidence:
    """Evaluate an explicit categorical constraint (colour, material, brand).

    Presence of the value is decisive.  Absence is decisive only for a *negative*
    constraint; for a positive one it is ``UNKNOWN`` because the metadata is not
    exhaustive and a different listed value does not prove the wanted one is absent.
    """
    if located is None:
        return _unknown(
            entry,
            metadata_present=True,
            reason_code=ReasonCode.INSUFFICIENT_METADATA,
            detail="no metadata field records this attribute for the candidate",
        )

    field, value = located
    found = _covers(value, entry.value)

    if found:
        if entry.polarity is PreferencePolarity.AVOID:
            return _evidence(
                entry=entry,
                status=EvidenceStatus.VIOLATION,
                reason_code=ReasonCode.EXPLICIT_VALUE_CONFLICT,
                metadata_present=True,
                metadata_field=field,
                metadata_value=value,
                detail=f"{field} explicitly contains the avoided value",
            )
        return _evidence(
            entry=entry,
            status=EvidenceStatus.MATCH,
            reason_code=ReasonCode.EXACT_VALUE_MATCH,
            metadata_present=True,
            metadata_field=field,
            metadata_value=value,
            detail=f"{field} explicitly contains the preferred value",
        )

    if entry.polarity is PreferencePolarity.AVOID:
        # The forbidden value is not present in a field that exists: the constraint is
        # satisfied, but absence is not proof, so this is UNKNOWN rather than MATCH.
        return _unknown(
            entry,
            metadata_present=True,
            reason_code=ReasonCode.INSUFFICIENT_METADATA,
            metadata_field=field,
            metadata_value=value,
            detail="field present but the avoided value does not appear",
        )

    return _unknown(
        entry,
        metadata_present=True,
        reason_code=ReasonCode.PREFERRED_VALUE_ABSENT,
        metadata_field=field,
        metadata_value=value,
        detail="field present but the preferred value does not appear",
    )


def _match_partial(
    entry: PreferenceMemoryEntry,
    record: ProductMetadata,
    fields: Sequence[str],
) -> PreferenceEvidence:
    """Evaluate a partially-supported kind (feature, category).

    An occurrence of the preference value in free text is evidence of a match; its
    absence proves nothing, so the miss is ``UNKNOWN`` and never a violation.  This
    kind is why the support level is ``PARTIALLY_SUPPORTED`` rather than ``SUPPORTED``.
    """
    found, field, value = _evaluate_text_fields(record, fields, entry.value)
    if found:
        if entry.polarity is PreferencePolarity.AVOID:
            return _evidence(
                entry=entry,
                status=EvidenceStatus.VIOLATION,
                reason_code=ReasonCode.EXPLICIT_VALUE_CONFLICT,
                metadata_present=True,
                metadata_field=field,
                metadata_value=value,
                detail=f"{field} contains the avoided value",
            )
        return _evidence(
            entry=entry,
            status=EvidenceStatus.MATCH,
            reason_code=ReasonCode.EXACT_VALUE_MATCH,
            metadata_present=True,
            metadata_field=field,
            metadata_value=value,
            detail=f"{field} contains the preferred value",
        )
    return _unknown(
        entry,
        metadata_present=True,
        reason_code=ReasonCode.INSUFFICIENT_METADATA,
        detail="no text field mentions the value; absence is not proof",
    )


def _match_numeric(
    entry: PreferenceMemoryEntry,
    record: ProductMetadata,
) -> PreferenceEvidence:
    """Evaluate a price bound against the candidate's price metadata."""
    raw = record.price_text
    if raw is None:
        return _unknown(
            entry,
            metadata_present=True,
            reason_code=ReasonCode.INSUFFICIENT_METADATA,
            metadata_field=METADATA_FIELD_PRICE,
            detail="candidate metadata carries no price",
        )

    try:
        bound = float(entry.value)
    except (TypeError, ValueError):
        # A stored bound that is not numeric cannot be compared.  This should be
        # unreachable through the M9 extraction path; it is reported rather than
        # guessed at.
        return _unknown(
            entry,
            metadata_present=True,
            reason_code=ReasonCode.METADATA_UNPARSEABLE,
            metadata_field=METADATA_FIELD_PRICE,
            metadata_value=raw,
            detail="preference bound is not numeric",
        )

    price = _parse_price(raw)
    if price is None:
        return _unknown(
            entry,
            metadata_present=True,
            reason_code=ReasonCode.METADATA_UNPARSEABLE,
            metadata_field=METADATA_FIELD_PRICE,
            metadata_value=raw,
            detail="candidate price token is not numeric",
        )

    if entry.kind is PreferenceKind.PRICE_MAX:
        within = price <= bound
        return _evidence(
            entry=entry,
            status=EvidenceStatus.MATCH if within else EvidenceStatus.VIOLATION,
            reason_code=(
                ReasonCode.NUMERIC_WITHIN_LIMIT
                if within
                else ReasonCode.NUMERIC_EXCEEDS_LIMIT
            ),
            metadata_present=True,
            metadata_field=METADATA_FIELD_PRICE,
            metadata_value=raw,
            detail=f"price {_format_number(price)} vs maximum {entry.value}",
        )

    at_least = price >= bound
    return _evidence(
        entry=entry,
        status=EvidenceStatus.MATCH if at_least else EvidenceStatus.VIOLATION,
        reason_code=(
            ReasonCode.NUMERIC_WITHIN_LIMIT
            if at_least
            else ReasonCode.NUMERIC_BELOW_MINIMUM
        ),
        metadata_present=True,
        metadata_field=METADATA_FIELD_PRICE,
        metadata_value=raw,
        detail=f"price {_format_number(price)} vs minimum {entry.value}",
    )


def _evaluate(
    entry: PreferenceMemoryEntry,
    record: ProductMetadata | MissingMetadata | None,
) -> PreferenceEvidence:
    """Evaluate one active preference against one candidate's metadata.

    ``None`` is accepted as well as :class:`MissingMetadata`, because the accepted M8
    enrichment represents an uncovered candidate with ``metadata=None`` while the M8
    catalogue lookup uses ``MissingMetadata``.  Both mean the same thing here -- no
    metadata -- and both must yield ``UNKNOWN``.
    """
    # An unsupported kind is decided before metadata availability: no metadata could
    # ever resolve it, so reporting "metadata_missing" would wrongly imply that better
    # data would help.
    if matching_support_for(entry.kind) is MatchingSupport.UNSUPPORTED_FOR_MATCHING:
        return _unknown(
            entry,
            metadata_present=record is not None and not isinstance(record, MissingMetadata),
            reason_code=ReasonCode.UNSUPPORTED_PREFERENCE_KIND,
            detail="no accepted metadata field can express this preference kind",
        )

    if record is None or isinstance(record, MissingMetadata):
        return _unknown(
            entry,
            metadata_present=False,
            reason_code=ReasonCode.METADATA_MISSING,
            detail="the catalogue holds no metadata record for this candidate",
        )

    kind = entry.kind
    if kind is PreferenceKind.COLOR:
        return _match_categorical(entry, record, _lookup_detail(record, ("Color", "Colour")))
    if kind is PreferenceKind.MATERIAL:
        return _match_categorical(entry, record, _lookup_detail(record, ("Material",)))
    if kind is PreferenceKind.BRAND:
        return _match_categorical(entry, record, _brand_value(record))
    if kind in (PreferenceKind.PRICE_MAX, PreferenceKind.PRICE_MIN):
        return _match_numeric(entry, record)
    if kind is PreferenceKind.FEATURE:
        return _match_partial(entry, record, _PARTIAL_FEATURE_FIELDS)
    if kind is PreferenceKind.CATEGORY:
        return _match_partial(entry, record, _PARTIAL_CATEGORY_FIELDS)

    # Defensive: every supported kind is handled above, so reaching here means the
    # matrix and the dispatch have drifted apart.  It is reported rather than ignored.
    return _unknown(
        entry,
        metadata_present=True,
        reason_code=ReasonCode.UNSUPPORTED_PREFERENCE_KIND,
        detail="no matching rule is registered for this preference kind",
    )


def active_preferences(snapshot: PreferenceMemorySnapshot) -> tuple[PreferenceMemoryEntry, ...]:
    """Return only the ACTIVE entries of a snapshot, in snapshot order.

    Superseded and removed entries are excluded here, which is what keeps inactive
    preference history out of matching decisions.
    """
    return tuple(entry for entry in snapshot.entries if entry.is_active)


def match_candidates(
    *,
    candidates: Sequence[Any],
    preferences: PreferenceMemorySnapshot | Sequence[PreferenceMemoryEntry],
) -> PreferenceEvidenceReport:
    """Produce evidence for every candidate against every ACTIVE preference.

    Parameters
    ----------
    candidates:
        The accepted M8 enrichment items (``EnrichedRecommendation``), already in
        SASRec order.  Only these candidates are ever inspected; the matcher has no
        catalogue handle, no retriever and no store, so it cannot reach another
        product.
    preferences:
        Either an M9 ``PreferenceMemorySnapshot`` (only its ACTIVE entries are used) or
        an explicit sequence of entries (already filtered by the caller).

    Returns
    -------
    PreferenceEvidenceReport
        Candidates in the exact input order, each carrying its original rank, item id,
        ``parent_asin`` and raw SASRec score unchanged, plus one evidence record per
        active preference.
    """
    if isinstance(preferences, PreferenceMemorySnapshot):
        active = active_preferences(preferences)
    else:
        # An explicit sequence is honoured as given, but inactive entries passed by
        # mistake are still filtered out so history can never influence matching.
        active = tuple(entry for entry in preferences if entry.is_active)

    evaluated: list[CandidatePreferenceEvidence] = []
    totals = CandidateEvidenceCounts()
    for item in candidates:
        evidence = tuple(_evaluate(entry, item.metadata) for entry in active)
        counts = CandidateEvidenceCounts(
            match_count=sum(1 for record in evidence if record.status is EvidenceStatus.MATCH),
            violation_count=sum(
                1 for record in evidence if record.status is EvidenceStatus.VIOLATION
            ),
            unknown_count=sum(
                1 for record in evidence if record.status is EvidenceStatus.UNKNOWN
            ),
        )
        totals = CandidateEvidenceCounts(
            match_count=totals.match_count + counts.match_count,
            violation_count=totals.violation_count + counts.violation_count,
            unknown_count=totals.unknown_count + counts.unknown_count,
        )
        evaluated.append(
            CandidatePreferenceEvidence(
                # Original recommendation fields, copied exactly and never recomputed.
                original_rank=item.rank,
                item_id=item.recommendation.item_id,
                parent_asin=item.parent_asin,
                sasrec_score=item.score,
                evidence=evidence,
            )
        )

    return PreferenceEvidenceReport(
        candidates=tuple(evaluated),
        active_preference_count=len(active),
        counts=totals,
    )


class PreferenceCandidateMatcher:
    """Stateless matcher object; the public seam for M10B and the Agent.

    It holds no store, no retriever and no mutable state, so the same instance can be
    shared while remaining deterministic.  Active preferences are always supplied by
    the caller, which keeps memory lifecycle separate from evidence semantics.
    """

    def match(
        self,
        *,
        candidates: Sequence[Any],
        preferences: PreferenceMemorySnapshot | Sequence[PreferenceMemoryEntry],
    ) -> PreferenceEvidenceReport:
        """Return evidence for ``candidates`` against the active ``preferences``."""
        return match_candidates(candidates=candidates, preferences=preferences)

    @staticmethod
    def support_for(kind: PreferenceKind) -> MatchingSupport:
        """Expose the declared support level for a preference kind."""
        return matching_support_for(kind)
