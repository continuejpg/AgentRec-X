"""Evidence schemas for preference-candidate matching (Milestone 10A).

M10A sits between the M9 preference memory and the future M10B reranker:

    SASRec candidates -> M8 candidate-scoped metadata -> M9 ACTIVE preferences
        -> M10A evidence -> (later) M10B policy

This module defines the **evidence contract only**.  It contains no scoring, no
weighting, no ordering and no candidate selection.  The three-state status is
deliberately not a boolean: "the metadata does not tell us" is a distinct and common
outcome, and collapsing it into *does not match* would fabricate a negative fact.

Status semantics
----------------
``MATCH``
    The supplied metadata explicitly **satisfies** the active preference.  Requires
    positive evidence in a field that can express the preference.

``VIOLATION``
    The supplied metadata explicitly **breaks** the active preference.  Only produced
    for constraints that forbid a value (an ``avoid`` categorical preference whose
    forbidden value is present, or a numeric bound the value falls outside).

``UNKNOWN``
    The available metadata is **insufficient** to decide.  Missing fields, absent
    values, unparseable numbers and preference kinds no metadata field can express all
    land here.  A positive preference whose value is simply absent from a fully
    readable field is ``UNKNOWN`` rather than a violation, because M8 metadata is not
    an exhaustive product specification.

Every record keeps the preference provenance needed to answer "why was this candidate
marked this way?" without guessing, and the candidate's original rank, item id,
``parent_asin`` and raw SASRec score are carried unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation.memory.schemas import PreferenceKind, PreferencePolarity

__all__ = [
    "MATCHING_SUPPORT",
    "METADATA_FIELD_BRAND",
    "METADATA_FIELD_CATEGORIES",
    "METADATA_FIELD_COLOR",
    "METADATA_FIELD_FEATURES",
    "METADATA_FIELD_MATERIAL",
    "METADATA_FIELD_PRICE",
    "METADATA_FIELD_TITLE",
    "CandidateEvidenceCounts",
    "CandidatePreferenceEvidence",
    "EvidenceStatus",
    "MatchingSupport",
    "PreferenceEvidence",
    "PreferenceEvidenceReport",
    "ReasonCode",
    "matching_support_for",
]


class EvidenceStatus(str, Enum):
    """Three-state matching outcome.  Never collapsed to a boolean."""

    MATCH = "match"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class ReasonCode(str, Enum):
    """Deterministic, machine-readable justification for a status.

    The primary contract is the code, not the prose: a diagnostic string may accompany
    it, but nothing should have to parse natural language to act on evidence.
    """

    EXACT_VALUE_MATCH = "exact_value_match"
    #: The forbidden value is explicitly present (an ``avoid`` categorical conflict).
    EXPLICIT_VALUE_CONFLICT = "explicit_value_conflict"
    #: The metadata describes the field, but not with the preferred value.  For a
    #: positive preference this is *not* a violation: M8 metadata is not exhaustive.
    PREFERRED_VALUE_ABSENT = "preferred_value_absent"
    NUMERIC_WITHIN_LIMIT = "numeric_within_limit"
    NUMERIC_EXCEEDS_LIMIT = "numeric_exceeds_limit"
    NUMERIC_BELOW_MINIMUM = "numeric_below_minimum"
    #: The candidate has no metadata record at all.
    METADATA_MISSING = "metadata_missing"
    #: The field exists but the value cannot be read as a comparable number.
    METADATA_UNPARSEABLE = "metadata_unparseable"
    #: No metadata field in the accepted M8 schema can express this preference kind, so
    #: the constraint can never be decided from available data.
    UNSUPPORTED_PREFERENCE_KIND = "unsupported_preference_kind"
    #: The relevant field is readable but absent/empty on this candidate.
    INSUFFICIENT_METADATA = "insufficient_metadata"


class MatchingSupport(str, Enum):
    """How far the accepted M8 metadata can support deciding a preference kind."""

    #: A dedicated, structured metadata field carries the fact.
    SUPPORTED = "supported"
    #: Free text can contain the value, so an occurrence is evidence but an absence
    #: proves nothing.
    PARTIALLY_SUPPORTED = "partially_supported"
    #: No metadata field can express the preference at all.
    UNSUPPORTED_FOR_MATCHING = "unsupported_for_matching"


#: M8 fields used as the evidence source for each preference kind.  These name real
#: ``ProductMetadata`` attributes; no field is invented.
METADATA_FIELD_COLOR = "details.Color"
METADATA_FIELD_MATERIAL = "details.Material"
METADATA_FIELD_BRAND = "store"
METADATA_FIELD_PRICE = "price_text"
METADATA_FIELD_FEATURES = "features"
METADATA_FIELD_CATEGORIES = "categories"
METADATA_FIELD_TITLE = "title"

#: Preferred detail keys per preference kind, tried in order.  ``details`` keeps the
#: source's own attribute names, so lookups fall back to a case-insensitive scan.
_PREFERRED_DETAIL_KEYS: dict[PreferenceKind, tuple[str, ...]] = {
    PreferenceKind.COLOR: ("Color", "Colour"),
    PreferenceKind.MATERIAL: ("Material"),
}


#: The supported / partially-supported / unsupported matrix, derived from the accepted
#: M9 ontology and the accepted M8 metadata schema.  No kind is added or removed here.
MATCHING_SUPPORT: dict[PreferenceKind, MatchingSupport] = {
    PreferenceKind.COLOR: MatchingSupport.SUPPORTED,
    PreferenceKind.MATERIAL: MatchingSupport.SUPPORTED,
    PreferenceKind.BRAND: MatchingSupport.SUPPORTED,
    PreferenceKind.PRICE_MAX: MatchingSupport.SUPPORTED,
    PreferenceKind.PRICE_MIN: MatchingSupport.SUPPORTED,
    PreferenceKind.FEATURE: MatchingSupport.PARTIALLY_SUPPORTED,
    PreferenceKind.CATEGORY: MatchingSupport.PARTIALLY_SUPPORTED,
    PreferenceKind.FREE_FORM_CONSTRAINT: MatchingSupport.UNSUPPORTED_FOR_MATCHING,
}


def matching_support_for(kind: PreferenceKind) -> MatchingSupport:
    """Return the declared support level for a preference kind."""
    return MATCHING_SUPPORT.get(kind, MatchingSupport.UNSUPPORTED_FOR_MATCHING)


class PreferenceEvidence(BaseModel):
    """One active preference evaluated against one candidate's metadata.

    Provenance is preserved in full so the record is self-explanatory: the preference
    id, kind, polarity, value, source span, source turn and logical sequence are all
    carried, together with the metadata field and value the decision was based on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- preference provenance --------------------------------------------- #
    preference_id: str = Field(..., description="memory_id of the active preference.")
    preference_kind: PreferenceKind
    preference_polarity: PreferencePolarity
    preference_value: str
    preference_source_text: str = Field(
        ..., description="Exact user span that produced the preference."
    )
    preference_source_turn_id: str
    preference_logical_seq: int = Field(..., ge=1)

    # -- decision ---------------------------------------------------------- #
    status: EvidenceStatus
    reason_code: ReasonCode
    support: MatchingSupport

    # -- metadata provenance ------------------------------------------------ #
    metadata_field: str | None = Field(
        default=None,
        description=(
            "Accepted M8 metadata field the decision read, e.g. 'details.Color'. None "
            "when no field could express the preference."
        ),
    )
    metadata_value: str | None = Field(
        default=None, description="Exact metadata text the decision read, if any."
    )
    metadata_present: bool = Field(
        default=False, description="True when the candidate has a metadata record at all."
    )
    detail: str | None = Field(
        default=None, description="Short human-readable note; never the primary contract."
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidatePreferenceEvidence(BaseModel):
    """Evidence for one candidate, with its original recommendation fields unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_rank: int = Field(..., ge=1, description="SASRec rank, copied unchanged.")
    item_id: int = Field(..., description="Internal model item id, copied unchanged.")
    parent_asin: str = Field(..., description="External identity, copied unchanged.")
    sasrec_score: float = Field(
        ..., description="Raw SASRec score, copied exactly; never normalised or combined."
    )
    evidence: tuple[PreferenceEvidence, ...] = ()

    def statuses(self) -> tuple[EvidenceStatus, ...]:
        """Statuses in evidence order."""
        return tuple(record.status for record in self.evidence)

    def count(self, status: EvidenceStatus) -> int:
        """Number of evidence records with ``status``."""
        return sum(1 for record in self.evidence if record.status is status)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidateEvidenceCounts(BaseModel):
    """Descriptive totals for one candidate.

    These are counts only.  They are **not** a score: no weighting, no sign, no sum and
    no ordering implication.  M10A deliberately does not collapse evidence to a scalar.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)


class PreferenceEvidenceReport(BaseModel):
    """Full M10A output: every candidate, in the exact order it was supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[CandidatePreferenceEvidence, ...] = ()
    active_preference_count: int = Field(default=0, ge=0)
    counts: CandidateEvidenceCounts = Field(default_factory=CandidateEvidenceCounts)

    @property
    def parent_asins(self) -> tuple[str, ...]:
        """Candidate identities, in preserved order."""
        return tuple(candidate.parent_asin for candidate in self.candidates)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Original ranks, in preserved order."""
        return tuple(candidate.original_rank for candidate in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()
