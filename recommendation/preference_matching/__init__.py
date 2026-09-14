"""Preference-candidate evidence matching (Milestone 10A).

Produces structured, attributable evidence describing whether each SASRec candidate's
**already-attached** metadata satisfies, violates or is inconclusive about each
**ACTIVE** preference::

    trusted interaction history
        -> RecommendationTool / SASRec           (M6/M7A, unchanged)
        -> ordered candidate list
        -> M8 candidate-scoped metadata          (frozen boundary)
        -> M9 ACTIVE preference memory           (read-only)
        -> M10A preference-candidate matcher     (this package)
        -> structured evidence only

M10A is an **evidence** layer.  It does not rank:

* no candidate is added, dropped, replaced or reordered;
* ``original_rank``, ``item_id``, ``parent_asin`` and the raw SASRec score are copied
  through unchanged;
* no preference score, weight, sign or combined number is produced;
* missing metadata is ``UNKNOWN``, never silently treated as a match or a violation;
* the matcher reads only the metadata the accepted M8 enricher attached to the current
  candidates -- there is no catalogue handle, no retriever and no store, so no new
  product can enter the candidate universe.

See ``README.md`` for the MATCH / VIOLATION / UNKNOWN semantics, the supported
preference-kind matrix and the M10A/M10B boundary.
"""

from __future__ import annotations

from .matcher import (
    PreferenceCandidateMatcher,
    active_preferences,
    match_candidates,
    tokenize,
)
from .schemas import (
    MATCHING_SUPPORT,
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
    "PreferenceCandidateMatcher",
    "PreferenceEvidence",
    "PreferenceEvidenceReport",
    "ReasonCode",
    "active_preferences",
    "match_candidates",
    "matching_support_for",
    "tokenize",
]

__version__ = "0.1.0"
