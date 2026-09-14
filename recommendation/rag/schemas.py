"""Evidence and enrichment schemas for candidate-scoped product RAG (Milestone 8-B).

Grounding rules encoded here
----------------------------
* Every evidence fragment carries the ``parent_asin`` it came from, the metadata
  field it was taken from, and a provenance label.  Text from different products is
  never merged into an unattributed blob, and a feature of candidate A can never
  become evidence for candidate B.
* A fragment is copied verbatim from normalized metadata.  Nothing is paraphrased,
  summarised, generated or extended.
* Retrieval happens **only** over metadata belonging to the current SASRec
  candidates.  The candidate set is supplied by the accepted M7A Tool; this layer
  cannot add, remove or reorder products.
* A candidate with no metadata, or with metadata but no searchable text, is
  represented explicitly rather than being dropped or filled in.
* ``retrieval_score`` ranks *evidence passages within the allowed candidate scope*
  (its documented tie-break is candidate submission order, i.e. the SASRec rank).
  It is not a recommendation score, not a probability and not comparable to the
  SASRec ``score`` on :class:`~recommendation.tools.schemas.ToolRecommendation`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from recommendation.tools.schemas import ToolRecommendation

from ..catalog.schemas import ProductMetadata

__all__ = [
    "EVIDENCE_FIELDS",
    "FALLBACK_REASONS",
    "CandidateEvidence",
    "CandidateEvidenceStatus",
    "EnrichedRecommendation",
    "EnrichmentResult",
    "ProductEvidence",
    "RagQuery",
    "SearchableField",
]

#: Metadata fields that may contribute searchable text, in a fixed evaluation order.
#: Only fields that really exist in the normalized schema are listed.
#: ``details`` contributes one evidence fragment per source attribute (for example
#: ``"Brand Name"``, ``"Color"``), each still attributed to its own key.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
    "details",
)

#: The subset of :data:`EVIDENCE_FIELDS` that a single value maps to 1:1.
SearchableField = Literal[
    "title",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
    "details",
]

#: Why no evidence was produced for a candidate.  Explicit, never silent.
FALLBACK_REASONS = ("no_metadata", "no_searchable_text", "no_lexical_match")

#: Evidence status for one candidate.
CandidateEvidenceStatus = Literal["evidence", "none"]


class ProductEvidence(BaseModel):
    """One attributable, verbatim fragment of normalized product metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str = Field(..., description="Candidate this fragment belongs to.")
    field: SearchableField = Field(
        ..., description="Normalized metadata field the fragment was taken from."
    )
    text: str = Field(..., description="Verbatim fragment; never paraphrased or generated.")
    retrieval_score: float = Field(
        ...,
        ge=0.0,
        description=(
            "Lexical relevance of this fragment to the query, within the current "
            "candidate scope only. NOT a recommendation score and NOT a probability."
        ),
    )
    provenance: str = Field(
        ...,
        description=(
            "Where the fact came from, e.g. 'amazon_reviews_2023:meta_categories/"
            "Sports_and_Outdoors#details:Brand Name'."
        ),
    )
    detail_key: str | None = Field(
        default=None,
        description="Source attribute name when the fragment came from `details`, else None.",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidateEvidence(BaseModel):
    """Retrieved evidence for exactly one candidate, or an explicit absence.

    A candidate is always present here, whether or not metadata exists for it, so
    the enrichment result stays positionally aligned with the Tool result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str
    rank: int = Field(..., ge=1, description="SASRec rank, copied unchanged from the Tool result.")
    status: CandidateEvidenceStatus
    metadata_status: Literal["found", "missing"]
    evidence: tuple[ProductEvidence, ...] = ()
    fallback_reason: Literal["no_metadata", "no_searchable_text", "no_lexical_match"] | None = None

    @property
    def has_evidence(self) -> bool:
        """True when at least one attributable fragment was retrieved."""
        return bool(self.evidence)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class EnrichedRecommendation(BaseModel):
    """A Tool recommendation plus its candidate-scoped evidence.

    ``recommendation`` is copied through **unchanged** -- same object semantics and
    same field values -- so rank, ``parent_asin`` and raw SASRec ``score`` survive
    enrichment exactly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation: ToolRecommendation
    metadata_status: Literal["found", "missing"]
    metadata: ProductMetadata | None = None
    evidence: tuple[ProductEvidence, ...] = ()
    fallback_reason: Literal["no_metadata", "no_searchable_text", "no_lexical_match"] | None = None

    @property
    def parent_asin(self) -> str:
        """Candidate identity, read straight from the untouched recommendation."""
        return self.recommendation.parent_asin

    @property
    def rank(self) -> int:
        """Candidate rank, read straight from the untouched recommendation."""
        return self.recommendation.rank

    @property
    def score(self) -> float:
        """Raw SASRec score, read straight from the untouched recommendation."""
        return self.recommendation.score

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class EnrichmentResult(BaseModel):
    """Result of enriching one Tool result: same candidates, same order, plus evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[EnrichedRecommendation, ...] = ()
    requested_k: int
    returned_k: int
    metadata_found: int
    metadata_missing: int
    evidence_count: int
    query_used: str | None = Field(
        default=None, description="The retrieval query actually used (None when no query was given)."
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def parent_asins(self) -> tuple[str, ...]:
        """Candidate identities in recommendation order."""
        return tuple(item.parent_asin for item in self.items)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class RagQuery(BaseModel):
    """A retrieval query.

    The query is **untrusted text**: it only selects evidence from the current
    candidates' metadata.  It cannot change the candidate universe, alter trusted
    interaction history, inject metadata records, or trigger any file or network
    access.  A ``RagQuery`` has no field for history or candidate identifiers, so
    a malicious query string has nowhere to put them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""

    @property
    def terms(self) -> tuple[str, ...]:
        """Whitespace-normalised query terms (tokenisation happens in retrieval)."""
        return tuple(self.text.split())

    @property
    def is_blank(self) -> bool:
        """True when the query carries no usable text."""
        return not self.text.strip()
