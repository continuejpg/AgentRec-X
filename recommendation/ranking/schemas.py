"""Typed contracts for the M6.1 belief-aware ranking foundation (offline lab).

M6.1 joins three trusted domains that already exist in this repository:

* a **frozen candidate snapshot** produced by the trusted recommendation pipeline
  (identity, original order, retrieval provenance, catalogue metadata, eligibility);
* the **M5 ranker projection** (``for_ranker``) of the unified decision state;
* the accepted **M10A/M10B** explicit-preference evidence and reranking components.

Nothing in this module ranks, filters, scores or persists anything.  It defines the
immutable records exchanged by the offline resolver / feature builder / ranking policies /
paired evaluator, so every comparison is over the *same* candidate set.

Boundary rules encoded here
---------------------------
* A candidate's identity is ``parent_asin`` plus the trusted integer ``item_id``; PAD
  (``item_id = 0``) is never a candidate.  Digests are *verification* material, never a
  reconstruction source.
* A snapshot is valid only for the user/task/turn it was built for.  Cross-user,
  cross-task, stale or digest-mismatched snapshots are rejected by the resolver.
* Candidate identity is preserved across every arm: a ranking arm may reorder, never add,
  drop or replace a candidate.
* Preference evidence keeps M10A's three-state semantics: ``UNKNOWN`` is not a negative
  match, and missing catalogue attributes are represented explicitly.
* Belief strength is ordinal, never a probability.  Ranking policies use ordinal
  tie-break keys only; no raw model score or uncalibrated weight is arithmetically mixed.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from recommendation.catalog.schemas import ProductMetadata
from recommendation.preference_matching.schemas import PreferenceEvidence
from recommendation.state.state_projector import RankerStateProjection

__all__ = [
    "CANDIDATE_SNAPSHOT_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "ArmEvaluation",
    "ArmRanking",
    "CandidateFeatureSet",
    "CandidateRankingFeatures",
    "CandidateRecord",
    "CandidateSnapshot",
    "CatalogueCoverage",
    "CatalogueFields",
    "EligibilityState",
    "EvaluationReport",
    "EvidenceState",
    "FrozenRankingTask",
    "InferredPreferenceEvidence",
    "PairedDelta",
    "PreferenceStrengthMetadata",
    "QualityMetrics",
    "RankedCandidate",
    "RankingArm",
    "SnapshotExpectation",
    "SourceRank",
    "TaskLabel",
]

#: Versions of the offline contracts.  Bumped when a stored report must be re-read.
CANDIDATE_SNAPSHOT_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


class EligibilityState(str, Enum):
    """Task-scoped feasibility of one candidate, mirroring the accepted evaluator.

    The string values intentionally match
    :class:`recommendation.control.constraint_eligibility.CandidateEligibility` so a
    trusted adapter can map one onto the other without inventing a fourth state.
    ``NOT_ASSESSED`` is this offline contract's explicit "no eligibility view was
    supplied" value; it must never be read as "eligible" when an assessment did run.
    """

    NOT_ASSESSED = "not_assessed"
    VERIFIED_ELIGIBLE = "verified_eligible"
    INELIGIBLE = "ineligible"
    UNRESOLVED = "unresolved"

    @property
    def is_feasible(self) -> bool:
        """True only for proved-compliant candidates."""
        return self is EligibilityState.VERIFIED_ELIGIBLE

    @property
    def was_assessed(self) -> bool:
        """True when a trusted eligibility decision exists for this candidate."""
        return self is not EligibilityState.NOT_ASSESSED


class EvidenceState(str, Enum):
    """Three-state preference evidence; the values match the accepted M10A statuses."""

    MATCH = "match"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class RankingArm(str, Enum):
    """The three preregistered offline ranking arms."""

    #: Existing candidate ordering (the RRF / original baseline).
    A0_ORIGINAL = "a0_original"
    #: The accepted M10A evidence + M10B reranker over explicit preferences.
    A1_EXPLICIT_RERANK = "a1_explicit_rerank"
    #: A1 plus a bounded ordinal inferred-preference tie-break.
    A2_BELIEF_TIEBREAK = "a2_belief_tiebreak"


class SourceRank(BaseModel):
    """One retrieval source's rank for a candidate.  Provenance, never a score to add."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(..., min_length=1, max_length=120)
    rank: int = Field(..., ge=1)
    score: float | None = Field(
        default=None,
        description=(
            "The source's own score, recorded only for provenance.  Scores from different "
            "sources are never comparable and no policy may read this numerically."
        ),
    )
    score_kind: str | None = Field(
        default=None, description="What score measures, e.g. 'sasrec_logit' / 'bm25'."
    )


class CandidateRecord(BaseModel):
    """One trusted candidate inside a frozen snapshot.

    Every field is copied from the trusted pipeline; nothing here is reconstructed from
    a digest.  ``metadata_status`` is explicit so a missing catalogue record is visible
    rather than silently represented by ``metadata=None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., min_length=1, max_length=200)
    item_id: int = Field(..., ge=1, description="Trusted integer item id; PAD=0 is never valid.")
    original_rank: int = Field(..., ge=1, description="Trusted upstream rank/order, 1-based.")
    score: float = Field(..., description="Copy of the ordering score; never modified.")
    score_kind: str = Field(default="unspecified", max_length=80)
    metadata_status: Literal["found", "missing"] = "missing"
    metadata: ProductMetadata | None = None
    source_ranks: tuple[SourceRank, ...] = ()
    rrf_score: float | None = Field(
        default=None, description="RRF fused score when the trusted pipeline produced one."
    )
    rrf_score_kind: str = Field(default="rrf_rank_fusion", max_length=80)
    eligibility: EligibilityState = EligibilityState.NOT_ASSESSED
    eligibility_detail: str | None = Field(default=None, max_length=300)

    @field_validator("candidate_id")
    @classmethod
    def _validate_candidate_id(cls, value: object) -> str:
        text = " ".join(str(value).split())
        if not text:
            raise ValueError("candidate_id must not be blank")
        return text

    @model_validator(mode="after")
    def _validate_metadata_consistency(self) -> CandidateRecord:
        if self.metadata is not None:
            if self.metadata_status != "found":
                raise ValueError("metadata present but metadata_status is not 'found'")
            if self.metadata.parent_asin != self.candidate_id:
                raise ValueError(
                    "catalogue metadata belongs to a different parent_asin than the candidate"
                )
        elif self.metadata_status != "missing":
            raise ValueError("metadata_status is 'found' but no metadata is attached")
        return self


class CandidateSnapshot(BaseModel):
    """Immutable, trusted candidate set for exactly one user/task/turn.

    The snapshot is supplied by the evaluation pipeline (it is not persisted in the M5
    event store and no digest is used to reconstruct rows).  Its reference/digest fields
    are verification material against the M5 candidate/ranking snapshot events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(..., min_length=1, max_length=200)
    user_key: str = Field(..., min_length=1, max_length=200)
    task_id: str = Field(..., min_length=1, max_length=200)
    turn_key: str = Field(..., min_length=1, max_length=200)
    candidates: tuple[CandidateRecord, ...] = ()
    source: str = Field(default="trusted_candidate_snapshot", max_length=120)
    schema_version: int = CANDIDATE_SNAPSHOT_SCHEMA_VERSION
    candidate_pool_ref: str | None = Field(default=None, max_length=200)
    ranking_snapshot_ref: str | None = Field(default=None, max_length=200)
    candidate_pool_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ranking_snapshot_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_candidate_set(self) -> CandidateSnapshot:
        if self.schema_version != CANDIDATE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported candidate snapshot schema version")
        identities = [candidate.candidate_id for candidate in self.candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("a candidate snapshot may not contain duplicate identities")
        item_ids = [candidate.item_id for candidate in self.candidates]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("a candidate snapshot may not contain duplicate item ids")
        ranks = [candidate.original_rank for candidate in self.candidates]
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(
                "candidate original ranks must be contiguous 1..n in snapshot order"
            )
        return self

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidate identities, in trusted original order."""
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def candidate_count(self) -> int:
        """Number of candidates in the snapshot."""
        return len(self.candidates)

    def candidate_pool_digest_value(self) -> str:
        """Order-insensitive SHA-256 of the candidate identity set (M5 algorithm)."""
        from recommendation.state.shopping_adapter import candidate_pool_snapshot

        return candidate_pool_snapshot(self.candidate_ids).digest

    def ranking_digest_value(self) -> str:
        """Order-sensitive SHA-256 of the trusted original order (M5 algorithm)."""
        from recommendation.state.shopping_adapter import ranking_snapshot

        return ranking_snapshot(self.candidate_ids).digest


class SnapshotExpectation(BaseModel):
    """The identity/context a trusted consumer expects a snapshot to belong to.

    Supplying this is how a caller proves ownership and detects staleness: a snapshot
    whose user, task, turn, count, references or digests disagree is rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_key: str = Field(..., min_length=1, max_length=200)
    task_id: str = Field(..., min_length=1, max_length=200)
    turn_key: str = Field(..., min_length=1, max_length=200)
    candidate_count: int | None = Field(default=None, ge=0)
    candidate_pool_ref: str | None = Field(default=None, max_length=200)
    ranking_snapshot_ref: str | None = Field(default=None, max_length=200)
    candidate_pool_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ranking_snapshot_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PreferenceStrengthMetadata(BaseModel):
    """Ordinal, non-calibrated preference strength metadata from the M5 projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_id: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., max_length=120)
    polarity: str = Field(..., max_length=20)
    direction: str | None = Field(default=None, max_length=20)
    scope: str = Field(..., max_length=20)
    lifecycle: str = Field(..., max_length=40)
    strength_rank: int = Field(..., ge=1, le=3)
    is_soft: bool


class InferredPreferenceEvidence(BaseModel):
    """Evidence for one *inferred* preference belief against one candidate.

    Directional numeric beliefs also carry ordinal peer-comparison counts computed over
    the frozen candidate set.  They are ordinal comparisons inside one set, not
    probabilities and not cross-model scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_id: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., max_length=120)
    polarity: str = Field(..., max_length=20)
    direction: str | None = Field(default=None, max_length=20)
    scope: str = Field(..., max_length=20)
    lifecycle: str = Field(default="inferred_hypothesis", max_length=40)
    strength_rank: int = Field(default=1, ge=1, le=3)
    status: EvidenceState
    reason_code: str = Field(..., min_length=1, max_length=80)
    metadata_field: str | None = Field(default=None, max_length=120)
    metadata_value: str | None = Field(default=None, max_length=200)
    ordinal_value: float | None = None
    #: Peer-comparison counts among *known* values in the same frozen candidate set.
    #: ``better_than_count`` counts other candidates whose grounded value is strictly
    #: better for this belief's direction (they outrank this candidate); it contributes to
    #: ``inferred_ordinal_penalty``.  ``worse_than_count`` counts others whose value is
    #: strictly worse (this candidate outranks them); it contributes to
    #: ``inferred_ordinal_support``.  Both are zero for an UNKNOWN observed attribute.
    better_than_count: int = Field(default=0, ge=0)
    worse_than_count: int = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=300)


class CatalogueFields(BaseModel):
    """Which catalogue attributes were actually available for one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    present: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidateRankingFeatures(BaseModel):
    """Typed, deterministic per-candidate features consumed by the ranking policies.

    Explicit evidence is the accepted M10A record (reused unchanged).  Inferred evidence
    is M5 soft-belief evidence with ordinal peer comparisons.  No field is a probability
    and no field combines scores from different models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., min_length=1, max_length=200)
    item_id: int = Field(..., ge=1)
    original_rank: int = Field(..., ge=1)
    source_ranks: tuple[SourceRank, ...] = ()
    rrf_score: float | None = None
    rrf_score_kind: str = "rrf_rank_fusion"
    explicit_matches: int = Field(default=0, ge=0)
    explicit_violations: int = Field(default=0, ge=0)
    explicit_unknown: int = Field(default=0, ge=0)
    inferred_matches: int = Field(default=0, ge=0)
    inferred_violations: int = Field(default=0, ge=0)
    inferred_unknown: int = Field(default=0, ge=0)
    #: Ordinal directional support/penalty summed over grounded inferred beliefs.
    inferred_ordinal_support: int = Field(default=0, ge=0)
    inferred_ordinal_penalty: int = Field(default=0, ge=0)
    explicit_evidence: tuple[PreferenceEvidence, ...] = ()
    inferred_evidence: tuple[InferredPreferenceEvidence, ...] = ()
    preference_strength_metadata: tuple[PreferenceStrengthMetadata, ...] = ()
    eligibility: EligibilityState = EligibilityState.NOT_ASSESSED
    eligibility_detail: str | None = Field(default=None, max_length=300)
    catalogue_fields: CatalogueFields = Field(default_factory=CatalogueFields)

    @property
    def explicit_key(self) -> tuple[int, int]:
        """The accepted M10B evidence key part: (violations ASC, -matches ASC)."""
        return (self.explicit_violations, -self.explicit_matches)

    @property
    def inferred_key(self) -> tuple[int, int]:
        """A2's ordinal tie-break part: (ordinal penalty ASC, -support ASC).

        UNKNOWN inferred evidence contributes nothing (0/0), so missing catalogue data is
        never counted as a violation.
        """
        return (self.inferred_ordinal_penalty, -self.inferred_ordinal_support)

    @property
    def tie_break_key(self) -> tuple[int, int, int, int]:
        """Explicit key, then inferred key; original rank/item id are appended by policies."""
        return (*self.explicit_key, *self.inferred_key)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CatalogueCoverage(BaseModel):
    """Catalogue feature availability and unsupported-attribute findings (diagnostics)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(default=0, ge=0)
    metadata_found_count: int = Field(default=0, ge=0)
    metadata_missing_count: int = Field(default=0, ge=0)
    #: Attribute -> number of candidates with a readable value (bounded field names).
    readable_fields: dict[str, int] = Field(default_factory=dict)
    unsupported_inferred_features: tuple[str, ...] = ()
    supported_inferred_features: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidateFeatureSet(BaseModel):
    """The builder's complete output for one frozen ranking task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    task_id: str
    user_key: str
    turn_key: str
    snapshot_id: str
    snapshot_candidate_count: int = Field(..., ge=0)
    candidate_pool_digest: str
    ranking_digest: str
    features: tuple[CandidateRankingFeatures, ...] = ()
    coverage: CatalogueCoverage = Field(default_factory=CatalogueCoverage)
    ranker_projection: RankerStateProjection | None = None

    @model_validator(mode="after")
    def _validate_features(self) -> CandidateFeatureSet:
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema version")
        identities = [feature.candidate_id for feature in self.features]
        if len(set(identities)) != len(identities):
            raise ValueError("feature set contains duplicate candidate identities")
        if len(identities) != self.snapshot_candidate_count:
            raise ValueError("feature set does not cover exactly the snapshot candidates")
        return self

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Candidate identities in snapshot/original order."""
        return tuple(feature.candidate_id for feature in self.features)


class TaskLabel(BaseModel):
    """One task-scoped relevance label for offline quality metrics.

    A label must come from a documented protocol; historical purchases are relevance
    labels for the retrieval target, not preference-conditioned satisfaction labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., min_length=1, max_length=200)
    relevance: float = Field(..., ge=0.0)
    label_source: str = Field(..., min_length=1, max_length=120)


class FrozenRankingTask(BaseModel):
    """One frozen paired-comparison task: identical candidates for every arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(..., min_length=1, max_length=200)
    user_key: str = Field(..., min_length=1, max_length=200)
    turn_key: str = Field(..., min_length=1, max_length=200)
    snapshot: CandidateSnapshot
    ranker_projection: RankerStateProjection
    labels: tuple[TaskLabel, ...] = ()
    label_protocol: str | None = Field(default=None, max_length=200)
    dataset_source: str = Field(default="synthetic", max_length=120)

    @model_validator(mode="after")
    def _validate_task_identity(self) -> FrozenRankingTask:
        if self.snapshot.user_key != self.user_key:
            raise ValueError("snapshot user_key does not match the task")
        if self.snapshot.task_id != self.task_id:
            raise ValueError("snapshot task_id does not match the task")
        if self.snapshot.turn_key != self.turn_key:
            raise ValueError("snapshot turn_key does not match the task")
        known = set(self.snapshot.candidate_ids)
        for label in self.labels:
            if label.candidate_id not in known:
                raise ValueError("a task label names a candidate outside the snapshot")
        return self


class RankedCandidate(BaseModel):
    """One candidate as placed by one arm.  Counts are copied, not rescored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    item_id: int = Field(..., ge=1)
    rank: int = Field(..., ge=1)
    original_rank: int = Field(..., ge=1)
    explicit_violations: int = Field(default=0, ge=0)
    explicit_matches: int = Field(default=0, ge=0)
    inferred_violations: int = Field(default=0, ge=0)
    inferred_matches: int = Field(default=0, ge=0)
    eligibility: EligibilityState = EligibilityState.NOT_ASSESSED
    reason: str = Field(default="", max_length=300)


class ArmRanking(BaseModel):
    """The same candidate set, ordered by one arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: RankingArm
    task_id: str
    candidate_ids: tuple[str, ...] = ()
    ranked: tuple[RankedCandidate, ...] = ()
    candidate_count: int = Field(default=0, ge=0)
    candidate_set_digest: str = ""
    policy_description: str = ""

    def rank_of(self, candidate_id: str) -> int | None:
        """Return the arm's rank for one candidate, or ``None`` when absent."""
        for item in self.ranked:
            if item.candidate_id == candidate_id:
                return item.rank
        return None


class ArmContractMetrics(BaseModel):
    """Contract/behavioural diagnostics for one arm.  Never quality evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(default=0, ge=0)
    candidate_identity_preserved: bool = True
    candidate_set_equal_to_snapshot: bool = True
    #: True when the arm ranks exactly the expected pool after the shared eligibility
    #: projection (the snapshot when nothing was assessed, otherwise only the verified
    #: eligible candidates).  ``candidate_set_equal_to_snapshot`` is False when the trusted
    #: eligibility decision legitimately removed candidates from the ranking pool.
    candidate_set_equal_to_expected_pool: bool = True
    ordering_deterministic: bool = True
    eligibility_invariant_violations: int = Field(default=0, ge=0)
    change_rate_vs_a0: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k_overlap_vs_a0: dict[str, float] = Field(default_factory=dict)
    change_rate_vs_a1: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k_overlap_vs_a1: dict[str, float] = Field(default_factory=dict)
    explicit_violations_total: int = Field(default=0, ge=0)
    explicit_matches_total: int = Field(default=0, ge=0)
    inference_covered_candidates: int = Field(default=0, ge=0)
    inference_unknown_candidates: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class QualityMetrics(BaseModel):
    """Offline ranking-quality metrics.  ``MEASURED`` only when labels were supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["MEASURED", "NOT_EXECUTED_NO_LABELS", "NOT_APPLICABLE"] = (
        "NOT_EXECUTED_NO_LABELS"
    )
    label_protocol: str | None = Field(default=None, max_length=200)
    task_count_with_labels: int = Field(default=0, ge=0)
    mrr: float | None = None
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    note: str = Field(default="", max_length=400)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class ArmEvaluation(BaseModel):
    """One arm's contract and (when labels exist) quality evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: RankingArm
    contract: ArmContractMetrics
    quality: QualityMetrics = Field(default_factory=QualityMetrics)


class PairedDelta(BaseModel):
    """Paired difference of one metric between two arms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    comparison: str = Field(..., description="e.g. 'a2_belief_tiebreak - a1_explicit_rerank'")
    mean_delta: float
    ci_low: float | None = None
    ci_high: float | None = None
    sample_size: int = Field(default=0, ge=0)
    status: Literal["MEASURED", "NOT_EXECUTED_NO_LABELS", "NOT_COMPUTABLE"] = "NOT_COMPUTABLE"
    note: str = Field(default="", max_length=400)


class EvaluationReport(BaseModel):
    """Frozen paired offline evaluation output for the synthetic/contract protocol.

    ``quality_status`` is the only field that may justify a quality claim; when it is not
    ``MEASURED`` the deltas describe ranking behaviour only and no effectiveness claim is
    permitted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_schema_version: int = EVALUATION_SCHEMA_VERSION
    protocol_name: str
    protocol_digest: str
    dataset_source: str
    task_count: int = Field(default=0, ge=0)
    candidate_count_total: int = Field(default=0, ge=0)
    arms: tuple[ArmEvaluation, ...] = ()
    paired_deltas: tuple[PairedDelta, ...] = ()
    quality_status: Literal["MEASURED", "NOT_EXECUTED_NO_LABELS"] = (
        "NOT_EXECUTED_NO_LABELS"
    )
    notes: tuple[str, ...] = ()
    report_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()

    def with_digest(self) -> EvaluationReport:
        """Return a copy with ``report_digest`` set over the canonical JSON body."""

        payload = self.model_dump(mode="json")
        payload.pop("report_digest", None)
        from recommendation.state.schemas import canonical_json

        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return self.model_copy(update={"report_digest": digest})
