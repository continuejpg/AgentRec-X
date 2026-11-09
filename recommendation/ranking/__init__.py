"""M6.1 — belief-aware offline ranking foundation (lab package).

This package is **offline only**.  It joins a frozen trusted candidate snapshot, the M5
``for_ranker`` projection and the accepted M10A/M10B preference evidence into typed
features, ranks them with three preregistered policies (A0/A1/A2), and evaluates them
under a frozen paired protocol.  It is never imported by the serving path, never changes
Serving V1, and makes no recommendation-quality claim unless labels from a documented
protocol exist.

See ``README.md`` and ``docs/M6_1_BELIEF_AWARE_RANKING.md`` /
``docs/M6_1_RANKING_PROTOCOL.md``.
"""

from __future__ import annotations

from .errors import (
    CandidateSnapshotError,
    FeatureBuildError,
    RankingError,
    RankingPolicyError,
    RankingProtocolError,
)
from .schemas import (
    ArmContractMetrics,
    ArmEvaluation,
    ArmRanking,
    CandidateFeatureSet,
    CandidateRankingFeatures,
    CandidateRecord,
    CandidateSnapshot,
    CatalogueCoverage,
    CatalogueFields,
    EligibilityState,
    EvaluationReport,
    EvidenceState,
    FrozenRankingTask,
    InferredPreferenceEvidence,
    PairedDelta,
    PreferenceStrengthMetadata,
    QualityMetrics,
    RankedCandidate,
    RankingArm,
    SnapshotExpectation,
    SourceRank,
    TaskLabel,
)

__all__ = [
    "ArmContractMetrics",
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
    "SourceRank",
    "SnapshotExpectation",
    "TaskLabel",
    "CandidateSnapshotError",
    "FeatureBuildError",
    "RankingError",
    "RankingPolicyError",
    "RankingProtocolError",
]

__version__ = "0.1.0"
