"""AgentRec-X 2.0-alpha control plane: a bounded agent loop over a frozen pipeline.

Stage 1 changes **who decides the next step**.  It does not change how a recommendation is
produced, and this package never scores, ranks, retrieves, matches, reranks, renders or
commits anything itself - it calls the accepted components.

::

    PolicyContext  --(AgentPolicy.choose)-->  ActionProposal        untrusted
                                                    |
                                          ActionValidator            controller-owned
                                                    |
                                              ValidatedAction        stamped metadata
                                                    |
                                 RecommendFromHistoryCapability  accepted pipeline
                                                    |
                                         RecommendationDomainResult  raw, NOT policy-visible
                                                    |
                                          ResultVerifier             structural/identity
                                                    |
                                           ObservationAdapter        minimised
                                                    |
                                        RecommendationObservation    what the policy may see
                                                    |
                                            LoopController           owns the loop

Two policies ship: :class:`~recommendation.control.policy.RuleBasedPolicy` (deterministic,
Stage 1's default) and any future ``LLMPolicy`` implementing the same one method -
``choose(context) -> ActionProposal`` - which is the only thing the loop requires.

Deliberately absent in Stage 1: tool search, clarification, planning, multiple
recommendation sources, autonomous commerce actions, and any recommendation-semantics
change whatsoever.
"""

from __future__ import annotations

from .arguments import (
    AskClarificationArguments,
    BundleArguments,
    CandidateSource,
    CheckCompatibilityArguments,
    CompareArguments,
    GetDetailsArguments,
    SearchCatalogArguments,
    SelectSourceArguments,
    TradeOffArguments,
)
from .candidate_ledger import (
    CandidateLedger,
    CandidateProvenance,
    CandidateVerificationStatus,
    LedgerEntry,
    LedgerSourceError,
    reciprocal_rank_fusion,
)
from .candidate_plane import (
    CANDIDATE_PLANE_VERSION,
    CandidatePlane,
    CandidateSourceTool,
    CandidateSourceUnavailable,
)
from .capability import (
    CAPABILITY_NAME,
    RecommendFromHistoryCapability,
    TrustedHistoryReader,
)
from .catalog_search import (
    CATALOG_SEARCH_SCORE_KIND,
    CatalogSearchHit,
    CatalogSearchResult,
    CatalogSearchSource,
)
from .completion import CompletionGuard
from .context import (
    CandidateState,
    PolicyContext,
    project_constraints,
    project_intent,
)
from .grounded_reasoning import (
    GROUNDED_FACTS_VERSION,
    UNKNOWN,
    ComparisonResult,
    CompatibilityRequirement,
    CompatibilityVerdict,
    ConstraintKind,
    ConstraintReport,
    ConstraintVerdict,
    GroundedFacts,
    GroundedReasoner,
    parse_dimensions_cm,
    parse_price,
    parse_weight_kg,
)
from .grounding import (
    GroundingReport,
    GroundingVerifier,
    IdentityMapLike,
    MappingOnlyCatalog,
    UngroundedCandidate,
)
from .loop import LOOP_CONTROLLER_VERSION, LoopController, LoopResult, build_run_id
from .policy import RECOMMENDATION_FIRST, RuleBasedPolicy
from .schemas import (
    ARGUMENTS_BY_ACTION,
    CANDIDATE_ACTIONS,
    CONTROL_PLANE_VERSION,
    NON_EXECUTING_ACTIONS,
    READ_ONLY_ACTIONS,
    OBSERVATION_VERSION,
    STAGE_1_ACTIONS,
    ActionKind,
    ActionProposal,
    AgentPolicy,
    CandidateSetObservation,
    ClarificationObservation,
    ControlState,
    DomainResult,
    FailureObservation,
    LoopLimits,
    Observation,
    PolicyActionError,
    RecommendationDomainResult,
    RecommendationObservation,
    RunStatus,
    StateChange,
    TerminationReason,
    TrajectoryStep,
    ValidatedAction,
    VerificationResult,
)
from .memory_proposal import (
    MEMORY_PROPOSAL_VERSION,
    MemoryProposalValidator,
    MemoryWriteOperation,
    MemoryWriteProposal,
    ProposalValidation,
    ProposalVerdict,
    approved_candidates,
    proposals_from_extraction,
)
from .task_state import (
    TASK_STATE_VERSION,
    ConstraintOrigin,
    PreferenceLifecycle,
    PreferenceScope,
    TaskConstraint,
    TaskIntent,
    TaskState,
)
from .trajectory import TrajectoryRecorder
from .validation import ActionValidator
from .verification import ObservationAdapter, ResultVerifier

__all__ = [
    "ARGUMENTS_BY_ACTION",
    "CAPABILITY_NAME",
    "ConstraintOrigin",
    "MEMORY_PROPOSAL_VERSION",
    "TASK_STATE_VERSION",
    "CATALOG_SEARCH_SCORE_KIND",
    "CANDIDATE_ACTIONS",
    "CANDIDATE_PLANE_VERSION",
    "NON_EXECUTING_ACTIONS",
    "READ_ONLY_ACTIONS",
    "CONTROL_PLANE_VERSION",
    "LOOP_CONTROLLER_VERSION",
    "OBSERVATION_VERSION",
    "RECOMMENDATION_FIRST",
    "STAGE_1_ACTIONS",
    "ActionKind",
    "ActionProposal",
    "AskClarificationArguments",
    "ActionValidator",
    "AgentPolicy",
    "CandidateLedger",
    "CandidatePlane",
    "CandidateProvenance",
    "CandidateSetObservation",
    "CandidateSource",
    "CandidateSourceTool",
    "CandidateSourceUnavailable",
    "CandidateState",
    "CandidateVerificationStatus",
    "ComparisonResult",
    "CompatibilityRequirement",
    "CompatibilityVerdict",
    "ConstraintKind",
    "ConstraintReport",
    "ConstraintVerdict",
    "CatalogSearchHit",
    "CatalogSearchResult",
    "CatalogSearchSource",
    "ClarificationObservation",
    "CompletionGuard",
    "CompareArguments",
    "ControlState",
    "DomainResult",
    "FailureObservation",
    "GroundedFacts",
    "GroundedReasoner",
    "GroundingReport",
    "GroundingVerifier",
    "IdentityMapLike",
    "LedgerEntry",
    "GetDetailsArguments",
    "LedgerSourceError",
    "LoopController",
    "LoopLimits",
    "LoopResult",
    "MappingOnlyCatalog",
    "MemoryProposalValidator",
    "MemoryWriteOperation",
    "MemoryWriteProposal",
    "Observation",
    "ObservationAdapter",
    "PolicyActionError",
    "PolicyContext",
    "PreferenceLifecycle",
    "PreferenceScope",
    "ProposalValidation",
    "ProposalVerdict",
    "RecommendFromHistoryCapability",
    "RecommendationDomainResult",
    "RecommendationObservation",
    "ResultVerifier",
    "RuleBasedPolicy",
    "RunStatus",
    "SearchCatalogArguments",
    "SelectSourceArguments",
    "StateChange",
    "TaskConstraint",
    "TaskIntent",
    "TaskState",
    "TradeOffArguments",
    "UNKNOWN",
    "TerminationReason",
    "TrajectoryRecorder",
    "TrajectoryStep",
    "TrustedHistoryReader",
    "UngroundedCandidate",
    "ValidatedAction",
    "VerificationResult",
    "approved_candidates",
    "build_run_id",
    "parse_dimensions_cm",
    "parse_price",
    "parse_weight_kg",
    "project_constraints",
    "project_intent",
    "proposals_from_extraction",
    "reciprocal_rank_fusion",
]

__version__ = "0.1.0"
