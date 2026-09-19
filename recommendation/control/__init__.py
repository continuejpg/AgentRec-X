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

from .capability import (
    CAPABILITY_NAME,
    RecommendFromHistoryCapability,
    TrustedHistoryReader,
)
from .completion import CompletionGuard
from .context import CandidateState, PolicyContext
from .loop import LOOP_CONTROLLER_VERSION, LoopController, LoopResult, build_run_id
from .policy import RECOMMENDATION_FIRST, RuleBasedPolicy
from .schemas import (
    CONTROL_PLANE_VERSION,
    OBSERVATION_VERSION,
    STAGE_1_ACTIONS,
    ActionKind,
    ActionProposal,
    AgentPolicy,
    ControlState,
    DomainResult,
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
from .trajectory import TrajectoryRecorder
from .validation import ActionValidator
from .verification import ObservationAdapter, ResultVerifier

__all__ = [
    "CAPABILITY_NAME",
    "CONTROL_PLANE_VERSION",
    "LOOP_CONTROLLER_VERSION",
    "OBSERVATION_VERSION",
    "RECOMMENDATION_FIRST",
    "STAGE_1_ACTIONS",
    "ActionKind",
    "ActionProposal",
    "ActionValidator",
    "AgentPolicy",
    "CandidateState",
    "CompletionGuard",
    "ControlState",
    "DomainResult",
    "LoopController",
    "LoopLimits",
    "LoopResult",
    "Observation",
    "ObservationAdapter",
    "PolicyActionError",
    "PolicyContext",
    "RecommendFromHistoryCapability",
    "RecommendationDomainResult",
    "RecommendationObservation",
    "ResultVerifier",
    "RuleBasedPolicy",
    "RunStatus",
    "StateChange",
    "TerminationReason",
    "TrajectoryRecorder",
    "TrajectoryStep",
    "TrustedHistoryReader",
    "ValidatedAction",
    "VerificationResult",
    "build_run_id",
]

__version__ = "0.1.0"
