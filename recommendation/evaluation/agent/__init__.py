"""Agent / trajectory evaluation plane (AgentRec-X 2.0-alpha Stage 6).

This package is the **third** evaluation plane, and it exists precisely so that the three are
not collapsed into one number:

:mod:`recommendation.evaluation.metrics` / :mod:`~recommendation.evaluation.evaluator`
    **Recommender quality** - HR@K, Recall@K, NDCG@K under the accepted full-catalogue,
    temporal leave-two-out protocol.  Unchanged, and *not* extended here.

:mod:`recommendation.evaluation.agent.cases` / :mod:`~recommendation.evaluation.agent.runner`
    **Agent / trajectory quality** - did the run choose sensible actions, respect its
    budgets, recover when a source came back empty, avoid using a verification it never
    performed, and terminate for a defensible reason.

The distinction matters because the metrics are not interchangeable.  NDCG measures where the
target item lands in a ranking; it says nothing about whether an agent asked a useful question
or recovered from an empty retrieval.  A trajectory metric says nothing about recommendation
accuracy.  Reporting either as if it measured the other is the error this plane is designed to
make impossible to commit accidentally.

What is deliberately absent
---------------------------
No single aggregate "agent score".  :class:`~recommendation.evaluation.agent.metrics.TrajectoryMetrics`
reports each dimension separately and
:class:`~recommendation.evaluation.agent.attribution.FailureAttribution` assigns each failure
to a component so a number cannot hide which part of the system produced it.
"""

from __future__ import annotations

from .attribution import (
    ATTRIBUTION_COMPONENTS,
    AttributionComponent,
    FailureAttribution,
)
from .cases import EVALUATION_CASES, EvaluationCase, case_by_id, load_cases
from .metrics import TrajectoryMetrics, compute_metrics
from .model_policy_adapter import (  # noqa: F401 - exported for the evaluation smoke
    ObservationReactiveModel,
    build_live_model_policy_factory,
    build_model_policy_factory,
)
from .runner import (
    ABLATION_ADAPTIVE,
    ABLATION_DECIDE_ONCE,
    ABLATION_LIVE_MODEL,
    ABLATION_MODEL_POLICY,
    CaseOutcome,
    CaseRunner,
    DecidingOncePolicy,
    run_case,
    run_suite,
)
from .schemas import (
    EVALUATION_PLANE_VERSION,
    AgentTrajectory,
    TerminalOutcome,
    TrajectoryRecord,
)

__all__ = [
    "ABLATION_ADAPTIVE",
    "ABLATION_DECIDE_ONCE",
    "ABLATION_LIVE_MODEL",
    "ABLATION_MODEL_POLICY",
    "ATTRIBUTION_COMPONENTS",
    "EVALUATION_CASES",
    "EVALUATION_PLANE_VERSION",
    "AgentTrajectory",
    "AttributionComponent",
    "CaseOutcome",
    "CaseRunner",
    "DecidingOncePolicy",
    "EvaluationCase",
    "FailureAttribution",
    "ObservationReactiveModel",
    "build_live_model_policy_factory",
    "TerminalOutcome",
    "TrajectoryMetrics",
    "TrajectoryRecord",
    "build_model_policy_factory",
    "case_by_id",
    "compute_metrics",
    "load_cases",
    "run_case",
    "run_suite",
]

__version__ = "0.1.0"
