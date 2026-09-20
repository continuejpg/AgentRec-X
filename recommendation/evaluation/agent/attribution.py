"""Failure attribution for the agent plane (Stage 6).

A single "agent score" is useless for engineering: it says something went wrong and nothing
about where.  This module maps each measured failure dimension onto the **component** that
owns it, so a report can say "candidate grounding failed in the trusted boundary" rather than
"the agent scored 0.7".

The mapping is data, and a test asserts it is total: every failure name
:meth:`~recommendation.evaluation.agent.metrics.TrajectoryMetrics.failures` can produce has a
component here.  That is what stops a new metric from being added without deciding who owns it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ATTRIBUTION_COMPONENTS",
    "AttributionComponent",
    "FailureAttribution",
]


class AttributionComponent(str, Enum):
    """The component a failure is attributed to.

    These are the *components*, not the symptoms: attributing a failure to "the agent" is the
    thing this enum exists to avoid.
    """

    #: A trusted candidate tool returned nothing useful, or too little.
    CANDIDATE_RETRIEVAL = "candidate_retrieval"
    #: Cross-source ordering or agreement went wrong.
    CANDIDATE_FUSION = "candidate_fusion"
    #: Identity was not confirmed against trusted data.
    CANDIDATE_GROUNDING = "candidate_grounding"
    #: The policy chose the wrong action or the wrong source.
    POLICY_SELECTION = "policy_selection"
    #: A question should have been asked, or the wrong one was.
    CLARIFICATION = "clarification"
    #: Constraint checking was skipped or produced the wrong verdict.
    CONSTRAINT_VERIFICATION = "constraint_verification"
    #: Completion was authorised when it should not have been, or refused when it should not.
    COMPLETION = "completion"
    #: A task-scoped instruction was persisted, or a durable one was not.
    MEMORY_SCOPE = "memory_scope"
    #: A prerequisite action was never taken.
    SEQUENCING = "sequencing"
    #: The run exceeded its budget or took steps that changed nothing.
    TRAJECTORY_EFFICIENCY = "trajectory_efficiency"
    #: The action protocol itself was violated.
    PROTOCOL = "protocol"


#: Failure dimension -> owning component.  Total over every name ``failures()`` emits.
ATTRIBUTION_COMPONENTS: dict[str, AttributionComponent] = {
    "terminal_not_allowed": AttributionComponent.COMPLETION,
    "tool_selection": AttributionComponent.POLICY_SELECTION,
    "required_action_missing": AttributionComponent.SEQUENCING,
    "forbidden_action_used": AttributionComponent.POLICY_SELECTION,
    "budget_overrun": AttributionComponent.TRAJECTORY_EFFICIENCY,
    "candidate_grounding": AttributionComponent.CANDIDATE_GROUNDING,
    "completion": AttributionComponent.COMPLETION,
    "memory_scope": AttributionComponent.MEMORY_SCOPE,
    "clarification": AttributionComponent.CLARIFICATION,
    # Declared for completeness even though no current metric emits them: a fusion or
    # protocol regression would surface through these names rather than being misfiled.
    "fusion_disagreement": AttributionComponent.CANDIDATE_FUSION,
    "retrieval_empty": AttributionComponent.CANDIDATE_RETRIEVAL,
    "protocol_violation": AttributionComponent.PROTOCOL,
    "constraint_violated": AttributionComponent.CONSTRAINT_VERIFICATION,
}


class FailureAttribution(BaseModel):
    """Failures of one run, grouped by the component that owns them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    variant: str
    #: Failure dimension -> owning component.
    attributed: dict[str, str] = Field(default_factory=dict)
    #: Components involved, sorted, for a compact report line.
    components: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """True when nothing failed."""
        return not self.attributed

    def for_component(self, component: AttributionComponent) -> tuple[str, ...]:
        """Return the failure dimensions attributed to ``component``."""
        return tuple(
            name for name, owner in self.attributed.items() if owner == component.value
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view."""
        return {
            "scenario_id": self.scenario_id,
            "variant": self.variant,
            "passed": self.passed,
            "attributed": dict(self.attributed),
            "components": list(self.components),
        }


def attribute(scenario_id: str, variant: str, failures: tuple[str, ...]) -> FailureAttribution:
    """Attribute a tuple of failure dimensions to their owning components.

    An unknown dimension is attributed to :attr:`AttributionComponent.PROTOCOL` rather than
    dropped: a metric that failed without a declared owner is itself a finding, and silently
    losing it would make a report look cleaner than the system is.
    """
    attributed: dict[str, str] = {}
    for name in failures:
        owner = ATTRIBUTION_COMPONENTS.get(name, AttributionComponent.PROTOCOL)
        attributed[name] = owner.value
    components = tuple(sorted(set(attributed.values())))
    return FailureAttribution(
        scenario_id=scenario_id,
        variant=variant,
        attributed=attributed,
        components=components,
    )
