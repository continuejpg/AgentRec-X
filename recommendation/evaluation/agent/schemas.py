"""Trajectory schema for the agent evaluation plane (Stage 6).

A trajectory is the *reconstructable record of one evaluated run*: what the policy was shown,
what it chose, whether that choice was authorised, what executed, what verification said, what
the state change was, and why the run stopped.  Stage 1's
:class:`~recommendation.control.trajectory.TrajectoryRecorder` already records most of this at
run time; this module is the **evaluation-facing projection** of it, with three differences:

* it is validated and versioned, so an offline analysis cannot drift from the runtime shape;
* it is explicitly *summarised* - the projection keeps the fields an evaluator needs and drops
  the rest, rather than re-exporting whatever the runtime happened to store;
* it carries the terminal outcome as a first-class value, so a suite can compare runs without
  re-deriving "did it work" from the step list.

What a trajectory must never contain, and therefore has no field for: the user's behavioural
history, the memory ``user_key``, memory entry identifiers, credentials, or a full hidden
prompt.  Those are excluded by the same serialization boundary the demo layer already
enforces, and the projection is built from whitelisted fields rather than by dumping runtime
state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EVALUATION_PLANE_VERSION",
    "AgentTrajectory",
    "TerminalOutcome",
    "TrajectoryRecord",
]

#: Version of the evaluation-plane contract.  Bumped when the trajectory or case shape changes,
#: so a stored report can be read against the schema that produced it.
EVALUATION_PLANE_VERSION = 1


class TerminalOutcome(str, Enum):
    """How a run ended, in the vocabulary an evaluator reasons about.

    Deliberately coarser than :class:`~recommendation.control.schemas.RunStatus`: an evaluator
    cares about *categories* of ending, and mapping them here keeps a suite from depending on
    runtime enum members that may be reorganised.
    """

    #: Completed through an accepted FINISH.
    COMPLETED = "completed"
    #: Suspended to ask the user a question.  A legal, non-failure ending.
    ASKED_CLARIFICATION = "asked_clarification"
    #: Stopped by a budget (step, tool-call or retry).
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: Stopped because the policy proposed something the system did not authorise.
    UNAUTHORISED_ACTION = "unauthorised_action"
    #: Stopped because completion was refused and could not be retried.
    COMPLETION_REFUSED = "completion_refused"
    #: Stopped by an execution failure.
    FAILED = "failed"


class TrajectoryRecord(BaseModel):
    """One step of a trajectory, in evaluation-facing form.

    Every field is a summary or a verdict.  ``observation_summary`` holds a *small* projection
    of the observation (kind, status, counts) rather than the observation itself, so a
    reasoning observation's grounded facts do not flow into an evaluation report by accident.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(..., ge=0)
    action_id: str
    #: The action the policy proposed, or ``None`` when it proposed nothing.
    proposed_action: str | None = None
    #: True when the validator authorised the proposal.
    authorised: bool = False
    #: Stable code from validation or verification; ``None`` when nothing refused.
    refusal_code: str | None = None
    #: The action that actually executed (differs from the proposal when refused).
    executed_action: str | None = None
    #: Observation kind and status, e.g. ``("candidate_set", "empty")``.
    observation_kind: str | None = None
    observation_status: str | None = None
    #: Counts a suite compares on: candidates returned, ungrounded identities seen.
    returned_k: int = Field(default=0, ge=0)
    ungrounded_count: int = Field(default=0, ge=0)
    #: Structural delta: did this step change the candidate set?
    produced_candidates: bool = False
    #: Whether the step consumed a tool call (candidate actions do; reasoning does not).
    consumed_tool_call: bool = False
    note: str | None = Field(default=None, max_length=200)


class AgentTrajectory(BaseModel):
    """The evaluated record of one run.

    ``records`` is the per-step list; ``terminal`` is how it ended.  ``scenario`` names the
    evaluation case, so a suite of trajectories is self-describing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = EVALUATION_PLANE_VERSION
    scenario_id: str
    #: Which control-plane configuration produced this run, for ablations.
    variant: str = "adaptive"
    terminal: TerminalOutcome
    #: The runtime's own termination reason, kept verbatim for audit.
    termination_reason: str | None = None
    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    records: tuple[TrajectoryRecord, ...] = ()
    #: Route the run reported, e.g. ``recommend`` or ``direct``.
    route: str = "direct"
    #: Whether the run produced a grounded candidate set at any point.
    produced_candidates: bool = False

    # -- derived views ----------------------------------------------------- #

    def action_sequence(self) -> tuple[str, ...]:
        """The actions actually executed, in order."""
        return tuple(
            record.executed_action
            for record in self.records
            if record.executed_action is not None
        )

    def proposed_sequence(self) -> tuple[str, ...]:
        """The actions proposed, in order (including refused ones)."""
        return tuple(
            record.proposed_action
            for record in self.records
            if record.proposed_action is not None
        )

    def refusal_codes(self) -> tuple[str, ...]:
        """Every refusal code recorded, in order."""
        return tuple(
            record.refusal_code for record in self.records if record.refusal_code
        )

    def used_action(self, action: str) -> bool:
        """True when ``action`` executed at least once."""
        return action in self.action_sequence()

    def executed_action_count(self, action: str) -> int:
        """How many times ``action`` executed."""
        return sum(1 for name in self.action_sequence() if name == action)

    def ungrounded_seen(self) -> int:
        """Total identities a source returned that the trusted side did not know.

        A non-zero value is a finding about a candidate source: it drifted or hallucinated.
        """
        return sum(record.ungrounded_count for record in self.records)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()
