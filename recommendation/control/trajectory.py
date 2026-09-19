"""In-memory trajectory recording for the bounded agent loop (AgentRec-X 2.0-alpha).

Trajectory logging exists from the first version, not as a later addition, because the
only honest way to justify a control plane is to be able to answer, for every step:

* what did the policy see?
* what did it choose?
* why was the action accepted or refused?
* what did the capability do?
* did the result pass verification?
* what observation was produced?
* what changed in the state?
* why did the next step happen?

What is recorded
----------------
Summaries and references - never payloads.  Specifically: the policy-context *summary*
(booleans, counts, action names), the proposal, the validation verdict, the validated
action's controller-generated metadata, an opaque reference to the raw domain result, the
verification verdict, the minimised observation, and a compact state delta.

What is never recorded
----------------------
Secrets, credentials, API keys, trusted interaction history (not even a digest), the
memory ``user_key``, preference values, candidate identities or ``parent_asin`` values,
raw scores, product metadata, hidden prompts, and filesystem paths.

The recorder is a plain in-memory list, bounded by the loop's own step budget, so it can
neither grow without limit nor become a second source of truth: it observes the control
plane, it does not participate in it.
"""

from __future__ import annotations

from typing import Any, Sequence

from .schemas import (
    ActionProposal,
    StateChange,
    TrajectoryStep,
    ValidatedAction,
    VerificationResult,
)

__all__ = ["TrajectoryRecorder"]


class TrajectoryRecorder:
    """Append-only, bounded record of one loop run's control decisions."""

    def __init__(self, *, run_id: str, turn_id: str | None = None) -> None:
        self._run_id = run_id
        self._turn_id = turn_id
        self._steps: list[TrajectoryStep] = []

    # -- metadata ---------------------------------------------------------- #

    @property
    def run_id(self) -> str:
        """The controller-owned run identity."""
        return self._run_id

    @property
    def steps(self) -> tuple[TrajectoryStep, ...]:
        """The recorded steps, in execution order."""
        return tuple(self._steps)

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self._steps)

    # -- recording --------------------------------------------------------- #

    def record(
        self,
        *,
        step_index: int,
        action_id: str,
        policy_context_summary: dict[str, Any],
        action_proposal: ActionProposal | None = None,
        validation_result: VerificationResult | None = None,
        validated_action: ValidatedAction | None = None,
        tool_result_ref: str | None = None,
        verification_result: VerificationResult | None = None,
        observation: Any = None,
        state_delta: StateChange | None = None,
        note: str | None = None,
    ) -> TrajectoryStep:
        """Record one loop step and return the immutable record.

        Optional fields default to absent so a step that stopped before executing is
        represented honestly rather than with placeholder values that look like results.
        """
        step = TrajectoryStep(
            run_id=self._run_id,
            turn_id=self._turn_id,
            step_index=step_index,
            action_id=action_id,
            policy_context_summary=dict(policy_context_summary),
            action_proposal=(
                None
                if action_proposal is None
                else {
                    "action": action_proposal.action.value,
                    "k": action_proposal.k,
                    "rationale": action_proposal.rationale,
                    "version": action_proposal.version,
                }
            ),
            validation_result=validation_result,
            validated_action=(
                None
                if validated_action is None
                else {
                    "action": validated_action.action.value,
                    "action_id": validated_action.action_id,
                    "step_index": validated_action.step_index,
                    "k": validated_action.k,
                    "version": validated_action.version,
                }
            ),
            tool_result_ref=tool_result_ref,
            verification_result=verification_result,
            observation=(None if observation is None else observation.model_dump()),
            state_delta=state_delta,
            note=note,
        )
        self._steps.append(step)
        return step

    # -- export ------------------------------------------------------------ #

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return the trajectory as JSON-serialisable dictionaries."""
        return [step.model_dump() for step in self._steps]

    def actions(self) -> tuple[str, ...]:
        """The action kinds recorded, in order - a compact audit of the decision path."""
        return tuple(
            str(step.action_proposal["action"])
            for step in self._steps
            if step.action_proposal is not None
        )

    def refusals(self) -> tuple[str, ...]:
        """The verification/validation refusal codes recorded, in order."""
        codes: list[str] = []
        for step in self._steps:
            for result in (step.validation_result, step.verification_result):
                if result is not None and not result.verified:
                    codes.append(result.code)
        return tuple(codes)

    def to_sequence(self) -> Sequence[dict[str, Any]]:
        """Alias for :meth:`as_dicts`, for callers that treat it as a sequence."""
        return self.as_dicts()
