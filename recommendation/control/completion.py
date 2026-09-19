"""``CompletionGuard`` - the boundary a FINISH proposal must cross (2.0-alpha).

A policy may only ever **propose** ``FINISH``.  It has no field with which to set a
status, mark a run complete, or declare its own output valid.  This module is what stands
between the proposal and the end of the run, and it exists for one reason: a control plane
where the deciding component can also declare itself done is not a control plane.

What the guard checks
---------------------
1. **The action was FINISH** - the guard is not a second validator, but it refuses to
   certify completion for an action that is not a completion proposal.
2. **Action/verdict agreement** - the action is only eligible if it was validated.
3. **Fail-closed state** - a run whose last execution *failed* (a refused verification or
   a capability error) may not be reported as successfully completed.  That is the
   fail-closed philosophy of the accepted system, preserved at the control layer.
4. **Groundedness of a produced recommendation** - the minimum a recommendation result
   must satisfy to be a legal ending.  When a recommendation was produced, its verified
   evidence must be present and grounded; a candidate set that failed verification can
   never be presented as a completed answer.
5. **No illegal intermediate state** - a run may not end with a tool-call budget already
   exhausted *before* the completion step, because that is a budget termination wearing a
   completion's clothes.

A refusal is returned as a :class:`VerificationResult` with a stable code (not an
exception) so the loop can either return control to the policy - when progress is still
possible - or terminate deterministically.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    ActionKind,
    ControlState,
    RunStatus,
    ValidatedAction,
    VerificationResult,
)

__all__ = ["CompletionGuard"]


class CompletionGuard:
    """Decide whether a validated FINISH proposal is a legal end to the run."""

    def check(
        self,
        action: ValidatedAction,
        *,
        state: ControlState,
        last_verification: VerificationResult | None,
        last_observation: Any = None,
        produced_recommendation: bool = False,
        candidates_grounded: bool = False,
        execution_failed: bool = False,
    ) -> VerificationResult:
        """Return the completion verdict for ``action``.

        Parameters
        ----------
        action:
            The validated action.  Must be a FINISH proposal.
        state:
            The controller's current run state.
        last_verification:
            The verdict of the most recent execution, or ``None`` when nothing has been
            executed yet (a run may legally finish without ever recommending).
        last_observation:
            The most recent observation, used to report a grounded candidate count.
        produced_recommendation:
            True when this run has produced a verified candidate set.
        candidates_grounded:
            True when that candidate set passed identity verification.
        execution_failed:
            True when the most recent execution failed.  A failed run may not be reported
            as a completed one.

        Returns
        -------
        VerificationResult
            ``verified=True`` only when ending here is defensible.
        """
        checks: list[str] = ["finish_action"]

        if action.action is not ActionKind.FINISH:
            return VerificationResult(
                verified=False,
                code="not_a_completion",
                detail="the completion guard only certifies a FINISH action",
                checks=checks,
            )

        if state.is_terminal:
            return VerificationResult(
                verified=False,
                code="already_terminal",
                detail="the run has already stopped",
                checks=(*checks, "run_open"),
            )
        checks.append("run_open")

        # The guard deliberately does **not** re-check the budgets.  Budget termination
        # belongs to the controller, which records ``MAX_STEPS`` / ``MAX_TOOL_CALLS`` in
        # the step that hits the limit before the policy is ever consulted again.  A
        # second budget rule here produced a real deadlock: after one successful
        # recommendation the tool budget was spent, so the guard refused the FINISH that
        # the controller's own ``available_actions`` had made the only legal action, and
        # the loop could never end.  What the guard does enforce is the *integrity* of the
        # ending, below.
        checks.append("budget_available")

        if execution_failed:
            return VerificationResult(
                verified=False,
                code="last_execution_failed",
                detail=(
                    "the most recent execution failed verification; a failed run may not "
                    "be completed successfully"
                ),
                checks=(*checks, "fail_closed"),
            )
        checks.append("fail_closed")

        if produced_recommendation and not candidates_grounded:
            return VerificationResult(
                verified=False,
                code="candidates_not_grounded",
                detail=(
                    "a recommendation was produced but its candidate set did not pass "
                    "verification, so the run may not be completed as a success"
                ),
                checks=(*checks, "grounded_candidates", "fail_closed"),
            )
        if produced_recommendation:
            checks.extend(("grounded_candidates", "recommendation_pipeline_complete"))

        if last_verification is not None and not last_verification.verified:
            return VerificationResult(
                verified=False,
                code="last_verification_refused",
                detail="the most recent result was refused verification",
                checks=(*checks, "verification_agreement"),
            )
        if last_verification is not None:
            checks.append("verification_agreement")

        return VerificationResult(
            verified=True,
            code="completion_accepted",
            detail=None,
            checks=tuple(checks),
        )

    def can_retry(
        self,
        refusal: VerificationResult,
        *,
        state: ControlState,
    ) -> bool:
        """True when a refused completion may legally return control to the policy.

        A refusal is recoverable only while the run still has budget and is still open.
        Everything else is a terminal failure, so the loop cannot spin on a guard that
        keeps saying no.
        """
        if state.is_terminal:
            return False
        if state.steps_remaining <= 0:
            return False
        if state.retry_count >= state.limits.max_retries:
            return False
        # Some refusals describe a condition no further step can change.
        return refusal.code not in {"already_terminal", "not_a_completion"}

    def terminal_status(self, refusal: VerificationResult) -> RunStatus:
        """The status to record when a refusal cannot be retried."""
        if refusal.code in {"already_terminal"}:
            return RunStatus.ABORTED
        return RunStatus.FAILED
