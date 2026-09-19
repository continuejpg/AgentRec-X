"""``ActionValidator`` - the boundary an ``ActionProposal`` must cross (2.0-alpha).

The validator is the only component that can turn an untrusted proposal into a
:class:`~recommendation.control.schemas.ValidatedAction`.  It does four things and
nothing else:

1. **Availability.** The proposed action must be one the *system* currently offers
   (``context.available_actions``).  A policy that names an action the run does not
   support is refused, not accommodated.
2. **Argument validation.** ``k`` is re-validated against the accepted Tool range, and
   rejected outright if it is not a strict integer.  This reuses the Tool's own bounds;
   a policy cannot widen them.
3. **Stamping.** ``action_id``, ``step_index``, ``run_id`` and ``turn_id`` are generated
   *here*, by the controller side.  A proposal has no field for any of them, so execution
   metadata is controller-owned by construction.
4. **Refusal.** Every rejection is a :class:`~recommendation.control.schemas.VerificationResult`
   with a stable code.  There is deliberately no default action: a refused proposal ends
   the run deterministically rather than being replaced by a guess.

What the validator does **not** do: it does not execute anything, it does not read or
write trusted state, and it cannot repair a proposal into a different action.
"""

from __future__ import annotations

from .schemas import (
    ActionKind,
    ActionProposal,
    PolicyActionError,
    ValidatedAction,
    VerificationResult,
)
from recommendation.tools.schemas import MIN_K

__all__ = ["ActionValidator", "ACTION_ID_PREFIX"]


#: Prefix for controller-generated action identities, so a stamped action is
#: recognisable as controller-owned in a trajectory or log.
ACTION_ID_PREFIX = "act"


class ActionValidator:
    """Validate proposals against the system-provided action space.

    Stateless and deterministic: the same ``(proposal, context)`` pair always produces
    the same verdict and the same stamped ``action_id``.
    """

    def __init__(self, *, action_id_prefix: str = ACTION_ID_PREFIX) -> None:
        self._prefix = action_id_prefix

    def action_id(self, *, run_id: str, step_index: int, action: ActionKind) -> str:
        """Return the deterministic, controller-owned identity of one action."""
        return f"{self._prefix}:{run_id}:{step_index}:{action.value}"

    def validate(
        self,
        proposal: ActionProposal,
        *,
        run_id: str,
        turn_id: str | None,
        step_index: int,
        available_actions: tuple[ActionKind, ...],
    ) -> tuple[ValidatedAction | None, VerificationResult]:
        """Return ``(validated_action, result)``.

        Exactly one of the two is meaningful: when ``result.verified`` is true the
        validated action is present; otherwise it is ``None`` and ``result.code`` says
        why.  Both are returned so the caller records the refusal in the trajectory
        without having to catch an exception for an expected outcome.
        """
        if not isinstance(proposal, ActionProposal):
            return None, VerificationResult(
                verified=False,
                code="not_a_proposal",
                detail=f"expected an ActionProposal, got {type(proposal).__name__}",
                checks=("proposal_type",),
            )

        if proposal.action not in available_actions:
            return None, VerificationResult(
                verified=False,
                code="action_not_available",
                detail=(
                    f"'{proposal.action.value}' is not available in this state; "
                    f"allowed: {', '.join(a.value for a in available_actions) or 'none'}"
                ),
                checks=("availability",),
            )

        # ``k`` was already range-checked by the proposal schema; this is the trusted
        # re-check at the boundary, including the strict-integer rule the Tool applies.
        if proposal.action is ActionKind.RECOMMEND_FROM_HISTORY:
            k = proposal.k
            if isinstance(k, bool) or not isinstance(k, int):
                return None, VerificationResult(
                    verified=False,
                    code="invalid_k",
                    detail=f"k must be a strict integer, got {type(k).__name__}",
                    checks=("availability", "k_type"),
                )
            try:
                resolved_k = proposal.requested_k
            except PolicyActionError as exc:  # pragma: no cover - schema prevents this
                return None, VerificationResult(
                    verified=False,
                    code="invalid_k",
                    detail=str(exc),
                    checks=("availability", "k_present"),
                )
        else:
            resolved_k = 0

        validated = ValidatedAction(
            action=proposal.action,
            action_id=self.action_id(
                run_id=run_id, step_index=step_index, action=proposal.action
            ),
            step_index=step_index,
            run_id=run_id,
            turn_id=turn_id,
            # ``k`` is only meaningful for RECOMMEND_FROM_HISTORY.  For FINISH the
            # validated action keeps the schema's minimum and nothing ever reads it -
            # the capability refuses any action that is not RECOMMEND_FROM_HISTORY.
            k=resolved_k if proposal.action is ActionKind.RECOMMEND_FROM_HISTORY else MIN_K,
            rationale=proposal.rationale,
        )
        return validated, VerificationResult(
            verified=True,
            code="accepted",
            detail=None,
            checks=(
                ("availability", "arguments")
                if proposal.action is ActionKind.RECOMMEND_FROM_HISTORY
                else ("availability",)
            ),
        )
