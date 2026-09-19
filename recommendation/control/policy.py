"""Stage 1 policies for the bounded agent loop (AgentRec-X 2.0-alpha).

``RuleBasedPolicy`` is a real, deterministic, production-quality policy - not a test
double.  It is the same kind of component as
:class:`~recommendation.demo.decision.DemoDecisionModel`, moved one level up: where that
model chose between two *routes* in a frozen DAG, this policy chooses the **next action**
in a loop.

The interface is the whole point::

    choose(context: PolicyContext) -> ActionProposal

An ``LLMPolicy`` implementing the same single method is a drop-in replacement.  Nothing
else in the control plane would change - not the validator, not the capability, not the
verifier, not the completion guard, not the loop controller, not the trajectory.  That
interchangeability is what Stage 1 is buying, and the policy test suite asserts it by
running the *identical* loop against two different policy implementations.

What the rule is
----------------
A recommendation is wanted when all of the following hold:

* the system currently permits ``RECOMMEND_FROM_HISTORY``;
* an application-owned trusted history exists (the accepted recommender has no cold-start
  fallback, so proposing a recommendation without history would guarantee a failure);
* no grounded candidate set has been produced for this run yet;
* the run is not in the state that follows a rejected proposal (a reflection signal: after
  a refusal the policy finishes rather than re-proposing and burning the loop budget).

Otherwise it proposes ``FINISH``.  There is no third outcome and no randomness.

What the policy cannot do
-------------------------
It cannot execute anything, write state, count its own steps, touch trusted history, name
a tool, supply candidate ids, commit memory, validate its own output or declare the run
finished.  It has no collaborator to do any of that with: its only input is a
:class:`~recommendation.control.context.PolicyContext` and its only output is an
:class:`~recommendation.control.schemas.ActionProposal`.
"""

from __future__ import annotations

from typing import Sequence

from .context import PolicyContext
from .schemas import ActionKind, ActionProposal, PolicyActionError

__all__ = ["DEFAULT_POLICY_K", "RECOMMENDATION_FIRST", "RuleBasedPolicy"]

#: Candidate count a policy asks for when it decides to recommend.  Kept as a literal so
#: the policy module imports nothing at all: the policy has no collaborator, no constant
#: table and no executable dependency, which is what makes "a policy can only choose" a
#: structural property rather than a promise.  The Tool re-validates the value against its
#: own accepted range at the trusted boundary.
DEFAULT_POLICY_K = 10


#: Documented preference order used when a policy may choose freely among available
#: actions: gather the evidence a recommendation needs before ending the turn.
RECOMMENDATION_FIRST: tuple[ActionKind, ...] = (
    ActionKind.RECOMMEND_FROM_HISTORY,
    ActionKind.FINISH,
)


class RuleBasedPolicy:
    """Deterministic Stage 1 policy: recommend once from history, then finish.

    Parameters
    ----------
    default_k:
        Candidate count requested when the policy decides to recommend.  A policy is
        allowed to *ask* for a count; the validator re-checks it against the accepted
        Tool range and the Tool remains the only component that can act on it.
    preference_order:
        The action preference order.  Exposed so a test can pin the policy's decision
        rule without re-implementing it.
    """

    def __init__(
        self,
        *,
        default_k: int = DEFAULT_POLICY_K,
        preference_order: Sequence[ActionKind] = RECOMMENDATION_FIRST,
    ) -> None:
        if not isinstance(default_k, int) or isinstance(default_k, bool) or default_k < 1:
            raise ValueError("default_k must be a positive integer")
        if not preference_order:
            raise ValueError("preference_order must not be empty")
        self._default_k = default_k
        self._preference_order = tuple(preference_order)
        self._call_count = 0

    # -- metadata ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Stable policy identity, recorded in the trajectory."""
        return "rule_based"

    @property
    def default_k(self) -> int:
        """The candidate count this policy asks for."""
        return self._default_k

    @property
    def call_count(self) -> int:
        """How many proposals this policy produced (diagnostics only)."""
        return self._call_count

    # -- the seam ---------------------------------------------------------- #

    def choose(self, context: PolicyContext) -> ActionProposal:
        """Return exactly one proposal drawn from ``context.available_actions``.

        Raises
        ------
        PolicyActionError
            No action is available at all.  The policy refuses rather than inventing one;
            the controller treats this as a deterministic termination.
        """
        self._call_count += 1
        available = context.available_actions
        if not available:
            raise PolicyActionError("no action is available in this control state")

        for action in self._preference_order:
            if action not in available:
                continue
            if action is ActionKind.RECOMMEND_FROM_HISTORY:
                if self._wants_recommendation(context):
                    return ActionProposal(
                        action=action,
                        k=self._default_k,
                        rationale=(
                            "trusted history is available and no verified candidate set "
                            "exists for this run"
                        ),
                    )
                continue
            if action is ActionKind.FINISH:
                return ActionProposal(
                    action=action,
                    rationale=self._finish_rationale(context),
                )

        # Every available action is one this policy deliberately declined.  Finish is the
        # only safe fallback, and it is still only a proposal: the CompletionGuard decides
        # whether ending here is legal.
        if context.action_available(ActionKind.FINISH):
            return ActionProposal(
                action=ActionKind.FINISH,
                rationale="no remaining action met this policy's conditions",
            )
        raise PolicyActionError(
            "no available action satisfied this policy: "
            + ", ".join(a.value for a in available)
        )

    # -- decision rule ----------------------------------------------------- #

    @staticmethod
    def _wants_recommendation(context: PolicyContext) -> bool:
        """The documented Stage 1 decision rule."""
        if context.last_proposal_rejected:
            # A refusal means the previous step did not do what the policy expected.
            # Re-proposing would spend the loop budget reproducing the same failure, so
            # the policy ends the turn instead of looping.
            return False
        if not context.has_trusted_history:
            return False
        if context.candidate_state.grounded:
            return False
        return True

    @staticmethod
    def _finish_rationale(context: PolicyContext) -> str:
        """A factual, payload-free reason for proposing FINISH."""
        if not context.has_trusted_history:
            return "no trusted history is available, so no recommendation can be produced"
        if context.candidate_state.grounded:
            return "a verified candidate set already exists for this run"
        if context.last_proposal_rejected:
            return "the previous proposal was refused; ending rather than repeating it"
        return "no further action is needed"
