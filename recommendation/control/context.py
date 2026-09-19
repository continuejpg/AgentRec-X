"""The controlled projection a policy is allowed to see (AgentRec-X 2.0-alpha).

``PolicyContext != AgentGraphState``
------------------------------------
The agent's trusted runtime state carries ``trusted_user_history``, a preference
snapshot, the Tool result, enrichment, evidence and the reranking report.  **None of
that is exposed here.**  A policy that could read the trusted history could copy it into
a tool call; a policy that could read candidate ids could name products directly.  Stage
1 removes both possibilities by construction rather than by instructing the policy not
to do it.

What a policy sees is what a *control* decision actually needs:

* the user's request text (untrusted, already visible to it today);
* the actions the **system** currently permits (never self-assumed);
* whether a trusted history exists at all - a boolean, never the history;
* how many explicit preferences are active - a count, never the entries;
* whether a verified candidate set already exists - a boolean plus an opaque reference,
  never the candidates;
* the last observation, already minimised;
* the remaining step and tool-call budgets, so a policy can behave sensibly near a limit
  even though it can never raise one.

What it must never expose, and therefore has no field for: database handles, credentials,
``user_key``, the memory store, the preference entries themselves, the metadata index, the
SASRec engine, the Tool, item ids, ``parent_asin`` values, raw scores, mapping or
checkpoint internals, and secret configuration of any kind.

The context is frozen: a policy cannot mutate the thing it was handed, and the controller
builds a fresh one for every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import ActionKind, Observation, RunStatus

__all__ = ["CandidateState", "PolicyContext"]


@dataclass(frozen=True)
class CandidateState:
    """What a policy may know about the candidate set: that it exists, and how big it is.

    ``candidate_set_ref`` is an opaque, controller-generated identity used only to let a
    policy tell "the same verified set I already saw" from "a different one".  It is not
    a candidate id, is not reversible into one, and cannot be passed to a tool.
    """

    grounded: bool = False
    candidate_count: int = 0
    candidate_set_ref: str = ""
    verification_status: str = "unverified"


@dataclass(frozen=True)
class PolicyContext:
    """The controller-built, policy-facing view of one loop step.

    Constructed by the controller only.  The policy receives it as an argument and cannot
    construct an authoritative one - the loop always rebuilds the context from trusted
    state, so a forged context is simply never read.
    """

    #: The user's natural-language request.  Untrusted text; the policy already receives
    #: the equivalent today through the decision prompt.
    user_request: str

    #: The actions the system currently permits, **computed by the controller** from the
    #: run state.  A policy chooses from this set; it never assumes what is supported.
    available_actions: tuple[ActionKind, ...]

    #: Whether an application-owned trusted interaction history exists.  A boolean, never
    #: the history itself.
    has_trusted_history: bool

    #: How many explicit preferences are active.  A count, never the entries: preference
    #: *values* are not needed to decide between "recommend" and "finish".
    active_preference_count: int = 0

    #: Whether a verified candidate set already exists for this run.
    candidate_state: CandidateState = field(default_factory=CandidateState)

    #: The previous step's observation, already minimised, or ``None`` on the first step.
    last_observation: Observation | None = None

    #: Budget the controller will enforce.  Read-only information: a policy can see that
    #: it is near a limit, and has no way to raise one.
    remaining_steps: int = 0
    remaining_tool_calls: int = 0

    #: Counters, for policies that want to behave differently after a failure.  These are
    #: observations about the loop, not control channels.
    step_index: int = 0
    run_status: RunStatus = RunStatus.RUNNING

    #: True when the controller is asking after a refusal, i.e. the previous proposal was
    #: invalid or completion was denied.  Lets a policy switch strategy without needing
    #: to inspect the reason in detail.
    last_proposal_rejected: bool = False

    def action_available(self, action: ActionKind) -> bool:
        """True when the system currently permits ``action``."""
        return action in self.available_actions

    def summary(self) -> dict[str, Any]:
        """Return a compact, payload-free summary for the trajectory record.

        Contains counts, booleans and action *names* only - never the user request text,
        never a candidate id, never a preference value.
        """
        return {
            "step_index": self.step_index,
            "available_actions": tuple(a.value for a in self.available_actions),
            "has_trusted_history": self.has_trusted_history,
            "active_preference_count": self.active_preference_count,
            "candidates_grounded": self.candidate_state.grounded,
            "candidate_count": self.candidate_state.candidate_count,
            "remaining_steps": self.remaining_steps,
            "remaining_tool_calls": self.remaining_tool_calls,
            "last_observation_kind": (
                getattr(self.last_observation, "kind", None)
                if self.last_observation is not None
                else None
            ),
            "last_proposal_rejected": self.last_proposal_rejected,
            "run_status": self.run_status.value,
        }
