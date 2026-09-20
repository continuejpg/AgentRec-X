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

__all__ = ["CandidateState", "PolicyContext", "project_constraints", "project_intent"]


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

    # -- Stage 3: task and memory projection ------------------------------- #
    #: What the user appears to want from this task.  A small vocabulary, and
    #: ``UNKNOWN`` is a legitimate value rather than a failure.
    task_intent: str = "unknown"
    #: The constraints in force **for this task**, already filtered to those a policy may
    #: act on, described as short attributed strings (``"color: avoids red"``).
    #:
    #: These are *not* the whole memory store and not raw memory entries: they are the
    #: active, task-relevant projection.  A policy needs to know what the user wants now to
    #: choose an action; it does not need stored ``memory_id`` values, supersession chains,
    #: the owning ``user_key``, or preferences that this task has overridden.
    task_constraints: tuple[str, ...] = ()
    #: How many of the projected constraints are explicit and task-scoped, i.e. strong
    #: enough to define feasibility.  A count, not the values.
    hard_constraint_count: int = 0
    #: How many came only from inference.  Soft evidence: it may inform a question or an
    #: ordering and can never exclude a product.
    inferred_constraint_count: int = 0
    #: True when the task is suspended awaiting a user answer, so a policy knows why it is
    #: being asked to decide again.
    awaiting_user: bool = False
    #: The identities of the run's **grounded** candidates, in the run's own order.
    #:
    #: Stage 4 requires this: ``GET_DETAILS``, ``COMPARE``, ``TRADE_OFF``,
    #: ``CHECK_COMPATIBILITY`` and ``BUNDLE`` all name the products they reason about, and a
    #: policy cannot name a product it cannot see.  Exposing them does not widen the candidate
    #: set, because every identity here was already produced by a trusted tool and confirmed
    #: by the :class:`~recommendation.control.grounding.GroundingVerifier`; the policy can only
    #: *refer* to them, and only read-only actions accept them.
    #:
    #: What is deliberately excluded: the ledger's ``UNGROUNDED`` audit entries.  Those are a
    #: finding about a misbehaving source, not candidates, and a policy must never be able to
    #: ask about one as though it were real.
    grounded_parent_asins: tuple[str, ...] = ()

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
            "task_intent": self.task_intent,
            "task_constraint_count": len(self.task_constraints),
            "hard_constraint_count": self.hard_constraint_count,
            "inferred_constraint_count": self.inferred_constraint_count,
            "awaiting_user": self.awaiting_user,
        }


def project_intent(task_state: Any) -> str:
    """Return the task intent's value as a plain string, tolerating an absent state."""
    intent = getattr(task_state, "intent", None)
    if intent is None:
        return "unknown"
    return getattr(intent, "value", str(intent))


def project_constraints(task_state: Any, *, limit: int = 12) -> tuple[tuple[str, ...], int, int]:
    """Project task state and memory into bounded, policy-visible constraint strings.

    Returns ``(descriptions, hard_count, inferred_count)``.

    Selection rules, in the order they matter:

    * **task-scoped explicit constraints come first** - they are what the user asked for
      *now*, and they outrank a stored default for this task;
    * then constraints read from validated persistent memory;
    * inferred signals are included only as *soft* evidence, and are counted separately so a
      policy can tell a stated preference from a guess.

    The result is bounded by ``limit`` so a policy's context cannot grow without limit, and
    it contains descriptions rather than entry objects: no ``memory_id``, no ``user_key``,
    no supersession chain, no store handle.

    This is **evidence selection, not memory mutation.**  Nothing here writes, retires or
    reorders stored preferences; a task-scoped override suppresses within the task by not
    being present in this list twice, and the stored entry is untouched.
    """
    if task_state is None:
        return (), 0, 0

    seen: set[str] = set()
    hard: list[str] = []
    soft: list[str] = []
    inferred = 0

    for constraint in getattr(task_state, "constraints", ()) or ():
        described = constraint.describe()
        if described in seen:
            continue
        seen.add(described)
        if getattr(constraint, "is_soft", False):
            soft.append(described)
            inferred += 1
        else:
            hard.append(described)

    ordered = (*hard, *soft)
    return tuple(ordered[:limit]), min(len(hard), limit), min(inferred, limit)
