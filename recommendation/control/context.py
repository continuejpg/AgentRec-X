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

__all__ = [
    "CandidateReference",
    "CandidateState",
    "DEFAULT_CANDIDATE_REFERENCE_LIMIT",
    "PolicyContext",
    "project_constraints",
    "project_intent",
]

#: How many candidate references a policy may be shown in one context.
#:
#: Bounded deliberately, and equal to the largest identity list any reasoning action accepts
#: (``MAX_COMPARE_ITEMS``), so the projection can never be larger than the action it feeds.  A
#: context whose size tracked the candidate set would let a broad retrieval turn into an
#: unbounded prompt.
DEFAULT_CANDIDATE_REFERENCE_LIMIT = 6


@dataclass(frozen=True)
class CandidateReference:
    """One **legal reasoning target**: a grounded run candidate a policy may name.

    This is the reference layer, and it is deliberately thin.  The reference *is* the trusted
    identity - the run's own ``parent_asin`` - rather than an opaque handle, because the
    repository already carried these identities in a single bounded projection and an alias
    table would add a resolution step without removing any exposure.  What matters is not the
    spelling of the reference but **who validates it**: every one of these was produced by a
    trusted candidate source, confirmed by the grounding verifier, and is re-checked against
    the run's live allowlist at execution time.

    What a reference carries, and why:

    * ``parent_asin`` - the identity a reasoning action may name.  Necessary, because the
      action's own argument model is keyed by it.
    * ``eligibility`` - Phase 2's task-scoped state, so a policy can tell a candidate it may
      present from one that is only inspectable.  This is what makes evidence acquisition
      targetable: an ``unresolved`` candidate is exactly the one worth reading.
    * ``evidence`` - a coarse three-way summary of the constraint/evidence position
      (``unconstrained`` / ``satisfied`` / ``attention``), so a policy can prefer a candidate
      whose facts are missing without being handed catalogue values.

    What a reference deliberately does **not** carry: product titles, prices, weights, feature
    text, catalogue records, raw retrieval scores, or provenance objects.  A policy choosing a
    target needs to know *which candidate* and *what is known*, never *what the catalogue says*
    - reading the catalogue is the action it is about to propose.
    """

    #: The run's grounded identity for this candidate.  Valid only within the run that
    #: produced it: a new run builds its own allowlist from its own candidate set.
    parent_asin: str
    #: Phase 2 task-scoped state: ``verified_eligible`` / ``ineligible`` / ``unresolved``, or
    #: ``unknown`` when no constraint is active or the candidate was not assessed.
    eligibility: str = "unknown"
    #: Coarse evidence position, so a policy can prefer an under-evidenced candidate.
    evidence: str = "unconstrained"

    @property
    def is_reasoning_target(self) -> bool:
        """True for every reference: a reference exists only because it is inspectable.

        An ineligible candidate is still a legal reasoning target - explaining "why was this
        excluded" requires reading it - but it is never a legal *recommendation*, which
        separates reasoning authority from recommendation membership.
        """
        return True

    def as_dict(self) -> dict[str, str]:
        """Return the compact, catalogue-free view a prompt payload carries."""
        return {
            "parent_asin": self.parent_asin,
            "eligibility": self.eligibility,
            "evidence": self.evidence,
        }


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

    # -- Phase 2: task-scoped feasibility ---------------------------------- #
    #: How many of the run's grounded candidates are **proved** to satisfy every active hard
    #: constraint.  Counts only: a policy needs to know whether anything compliant exists in
    #: order to decide between finishing, gathering evidence and retrieving more, and it does
    #: not need the identities to make that decision.
    verified_eligible_count: int = 0
    #: How many are proved to violate at least one active hard constraint.  These candidates
    #: remain in the ledger for audit and remain readable by a reasoning action; they are
    #: simply not eligible to be presented as a compliant recommendation.
    ineligible_count: int = 0
    #: How many have no violation but at least one constraint the trusted catalogue could not
    #: decide.  Non-zero is the signal that gathering the missing fact is worth a step, and
    #: it is deliberately *not* reported as either compliance or violation.
    unresolved_count: int = 0
    #: The feasible subset of :attr:`grounded_parent_asins`, in the run's own order.
    #:
    #: This is a **narrowing of an allowlist the policy already has**, never a new source of
    #: identity: every member was already grounded, and a policy still cannot name an identity
    #: that is not in one of these two tuples.  It is exposed because a compliant
    #: recommendation can only be built from these candidates, so a policy reasoning about
    #: *what it may present* needs to distinguish them from the excluded ones.
    feasible_parent_asins: tuple[str, ...] = ()
    #: Short ``kind<=expected`` labels for the active hard constraints, so a policy can tell
    #: which dimension is unresolved.  A constraint dimension and the user's own stated value
    #: - never a product fact, and never a soft preference.
    active_constraints: tuple[str, ...] = ()

    #: Phase 2.1: the bounded set of **legal reasoning targets**.
    #:
    #: This is the reference layer the reasoning actions are validated against.  Every entry is
    #: a candidate the run actually holds - produced by a trusted source and confirmed by the
    #: grounding verifier - so a policy can *choose which* candidate to inspect without being
    #: able to name one the run does not have.  It is a narrowing of
    #: :attr:`grounded_parent_asins` plus Phase-2 eligibility, never a second source of
    #: identity, and it is bounded by :data:`DEFAULT_CANDIDATE_REFERENCE_LIMIT`.
    #:
    #: An ineligible candidate appears here on purpose: it is a legal *reasoning* target even
    #: though it is not a legal recommendation, which is what lets an agent explain an
    #: exclusion.  It cannot re-enter the recommendation set by being inspected.
    grounded_candidates: tuple[CandidateReference, ...] = ()

    #: Phase 4: attributes the configured evidence source can be asked about, so a policy knows
    #: which questions are answerable.  Empty when no evidence source is configured, which is how
    #: a policy tells "evidence is not available here" from "I have not asked yet".
    evidence_attributes: tuple[str, ...] = ()
    #: Phase 4: how many distinct facts have been acquired for this run.  A count, never the
    #: values: the facts belong to the constraint report, attached to the verdict they produced.
    acquired_evidence_count: int = 0
    #: Phase 4: how many candidates the evidence overlay holds facts about.  Lets a policy tell
    #: "some candidates are still unevidenced" from "evidence is exhausted".
    evidenced_candidate_count: int = 0
    #: Phase 4: whether a similar-item source is configured, so the policy knows whether
    #: ``FIND_SIMILAR`` can be proposed at all.  Derived from the offered menu, not asserted.
    similar_item_available: bool = False

    def reasoning_targets(self) -> tuple[str, ...]:
        """The identities a reasoning action may legally name, in the run's own order.

        The one function a policy needs to build a legal reference.  Kept separate from
        :attr:`feasible_parent_asins` because the two answer different questions: this is
        "what may I inspect", that is "what may I present".
        """
        return tuple(reference.parent_asin for reference in self.grounded_candidates)

    def candidate_reference(self, parent_asin: str) -> CandidateReference | None:
        """The reference for one identity, or ``None`` when it is not a legal target."""
        for reference in self.grounded_candidates:
            if reference.parent_asin == parent_asin:
                return reference
        return None

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
            "verified_eligible_count": self.verified_eligible_count,
            "ineligible_count": self.ineligible_count,
            "unresolved_count": self.unresolved_count,
            "active_constraint_count": len(self.active_constraints),
            # The summary carries the *count* of legal reasoning targets, never the identities:
            # the trajectory record is a different boundary from the policy's own view.
            "reasoning_target_count": len(self.grounded_candidates),
            # Phase 4 counts only.  The acquired *values* stay in the constraint report.
            "evidence_attribute_count": len(self.evidence_attributes),
            "acquired_evidence_count": self.acquired_evidence_count,
            "evidenced_candidate_count": self.evidenced_candidate_count,
            "similar_item_available": self.similar_item_available,
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
