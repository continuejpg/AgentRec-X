"""Memory write proposals and their validator (AgentRec-X 2.0-alpha Stage 3).

The pipeline this module implements::

    user text
        |
    Preference Interpreter  (interpreter proposes; may be rule-based or a future LLM)
        |
    MemoryWriteProposal     <- what to write, why, from where, how confident
        |
    MemoryValidator         <- validates grounding, attribution, scope and lifecycle
        |
    PreferenceMemoryService (the accepted, already-trusted writer) -> commit / reject

Two rules do the work, and both are structural rather than advisory.

**A proposal is not a commit.**  The interpreter produces
:class:`MemoryWriteProposal` objects.  Nothing in this module can write the store: the only
thing that reaches persistent memory is a proposal the validator marked approved, handed to
the accepted :class:`~recommendation.memory.service.PreferenceMemoryService`.  There is no
``write()`` here, and no store handle.

**An inference is not truth.**  A proposal carries a
:class:`~recommendation.control.task_state.PreferenceLifecycle`.  Only ``EXPLICIT`` (the user
said it) and ``CONFIRMED`` (the user confirmed a hypothesis) may be committed.  An
``INFERRED_HYPOTHESIS`` is validated as *soft evidence* and refused a durable commit - so an
inferred long-term preference can never quietly become hard truth.  It may still inform the
current task through :class:`~recommendation.control.task_state.TaskState`, which is where
soft signals belong.

Validation performed (each refusal carries a stable code):

* the proposal's value is non-blank and bounded;
* the supporting span actually occurs in the user-authored turn it claims to come from -
  this is the grounding check that makes a proposal auditable rather than merely typed;
* the scope and lifecycle permit a durable commit at all;
* the operation is a legal one for the shape of the proposal (a removal carries no value;
  an addition does);
* a claim the interpreter marked as *not grounded in the user's own words* is refused when it
  asks for a durable commit.

The validator never *repairs* a proposal.  A refusal is a first-class result, and the
proposal is retained for the trajectory so a misbehaving interpreter is visible.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from recommendation.memory.schemas import (
    MAX_SOURCE_TEXT_LENGTH,
    MAX_VALUE_LENGTH,
    PreferenceCandidate,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)
from recommendation.memory.schemas import (
    PreferenceExtraction,
)

from .task_state import PreferenceLifecycle, PreferenceScope

__all__ = [
    "MEMORY_PROPOSAL_VERSION",
    "MemoryProposalValidator",
    "MemoryWriteOperation",
    "MemoryWriteProposal",
    "ProposalValidation",
    "ProposalVerdict",
    "approved_candidates",
    "proposals_from_extraction",
]

#: Version of the proposal contract.
MEMORY_PROPOSAL_VERSION = 1


class MemoryWriteOperation(str, Enum):
    """What a proposal asks the memory service to do.

    Mirrors the accepted ADD / REPLACE / REMOVE lifecycle.  The distinction that matters
    most for this stage is the one *not* in this enum: a **negative preference** is an
    ``ADD`` of an ``AVOID``-polarity value, never a ``REMOVE``.  "I don't like red" adds a
    durable avoidance; "forget my colour preferences" retracts.  Collapsing the two is the
    classic memory bug, so they are separated here and tested.
    """

    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"

    @property
    def requires_value(self) -> bool:
        """True when the operation must carry a value to be meaningful."""
        return self is not MemoryWriteOperation.REMOVE


class MemoryWriteProposal(BaseModel):
    """One proposed change to persistent preference memory.

    Carries everything the validator needs to decide, and enough provenance for the
    trajectory to explain the decision later.

    It deliberately does **not** carry: a ``user_key`` (the controller holds the namespace,
    not the interpreter), a store handle, a product identity, or a free-form instruction
    string.  ``source_span`` is the user's own words, quoted for grounding.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = MEMORY_PROPOSAL_VERSION

    operation: MemoryWriteOperation
    kind: PreferenceKind
    value: str = Field(default="", max_length=MAX_VALUE_LENGTH)
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    mode: PreferenceMode = PreferenceMode.ADD
    replaces: str | None = Field(default=None, max_length=MAX_VALUE_LENGTH)

    scope: PreferenceScope = PreferenceScope.LONG_TERM
    lifecycle: PreferenceLifecycle = PreferenceLifecycle.EXPLICIT

    #: The user's own words this proposal was read from, verbatim.  The grounding check
    #: tests that this span really occurs in the turn the proposal claims.
    source_span: str = Field(default="", max_length=MAX_SOURCE_TEXT_LENGTH)
    #: Which turn the span came from.  Controller-supplied; a proposal cannot invent one.
    turn_id: str | None = None
    #: Who produced the proposal, e.g. ``"rule_based_preference_extractor"``.
    interpreter: str = Field(default="unspecified", max_length=80)
    #: The interpreter's own confidence in *its* reading.  Recorded for audit; it is NOT a
    #: licence to commit - confidence never overrides the lifecycle rule.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def asks_for_durable_commit(self) -> bool:
        """True when this proposal wants to reach persistent memory."""
        return not self.scope.is_ephemeral

    def signature(self) -> tuple[str, str, str, str]:
        """Return the identity used to collapse duplicate proposals within a turn."""
        return (
            self.kind.value,
            self.value.strip().lower(),
            self.polarity.value,
            self.mode.value,
        )


class ProposalVerdict(str, Enum):
    """The validator's decision about one proposal."""

    #: May be committed to persistent memory.
    APPROVED = "approved"
    #: May be used as soft evidence for this task, but must not be committed.
    SOFT_ONLY = "soft_only"
    #: Refused; must not be committed and must not be used as evidence either.
    REJECTED = "rejected"


class ProposalValidation(BaseModel):
    """The verdict for one proposal, with a stable code and an explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: MemoryWriteProposal
    verdict: ProposalVerdict
    code: str
    detail: str | None = Field(default=None, max_length=200)
    checks: tuple[str, ...] = ()

    @property
    def may_commit(self) -> bool:
        """True only for an approved proposal."""
        return self.verdict is ProposalVerdict.APPROVED

    @property
    def may_use_as_evidence(self) -> bool:
        """True when the signal may inform the current task as soft evidence."""
        return self.verdict in (ProposalVerdict.APPROVED, ProposalVerdict.SOFT_ONLY)


class MemoryProposalValidator:
    """Validate proposals before anything can reach persistent memory.

    Stateless and deterministic: the same ``(proposal, turn_text)`` pair always produces the
    same verdict, so a memory commit is reproducible from the trajectory.
    """

    def __init__(self, *, require_grounded_span: bool = True) -> None:
        #: When true, a durable commit requires the supporting span to occur in the turn.
        #: Kept configurable so a deployment whose interpreter quotes loosely can relax it
        #: deliberately, and so the strict behaviour is the default and is tested.
        self._require_grounded_span = bool(require_grounded_span)

    @property
    def requires_grounded_span(self) -> bool:
        """True when a durable commit requires a span grounded in the user's own words."""
        return self._require_grounded_span

    # -- validation -------------------------------------------------------- #

    def validate(
        self,
        proposal: MemoryWriteProposal,
        *,
        turn_text: str,
    ) -> ProposalValidation:
        """Return the verdict for one proposal against the turn it claims to come from.

        ``turn_text`` is the user-authored message.  It is used **only** as a grounding
        source for the quoted span; it is never re-interpreted here, so this validator
        cannot invent a preference the interpreter did not propose.
        """
        checks: list[str] = ["shape"]

        # -- shape --------------------------------------------------------- #
        if proposal.operation.requires_value and not proposal.value.strip():
            return ProposalValidation(
                proposal=proposal,
                verdict=ProposalVerdict.REJECTED,
                code="missing_value",
                detail="this operation requires a value",
                checks=(*checks, "value_present"),
            )
        if not proposal.operation.requires_value and proposal.value.strip():
            return ProposalValidation(
                proposal=proposal,
                verdict=ProposalVerdict.REJECTED,
                code="removal_carries_value",
                detail="a removal must not carry a value to store",
                checks=(*checks, "value_present"),
            )
        checks.append("value_present")

        # -- a negative preference is not a removal ------------------------- #
        # Structurally impossible to confuse here: polarity lives on the proposal and
        # operation is separate.  This check catches an interpreter that expressed an
        # avoidance *as* a removal - the exact collapse the stage forbids.
        if (
            proposal.operation is MemoryWriteOperation.REMOVE
            and proposal.polarity is PreferencePolarity.AVOID
        ):
            return ProposalValidation(
                proposal=proposal,
                verdict=ProposalVerdict.REJECTED,
                code="avoidance_expressed_as_removal",
                detail=(
                    "a negative preference is an ADD of an 'avoid' value; only an explicit "
                    "retraction is a removal"
                ),
                checks=(*checks, "negative_preference_not_removal"),
            )
        checks.append("negative_preference_not_removal")

        # -- lifecycle: an inference is never truth ------------------------- #
        if not proposal.lifecycle.may_be_committed:
            # Refused a durable commit, but explicitly usable as soft evidence for this
            # task - which is where a hypothesis belongs.
            return ProposalValidation(
                proposal=proposal,
                verdict=ProposalVerdict.SOFT_ONLY,
                code="inferred_signal_is_soft_evidence",
                detail=(
                    f"'{proposal.lifecycle.value}' may inform this task but may not be "
                    "committed as a durable preference"
                ),
                checks=(*checks, "lifecycle_may_commit"),
            )
        checks.append("lifecycle_may_commit")

        # -- scope ---------------------------------------------------------- #
        if not proposal.asks_for_durable_commit:
            return ProposalValidation(
                proposal=proposal,
                verdict=ProposalVerdict.SOFT_ONLY,
                code="ephemeral_scope_not_persisted",
                detail=(
                    f"scope '{proposal.scope.value}' applies to this task only and is "
                    "never persisted"
                ),
                checks=(*checks, "scope_allows_commit"),
            )
        checks.append("scope_allows_commit")

        # -- grounding: the span must really be the user's words ------------ #
        if self._require_grounded_span:
            span = proposal.source_span.strip()
            if not span:
                return ProposalValidation(
                    proposal=proposal,
                    verdict=ProposalVerdict.REJECTED,
                    code="ungrounded_span",
                    detail="a durable commit requires the user's own supporting words",
                    checks=(*checks, "grounded_span"),
                )
            if span.casefold() not in (turn_text or "").casefold():
                return ProposalValidation(
                    proposal=proposal,
                    verdict=ProposalVerdict.REJECTED,
                    code="span_not_in_turn",
                    detail="the quoted support does not occur in the user's message",
                    checks=(*checks, "grounded_span"),
                )
        checks.append("grounded_span")

        return ProposalValidation(
            proposal=proposal,
            verdict=ProposalVerdict.APPROVED,
            code="approved",
            detail=None,
            checks=tuple(checks),
        )

    def validate_all(
        self,
        proposals: tuple[MemoryWriteProposal, ...],
        *,
        turn_text: str,
    ) -> tuple[ProposalValidation, ...]:
        """Validate several proposals, collapsing exact duplicates within the turn.

        Deduplication happens on the *proposal* signature, not on the stored entry: the
        accepted memory service still owns idempotency and supersession.  This only stops
        one turn from proposing the same fact twice.
        """
        seen: set[tuple[str, str, str, str]] = set()
        results: list[ProposalValidation] = []
        for proposal in proposals:
            signature = proposal.signature()
            if signature in seen:
                continue
            seen.add(signature)
            results.append(self.validate(proposal, turn_text=turn_text))
        return tuple(results)


# --------------------------------------------------------------------------- #
# Interpreter output -> proposals
# --------------------------------------------------------------------------- #


def proposals_from_extraction(
    extraction: PreferenceExtraction,
    *,
    turn_id: str | None,
    interpreter: str,
    default_scope: PreferenceScope = PreferenceScope.LONG_TERM,
    lifecycle: PreferenceLifecycle = PreferenceLifecycle.EXPLICIT,
) -> tuple[MemoryWriteProposal, ...]:
    """Turn an accepted preference extraction into memory write proposals.

    This is the migration adapter for the stage: the accepted rule-based extractor keeps
    working unchanged and its typed output becomes proposals.  Nothing is inferred here -
    the function performs no interpretation of its own, so it cannot introduce a signal the
    extractor did not find.

    ``default_scope`` is the scope assumed for an extraction that carries no scope of its
    own (the accepted schema has none).  It defaults to ``LONG_TERM`` because that matches
    the accepted Milestone 9 behaviour: an explicit statement is remembered across turns.
    A caller that knows the statement is task-local passes ``TASK``.
    """
    proposals: list[MemoryWriteProposal] = []
    for candidate in extraction.preferences:
        proposals.append(
            MemoryWriteProposal(
                operation=_operation_for(candidate.mode),
                kind=candidate.kind,
                value=candidate.value,
                polarity=candidate.polarity,
                mode=candidate.mode,
                replaces=candidate.replaces,
                scope=default_scope,
                lifecycle=lifecycle,
                source_span=candidate.source_text,
                turn_id=turn_id,
                interpreter=candidate.extractor or interpreter,
            )
        )
    for removal in extraction.removals:
        proposals.append(
            MemoryWriteProposal(
                operation=MemoryWriteOperation.REMOVE,
                kind=removal.kind,
                polarity=PreferencePolarity.PREFER,
                value="",
                scope=default_scope,
                lifecycle=lifecycle,
                source_span=removal.source_text,
                turn_id=turn_id,
                interpreter=removal.extractor or interpreter,
            )
        )
    return tuple(proposals)


def approved_candidates(
    validations: tuple[ProposalValidation, ...],
) -> tuple[PreferenceCandidate, ...]:
    """Convert approved proposals into accepted :class:`PreferenceCandidate` values.

    Only approved proposals become candidates, and the conversion is total: nothing is
    dropped silently, nothing is repaired.  Rejected and soft-only proposals simply do not
    produce a candidate, which is what keeps an inference out of persistent memory.
    """
    candidates: list[PreferenceCandidate] = []
    for validation in validations:
        if not validation.may_commit:
            continue
        proposal = validation.proposal
        if proposal.operation is MemoryWriteOperation.REMOVE:
            # Removals are carried as retraction directives by the accepted service, not as
            # candidates; ``proposals_from_extraction`` mirrors that split.
            continue
        candidates.append(
            PreferenceCandidate(
                kind=proposal.kind,
                value=proposal.value,
                polarity=proposal.polarity,
                source_text=proposal.source_span or proposal.value,
                extractor=proposal.interpreter,
                mode=proposal.mode,
                replaces=proposal.replaces,
            )
        )
    return tuple(candidates)


def _operation_for(mode: PreferenceMode) -> MemoryWriteOperation:
    """Map an accepted preference mode onto a proposal operation."""
    if mode is PreferenceMode.REMOVE:
        return MemoryWriteOperation.REMOVE
    if mode is PreferenceMode.REPLACE:
        return MemoryWriteOperation.REPLACE
    return MemoryWriteOperation.ADD
