"""Task state, preference scope and lifecycle semantics (AgentRec-X 2.0-alpha Stage 3).

This module is where the three signal domains the stage requires are kept distinct instead
of being collapsed into one profile string:

``A. Behavioural preference``
    Application-owned interaction history.  Never here, never mutable by this module, and
    never written by the agent at all - it belongs to the host application.

``B. Long-term explicit preference``
    Validated persistent memory, owned by the accepted
    :class:`~recommendation.memory.service.PreferenceMemoryService`.  This module *reads*
    it through the accepted snapshot and never writes it: a commit is
    :class:`~recommendation.control.memory_proposal`'s job, and it goes through the accepted
    service so provenance and supersession handling stay in one place.

``C. Current task / session intent``
    :class:`TaskState`, owned by this module.  Mutable during the interaction, discarded
    when the task ends, and **immediately usable in the turn that produced it**.

Why task state has to exist separately
--------------------------------------
Long-term memory deliberately takes effect *from the next turn* - that is the accepted
Milestone 9 semantics, and it is what stops a turn from appearing to have let its own
statement influence the candidates it returned.  But a user who says "not the red one this
time" is talking about *this* request, and waiting a turn to honour that is simply wrong.

So the same sentence can produce two different things:

::

    "not red this time"        -> TaskState constraint (applies now), no memory write
    "never red again"          -> TaskState constraint (applies now)
                               + MemoryWriteProposal (committed for future tasks)

The distinction is ``PreferenceScope``.  A ``TASK``-scoped instruction never deletes a
``LONG_TERM`` preference: a temporary override *suppresses* within the task and leaves the
stored state intact.  Suppression is not removal, and this module makes that structural by
keeping task constraints in a separate object from memory entries rather than by mutating
memory with a "temporarily off" flag.

Authority
---------
``TaskState`` is written only by the controller, from the user's own message and from
validated proposals.  A policy may *read* the task projection; it cannot add a constraint to
task state, and it cannot reach long-term memory at all.  A policy can propose an intent,
which the validator accepts or refuses - the same proposal/authorization split as actions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from recommendation.memory.schemas import PreferenceKind, PreferencePolarity

__all__ = [
    "TASK_STATE_VERSION",
    "ConstraintOrigin",
    "PreferenceLifecycle",
    "PreferenceScope",
    "TaskConstraint",
    "TaskState",
    "TaskIntent",
]

#: Version of the task-state contract.
TASK_STATE_VERSION = 1


class PreferenceScope(str, Enum):
    """How long an explicit statement is meant to apply.

    This is the semantic that keeps a temporary instruction from silently rewriting a
    durable one.  It is decided at interpretation time and validated before any commit.

    * :attr:`SESSION` - applies to the current session / task only.  Never persisted.
    * :attr:`TASK` - applies to the current task / request.  Never persisted.  Treated the
      same as ``SESSION`` for storage purposes; kept distinct because the intent differs
      and evaluation needs to tell them apart.
    * :attr:`LONG_TERM` - crosses tasks, so it is proposed for persistent commit.
    """

    SESSION = "session"
    TASK = "task"
    LONG_TERM = "long_term"

    @property
    def is_ephemeral(self) -> bool:
        """True when this scope must never be persisted."""
        return self is not PreferenceScope.LONG_TERM


class PreferenceLifecycle(str, Enum):
    """Status of a preference signal, distinguishing what was said from what is believed.

    The accepted memory schema has ``ACTIVE`` / ``SUPERSEDED`` / ``REMOVED`` for *stored*
    entries.  This enum covers the wider ladder a *signal* moves through, including signals
    that are deliberately not treated as truth:

    * :attr:`EXPLICIT` - the user stated it.  The only kind that may be committed as a
      durable preference by the accepted service.
    * :attr:`INFERRED_HYPOTHESIS` - the system derived it and is **not** asserting it.  It
      is soft evidence: it may inform a question or an ordering, and it may never be
      written as a durable preference, and it never overrides an explicit statement.
    * :attr:`CONFIRMED` - an inferred signal the user subsequently confirmed.  Confirmation
      is what promotes a hypothesis to something durable; it never happens automatically.
    * :attr:`RETRACTED` - the signal was withdrawn.  Retraction is recorded, not deleted,
      so provenance survives.
    """

    EXPLICIT = "explicit"
    INFERRED_HYPOTHESIS = "inferred_hypothesis"
    CONFIRMED = "confirmed"
    RETRACTED = "retracted"

    @property
    def may_be_committed(self) -> bool:
        """True when this signal may become a durable preference.

        Only explicit statements and user-confirmed hypotheses qualify.  An
        unconfirmed inference never does - that is the rule that stops an inferred
        long-term preference from becoming hard truth.
        """
        return self in (PreferenceLifecycle.EXPLICIT, PreferenceLifecycle.CONFIRMED)

    @property
    def is_soft_evidence(self) -> bool:
        """True when this signal may only ever be soft evidence."""
        return self is PreferenceLifecycle.INFERRED_HYPOTHESIS


class ConstraintOrigin(str, Enum):
    """Where a constraint came from.  Provenance for precedence, not for display."""

    #: Stated by the user in this turn.
    CURRENT_TURN = "current_turn"
    #: Stated by the user in an earlier turn and still active in this session.
    SESSION_HISTORY = "session_history"
    #: Read from validated persistent memory.
    PERSISTENT_MEMORY = "persistent_memory"
    #: Derived by the system; soft evidence only.
    INFERRED = "inferred"


class TaskConstraint(BaseModel):
    """One constraint that applies to the current task.

    A constraint is a *statement about what the user wants now*, carrying enough provenance
    to justify precedence: which turn produced it, what the user actually said, and whether
    it came from this turn, this session, memory or inference.

    It deliberately does not carry a product.  A constraint narrows or orders candidates; it
    never names one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PreferenceKind
    value: str = Field(..., min_length=1, max_length=120)
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    scope: PreferenceScope = PreferenceScope.TASK
    lifecycle: PreferenceLifecycle = PreferenceLifecycle.EXPLICIT
    origin: ConstraintOrigin = ConstraintOrigin.CURRENT_TURN
    #: The user's own words this constraint was read from, verbatim.  Untrusted text kept
    #: for audit; never re-interpreted as an instruction and never a product fact.
    source_text: str = Field(default="", max_length=500)
    #: Which turn produced it, for trajectory reconstruction.
    turn_id: str | None = None

    def signature(self) -> tuple[str, str, str]:
        """Return the identity used to collapse duplicate constraints."""
        return (self.kind.value, self.polarity.value, self.value.strip().lower())

    @property
    def is_hard_candidate(self) -> bool:
        """True when this constraint is stated strongly enough to define feasibility.

        Only an **explicit** constraint from the current turn or this session can act as a
        hard constraint.  An inference never can, and neither can a value the user never
        actually stated - which is what stops a soft signal from removing products.
        """
        return (
            self.lifecycle.may_be_committed
            and self.origin in (ConstraintOrigin.CURRENT_TURN, ConstraintOrigin.SESSION_HISTORY)
        )

    @property
    def is_soft(self) -> bool:
        """True when this constraint may only order candidates, never exclude them."""
        return not self.is_hard_candidate

    def describe(self) -> str:
        """Return a short, attributed description for a policy projection."""
        marker = "prefers" if self.polarity is PreferencePolarity.PREFER else "avoids"
        return f"{self.kind.value}: {marker} {self.value}"


class TaskIntent(str, Enum):
    """What the user appears to be asking for in this task.

    A small, deliberately non-committal vocabulary: it records the *shape* of the request so
    a policy can choose an action (search, recommend, clarify), and it makes no claim about
    products.  ``UNKNOWN`` is a first-class value rather than a failure.
    """

    RECOMMEND = "recommend"
    SEARCH = "search"
    COMPARE = "compare"
    DETAILS = "details"
    COMPATIBILITY = "compatibility"
    BUNDLE = "bundle"
    UNKNOWN = "unknown"


class TaskState(BaseModel):
    """Everything that is true of the **current task** and nothing else.

    Mutable during the interaction by the controller; never persisted; discarded when the
    task ends.  It is the one place a same-turn instruction can take effect immediately,
    which is why it is separate from the accepted preference memory rather than a flag on it.

    What it deliberately does not hold: behavioural history (application-owned), the
    catalogue, the candidate ledger, or any product identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = TASK_STATE_VERSION
    #: Application-owned task identity.  Opaque; the agent never derives it.
    task_id: str | None = None
    #: The user's request as stated.  Untrusted text.
    request_text: str = Field(default="", max_length=1000)
    intent: TaskIntent = TaskIntent.UNKNOWN
    #: Constraints in force **for this task**.  Ordered by insertion; the last statement of
    #: a given kind/polarity wins for precedence purposes without deleting anything.
    constraints: tuple[TaskConstraint, ...] = ()
    #: Set when the run is waiting for the user, so a resumed run knows what it asked.
    pending_question: str | None = Field(default=None, max_length=400)
    #: How many times the user has been asked something in this task.  A controller-owned
    #: budget input, so clarification cannot become an unbounded interrogation.
    clarification_count: int = Field(default=0, ge=0)

    # -- constraint handling ---------------------------------------------- #

    def with_constraint(self, constraint: TaskConstraint) -> TaskState:
        """Return a new state with ``constraint`` added, collapsing duplicates.

        A repeated statement of the same kind/polarity/value updates the existing record
        rather than appending a second one, so the constraint list stays a set of current
        instructions rather than a transcript.
        """
        signature = constraint.signature()
        kept = [
            existing
            for existing in self.constraints
            if existing.signature() != signature
        ]
        return self.model_copy(update={"constraints": (*kept, constraint)})

    def with_constraints(self, constraints: tuple[TaskConstraint, ...]) -> TaskState:
        """Return a new state with several constraints applied in order."""
        state = self
        for constraint in constraints:
            state = state.with_constraint(constraint)
        return state

    def hard_constraints(self) -> tuple[TaskConstraint, ...]:
        """The explicit, task-scoped constraints that define feasibility."""
        return tuple(c for c in self.constraints if c.is_hard_candidate)

    def soft_constraints(self) -> tuple[TaskConstraint, ...]:
        """The constraints that may only order candidates."""
        return tuple(c for c in self.constraints if c.is_soft)

    def avoid_values(self, kind: PreferenceKind) -> tuple[str, ...]:
        """Explicit avoid-values of one kind in force for this task."""
        return tuple(
            constraint.value
            for constraint in self.constraints
            if constraint.kind is kind
            and constraint.polarity is PreferencePolarity.AVOID
            and constraint.is_hard_candidate
        )

    def describe(self) -> tuple[str, ...]:
        """Return short descriptions of every constraint, for a policy projection."""
        return tuple(constraint.describe() for constraint in self.constraints)

    # -- suspension -------------------------------------------------------- #

    def suspended(self, question: str) -> TaskState:
        """Return a new state marked as waiting for the user's answer."""
        return self.model_copy(
            update={
                "pending_question": question,
                "clarification_count": self.clarification_count + 1,
            }
        )

    def resumed(self) -> TaskState:
        """Return a new state with the pending question cleared."""
        return self.model_copy(update={"pending_question": None})

    @property
    def is_waiting_for_user(self) -> bool:
        """True when the task is suspended pending a user answer."""
        return bool(self.pending_question)
