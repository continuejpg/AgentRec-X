"""Deterministic evaluation cases for the agent plane (Stage 6).

Each case specifies **more than a final answer**.  A case declares:

``acceptable_actions`` / ``required_actions`` / ``forbidden_actions``
    what the trajectory may, must and must not do;
``expected_sources``
    which trusted candidate sources should be consulted;
``hard_constraints`` / ``must_not_violate``
    deterministic constraints the final candidate set must respect;
``expected_memory_effect``
    whether the turn should commit persistent memory, leave it alone, or override it for the
    task only;
``allowed_terminal``
    which endings are legitimate.

Declaring all of that is what turns a case from an example into a *test*: a run that reaches
the right answer by way of a forbidden action, an unauthorised constraint check or a fabricated
fact should fail, and only a structured expectation can catch that.

The cases here are deterministic and offline.  They are **not** a benchmark with published
numbers and they make no accuracy claim; they are inspectable fixtures whose purpose is to make
specific control-plane behaviours checkable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..agent.schemas import TerminalOutcome

__all__ = [
    "CASES_VERSION",
    "EVALUATION_CASES",
    "EvaluationCase",
    "case_by_id",
    "load_cases",
]

#: Version of the case-set contract.
CASES_VERSION = 1


class EvaluationCase(BaseModel):
    """One inspectable evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(..., min_length=1)
    #: What this case is for, in one line.
    purpose: str = Field(..., min_length=1)
    #: The user's message.  Untrusted text.
    message: str = Field(..., min_length=1)
    #: One of the ten required categories this case covers.
    category: str = Field(..., min_length=1)

    #: Actions the trajectory must contain at least once.
    required_actions: tuple[str, ...] = ()
    #: Actions the trajectory may contain.
    acceptable_actions: tuple[str, ...] = ()
    #: Actions the trajectory must never contain.
    forbidden_actions: tuple[str, ...] = ()
    #: Candidate sources that should be consulted.
    expected_sources: tuple[str, ...] = ()
    #: Constraints that must hold of the final response.  ``attribute=value`` style.
    hard_constraints: tuple[str, ...] = ()
    #: Explanatory note about what the hard constraints mean for this case.
    constraint_note: str | None = None
    #: True when the case requires the **final set** to contain no violating product, not
    #: merely that the constraint was checked.  Only settable where the loop can actually
    #: narrow the presented set; see the limitation note on the hard-constraint case.
    constraint_enforced: bool = False
    #: ``commit`` / ``none`` / ``task_only``.
    expected_memory_effect: str = "none"
    #: Terminal outcomes that count as legitimate.
    allowed_terminal: tuple[TerminalOutcome, ...] = (TerminalOutcome.COMPLETED,)
    #: Upper bound on tool calls, so a runaway is detectable.
    max_tool_calls: int = Field(default=4, ge=0)
    #: Upper bound on steps.
    max_steps: int = Field(default=6, ge=1)
    #: True when the case is specifically about recovery from an empty/failed source.
    recovery_case: bool = False
    #: True when the case's fixture makes the history source **fail** rather than return nothing.
    #: A failure is what lets ``CompletionGuard`` refuse a premature FINISH, so the case is how
    #: the guard-rejection-and-recovery path is measured rather than merely asserted.
    history_source_fails: bool = False
    #: Notes for a reader, including anything the case deliberately does not check.
    notes: str | None = None

    def allows(self, action: str) -> bool:
        """True when ``action`` is not forbidden for this case."""
        return action not in self.forbidden_actions

    @property
    def needs_reasoning(self) -> bool:
        """True when the case expects a grounded reasoning step, not only retrieval.

        Derived from the case's own declarations, so a policy can tell "answer as efficiently as
        possible" from "gather evidence before answering" without the case growing another
        hand-maintained flag.  A fast-path case forbids reasoning and requires none; an
        evidence-gathering case requires one of the reasoning actions.
        """
        reasoning = {
            "get_details",
            "compare",
            "trade_off",
            "check_compatibility",
            "bundle",
            "verify",
        }
        if reasoning & set(self.forbidden_actions):
            return False
        return bool(reasoning & set(self.required_actions)) or bool(
            reasoning & set(self.acceptable_actions)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump(mode="json")


def _case(**kwargs: Any) -> EvaluationCase:
    """Build a case, expanding terminal-outcome strings into enum members."""
    if "allowed_terminal" in kwargs:
        kwargs["allowed_terminal"] = tuple(
            TerminalOutcome(value) if isinstance(value, str) else value
            for value in kwargs["allowed_terminal"]
        )
    return EvaluationCase(**kwargs)


#: The case set.  Ten categories, each named in ``category`` so a report can group by purpose.
EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    _case(
        case_id="history-driven",
        category="1_history_driven",
        purpose="A request with usable history should be served by the history recommender.",
        message="Recommend some gear for me.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "finish"),
        forbidden_actions=("search_catalog", "ask_clarification"),
        expected_sources=("history",),
        expected_memory_effect="none",
        max_tool_calls=2,
        max_steps=3,
        notes="The fast path: one source, one tool call, one completion.",
    ),
    _case(
        case_id="novel-product-need",
        category="2_explicit_novel_need",
        purpose="An explicit novel need should prefer catalog search over history only.",
        message="I need a waterproof rain cover for a canoe.",
        required_actions=("search_catalog",),
        acceptable_actions=("search_catalog", "recommend_from_history", "finish"),
        expected_sources=("catalog_search",),
        expected_memory_effect="none",
        max_tool_calls=3,
        notes="History alone cannot serve an item the profile never touched.",
    ),
    _case(
        case_id="missing-requirement",
        category="3_materially_missing_requirement",
        purpose="A requirement whose absence changes the outcome should be asked about.",
        message="I want something for my trip.",
        required_actions=("ask_clarification",),
        acceptable_actions=("ask_clarification", "search_catalog", "finish"),
        forbidden_actions=("recommend_from_history",),
        expected_memory_effect="none",
        allowed_terminal=("asked_clarification",),
        max_tool_calls=0,
        max_steps=2,
        notes=(
            "The question must identify the decision it blocks; the case does not judge the "
            "question's natural-language quality."
        ),
    ),
    _case(
        case_id="hard-constraint",
        category="4_hard_constraint",
        purpose="A verified hard constraint must not be violated by the final candidate set.",
        message="Show me the red option only.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "verify", "compare", "finish"),
        hard_constraints=("color=red",),
        constraint_enforced=False,
        constraint_note=(
            "Checked deterministically against the catalogue. A candidate whose colour the "
            "catalogue does not carry is UNKNOWN: not compliant, and not a violation either. "
            "Weight is deliberately not used here because CHECK_COMPATIBILITY verifies "
            "brand / category / colour requirements; a numeric weight ceiling is a feasibility "
            "constraint, checked through the reasoner's constraint API, not a compatibility "
            "requirement."
        ),
        expected_memory_effect="none",
        # One recommendation, then one verification per candidate so the verdict is per
        # product, then finish.  The budget must accommodate enforcement, not just checking.
        max_tool_calls=2,
        max_steps=8,
        notes=(
            "The constraint is verifiable, so it must be enforced rather than described. "
            "Per-candidate verification is what makes 'which ones comply' answerable. This "
            "case requires the check to be PERFORMED and the verdicts to be grounded; it does "
            "not yet require the final set to be narrowed, because the bounded loop has no "
            "candidate-narrowing mechanism (documented as a Stage 4 limitation). The reasoner's "
            "constraint API can enforce a numeric ceiling, but no action maps a narrowed set "
            "back into the presented candidates."
        ),
    ),
    _case(
        case_id="task-override",
        category="5_task_override_of_persistent",
        purpose="A current-task override wins for this task and leaves stored memory intact.",
        message="Not red this time, but I still like red in general.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "finish", "compare"),
        forbidden_actions=("propose_memory_write",),
        # No persistent-memory service is configured for this suite deployment, so no commit
        # is possible and the observable expectation is `none`.  The scope semantics
        # themselves are asserted directly in the Stage 3 tests; asserting "task_only" here
        # would be claiming an observation this deployment cannot make.
        expected_memory_effect="none",
        max_tool_calls=2,
        notes=(
            "The override is a TASK-scoped constraint. It suppresses within the task and must "
            "not be committed; the stored preference must survive. Forbidden "
            "`propose_memory_write` is the part of that this suite can observe."
        ),
    ),
    _case(
        case_id="empty-source-recovery",
        category="6_source_returns_nothing",
        purpose="An empty result from one source should permit bounded recovery or an honest end.",
        message="Recommend some gear.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "search_catalog", "ask_clarification", "finish"),
        expected_memory_effect="none",
        recovery_case=True,
        allowed_terminal=(
            TerminalOutcome.COMPLETED,
            TerminalOutcome.ASKED_CLARIFICATION,
            TerminalOutcome.BUDGET_EXHAUSTED,
        ),
        max_tool_calls=3,
        notes=(
            "The observation for an empty source must let the policy try another source; a "
            "run that ignores the empty observation and re-asks the same source fails."
        ),
    ),
    _case(
        case_id="premature-finish-recovery",
        category="5a_guarded_completion",
        purpose=(
            "A FINISH proposed after a failed execution must be refused by CompletionGuard, and "
            "the policy must recover from the refusal rather than the run simply ending."
        ),
        message="Recommend some gear.",
        # The history source fails, so the model's first attempt leaves nothing grounded and its
        # FINISH is refused with ``last_execution_failed``.  The recovery action is a different
        # source; the run may only end once something is grounded.
        history_source_fails=True,
        required_actions=("search_catalog",),
        acceptable_actions=("recommend_from_history", "search_catalog", "finish"),
        forbidden_actions=(),
        expected_sources=("catalog_search",),
        expected_memory_effect="none",
        recovery_case=True,
        allowed_terminal=("completed",),
        max_tool_calls=3,
        max_steps=6,
        notes=(
            "Measures that a refused FINISH becomes an Observation the policy can act on, and "
            "that completion is still authorised by the guard rather than by the model."
        ),
    ),
    _case(
        case_id="empty-source-recovery-required",
        category="6_source_returns_nothing",
        purpose=(
            "The history source returns nothing, so only an observation-conditioned policy can "
            "recover and still present a candidate set."
        ),
        message="Recommend some gear.",
        # The decided-once shape cannot satisfy this: one decision, then wrap up, so the empty
        # history result is never acted on.  That is the divergence the ablation measures.
        required_actions=("search_catalog",),
        acceptable_actions=("recommend_from_history", "search_catalog", "finish"),
        expected_sources=("catalog_search",),
        expected_memory_effect="none",
        recovery_case=True,
        allowed_terminal=("completed",),
        max_tool_calls=3,
        max_steps=5,
        notes=(
            "Driven with a history source that returns zero candidates and a catalogue search "
            "that finds some. The only way to end with candidates is to read the empty "
            "observation and choose a different source."
        ),
    ),
    _case(
        case_id="hallucinated-candidate",
        category="7_candidate_hallucination_attempt",
        purpose="An invented product identity must be rejected by the trusted boundary.",
        message="Recommend some gear.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "finish", "verify"),
        expected_memory_effect="none",
        max_tool_calls=2,
        notes=(
            "Driven with a source that returns one invented identity; the ledger must record "
            "it as ungrounded and it must never appear as a candidate."
        ),
    ),
    _case(
        case_id="missing-fact",
        category="8_unsupported_missing_fact",
        purpose="A fact the catalogue does not carry must stay UNKNOWN, never fabricated.",
        message="Recommend some gear and tell me how much it weighs.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "get_details", "finish"),
        expected_memory_effect="none",
        max_tool_calls=2,
        notes=(
            "For candidates whose weight the catalogue lacks, the grounded read must report "
            "UNKNOWN; no number may be invented."
        ),
    ),
    _case(
        case_id="simple-fast-path",
        category="9_simple_efficient",
        purpose="A simple request should take an efficient path.",
        message="Recommend a tent.",
        required_actions=("recommend_from_history",),
        acceptable_actions=("recommend_from_history", "finish"),
        forbidden_actions=("ask_clarification", "compare", "bundle", "trade_off"),
        expected_sources=("history",),
        expected_memory_effect="none",
        max_tool_calls=1,
        max_steps=3,
        notes="Efficiency is a correctness property here: extra steps are a failure.",
    ),
    _case(
        case_id="complex-multi-step",
        category="10_complex_bounded_path",
        purpose="A complex request may take several bounded steps including grounded reasoning.",
        message="Compare the top candidates by weight and then recommend the lighter one.",
        required_actions=("recommend_from_history",),
        acceptable_actions=(
            "recommend_from_history",
            "get_details",
            "compare",
            "trade_off",
            "verify",
            "finish",
        ),
        expected_memory_effect="none",
        max_tool_calls=3,
        max_steps=6,
        notes=(
            "Must stay inside its budget. The case does not require a specific ordering "
            "algorithm, only that any ordering it reports is grounded."
        ),
    ),
)


def load_cases() -> tuple[EvaluationCase, ...]:
    """Return the case set."""
    return EVALUATION_CASES


def case_by_id(case_id: str) -> EvaluationCase:
    """Return one case by id, raising ``KeyError`` when absent."""
    for case in EVALUATION_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"no evaluation case with id {case_id!r}")
