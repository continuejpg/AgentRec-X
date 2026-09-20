"""Trajectory metrics for the agent plane (Stage 6).

Three evaluation planes exist and this module is the third:

======================  ==========================================================
plane                   what it measures
======================  ==========================================================
recommender             HR / Recall / NDCG under the accepted protocol
shopping quality        constraint adherence, grounding, trade-off correctness
**agent / trajectory**  tool selection, unnecessary calls, recovery, budget use
======================  ==========================================================

The metrics here deliberately do **not** produce a single score.  They report each dimension
separately, and :mod:`~recommendation.evaluation.agent.attribution` maps a failure to the
component that produced it, so a number cannot hide which part of the system went wrong.

Two claims this module refuses to make, stated in code rather than in prose:

* **NDCG does not measure orchestration.**  Nothing here reads a recommender metric to judge a
  trajectory, and nothing in :mod:`recommendation.evaluation.metrics` is modified to judge it.
* **Preference adherence does not prove relevance.**  A trajectory that respected a colour
  preference says nothing about whether the product was worth recommending, so no metric here
  claims relevance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .cases import EvaluationCase
from .schemas import AgentTrajectory, TerminalOutcome

__all__ = ["TrajectoryMetrics", "compute_metrics"]


class TrajectoryMetrics(BaseModel):
    """Per-dimension trajectory measurements for one run.

    Every field is a count, a ratio or a boolean that a reader can verify against the
    trajectory.  There is intentionally no ``score`` field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    variant: str

    # -- task outcome ------------------------------------------------------ #
    #: The run reached a terminal state the case allows.
    terminal_allowed: bool
    terminal_outcome: TerminalOutcome
    reached_completion: bool

    # -- tool / action selection ------------------------------------------- #
    tool_selection_correct: bool
    #: Actions the case required, and whether each appeared.
    required_actions_met: tuple[str, ...]
    required_actions_missing: tuple[str, ...]
    #: Actions the case forbids, and whether any appeared.
    forbidden_actions_used: tuple[str, ...]
    #: Tool calls beyond the case's bound.
    unnecessary_tool_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    steps: int = Field(default=0, ge=0)

    # -- clarification ----------------------------------------------------- #
    clarification_used: bool = False
    clarification_was_expected: bool = False

    # -- constraints and grounding ----------------------------------------- #
    #: Identities a source returned that the trusted side did not know.
    hallucinated_candidates: int = Field(default=0, ge=0)
    #: How many hard constraints the case declares.
    hard_constraints_declared: int = Field(default=0, ge=0)
    #: The constraints that a **final candidate was verified to violate**.  Non-empty means
    #: the run ended with a product the catalogue says does not satisfy the requirement.
    constraint_violations: tuple[str, ...] = ()
    #: Constraints the catalogue could not resolve for at least one final candidate.  Reported
    #: separately from violations: unresolved is not a failure, but it is not compliance
    #: either, and a report that merged the two would be claiming a check that never happened.
    constraint_unresolved: tuple[str, ...] = ()
    #: The constraints for which a deterministic verdict was actually obtained.  This is
    #: "was the hard constraint checked", which is separable from whether the final set
    #: respects it.
    #:
    #: A constraint that was checked and *satisfied* counts here: before Phase 2 only
    #: violations and unknowns were folded in, so a run whose candidates all complied was
    #: reported as an unverified completion - which made "checked and compliant"
    #: indistinguishable from "never checked".
    constraint_checked: tuple[str, ...] = ()
    #: True when the case requires the final set to respect its constraints and it does not:
    #: a **verified violation reached the presented set**.  This is measured from the candidates
    #: that were actually presented, not from whether an action ran, so it detects the
    #: enforcement gap directly.
    constraint_not_enforced: bool = False
    #: True when the run presented a candidate whose declared hard constraint could not be
    #: decided.  Unresolved is not a violation, but presenting it as a compliant match would
    #: claim a check that never happened, so it is measured separately.
    constraint_unresolved_in_output: bool = False
    #: True when the run completed while a declared hard constraint was neither verified nor
    #: reported as unresolved.
    unverified_completion: bool = False

    # -- memory ------------------------------------------------------------ #
    #: ``commit`` / ``none`` / ``task_only``, as actually observed.
    memory_effect: str = "none"
    memory_effect_correct: bool = True

    # -- efficiency -------------------------------------------------------- #
    #: True when the run took no step that changed nothing.
    no_wasted_steps: bool = True

    def failures(self) -> tuple[str, ...]:
        """Return the names of the dimensions that failed.

        This is the plugin point for attribution: every name here maps to a component in
        :data:`~recommendation.evaluation.agent.attribution.ATTRIBUTION_COMPONENTS`, so a
        report can say *where* a run went wrong rather than only that it did.
        """
        failed: list[str] = []
        if not self.terminal_allowed:
            failed.append("terminal_not_allowed")
        if not self.tool_selection_correct:
            failed.append("tool_selection")
        if self.required_actions_missing:
            failed.append("required_action_missing")
        if self.forbidden_actions_used:
            failed.append("forbidden_action_used")
        if self.unnecessary_tool_calls:
            failed.append("budget_overrun")
        if self.hallucinated_candidates:
            failed.append("candidate_grounding")
        if self.constraint_violations and self.constraint_not_enforced:
            failed.append("constraint_violated")
        if self.constraint_unresolved_in_output:
            failed.append("constraint_unresolved")
        if self.unverified_completion:
            failed.append("completion")
        if not self.memory_effect_correct:
            failed.append("memory_scope")
        if self.clarification_was_expected and not self.clarification_used:
            failed.append("clarification")
        return tuple(failed)

    @property
    def passed(self) -> bool:
        """True when no dimension failed."""
        return not self.failures()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, including the derived failure list."""
        payload = self.model_dump(mode="json")
        payload["failures"] = list(self.failures())
        payload["passed"] = self.passed
        return payload


def compute_metrics(
    trajectory: AgentTrajectory,
    case: EvaluationCase,
    *,
    memory_effect: str | None = None,
    required_actions_verified: tuple[str, ...] | None = None,
    constraint_report: tuple[Any, ...] = (),
    presented_constraint_report: tuple[Any, ...] = (),
) -> TrajectoryMetrics:
    """Measure one trajectory against its case.

    Parameters
    ----------
    trajectory:
        The run to measure.
    case:
        The case that declares what is acceptable.
    memory_effect:
        What the run actually did to persistent memory (``commit`` / ``none`` /
        ``task_only``).  ``None`` means "not observed", which is treated as ``none`` - the
        conservative reading, so an unobserved commit is not silently credited.
    required_actions_verified:
        For cases whose required actions are *evidence-dependent* (the hallucination and
        missing-fact cases), the actions that genuinely exercised the behaviour.  When
        supplied, requirements are checked against this list instead of the raw action
        sequence, so a run cannot satisfy "verify" by proposing it and never reading a fact.
    constraint_report:
        Verdicts for every candidate the run **evaluated**.  Drives ``constraint_checked``, so
        a run that narrowed its answer is still credited with having done the checking.
    presented_constraint_report:
        Verdicts for the candidates the run **presented**.  Drives ``constraint_violations``,
        ``constraint_unresolved_in_output`` and ``constraint_not_enforced``, because only a
        presented candidate is a claim of compliance.  Empty for a run that presented nothing,
        which is why an empty feasible set is not reported as an enforcement failure.
    """
    executed = trajectory.action_sequence()
    considered = executed if required_actions_verified is None else required_actions_verified

    required_present = tuple(a for a in case.required_actions if a in considered)
    required_missing = tuple(a for a in case.required_actions if a not in considered)
    forbidden_used = tuple(a for a in case.forbidden_actions if a in executed)

    observed_memory = memory_effect if memory_effect is not None else "none"

    # Hard constraints are checked against the **final** product the run presented, using the
    # catalogue, not against whether some action ran.  A run can hold every fact and still
    # never compare one against the limit, so "did an action run" would credit a check that
    # never happened.
    # Two different questions, kept apart on purpose:
    #
    # * **checked** - was a deterministic verdict obtained at all?  Measured over everything
    #   the run evaluated, because a run that narrowed its answer still *did* the checking,
    #   and reporting it as an unverified completion would punish correct behaviour.
    # * **enforced** - did a violation reach the presented set?  Measured over the candidates
    #   the response was built from, because only those are a claim of compliance.
    #
    # Before Phase 2 there was one report and one question, which made "checked and compliant"
    # indistinguishable from "never checked", and made "excluded the violation" look identical
    # to "presented it".
    checked_labels: list[str] = []
    for report in constraint_report:
        if getattr(getattr(report, "verdict", None), "value", None) is not None:
            label = f"{getattr(getattr(report, 'kind', None), 'value', '?')}="
            checked_labels.append(label + str(getattr(report, "expected", "")))

    presented_violations: list[str] = []
    presented_unresolved: list[str] = []
    for report in presented_constraint_report:
        verdict = getattr(getattr(report, "verdict", None), "value", None)
        label = f"{getattr(getattr(report, 'kind', None), 'value', '?')}="
        label += str(getattr(report, "expected", ""))
        if verdict == "violated":
            presented_violations.append(label)
        elif verdict == "unknown":
            presented_unresolved.append(label)

    checked = tuple(sorted({f.split("=", 1)[0] for f in checked_labels}))
    declared = tuple(sorted({c.split("=", 1)[0] for c in case.hard_constraints}))
    constraint_not_enforced = bool(presented_violations and case.hard_constraints)
    constraint_unresolved_in_output = bool(presented_unresolved and case.hard_constraints)
    # Reported from the presented set: these describe what the answer claimed.
    violations = presented_violations
    unresolved = presented_unresolved

    # A completion is "unverified" when the run completed with a declared hard constraint that
    # neither a violation nor an unresolved result accounts for - i.e. the constraint was
    # never actually tested.  That is the case the CompletionGuard exists to refuse, so
    # measuring it is how the plane detects a guard regression.
    unverified_completion = bool(
        trajectory.terminal is TerminalOutcome.COMPLETED
        and case.hard_constraints
        and not set(declared) <= set(checked)
    )

    return TrajectoryMetrics(
        scenario_id=trajectory.scenario_id,
        variant=trajectory.variant,
        terminal_allowed=trajectory.terminal in case.allowed_terminal,
        terminal_outcome=trajectory.terminal,
        reached_completion=trajectory.terminal is TerminalOutcome.COMPLETED,
        tool_selection_correct=not required_missing and not forbidden_used,
        required_actions_met=required_present,
        required_actions_missing=required_missing,
        forbidden_actions_used=forbidden_used,
        unnecessary_tool_calls=max(0, trajectory.tool_calls - case.max_tool_calls),
        tool_calls=trajectory.tool_calls,
        steps=trajectory.steps,
        clarification_used="ask_clarification" in executed,
        clarification_was_expected="ask_clarification" in case.required_actions,
        hallucinated_candidates=trajectory.ungrounded_seen(),
        hard_constraints_declared=len(case.hard_constraints),
        constraint_violations=tuple(violations),
        constraint_unresolved=tuple(unresolved),
        constraint_checked=checked,
        constraint_not_enforced=constraint_not_enforced,
        constraint_unresolved_in_output=constraint_unresolved_in_output,
        unverified_completion=unverified_completion,
        memory_effect=observed_memory,
        memory_effect_correct=observed_memory == case.expected_memory_effect,
        no_wasted_steps=all(
            record.produced_candidates
            or record.executed_action not in ("recommend_from_history", "search_catalog")
            for record in trajectory.records
        ),
    )
