"""Case runner and ablation harness for the agent plane (Stage 6).

The runner executes each evaluation case through a **real** control plane over the real
accepted pipeline, records an :class:`~recommendation.evaluation.agent.schemas.AgentTrajectory`,
and measures it.  Nothing is simulated at the loop level: the observations, the validator, the
budgets and the completion guard are the production components.

The ablation mechanism
----------------------
The brief requires comparing capability variants **without polluting the production runtime
with unsafe flags**.  The mechanism here is therefore evaluator-side policy injection: the
variant is a *policy* (and an injected capability set), not a runtime toggle.  Two variants are
provided:

``ABLATION_ADAPTIVE``
    The bounded loop the milestone built: the policy may change its next action because of the
    observation it was just given.

``ABLATION_DECIDE_ONCE``
    The historical baseline: the policy makes one decision up front and then finishes, never
    reading an observation.  This is the "decide once -> fixed workflow" shape expressed as a
    policy, so the comparison isolates *observation-conditioned control* and nothing else.

The comparison matters most on the recovery cases, and that is where a test asserts the two
variants diverge.  Neither variant is a strawman: the once-policy is a legitimate,
deterministic strategy that happens to be unable to react.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from recommendation.control import (
    ActionKind,
    ActionProposal,
    LoopLimits,
    LoopResult,
    PolicyContext,
    RunStatus,
    TerminationReason,
)
from recommendation.control.grounded_reasoning import ConstraintKind
from recommendation.control.arguments import (
    AskClarificationArguments,
    CheckCompatibilityArguments,
    CompareArguments,
    GetDetailsArguments,
    SearchCatalogArguments,
)

from .attribution import FailureAttribution, attribute
from .cases import EvaluationCase
from .metrics import TrajectoryMetrics, compute_metrics
from .schemas import AgentTrajectory, TerminalOutcome, TrajectoryRecord

__all__ = [
    "ABLATION_ADAPTIVE",
    "ABLATION_DECIDE_ONCE",
    "ABLATION_MODEL_POLICY",
    "CaseOutcome",
    "CaseRunner",
    "DecidingOncePolicy",
    "SuiteReport",
    "build_adaptive_policy",
    "run_case",
    "run_suite",
]

#: The observation-conditioned variant.
ABLATION_ADAPTIVE = "adaptive"
#: The decide-once baseline.
ABLATION_DECIDE_ONCE = "decide_once"
#: Phase 1: a model-driven next-action policy, supplied by the caller as a scripted model so
#: the comparison stays offline and reproducible.
ABLATION_MODEL_POLICY = "model_policy"


# --------------------------------------------------------------------------- #
# Policies used by the suite (evaluator-side; production code is untouched)
# --------------------------------------------------------------------------- #


class _RecommendThenFinish:
    """The Stage 1 behaviour: recommend once from history, then finish.

    Used by cases that only exercise the fast path.  It never reads an observation to decide -
    which is exactly why it is the right policy for cases where no adaptation is expected.
    """

    name = "suite-recommend-then-finish"

    def choose(self, context: PolicyContext) -> ActionProposal:
        if context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)


class _SearchCatalogFirst:
    """Search the catalogue, then finish.  Used by the novel-need case."""

    name = "suite-search-catalog-first"

    def __init__(self, terms: tuple[str, ...]) -> None:
        self._terms = terms

    def choose(self, context: PolicyContext) -> ActionProposal:
        if context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        if context.action_available(ActionKind.SEARCH_CATALOG):
            return ActionProposal(
                action=ActionKind.SEARCH_CATALOG,
                arguments=SearchCatalogArguments(terms=self._terms, limit=4),
            )
        return ActionProposal(action=ActionKind.FINISH)


class _AskWhenUnspecified:
    """Ask one material question, then finish.  Used by the clarification case."""

    name = "suite-ask-when-unspecified"

    def choose(self, context: PolicyContext) -> ActionProposal:
        if context.awaiting_user or context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        if context.action_available(ActionKind.ASK_CLARIFICATION):
            return ActionProposal(
                action=ActionKind.ASK_CLARIFICATION,
                arguments=AskClarificationArguments(
                    question="What will you use it for, so I can pick the right category?",
                    blocks="candidate_set",
                ),
            )
        return ActionProposal(action=ActionKind.FINISH)


class _RecommendThenVerify:
    """Recommend, then re-read grounded facts so a hard constraint can be checked."""

    name = "suite-recommend-then-verify"

    def choose(self, context: PolicyContext) -> ActionProposal:
        last = getattr(context.last_observation, "kind", None)
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        if last != "details" and context.action_available(ActionKind.GET_DETAILS):
            return ActionProposal(
                action=ActionKind.GET_DETAILS,
                arguments=GetDetailsArguments(parent_asins=_first_two(context)),
            )
        return ActionProposal(action=ActionKind.FINISH)


class _RecommendThenCheckConstraint:
    """Recommend, read grounded facts, then **verify the case's constraint**.

    Reading facts is not checking a constraint: a run can hold every weight the catalogue has
    and still never compare one against the limit.  This policy performs the check, so the
    hard-constraint case measures enforcement rather than retrieval of the same data.
    """

    name = "suite-recommend-then-check-constraint"

    def __init__(self, requirement: str) -> None:
        self._requirement = requirement

    def choose(self, context: PolicyContext) -> ActionProposal:
        last = getattr(context.last_observation, "kind", None)
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        if last == "compatibility":
            return ActionProposal(action=ActionKind.FINISH)
        if last != "details" and context.action_available(ActionKind.GET_DETAILS):
            return ActionProposal(
                action=ActionKind.GET_DETAILS,
                arguments=GetDetailsArguments(parent_asins=context.grounded_parent_asins),
            )
        if context.action_available(ActionKind.CHECK_COMPATIBILITY):
            return ActionProposal(
                action=ActionKind.CHECK_COMPATIBILITY,
                arguments=CheckCompatibilityArguments(
                    parent_asins=context.grounded_parent_asins,
                    requirement=self._requirement,
                ),
            )
        return ActionProposal(action=ActionKind.FINISH)


class _ConstraintFilteringPolicy:
    """Recommend, verify the constraint per candidate, and present a compliant set.

    This is the difference between *checking* a hard constraint and *enforcing* it.  A run can
    check every weight, learn that two products exceed the limit, and still hand the user all
    four.  This policy closes that loop: it verifies each candidate individually, remembers
    which ones the catalogue says satisfy the requirement, and presents only those.

    The filtering is the policy's decision, taken from grounded verdicts.  The controller does
    not filter for it, and the metric independently checks the final set, so a policy that
    claims compliance while presenting a violating product still fails the case.
    """

    name = "suite-constraint-filtering"

    def __init__(self, requirement: str) -> None:
        self._requirement = requirement
        self._compliant: set[str] = set()
        self._checked: set[str] = set()

    def choose(self, context: PolicyContext) -> ActionProposal:
        # Record the previous step's verdict FIRST.  Reading it after deciding the next action
        # meant the product never left the unchecked set, so the policy re-verified the same
        # product until the step budget stopped it - a no-progress loop the controller bounded
        # but the policy should never have entered.
        self._record_verdict(context)

        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        if context.awaiting_user:
            return ActionProposal(action=ActionKind.FINISH)

        # Verify each candidate individually so the verdict is per product.
        unchecked = [
            identity
            for identity in context.grounded_parent_asins
            if identity not in self._checked
        ]
        if unchecked and context.action_available(ActionKind.CHECK_COMPATIBILITY):
            target = unchecked[0]
            return ActionProposal(
                action=ActionKind.CHECK_COMPATIBILITY,
                arguments=CheckCompatibilityArguments(
                    parent_asins=(target,), requirement=self._requirement
                ),
            )
        return ActionProposal(action=ActionKind.FINISH)

    def _record_verdict(self, context: PolicyContext) -> None:
        """Record the previous step's compatibility verdict, if there was one.

        Every verdict marks the product as *checked*, including UNKNOWN.  An UNKNOWN is not
        compliance and not a violation: it means the catalogue could not decide, and re-asking
        would not change that.
        """
        observation = context.last_observation
        if getattr(observation, "kind", None) != "compatibility":
            return
        verdict = getattr(observation, "verdict", None)
        for identity in tuple(getattr(observation, "members", ()) or ()):
            self._checked.add(identity)
            if verdict == "compatible":
                self._compliant.add(identity)

    @property
    def compliant(self) -> frozenset[str]:
        """The identities the catalogue verified as satisfying the requirement."""
        return frozenset(self._compliant)


class _RecommendThenCompare:
    """Recommend, then compare grounded attributes, then finish."""

    name = "suite-recommend-then-compare"

    def choose(self, context: PolicyContext) -> ActionProposal:
        last = getattr(context.last_observation, "kind", None)
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        if last != "comparison" and context.action_available(ActionKind.COMPARE):
            return ActionProposal(
                action=ActionKind.COMPARE,
                arguments=CompareArguments(
                    parent_asins=_first_two(context), attributes=("weight_text", "price_text")
                ),
            )
        return ActionProposal(action=ActionKind.FINISH)


class _RecoverAfterEmpty:
    """Recommend; if that yields nothing, search the catalogue; then finish.

    The adaptive case: the *second* action is chosen only because the observation said the
    first source came back empty.  A policy that does not read observations cannot do this,
    which is what the ablation measures.
    """

    name = "suite-recover-after-empty"

    def __init__(self, terms: tuple[str, ...] = ("redwidget",)) -> None:
        # One of the suite's synthetic catalogue tokens, so the alternate source really can
        # succeed.  On a real catalogue the same case uses ordinary product words; the case
        # tests *source switching*, not tokenisation.
        self._terms = terms

    def choose(self, context: PolicyContext) -> ActionProposal:
        if context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.FINISH)
        observation = context.last_observation
        status = getattr(observation, "status", None)
        if status == "empty" and context.action_available(ActionKind.SEARCH_CATALOG):
            return ActionProposal(
                action=ActionKind.SEARCH_CATALOG,
                arguments=SearchCatalogArguments(terms=self._terms, limit=4),
            )
        if context.action_available(ActionKind.RECOMMEND_FROM_HISTORY):
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        return ActionProposal(action=ActionKind.FINISH)


class DecidingOncePolicy:
    """The decide-once ablation: one decision, then finish, never reading an observation.

    This is the historical "fixed workflow" shape expressed as a policy.  It is included so the
    suite can compare *observation-conditioned* control against *decide-once* control on the
    same cases, with no runtime flag and no change to production code.

    It is deliberately not a strawman: deciding once is a legitimate deterministic strategy.  It
    simply cannot react, and the recovery cases are where that shows.
    """

    name = "ablation-decide-once"

    def __init__(self, delegate: Any, baseline: Any | None = None) -> None:
        self._delegate = delegate
        #: What to propose once the single decision is spent.  ``None`` means "finish".
        self._baseline = baseline
        self._decided = False
        self._committed: ActionProposal | None = None

    @property
    def delegate(self) -> Any:
        """The policy whose first decision is honoured."""
        return self._delegate

    def choose(self, context: PolicyContext) -> ActionProposal:
        if not self._decided:
            self._decided = True
            self._committed = self._delegate.choose(context)
            return self._committed
        # Never re-read the observation.  The baseline is a bare FINISH by default, which is
        # the historical decide-once behaviour: one action, then wrap up regardless of what it
        # returned.
        if self._baseline is not None:
            return self._baseline.choose(context)
        return ActionProposal(
            action=ActionKind.FINISH,
            rationale="decide-once ablation: the single decision was already taken",
        )


def build_adaptive_policy(case: EvaluationCase) -> Any:
    """Return the adaptive policy a case is designed to exercise."""
    if case.case_id == "novel-product-need":
        # The suite's synthetic catalogue indexes titles as single tokens (``redwidget``), so
        # the term is chosen to be one the fixture actually contains.  On a real catalogue the
        # same case uses ordinary product words; the case tests *source selection*, not
        # tokenisation.
        return _SearchCatalogFirst(("redwidget",))
    if case.case_id == "missing-requirement":
        return _AskWhenUnspecified()
    if case.case_id == "hard-constraint":
        return _ConstraintFilteringPolicy("weight_max: 2 Kilograms")
    if case.case_id == "complex-multi-step":
        return _RecommendThenCompare()
    if case.case_id in ("empty-source-recovery", "empty-source-recovery-required"):
        return _RecoverAfterEmpty()
    if case.case_id == "missing-fact":
        return _RecommendThenVerify()
    if case.case_id == "hallucinated-candidate":
        return _RecommendThenFinish()
    return _RecommendThenFinish()


def _first_two(context: PolicyContext) -> tuple[str, ...]:
    """Return up to two grounded identities for a facts question.

    The identities come from the policy's own projection - the run's verified candidate set -
    so a reasoning action can only ever name a product the run already holds.
    """
    return tuple(context.grounded_parent_asins[:2])


@runtime_checkable
class _ControllerLike(Protocol):
    """The slice of the controller the runner uses."""

    def run(self, message: str, history: Any, **kwargs: Any) -> LoopResult:
        """Run one turn."""
        ...


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class CaseOutcome:
    """One case result: trajectory, metrics and attribution."""

    __slots__ = ("case", "trajectory", "metrics", "attribution")

    def __init__(
        self,
        *,
        case: EvaluationCase,
        trajectory: AgentTrajectory,
        metrics: TrajectoryMetrics,
        attribution: FailureAttribution,
    ) -> None:
        self.case = case
        self.trajectory = trajectory
        self.metrics = metrics
        self.attribution = attribution

    @property
    def passed(self) -> bool:
        """True when the case's metrics all passed."""
        return self.metrics.passed

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "case_id": self.case.case_id,
            "category": self.case.category,
            "passed": self.passed,
            "metrics": self.metrics.as_dict(),
            "attribution": self.attribution.as_dict(),
            "actions": list(self.trajectory.action_sequence()),
            "terminal": self.trajectory.terminal.value,
        }


class CaseRunner:
    """Run evaluation cases against a control plane built by an injected factory.

    Parameters
    ----------
    controller_factory:
        ``(case, policy, limits) -> controller``.  Injected so the suite can run against the
        synthetic fixture control plane in tests **or** a composed demo runtime, without this
        module depending on either.  It is also the seam that keeps the ablation mechanism out
        of production code: a variant is a different policy handed to the same factory.
    variant:
        A label recorded on every trajectory, so a report groups by variant.
    """

    def __init__(
        self,
        controller_factory: Callable[[EvaluationCase, Any, LoopLimits], Any],
        *,
        variant: str = ABLATION_ADAPTIVE,
        identities_provider: Callable[[Any], tuple[str, ...]] | None = None,
        policy_factory: Callable[[EvaluationCase], Any] | None = None,
    ) -> None:
        self._factory = controller_factory
        self._variant = variant
        self._identities_provider = identities_provider
        #: When set, the runner builds a fresh policy per case itself.  A variant is therefore
        #: *a policy*, injected through the same seam as any other - which is what keeps the
        #: production runtime free of ablation switches.
        self._policy_factory = policy_factory

    @property
    def variant(self) -> str:
        """The variant label recorded on trajectories."""
        return self._variant

    def run(
        self,
        case: EvaluationCase,
        *,
        policy: Any | None = None,
        reasoner: Any | None = None,
    ) -> CaseOutcome:
        """Run one case and return its measured outcome.

        The policy is supplied by the caller: the case declares *what is acceptable*, the
        policy declares *how to decide*, and conflating the two would make the suite grade its
        own homework.
        """
        if policy is not None:
            chosen = policy
        elif self._policy_factory is not None:
            chosen = self._policy_factory(case)
        else:
            chosen = build_adaptive_policy(case)
        limits = LoopLimits(
            max_steps=case.max_steps,
            max_tool_calls=max(1, case.max_tool_calls),
            max_retries=1,
        )
        controller = self._factory(case, chosen, limits)
        result = controller.run(case.message, _HISTORY, turn_id=f"eval-{case.case_id}")

        trajectory = self._project(case, result, controller)
        report = self._check_constraints(case, result, reasoner)
        metrics = compute_metrics(trajectory, case, constraint_report=report)
        return CaseOutcome(
            case=case,
            trajectory=trajectory,
            metrics=metrics,
            attribution=attribute(case.case_id, self._variant, metrics.failures()),
        )

    def _check_constraints(
        self, case: EvaluationCase, result: LoopResult, reasoner: Any | None
    ) -> tuple[Any, ...]:
        """Check the case's hard constraints against the run's **final** candidate set.

        Returns an empty tuple when the case declares no constraints or no reasoner was
        supplied.  An empty tuple is the honest answer in that situation: it means *nothing was
        checked here*, and the metrics treat an uncovered declared constraint as unverified
        rather than as compliance.
        """
        if not case.hard_constraints or reasoner is None:
            return ()
        tool_result = result.state.get("tool_result")
        if tool_result is None:
            return ()
        identities = [item.parent_asin for item in tool_result.recommendations]
        if not identities:
            return ()
        reports: list[Any] = []
        for declaration in case.hard_constraints:
            kind_name, _, expected = declaration.partition("=")
            try:
                kind = ConstraintKind(kind_name.strip())
            except ValueError:
                continue  # an unrecognised declaration is skipped, not guessed
            reports.extend(
                reasoner.check_constraint(identity, kind=kind, expected=expected.strip())
                for identity in identities
            )
        return tuple(reports)

    # -- projection -------------------------------------------------------- #

    def _project(self, case: EvaluationCase, result: LoopResult, controller: Any) -> AgentTrajectory:
        """Project a runtime result into the evaluation-facing trajectory."""
        records: list[TrajectoryRecord] = []
        for step in result.trajectory.steps:
            proposal = step.action_proposal or {}
            validated = step.validated_action or {}
            # ``TrajectoryStep`` stores the observation as a dumped mapping but the
            # verification result as the model itself, so the two are read differently.
            verification = step.verification_result
            observation = step.observation or {}
            refused = verification is not None and not verification.verified
            records.append(
                TrajectoryRecord(
                    step_index=step.step_index,
                    action_id=step.action_id,
                    policy_name=step.policy_metadata.get("policy"),
                    proposed_action=proposal.get("action"),
                    authorised=validated.get("action") is not None,
                    refusal_code=verification.code if refused else None,
                    executed_action=validated.get("action"),
                    observation_kind=observation.get("kind"),
                    observation_status=observation.get("status"),
                    returned_k=int(observation.get("returned_k", 0) or 0),
                    ungrounded_count=int(observation.get("ungrounded_count", 0) or 0),
                    produced_candidates=bool(
                        getattr(step.state_delta, "produced_candidates", False)
                    ),
                    consumed_tool_call=validated.get("action")
                    in ("recommend_from_history", "search_catalog", "select_source", "find_similar"),
                    note=step.note,
                )
            )
        return AgentTrajectory(
            scenario_id=case.case_id,
            variant=self._variant,
            terminal=_terminal_outcome(result),
            termination_reason=(
                None
                if result.control.termination_reason is None
                else str(result.control.termination_reason.value)
            ),
            steps=result.control.step_count,
            tool_calls=result.control.tool_call_count,
            records=tuple(records),
            route=result.route,
            produced_candidates=any(record.produced_candidates for record in records),
        )


def _terminal_outcome(result: LoopResult) -> TerminalOutcome:
    """Map a runtime result onto the evaluation vocabulary."""
    if result.status is RunStatus.WAITING_FOR_USER:
        return TerminalOutcome.ASKED_CLARIFICATION
    if result.status is RunStatus.FINISHED:
        return TerminalOutcome.COMPLETED
    reason = result.control.termination_reason
    if reason in (TerminationReason.MAX_STEPS, TerminationReason.MAX_TOOL_CALLS):
        return TerminalOutcome.BUDGET_EXHAUSTED
    if reason is TerminationReason.INVALID_ACTION:
        return TerminalOutcome.UNAUTHORISED_ACTION
    if reason is TerminationReason.COMPLETION_REFUSED:
        return TerminalOutcome.COMPLETION_REFUSED
    return TerminalOutcome.FAILED


#: A fixed trusted history for the suite.  Application-owned; never from a model.
_HISTORY: tuple[str, ...] = ("B000000001", "B000000002", "B000000003")


class SuiteReport:
    """The result of running a case set under one variant."""

    __slots__ = ("variant", "outcomes")

    def __init__(self, variant: str, outcomes: list[CaseOutcome]) -> None:
        self.variant = variant
        self.outcomes = outcomes

    @property
    def passed(self) -> int:
        """How many cases passed."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def failed(self) -> int:
        """How many cases failed."""
        return len(self.outcomes) - self.passed

    def failures_by_component(self) -> dict[str, int]:
        """Count failures per attributed component.

        The report shape the brief asks for: never one aggregate number, always per component.
        """
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            for component in outcome.attribution.components:
                counts[component] = counts.get(component, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "variant": self.variant,
            "total": len(self.outcomes),
            "passed": self.passed,
            "failed": self.failed,
            "failures_by_component": self.failures_by_component(),
            "cases": [outcome.as_dict() for outcome in self.outcomes],
        }


def run_case(
    case: EvaluationCase,
    *,
    controller_factory: Callable[[EvaluationCase, Any, LoopLimits], Any],
    policy: Any | None = None,
    variant: str = ABLATION_ADAPTIVE,
    reasoner: Any | None = None,
) -> CaseOutcome:
    """Run one case with a one-shot runner."""
    return CaseRunner(controller_factory, variant=variant).run(
        case, policy=policy, reasoner=reasoner
    )


def run_suite(
    cases: tuple[EvaluationCase, ...],
    *,
    controller_factory: Callable[[EvaluationCase, Any, LoopLimits], Any],
    variant: str = ABLATION_ADAPTIVE,
    policy_factory: Callable[[EvaluationCase], Any] | None = None,
    reasoner_factory: Callable[[EvaluationCase], Any] | None = None,
    model_policy_factory: Callable[[EvaluationCase], Any] | None = None,
) -> SuiteReport:
    """Run a case set under one variant and return the aggregated report.

    ``model_policy_factory`` runs the set under a model-driven policy: it is called per case so
    each case gets a fresh scripted model, and it takes precedence over ``policy_factory``.

    ``reasoner_factory`` is optional: without a catalogue reasoner the hard-constraint case
    reports its constraints as unverified rather than as satisfied, which is the honest
    outcome when nothing could check them.
    """
    runner = CaseRunner(
        controller_factory,
        variant=variant,
        policy_factory=None if model_policy_factory is not None else policy_factory,
    )
    outcomes = [
        runner.run(
            case,
            policy=None if model_policy_factory is None else model_policy_factory(case),
            reasoner=None if reasoner_factory is None else reasoner_factory(case),
        )
        for case in cases
    ]
    return SuiteReport(variant, outcomes)
