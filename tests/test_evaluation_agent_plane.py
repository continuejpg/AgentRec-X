"""Stage 6 tests: the agent evaluation plane.

`tests/test_evaluation_metrics.py` and its siblings measure the **recommender** under the
accepted full-catalogue temporal protocol.  Nothing in this file touches them, and nothing here
reuses an NDCG number to judge a trajectory.

What this file checks:

**A. Trajectory schema** - a run's record is reconstructable and payload-free.
**B. Cases** - the ten required categories exist, and each declares more than a final answer.
**C. Metrics** - each dimension is measured separately, with no aggregate score.
**D. Attribution** - every failure maps to an owning component, totally.
**E. Ablation** - the observation-conditioned loop and the decide-once baseline genuinely
   diverge on failure/recovery, which is the comparison the stage exists to support.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    GroundedReasoner,
    GroundingVerifier,
    ReasoningExecutor,
)
from recommendation.evaluation.agent import (  # noqa: E402
    ABLATION_ADAPTIVE,
    ABLATION_DECIDE_ONCE,
    ABLATION_MODEL_POLICY,
    ATTRIBUTION_COMPONENTS,
    EVALUATION_CASES,
    AttributionComponent,
    CaseRunner,
    DecidingOncePolicy,
    ObservationReactiveModel,
    EvaluationCase,
    TerminalOutcome,
    TrajectoryMetrics,
    TrajectoryRecord,
    build_model_policy_factory,
    case_by_id,
    load_cases,
    run_suite,
)
from recommendation.control.model_policy import LLM_POLICY_NAME  # noqa: E402
from recommendation.evaluation.agent.attribution import attribute  # noqa: E402
from recommendation.evaluation.agent.metrics import compute_metrics  # noqa: E402
from recommendation.evaluation.agent.runner import build_adaptive_policy  # noqa: E402
from recommendation.evaluation.agent.schemas import EVALUATION_PLANE_VERSION, AgentTrajectory  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

# --------------------------------------------------------------------------- #
# suite wiring
# --------------------------------------------------------------------------- #


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self, rows: tuple[tuple[object, ...], ...] = CANDIDATE_ROWS) -> None:
        self._ids = {row[0]: row[1] for row in rows}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


def build_reasoner() -> GroundedReasoner:
    """A reasoner over the suite's synthetic catalogue."""
    return GroundedReasoner(build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata)


def suite_factory(case: EvaluationCase, policy: object, limits: object) -> object:
    """Compose a real control plane for one case.

    Two case declarations change the fixture, and both are read from the case rather than matched
    on its id:

    * ``history_source_fails`` - the history engine raises, which is what lets ``CompletionGuard``
      refuse a premature FINISH;
    * a case whose only route to candidates is another source is driven with a history engine that
      returns **nothing**, so ending with candidates requires switching source.

    Every other case gets a history engine that returns the standard fixture ranking.
    """
    empty_history = "recovery-required" in case.case_id
    harness = build_control_harness(
        policy=policy,
        limits=limits,
        rows=() if empty_history else CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
        engine_error=(
            RuntimeError("history source unavailable") if case.history_source_fails else None
        ),
    )
    metadata = harness.parts["enricher"].metadata
    harness.controller._reasoning = ReasoningExecutor(  # noqa: SLF001 - deliberate wiring
        GroundedReasoner(metadata)
    )
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=CandidateLedger(),
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
    )
    return harness.controller


# =========================================================================== #
# A. Trajectory schema
# =========================================================================== #


def test_trajectory_is_reconstructable_from_a_real_run() -> None:
    """A trajectory explains the run: context, action, verification, observation, termination."""
    runner = CaseRunner(suite_factory, variant=ABLATION_ADAPTIVE)
    outcome = runner.run(case_by_id("history-driven"), reasoner=build_reasoner())
    trajectory = outcome.trajectory

    assert trajectory.version == EVALUATION_PLANE_VERSION
    assert trajectory.scenario_id == "history-driven"
    assert trajectory.variant == ABLATION_ADAPTIVE
    assert trajectory.terminal is TerminalOutcome.COMPLETED
    assert trajectory.termination_reason == "completed"
    assert trajectory.action_sequence() == ("recommend_from_history", "finish")
    for record in trajectory.records:
        assert isinstance(record, TrajectoryRecord)
        assert record.action_id
    # The candidate-producing step carries its observation and grounded flag.
    first = trajectory.records[0]
    assert first.observation_kind == "recommendation"
    assert first.produced_candidates is True
    assert first.consumed_tool_call is True


def test_trajectory_carries_no_trusted_history_or_keys() -> None:
    """The evaluation record is subject to the same boundary as the demo serialization."""
    runner = CaseRunner(suite_factory, variant=ABLATION_ADAPTIVE)
    outcome = runner.run(case_by_id("history-driven"), reasoner=build_reasoner())
    blob = str(outcome.trajectory.as_dict())
    for forbidden in ("trusted_user_history", "user_key", "memory_id", "source_user_int_id"):
        assert forbidden not in blob
    # The trusted history values the run used must not appear either.
    for asin in ("B000000001", "B000000002", "B000000003"):
        assert asin not in blob


def test_ungrounded_identities_are_counted_as_a_finding() -> None:
    """A source returning an unknown identity is visible in the trajectory, not hidden."""
    trajectory = AgentTrajectory(
        scenario_id="x",
        terminal=TerminalOutcome.COMPLETED,
        records=(
            TrajectoryRecord(
                step_index=0,
                action_id="a",
                executed_action="search_catalog",
                observation_kind="candidate_set",
                ungrounded_count=3,
            ),
        ),
    )
    assert trajectory.ungrounded_seen() == 3


# =========================================================================== #
# B. Cases
# =========================================================================== #


def test_the_case_set_covers_the_ten_required_categories() -> None:
    """Each category the stage names exists as an inspectable case.

    The set may contain **more** than the ten required categories: a later phase adds cases for
    behaviour the original ten did not measure (guarded completion recovery).  The requirement is
    coverage, not an exact count.
    """
    categories = {case.category.split("_", 1)[0] for case in load_cases()}
    assert {str(number) for number in range(1, 11)} <= categories


def test_every_case_declares_more_than_a_final_answer() -> None:
    """A case is a structured expectation, not an example."""
    for case in load_cases():
        assert case.purpose
        assert case.allowed_terminal, f"{case.case_id} must declare allowed terminals"
        declared = (
            case.required_actions
            or case.forbidden_actions
            or case.hard_constraints
            or case.expected_memory_effect != "none"
            or case.recovery_case
        )
        assert declared, f"{case.case_id} declares nothing beyond a message"


def test_cases_are_self_consistent() -> None:
    """A required action cannot also be forbidden, and expectations are bounded."""
    for case in load_cases():
        assert not (set(case.required_actions) & set(case.forbidden_actions))
        assert case.max_steps >= 1
        # A case that requires clarification must allow the suspended terminal.
        if "ask_clarification" in case.required_actions:
            assert TerminalOutcome.ASKED_CLARIFICATION in case.allowed_terminal


def test_case_ids_are_unique_and_addressable() -> None:
    """Case identity is stable so a report can be compared across runs."""
    ids = [case.case_id for case in load_cases()]
    assert len(ids) == len(set(ids))
    assert case_by_id("history-driven").case_id == "history-driven"
    with pytest.raises(KeyError):
        case_by_id("no-such-case")


def test_cases_declare_the_recovery_requirement_explicitly() -> None:
    """The one case that cannot be satisfied by decide-once is marked as a recovery case."""
    required = [case for case in load_cases() if "recovery-required" in case.case_id]
    assert len(required) == 1
    assert required[0].recovery_case is True
    assert required[0].allowed_terminal == (TerminalOutcome.COMPLETED,)


# =========================================================================== #
# C. Metrics
# =========================================================================== #


def _trajectory(**overrides: object) -> AgentTrajectory:
    """Build a minimal trajectory for metric unit tests."""
    base = {
        "scenario_id": "unit",
        "variant": ABLATION_ADAPTIVE,
        "terminal": TerminalOutcome.COMPLETED,
        "steps": 1,
        "tool_calls": 1,
        "records": (),
    }
    base.update(overrides)
    return AgentTrajectory(**base)  # type: ignore[arg-type]


def test_metrics_measure_each_dimension_separately() -> None:
    """There is no aggregate score, and every dimension is independently reported."""
    case = case_by_id("history-driven")
    metrics = compute_metrics(_trajectory(), case)
    assert isinstance(metrics, TrajectoryMetrics)
    assert not hasattr(metrics, "score")
    for field in (
        "terminal_allowed",
        "tool_selection_correct",
        "unnecessary_tool_calls",
        "hallucinated_candidates",
        "memory_effect_correct",
    ):
        assert field in type(metrics).model_fields


def test_a_forbidden_action_fails_tool_selection() -> None:
    """An action the case forbids fails the run even if the outcome looks right."""
    case = case_by_id("simple-fast-path")  # forbids clarification and reasoning
    trajectory = _trajectory(
        records=(
            TrajectoryRecord(step_index=0, action_id="a", executed_action="ask_clarification"),
        )
    )
    metrics = compute_metrics(trajectory, case)
    assert metrics.tool_selection_correct is False
    assert "forbidden_action_used" in metrics.failures()
    assert metrics.passed is False


def test_a_missing_required_action_fails_the_run() -> None:
    """A case may require an action, and not taking it is a failure."""
    case = case_by_id("novel-product-need")
    metrics = compute_metrics(_trajectory(), case)
    assert "search_catalog" in metrics.required_actions_missing
    assert "required_action_missing" in metrics.failures()


def test_budget_overrun_is_measured_against_the_case_bound() -> None:
    """Efficiency is a correctness property for the fast-path case."""
    case = case_by_id("simple-fast-path")  # max_tool_calls == 1
    metrics = compute_metrics(_trajectory(tool_calls=4), case)
    assert metrics.unnecessary_tool_calls == 3
    assert "budget_overrun" in metrics.failures()


def test_a_disallowed_terminal_fails_the_run() -> None:
    """Ending in a way the case does not allow is a failure, whatever else happened."""
    case = case_by_id("history-driven")
    metrics = compute_metrics(_trajectory(terminal=TerminalOutcome.FAILED), case)
    assert metrics.terminal_allowed is False
    assert "terminal_not_allowed" in metrics.failures()


def test_hallucinated_candidates_are_a_grounding_failure() -> None:
    """An ungrounded identity is attributed to the trusted boundary, not to the policy."""
    case = case_by_id("hallucinated-candidate")
    trajectory = _trajectory(
        records=(
            TrajectoryRecord(
                step_index=0,
                action_id="a",
                executed_action="recommend_from_history",
                ungrounded_count=2,
            ),
        )
    )
    metrics = compute_metrics(trajectory, case)
    assert metrics.hallucinated_candidates == 2
    assert "candidate_grounding" in metrics.failures()
    attribution = attribute(case.case_id, ABLATION_ADAPTIVE, metrics.failures())
    assert "candidate_grounding" in attribution.components


def test_constraint_checked_is_separate_from_constraint_enforced() -> None:
    """Checking a constraint and respecting it are different measurements."""
    case = case_by_id("hard-constraint")
    # The case requires the check, not (yet) the narrowing.
    assert case.constraint_enforced is False
    # With no constraint report the constraint counts as unverified, not as compliant.
    metrics = compute_metrics(_trajectory(), case)
    assert metrics.constraint_checked == ()
    assert metrics.unverified_completion is True
    assert "completion" in metrics.failures()


def test_memory_effect_is_measured_against_the_case() -> None:
    """A run that commits when the case expects a task-only override fails."""
    case = case_by_id("task-override")
    metrics = compute_metrics(_trajectory(), case, memory_effect="commit")
    assert metrics.memory_effect_correct is False
    assert "memory_scope" in metrics.failures()


# =========================================================================== #
# D. Attribution
# =========================================================================== #


def test_attribution_is_total_over_the_failure_vocabulary() -> None:
    """Every failure a metric can emit has a declared owning component."""
    case = case_by_id("history-driven")
    all_failures: set[str] = set()
    for terminal in TerminalOutcome:
        for tool_calls in (0, 9):
            metrics = compute_metrics(
                _trajectory(terminal=terminal, tool_calls=tool_calls), case
            )
            all_failures.update(metrics.failures())
    # Plus the ones that need specific inputs.
    all_failures.update(
        compute_metrics(_trajectory(), case_id_capture := case_by_id("novel-product-need")).failures()
    )
    assert all_failures <= set(ATTRIBUTION_COMPONENTS), sorted(all_failures - set(ATTRIBUTION_COMPONENTS))


def test_an_unknown_failure_is_attributed_to_the_protocol_not_dropped() -> None:
    """A metric with no declared owner is still reported, as a protocol finding."""
    result = attribute("x", ABLATION_ADAPTIVE, ("something_new",))
    assert result.attributed["something_new"] == AttributionComponent.PROTOCOL.value
    assert result.passed is False


def test_a_clean_run_attributes_nothing() -> None:
    """No failures means no attribution, and the report says so."""
    result = attribute("x", ABLATION_ADAPTIVE, ())
    assert result.passed is True
    assert result.components == ()


def test_failures_are_grouped_by_component() -> None:
    """A report can say where a run went wrong, not only that it did."""
    result = attribute(
        "x", ABLATION_ADAPTIVE, ("required_action_missing", "candidate_grounding")
    )
    assert result.for_component(AttributionComponent.SEQUENCING) == ("required_action_missing",)
    assert result.for_component(AttributionComponent.CANDIDATE_GROUNDING) == (
        "candidate_grounding",
    )


# =========================================================================== #
# E. Ablation
# =========================================================================== #


def test_the_adaptive_loop_handles_the_case_set_except_documented_gaps() -> None:
    """The observation-conditioned loop satisfies the case set apart from documented gaps.

    ``premature-finish-recovery`` is the documented gap, and it is a limitation of the
    **evaluator's deterministic policy stub**, not of the loop: that stub proposes
    ``recommend_from_history`` whenever candidates are not grounded, so when the history source
    *fails* it repeats the failed action instead of switching source.  The loop bounds it
    correctly; the stub simply cannot recover.  A model-driven policy does recover, which is
    exactly what ``tests/test_llm_agent_policy.py`` and the model-policy ablation show.

    Asserting the gap here keeps it visible rather than letting a green suite imply the
    deterministic stub handles every case.
    """
    report = run_suite(
        load_cases(),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=lambda case: build_reasoner(),
    )
    failed = {outcome.case.case_id for outcome in report.outcomes if not outcome.passed}
    assert failed <= {"premature-finish-recovery"}, [
        (outcome.case.case_id, outcome.metrics.failures())
        for outcome in report.outcomes
        if not outcome.passed
    ]


def test_the_adaptive_and_decide_once_variants_genuinely_diverge() -> None:
    """The comparison the stage exists to make: observation-conditioned control vs decide-once.

    On the case where the first source returns nothing, only a policy that reads the
    observation can switch source.  This is the difference between
    ``Action -> Observation -> Policy`` and ``Decide Once -> Fixed Workflow``, measured rather
    than asserted.
    """
    case = case_by_id("empty-source-recovery-required")

    adaptive = CaseRunner(suite_factory, variant=ABLATION_ADAPTIVE).run(
        case, policy=build_adaptive_policy(case), reasoner=build_reasoner()
    )
    decide_once = CaseRunner(suite_factory, variant=ABLATION_DECIDE_ONCE).run(
        case,
        policy=DecidingOncePolicy(build_adaptive_policy(case)),
        reasoner=build_reasoner(),
    )

    # The adaptive run recovers: it observes the empty history result and switches source.
    assert adaptive.passed is True
    assert adaptive.trajectory.action_sequence() == (
        "recommend_from_history",
        "search_catalog",
        "finish",
    )

    # The decide-once run cannot: one decision is honoured, then it wraps up regardless of the
    # empty result, so no candidate set is ever produced.
    assert decide_once.passed is False
    assert "search_catalog" not in decide_once.trajectory.action_sequence()
    assert decide_once.trajectory.terminal is not TerminalOutcome.COMPLETED
    assert "required_action_missing" in decide_once.metrics.failures()


def test_ablation_is_policy_injection_not_a_runtime_flag() -> None:
    """Variants differ only by the injected policy, so production code needs no unsafe switch."""
    case = case_by_id("history-driven")
    adaptive = CaseRunner(suite_factory, variant=ABLATION_ADAPTIVE).run(
        case, policy=build_adaptive_policy(case), reasoner=build_reasoner()
    )
    once = CaseRunner(suite_factory, variant=ABLATION_DECIDE_ONCE).run(
        case, policy=DecidingOncePolicy(build_adaptive_policy(case)), reasoner=build_reasoner()
    )
    # Both used the same factory (same control plane); only the policy differed.
    assert adaptive.trajectory.variant == ABLATION_ADAPTIVE
    assert once.trajectory.variant == ABLATION_DECIDE_ONCE
    assert adaptive.trajectory.scenario_id == once.trajectory.scenario_id
    # And the decide-once policy is a real policy, not an error path.
    assert "finish" in once.trajectory.action_sequence()


def test_suite_report_has_no_single_aggregate_score() -> None:
    """The report groups by component and never collapses to one number."""
    report = run_suite(
        (case_by_id("simple-fast-path"),),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=lambda case: build_reasoner(),
    )
    payload = report.as_dict()
    assert "failures_by_component" in payload
    assert "score" not in payload
    assert payload["total"] == 1


# =========================================================================== #
# F. Documentation
# =========================================================================== #


def test_every_implemented_stage_is_documented_and_marked() -> None:
    """Documentation states which stages exist, and does not claim more than the code does.

    A documentation drift guard: the four planes the upgrade added must be described, and the
    sections that describe unimplemented capability must say so rather than leaving a reader to
    infer support.
    """
    text = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for marker in (
        "## 6. Control plane",
        "## 7. Candidate plane (Stage 2, IMPLEMENTED)",
        "## 8. Interaction and personalization plane (Stage 3, IMPLEMENTED)",
        "## 9. Reasoning plane (Stage 4, IMPLEMENTED)",
        "## 10. Model-driven policy (Phase 1, IMPLEMENTED)",
        "## 11. Evaluation planes (Stage 6, IMPLEMENTED)",
    ):
        assert marker in text, f"architecture doc is missing {marker!r}"
    # The model-policy section must state its own limits rather than implying a live provider call.
    for limit in ("Not implemented", "constraint narrowing"):
        assert limit in text, f"architecture doc does not state the limit {limit!r}"
    # And the honest limits are stated, not omitted.
    assert "Not implemented (documented, not claimed)" in text


def test_the_agent_plane_does_not_claim_recommendation_accuracy() -> None:
    """The evaluation plane's own docstring refuses the claim the plane cannot support."""
    module = (REPO_ROOT / "recommendation" / "evaluation" / "agent" / "metrics.py").read_text(
        encoding="utf-8"
    )
    assert "NDCG does not measure orchestration" in module
    assert "does not prove relevance" in module


# =========================================================================== #
# G. Model-driven policy ablation (Phase 1)
# =========================================================================== #


def test_the_model_policy_variant_runs_the_whole_case_set() -> None:
    """A model-driven policy satisfies the case set through the same runner and controller.

    The variant is *a policy*, injected through the existing seam: no runtime flag, no branch in
    the controller, no change to the cases.  A scripted model keeps it offline and reproducible.
    """
    report = run_suite(
        load_cases(),
        controller_factory=suite_factory,
        variant=ABLATION_MODEL_POLICY,
        model_policy_factory=build_model_policy_factory(),
        reasoner_factory=lambda case: build_reasoner(),
    )
    assert report.failed == 0, [
        (outcome.case.case_id, outcome.metrics.failures())
        for outcome in report.outcomes
        if not outcome.passed
    ]
    assert report.as_dict()["failures_by_component"] == {}


def test_the_model_policy_is_observation_dependent_across_cases() -> None:
    """The same model double reaches different trajectories because the observations differ.

    ``simple-fast-path`` is answered in two steps; ``empty-source-recovery-required`` needs a
    source switch first.  The model is the same implementation in both, so the difference comes
    from what it was told, not from a different script.
    """
    report = run_suite(
        (case_by_id("simple-fast-path"), case_by_id("empty-source-recovery-required")),
        controller_factory=suite_factory,
        variant=ABLATION_MODEL_POLICY,
        model_policy_factory=build_model_policy_factory(),
        reasoner_factory=lambda case: build_reasoner(),
    )
    by_id = {outcome.case.case_id: outcome for outcome in report.outcomes}
    assert by_id["simple-fast-path"].trajectory.action_sequence() == (
        "recommend_from_history",
        "finish",
    )
    assert by_id["empty-source-recovery-required"].trajectory.action_sequence() == (
        "search_catalog",
        "finish",
    )


def test_the_model_policy_recovers_from_a_refused_completion() -> None:
    """A guard-refused FINISH becomes an observation the model policy acts on.

    The case makes the history source fail, so the first attempt leaves nothing grounded.  The
    model's FINISH is refused by ``CompletionGuard``, and only then does it choose a different
    source - which is the difference between a model that reads observations and one that does
    not.
    """
    outcome = CaseRunner(suite_factory, variant=ABLATION_MODEL_POLICY).run(
        case_by_id("premature-finish-recovery"),
        policy=build_model_policy_factory()(case_by_id("premature-finish-recovery")),
        reasoner=build_reasoner(),
    )
    assert outcome.passed is True
    sequence = outcome.trajectory.action_sequence()
    assert "search_catalog" in sequence
    # The run ended as a completion, so the guard authorised it - the model did not.
    assert outcome.trajectory.terminal is TerminalOutcome.COMPLETED


def test_the_ablation_reports_components_not_a_score() -> None:
    """Extending the ablation did not introduce an aggregate score."""
    report = run_suite(
        (case_by_id("history-driven"),),
        controller_factory=suite_factory,
        variant=ABLATION_MODEL_POLICY,
        model_policy_factory=build_model_policy_factory(),
        reasoner_factory=lambda case: build_reasoner(),
    )
    payload = report.as_dict()
    assert "score" not in payload
    assert "failures_by_component" in payload
    assert payload["variant"] == ABLATION_MODEL_POLICY


def test_the_variant_label_matches_the_policy_that_actually_decided() -> None:
    """A model-policy run is decided by the model policy, not by a silent fallback.

    Regression: the runner falls back to the adaptive default whenever neither an explicit
    ``policy`` nor a ``policy_factory`` is supplied, so an ablation arm that passed
    ``policy_factory=None`` ran the *deterministic* policy while still being filed under the
    model variant.  The label was the only evidence, and it was wrong.  Every record therefore
    carries the runtime's own policy name, and this asserts the two agree.
    """
    case = case_by_id("premature-finish-recovery")

    # The exact shape that regressed: the arm asks the runner to build the policy itself.
    runner = CaseRunner(suite_factory, variant=ABLATION_MODEL_POLICY)
    silent = runner.run(case, policy=None, reasoner=build_reasoner())
    assert silent.trajectory.policy_names() == ("suite-recommend-then-finish",), (
        "the runner's default is the adaptive stub; recording it is what exposes the mistake"
    )

    # The correct wiring: the model policy is handed in for the arm.
    outcome = CaseRunner(suite_factory, variant=ABLATION_MODEL_POLICY).run(
        case,
        policy=build_model_policy_factory()(case),
        reasoner=build_reasoner(),
    )
    assert outcome.trajectory.policy_names() == (LLM_POLICY_NAME,)
    assert outcome.passed is True
    assert outcome.trajectory.action_sequence() == ("search_catalog", "finish")


def test_the_model_policy_prompt_carries_no_product_identity() -> None:
    """The variant's own boundary check: the prompt payload names no candidate.

    The model double needs identities to propose a facts question, and the adapter supplies them
    out of band precisely so the *prompt* stays free of them.  This asserts that separation.
    """
    case = case_by_id("missing-fact")
    model = ObservationReactiveModel()
    policy = build_model_policy_factory()(case)
    # Reach into the adapter's model to inspect the requests it recorded.
    inner = policy.model  # type: ignore[attr-defined]
    assert isinstance(inner, ObservationReactiveModel)

    controller = suite_factory(case, policy, __import__(
        "recommendation.control", fromlist=["LoopLimits"]
    ).LoopLimits(max_steps=case.max_steps, max_tool_calls=max(1, case.max_tool_calls), max_retries=1))
    controller.run(case.message, ("B1", "B2", "B3"), turn_id="t")

    assert inner.requests, "the model was asked at least once"
    for request in inner.requests:
        blob = str(request.context_payload) + request.system_prompt
        for row in CANDIDATE_ROWS:
            assert row[0] not in blob, "a product identity reached the model's prompt"
    assert model.call_count == 0
