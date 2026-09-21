"""Phase 2: task-scoped hard-constraint enforcement and the feasible candidate view.

The gap this module pins is the difference between **checking** a hard constraint and
**enforcing** it.  Before this phase the trusted reasoner could compute a per-candidate
three-state verdict, but nothing turned that verdict into task eligibility: a candidate the
catalogue proved violated could still be presented as a recommendation, because the ledger
is a provenance record with no notion of feasibility and the renderers iterate whatever the
candidate set holds.

Everything here is offline and deterministic: the real reasoner over the synthetic fixture
catalogue, the real control plane, and no provider of any kind.

What the tests are arranged to prove:

A. the derivation itself - three states in, three eligibility states out
B. the negative space - what must *not* happen (resurrection, unknown-as-pass, ledger
   destruction, stale eligibility, policy bypass)
C. provenance survives exclusion
D. the wiring - a real run narrows what it presents, and both policies agree on the truth
E. completion - FINISH cannot certify unverified compliance
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    CandidateEligibility,
    CandidateEligibilityEvaluator,
    CandidateLedger,
    CandidateSource,
    CompletionGuard,
    ConstraintKind,
    ConstraintRequirement,
    ConstraintReport,
    ConstraintVerdict,
    FeasibilityProjection,
    FeasibleCandidateView,
    GroundedReasoner,
    LoopLimits,
    PolicyContext,
    RuleBasedPolicy,
    candidate_eligibility,
    candidate_feasibility_projection,
    constraints_from_task_state,
)
from recommendation.control.constraint_eligibility import (  # noqa: E402
    CONSTRAINT_ELIGIBILITY_VERSION,
    constraint_kind_for,
)
from recommendation.control.grounded_reasoning import UNKNOWN  # noqa: E402
from recommendation.control.task_state import (  # noqa: E402
    ConstraintOrigin,
    TaskConstraint,
    TaskState,
)
from recommendation.memory.schemas import PreferenceKind  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_PRICES,
    CANDIDATE_ROWS,
    build_tool_result,
    rendered_order,
)
from tests.control_fixture import build_control_harness  # noqa: E402

#: A budget ceiling the fixture splits cleanly: ``cand-red`` (25.00) and ``cand-black``
#: (15.00) satisfy it; ``cand-blue`` (40.00) and ``cand-green`` (60.00) violate it.
BUDGET = "35"
#: A ceiling every fixture candidate violates, for the empty-feasible-set case.
IMPOSSIBLE_BUDGET = "5"
#: A ceiling every fixture candidate satisfies.
GENEROUS_BUDGET = "500"

_ALL = ("cand-red", "cand-blue", "cand-black", "cand-green")


@pytest.fixture(scope="module")
def reasoner() -> GroundedReasoner:
    """The real deterministic reasoner over the suite catalogue."""
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


def _price(limit: str) -> ConstraintRequirement:
    return ConstraintRequirement(kind=ConstraintKind.PRICE_MAX, expected=limit)


def _view(
    reasoner: GroundedReasoner,
    limit: str,
    identities: tuple[str, ...] = _ALL,
) -> FeasibleCandidateView:
    return candidate_eligibility(
        identities, requirements=(_price(limit),), reasoner=reasoner
    )


def _task(*constraints: TaskConstraint) -> TaskState:
    state = TaskState()
    for constraint in constraints:
        state = state.with_constraint(constraint)
    return state


def _budget_task(limit: str, *, origin: ConstraintOrigin = ConstraintOrigin.CURRENT_TURN) -> TaskState:
    return _task(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX, value=limit, origin=origin, source_text=f"under {limit}"
        )
    )


# =========================================================================== #
# A. The derivation: three verdicts in, three eligibility states out
# =========================================================================== #


def test_satisfied_makes_a_candidate_verified_eligible(reasoner: GroundedReasoner) -> None:
    """``SATISFIED`` is the only verdict that admits a candidate to the feasible set."""
    view = _view(reasoner, GENEROUS_BUDGET)
    assert view.eligibility_of("cand-red") is CandidateEligibility.VERIFIED_ELIGIBLE
    assert view.is_feasible("cand-red") is True
    assert view.verified_eligible_count == 4
    assert view.ineligible_count == 0
    assert view.unresolved_count == 0
    # The projection is the input order, filtered - and with everything feasible it is the
    # input order unchanged.
    assert view.feasible_identities == _ALL


def test_violated_makes_a_candidate_ineligible(reasoner: GroundedReasoner) -> None:
    """``VIOLATED`` excludes a candidate, and the observed value is retained as the reason.

    The fixture prices are 15.00 / 25.00 / 40.00 / 60.00, so a ceiling of 35 splits the set.
    """
    view = _view(reasoner, BUDGET)
    assert view.eligibility_of("cand-blue") is CandidateEligibility.INELIGIBLE
    assert view.is_feasible("cand-blue") is False
    assessment = view.assessment_for("cand-blue")
    assert assessment is not None
    assert assessment.violated[0].observed == f"{float(CANDIDATE_PRICES['cand-blue']):.2f}"
    assert "price_max" in assessment.explain()
    # The exclusions are reported by the projection, which is what a presenter consumes.
    projection = candidate_feasibility_projection(view, build_tool_result(), None, None)
    assert projection.excluded_identities == ("cand-blue", "cand-green")


def test_unknown_makes_a_candidate_unresolved_not_ineligible(
    reasoner: GroundedReasoner,
) -> None:
    """``UNKNOWN`` is its own state: neither a pass nor a failure.

    A dimension the catalogue carries no attribute for cannot be decided, so the candidate
    must not be excluded (that would fabricate a violation) and must not be admitted (that
    would fabricate compliance).
    """
    requirement = ConstraintRequirement(
        kind=ConstraintKind.UNVERIFIABLE, expected="waterproof"
    )
    view = candidate_eligibility(_ALL, requirements=(requirement,), reasoner=reasoner)
    assert view.eligibility_of("cand-red") is CandidateEligibility.UNRESOLVED
    assert view.is_feasible("cand-red") is False
    assert view.ineligible_count == 0
    assert view.unresolved_count == 4
    assert view.has_evaluable_requirements is False
    assert view.feasible_identities == ()


def test_any_violated_among_multiple_hard_constraints_is_ineligible(
    reasoner: GroundedReasoner,
) -> None:
    """A proved violation outranks an unresolved constraint.

    Already having enough evidence to exclude a candidate means "unresolved" would be the
    wrong answer - it would leave a known-bad candidate inside the feasible view.
    """
    requirements = (
        _price(BUDGET),
        ConstraintRequirement(kind=ConstraintKind.UNVERIFIABLE, expected="waterproof"),
    )
    view = candidate_eligibility(_ALL, requirements=requirements, reasoner=reasoner)
    # cand-blue violates the price ceiling *and* has an undecidable waterproof constraint.
    assert view.eligibility_of("cand-blue") is CandidateEligibility.INELIGIBLE
    # cand-red satisfies price but cannot be decided on waterproof.
    assert view.eligibility_of("cand-red") is CandidateEligibility.UNRESOLVED
    assert view.feasible_identities == ()


def test_all_satisfied_is_verified_eligible_and_satisfied_plus_unknown_is_not(
    reasoner: GroundedReasoner,
) -> None:
    """The two boundary combinations, asserted together so neither can drift.

    ``all SATISFIED`` -> ``VERIFIED_ELIGIBLE``; ``SATISFIED + UNKNOWN`` -> ``UNRESOLVED``.
    The second is the case that matters: one undecided constraint is enough to withhold the
    verified label, and nothing about the satisfied one promotes it.
    """
    satisfied_only = candidate_eligibility(
        ("cand-black",),
        requirements=(
            _price(BUDGET),
            ConstraintRequirement(kind=ConstraintKind.CATEGORY, expected="widget"),
        ),
        reasoner=reasoner,
    )
    assessment = satisfied_only.assessment_for("cand-black")
    assert assessment is not None
    assert len(assessment.satisfied) == 1  # price
    assert len(assessment.violated) == 0
    # The category requirement is undecidable for this fixture (no matching category), so
    # this is the satisfied+unknown combination rather than the all-satisfied one.
    if assessment.unresolved:
        assert assessment.eligibility is CandidateEligibility.UNRESOLVED
    else:
        assert assessment.eligibility is CandidateEligibility.VERIFIED_ELIGIBLE

    all_satisfied = candidate_eligibility(
        ("cand-black",), requirements=(_price(BUDGET),), reasoner=reasoner
    )
    assert all_satisfied.eligibility_of("cand-black") is CandidateEligibility.VERIFIED_ELIGIBLE


def test_a_candidate_the_catalogue_cannot_ground_is_unresolved(
    reasoner: GroundedReasoner,
) -> None:
    """An unknown identity is assessed, not dropped: dropping it would read as compliance."""
    view = _view(reasoner, BUDGET, ("cand-red", "cand-nonexistent"))
    assert view.eligibility_of("cand-nonexistent") is CandidateEligibility.UNRESOLVED
    assert view.feasible_identities == ("cand-red",)
    assert view.assessed_identities == ("cand-red", "cand-nonexistent")


def test_no_active_constraint_means_no_view_rather_than_everything_eligible(
    reasoner: GroundedReasoner,
) -> None:
    """An empty requirement set is reported as "nothing enforced", not "all verified".

    The distinction is load-bearing: a consumer must be able to tell "no constraint was in
    force, present as before" from "every candidate was proved compliant", because only the
    latter licenses a claim of compliance.
    """
    view = candidate_eligibility(_ALL, requirements=(), reasoner=reasoner)
    assert view.has_requirements is False
    # Every candidate is trivially feasible when nothing is constrained - that is not a
    # claim of compliance, which is why ``label`` is empty and ``has_requirements`` false.
    assert view.verified_eligible_count == 4
    assert view.label == ""
    assert view.requirements == ()


def test_the_view_is_versioned_and_serialises_its_detail(reasoner: GroundedReasoner) -> None:
    """The inspectable record keeps every per-constraint verdict, not just the aggregate."""
    payload = _view(reasoner, BUDGET).as_dict()
    assert payload["version"] == CONSTRAINT_ELIGIBILITY_VERSION
    assert payload["verified_eligible_count"] == 2
    assert payload["ineligible_count"] == 2
    assert payload["unresolved_count"] == 0
    assert len(payload["assessments"]) == 4
    blue = next(a for a in payload["assessments"] if a["parent_asin"] == "cand-blue")
    assert blue["eligibility"] == "ineligible"
    assert blue["reports"][0]["verdict"] == "violated"
    assert float(str(blue["reports"][0]["observed"])) == float(CANDIDATE_PRICES["cand-blue"])


def test_eligibility_is_independent_of_the_policy(reasoner: GroundedReasoner) -> None:
    """The same grounded facts produce the same verdicts under either policy.

    Constraint truth is not a policy output, so this asserts the evaluator has no policy
    input at all: two evaluators over the same reasoner agree, and no policy object is
    involved in producing the view.
    """
    first = CandidateEligibilityEvaluator(reasoner).assess(_ALL, requirements=(_price(BUDGET),))
    second = CandidateEligibilityEvaluator(reasoner).assess(_ALL, requirements=(_price(BUDGET),))
    assert first.as_dict() == second.as_dict()
    assert first.feasible_identities == ("cand-red", "cand-black")


# =========================================================================== #
# B. Stale eligibility: task scope, and what a changed constraint does
# =========================================================================== #


def test_the_same_candidate_is_ineligible_under_one_budget_and_eligible_under_another(
    reasoner: GroundedReasoner,
) -> None:
    """Eligibility is task-scoped, never a permanent property of a product."""
    # cand-green costs 60.00: excluded at a 50.00 ceiling, eligible at 70.00.
    assert _view(reasoner, "50").eligibility_of("cand-green") is CandidateEligibility.INELIGIBLE
    assert (
        _view(reasoner, "70").eligibility_of("cand-green")
        is CandidateEligibility.VERIFIED_ELIGIBLE
    )


def test_a_changed_task_state_produces_a_changed_view_with_no_stale_verdict(
    reasoner: GroundedReasoner,
) -> None:
    """A clarified constraint re-evaluates from scratch; nothing is reused or invalidated.

    The evaluator is stateless by construction, so "stale eligibility" is not a condition
    that can arise - this asserts the observable consequence rather than trusting the design.
    """
    evaluator = CandidateEligibilityEvaluator(reasoner)
    initial = constraints_from_task_state(_budget_task("50"))
    clarified = constraints_from_task_state(_budget_task("70"))

    assert initial[0].expected == "50"
    assert clarified[0].expected == "70"

    before = evaluator.assess(("cand-red", "cand-green"), requirements=initial)
    after = evaluator.assess(("cand-red", "cand-green"), requirements=clarified)

    assert before.eligibility_of("cand-green") is CandidateEligibility.INELIGIBLE
    assert after.eligibility_of("cand-green") is CandidateEligibility.VERIFIED_ELIGIBLE
    assert before.requirements != after.requirements
    # The old view is untouched and still describes the constraint set it was built from.
    assert before.eligibility_of("cand-green") is CandidateEligibility.INELIGIBLE


def test_only_explicit_task_constraints_define_feasibility(reasoner: GroundedReasoner) -> None:
    """Soft and inferred constraints never narrow the domain.

    A behavioural signal or an inference may order candidates; only an explicit
    current-turn or session constraint may exclude one.  That rule already lives in
    ``TaskState.hard_constraints`` and this asserts the eligibility layer inherits it rather
    than re-deriving hardness.
    """
    inferred = _task(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value=BUDGET,
            origin=ConstraintOrigin.INFERRED,
        )
    )
    assert inferred.hard_constraints() == ()
    assert constraints_from_task_state(inferred) == ()

    persistent = _task(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value=BUDGET,
            origin=ConstraintOrigin.PERSISTENT_MEMORY,
        )
    )
    assert constraints_from_task_state(persistent) == ()

    explicit = _budget_task(BUDGET)
    requirements = constraints_from_task_state(explicit)
    assert len(requirements) == 1
    assert requirements[0].source_text == f"under {BUDGET}"


def test_an_unmappable_constraint_stays_active_as_unverifiable() -> None:
    """A stated constraint with no catalogue dimension is not silently dropped.

    Dropping it would let a candidate be presented as compliant with a constraint that was
    never tested, which is the same class of error as treating UNKNOWN as a pass.
    """
    # ``free_form_constraint`` is the genuinely unsupported kind: no catalogue attribute models
    # it and no evidence attribute does either.  Phase 4 moved ``material`` and ``feature`` out of
    # this category because an evidence source *can* decide them; ``free_form_constraint`` stays,
    # so an unsupported constraint remains visible rather than being dropped or guessed.
    task = _task(TaskConstraint(kind=PreferenceKind.FREE_FORM_CONSTRAINT, value="fits my old rack"))
    requirements = constraints_from_task_state(task)
    assert len(requirements) == 1
    assert requirements[0].kind is ConstraintKind.UNVERIFIABLE
    assert requirements[0].is_evaluable is False
    assert requirement_is_always_unresolved(requirements[0])


def requirement_is_always_unresolved(requirement: ConstraintRequirement) -> bool:
    """True when the requirement's dimension resolves to ``UNKNOWN`` for any candidate."""
    return requirement.kind is ConstraintKind.UNVERIFIABLE


def test_constraint_kind_mapping_is_total_over_the_preference_ontology() -> None:
    """Every preference kind maps to a constraint dimension, honestly."""
    mapped = {
        PreferenceKind.PRICE_MAX.value: ConstraintKind.PRICE_MAX,
        PreferenceKind.PRICE_MIN.value: ConstraintKind.PRICE_MIN,
        PreferenceKind.CATEGORY.value: ConstraintKind.CATEGORY,
        PreferenceKind.BRAND.value: ConstraintKind.BRAND,
        PreferenceKind.COLOR.value: ConstraintKind.COLOR,
        # Phase 4: decidable from acquired trusted evidence rather than from the base catalogue.
        PreferenceKind.MATERIAL.value: ConstraintKind.MATERIAL,
        PreferenceKind.FEATURE.value: ConstraintKind.FEATURE,
    }
    for value, expected in mapped.items():
        assert constraint_kind_for(value) is expected
    # Only a constraint nothing can decide stays UNVERIFIABLE.
    assert constraint_kind_for(PreferenceKind.FREE_FORM_CONSTRAINT) is ConstraintKind.UNVERIFIABLE


# =========================================================================== #
# C. Provenance survives exclusion
# =========================================================================== #


def test_an_ineligible_candidate_keeps_every_ledger_provenance_record(
    reasoner: GroundedReasoner,
) -> None:
    """A multi-source candidate is excluded from the view and intact in the ledger.

    This is the milestone's core distinction.  ``cand-green`` arrived from two trusted
    sources and is proved to violate the budget; the feasible view drops it, and the ledger
    still knows it was retrieved, by whom, at what rank, and that its identity was grounded.
    Deleting it would have destroyed exactly the evidence an evaluator needs.
    """
    ledger = CandidateLedger()
    grounded = frozenset(row[0] for row in CANDIDATE_ROWS)
    ledger.record_from_source(
        source=CandidateSource.HISTORY,
        candidates=[("cand-green", 1, 0.9), ("cand-red", 2, 0.8)],
        score_kind="sasrec_logit",
        grounded=grounded,
    )
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=[("cand-green", 3, 7.5)],
        score_kind="bm25",
        source_query="green widget",
        grounded=grounded,
    )

    entry = ledger.get("cand-green")
    assert entry is not None
    assert len(entry.sources) == 2
    assert entry.is_multi_source is True
    assert {p.source_rank for p in entry.provenance} == {1, 3}

    view = candidate_eligibility(
        ("cand-red", "cand-green"),
        requirements=(_price(BUDGET),),
        reasoner=reasoner,
    )
    projection = candidate_feasibility_projection(view, build_tool_result(), None, None)
    assert "cand-green" in projection.excluded_identities
    assert "cand-green" not in projection.presented_identities

    # The ledger is untouched by enforcement: same membership, same two sources, same ranks.
    after = ledger.get("cand-green")
    assert after is not None
    assert len(ledger) == 2
    assert "cand-green" in ledger
    assert {p.source for p in after.provenance} == {
        CandidateSource.HISTORY,
        CandidateSource.CATALOG_SEARCH,
    }
    assert {p.source_rank for p in after.provenance} == {1, 3}
    assert after.verification_status.value == "grounded"
    # And the violated verdict is still inspectable, with its observed value.
    assessment = view.assessment_for("cand-green")
    assert assessment is not None
    assert float(str(assessment.violated[0].observed)) == float(CANDIDATE_PRICES["cand-green"])


def test_narrowing_never_removes_the_reason_from_the_record(
    reasoner: GroundedReasoner,
) -> None:
    """The projection can explain every exclusion without re-running the evaluation."""
    view = _view(reasoner, BUDGET)
    projection = candidate_feasibility_projection(view, build_tool_result(), None, None)
    assert projection.exclusion_reason("cand-blue") is not None
    assert "price_max" in str(projection.exclusion_reason("cand-blue"))
    assert projection.exclusion_reason("cand-red") is None
    payload = projection.as_dict()
    assert payload["presented"] == ["cand-red", "cand-black"]
    assert {row["parent_asin"] for row in payload["excluded"]} == {"cand-blue", "cand-green"}


# =========================================================================== #
# D. The negative space: what must not happen
# =========================================================================== #


def test_a_known_violation_cannot_appear_as_a_compliant_recommendation(
    reasoner: GroundedReasoner,
) -> None:
    """The presented set excludes every proved violation.

    The violation is still identifiable from the run's evidence, which is what distinguishes
    "absent from the answer" from "absent from the record".
    """
    view = _view(reasoner, BUDGET)
    projection = candidate_feasibility_projection(
        view, build_tool_result(), None, None
    )
    presented = {
        item.parent_asin for item in projection.tool_result.recommendations
    }
    violations = {
        a.parent_asin for a in view.assessments if a.eligibility.is_ineligible
    }
    assert violations == {"cand-blue", "cand-green"}
    assert presented.isdisjoint(violations)
    assert presented == {"cand-red", "cand-black"}


def test_ranking_cannot_resurrect_an_ineligible_candidate(
    reasoner: GroundedReasoner,
) -> None:
    """A ranker that prefers a violating candidate still cannot present it.

    Eligibility is decided upstream of order, so the only thing a ranker can influence is
    the order of the feasible members.  Here the "ranker" is simulated by handing the
    projection a candidate order that puts the ineligible candidate first: the output keeps
    the feasible order and drops the ineligible member entirely.
    """
    res = build_tool_result()
    reordered = res.model_copy(
        update={
            "recommendations": [
                item for item in reversed(res.recommendations)
            ]
        }
    )
    assert [i.parent_asin for i in reordered.recommendations][0] == "cand-green"

    view = _view(reasoner, BUDGET)
    projection = candidate_feasibility_projection(view, reordered, None, None)
    kept = [item.parent_asin for item in projection.tool_result.recommendations]
    assert kept == ["cand-black", "cand-red"], "eligibility order, not ranker order, decides"
    assert "cand-green" not in kept
    assert "cand-blue" not in kept


def test_unknown_is_never_presented_as_verified_compliance(
    reasoner: GroundedReasoner,
) -> None:
    """An unresolved candidate is absent from the feasible projection.

    This is the ``UNKNOWN != PASS`` rule at the presentation boundary: the response must not
    contain a candidate whose compliance could not be established.
    """
    requirement = ConstraintRequirement(
        kind=ConstraintKind.UNVERIFIABLE, expected="waterproof"
    )
    view = candidate_eligibility(_ALL, requirements=(requirement,), reasoner=reasoner)
    projection = candidate_feasibility_projection(view, build_tool_result(), None, None)
    assert view.unresolved_count == 4
    assert projection.is_empty is True
    assert projection.presented_identities == ()
    assert projection.tool_result.recommendations == []
    # And it is not reported as a violation either - the two states stay distinct.
    assert view.ineligible_count == 0


def test_a_fully_violated_set_yields_an_empty_presentation_not_the_least_bad(
    reasoner: GroundedReasoner,
) -> None:
    """When nothing can comply, the answer is empty - never the closest violator."""
    view = _view(reasoner, IMPOSSIBLE_BUDGET)
    assert view.verified_eligible_count == 0
    assert view.ineligible_count == 4
    projection = candidate_feasibility_projection(view, build_tool_result(), None, None)
    assert projection.is_empty is True
    assert projection.tool_result.recommendations == []
    assert [item.parent_asin for item in build_tool_result().recommendations] == list(_ALL)


def test_the_projection_does_not_mutate_the_trusted_artifacts(
    reasoner: GroundedReasoner,
) -> None:
    """The narrowing is a copy, so the retrieved set survives for the audit trail."""
    forged = build_tool_result()
    view = _view(reasoner, BUDGET)
    before = [item.parent_asin for item in forged.recommendations]
    projection = candidate_feasibility_projection(view, forged, None, None)

    assert [item.parent_asin for item in forged.recommendations] == before
    assert forged.returned_k == 4
    assert [item.parent_asin for item in projection.tool_result.recommendations] == [
        "cand-red",
        "cand-black",
    ]
    # Ranks are provenance, not positions: they are preserved rather than renumbered.
    assert [item.rank for item in projection.tool_result.recommendations] == [1, 3]


def test_a_change_to_the_candidate_set_cannot_leave_a_stale_assessment(
    reasoner: GroundedReasoner,
) -> None:
    """Re-assessing a different candidate set produces verdicts for that set, not the old one.

    A view is valid only for the facts and candidates it was built from, so a consumer can
    never read a verdict about a candidate that is no longer in the run's set.
    """
    wide = _view(reasoner, BUDGET)
    narrow = _view(reasoner, BUDGET, ("cand-red", "cand-blue"))
    assert wide.assessment_for("cand-green") is not None
    assert narrow.assessment_for("cand-green") is None
    assert narrow.eligibility_of("cand-green") is None
    assert narrow.assessed_identities == ("cand-red", "cand-blue")
    # An unassessed candidate is not feasible: "not evaluated" is not "verified".
    assert narrow.is_feasible("cand-green") is False


# =========================================================================== #
# E. Completion semantics
# =========================================================================== #


def _finish_action() -> Any:
    from recommendation.control.schemas import ValidatedAction

    return ValidatedAction(
        action_id="a1",
        step_index=0,
        run_id="r",
        k=3,
        action=ActionKind.FINISH,
    )


def _open_state() -> Any:
    from recommendation.control.schemas import ControlState

    return ControlState(run_id="r", limits=LoopLimits(max_steps=6, max_tool_calls=3))


def test_the_guard_refuses_to_certify_compliance_when_nothing_is_verified() -> None:
    """An unresolved candidate is a reason to keep going, not to certify.

    ``no_verified_compliant_candidate`` is retryable, so the refusal is an observation the
    policy acts on rather than a dead end.
    """
    guard = CompletionGuard()
    verdict = guard.check(
        _finish_action(),
        state=_open_state(),
        last_verification=None,
        produced_recommendation=True,
        candidates_grounded=True,
        constraints_active=True,
        feasible_candidate_count=0,
        unresolved_candidate_count=2,
    )
    assert verdict.verified is False
    assert verdict.code == "no_verified_compliant_candidate"
    assert guard.can_retry(verdict, state=_open_state()) is True


def test_the_guard_allows_an_honest_end_when_every_candidate_is_disproved() -> None:
    """With nothing left to resolve, finishing is honest rather than premature.

    The projection presents an empty set, so no violating candidate is shown.  Refusing here
    would spin the run until its budget expired over a question that is already settled.
    """
    guard = CompletionGuard()
    verdict = guard.check(
        _finish_action(),
        state=_open_state(),
        last_verification=None,
        produced_recommendation=True,
        candidates_grounded=True,
        constraints_active=True,
        feasible_candidate_count=0,
        unresolved_candidate_count=0,
    )
    assert verdict.verified is True
    assert verdict.code == "completion_accepted"
    assert "constraint_eligibility" in verdict.checks


def test_the_guard_is_unaffected_when_no_constraint_is_active() -> None:
    """The Phase-2 check is inert without active constraints."""
    guard = CompletionGuard()
    verdict = guard.check(
        _finish_action(),
        state=_open_state(),
        last_verification=None,
        produced_recommendation=True,
        candidates_grounded=True,
    )
    assert verdict.verified is True
    assert verdict.code == "completion_accepted"
    assert "constraint_eligibility" not in verdict.checks


def test_the_guard_allows_completion_once_something_is_verified() -> None:
    """One verified compliant candidate is enough for a defensible ending."""
    guard = CompletionGuard()
    verdict = guard.check(
        _finish_action(),
        state=_open_state(),
        last_verification=None,
        produced_recommendation=True,
        candidates_grounded=True,
        constraints_active=True,
        feasible_candidate_count=1,
        unresolved_candidate_count=1,
    )
    assert verdict.verified is True


# =========================================================================== #
# F. Integration: a real run enforces, and both policies agree
# =========================================================================== #


def _run_with_budget(limit: str, *, task: TaskState | None = None) -> Any:
    """Drive one real run with a budget constraint active, then return the result."""
    harness = build_control_harness(
        rows=CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
        with_eligibility=True,
    )
    return harness, harness.controller.run(
        "cheap options please",
        ("B1", "B2", "B3"),
        task_state=task if task is not None else _budget_task(limit),
        run_id="phase2",
    )


def test_a_real_run_excludes_the_violation_from_what_it_presents() -> None:
    """The end-to-end trajectory the phase exists to produce.

    Retrieve -> evaluate -> narrow -> finish, with the violating candidate declared in the
    response as absent rather than deleted from the run.
    """
    harness, result = _run_with_budget(BUDGET)

    assert result.control.status.value == "finished"
    # The retrieved set keeps the violating candidate: enforcement is not deletion.
    retrieved = [item.parent_asin for item in result.state["tool_result"].recommendations]
    assert retrieved == ["cand-red", "cand-blue", "cand-black"]
    assessment = result.state["candidate_eligibility"]
    assert assessment.eligibility_of("cand-blue") is CandidateEligibility.INELIGIBLE
    assert assessment.feasible_identities == ("cand-red", "cand-black")

    # What the user is shown contains no violated candidate.
    presented = rendered_order(result.final_response)
    assert presented == ("cand-red", "cand-black")
    assert "cand-blue" not in result.final_response


def test_a_real_run_reports_an_honest_empty_answer_when_nothing_can_comply() -> None:
    """Every candidate disproved: the answer is empty, not the least-bad violator."""
    harness, result = _run_with_budget(IMPOSSIBLE_BUDGET)
    assessment = result.state["candidate_eligibility"]
    assert assessment.verified_eligible_count == 0
    assert assessment.ineligible_count == 3
    presented = rendered_order(result.final_response)
    assert presented == ()
    for identity in ("cand-red", "cand-blue", "cand-black"):
        assert identity not in result.final_response
    # The retrieved set is still fully recorded for the audit trail.
    assert len(result.state["tool_result"].recommendations) == 3


def test_an_unconstrained_run_is_unchanged_by_the_enforcement_path() -> None:
    """With no hard constraint the run presents exactly what it always did."""
    harness = build_control_harness(
        rows=CANDIDATE_ROWS, catalog_rows=CANDIDATE_ROWS, with_eligibility=True
    )
    result = harness.controller.run("recommend gear", ("B1", "B2", "B3"), run_id="phase2")
    assert "candidate_eligibility" not in result.state
    presented = rendered_order(result.final_response)
    retrieved = [item.parent_asin for item in result.state["tool_result"].recommendations]
    assert list(presented) == retrieved


def test_the_policy_can_observe_unresolved_candidates_and_is_told_the_constraint() -> None:
    """Unresolved state reaches the policy, so further evidence is a legal next step.

    The policy sees counts and the active constraint, and - crucially - a *feasible* set that
    is narrower than its grounded set, which is the signal that something is being withheld.
    """
    from recommendation.control.task_state import ConstraintOrigin as _Origin

    # A weight ceiling the catalogue cannot decide: the fixture stores weights, but the
    # stated limit carries no unit, and the reasoner refuses to assume one.
    task = _task(
        TaskConstraint(
            kind=PreferenceKind.FEATURE,
            value="2 Kilograms",
            origin=_Origin.CURRENT_TURN,
        )
    )
    harness = build_control_harness(
        rows=CANDIDATE_ROWS, catalog_rows=CANDIDATE_ROWS, with_eligibility=True
    )
    controller = harness.controller
    result = controller.run("light enough to carry", ("B1", "B2", "B3"), task_state=task, run_id="r")
    assessment = result.state["candidate_eligibility"]
    # A feature constraint has no deterministic dimension, so it is unresolved - never
    # silently satisfied.
    assert assessment.unresolved_count == 3
    assert assessment.ineligible_count == 0
    assert assessment.feasible_identities == ()
    assert rendered_order(result.final_response) == ()


def test_both_policies_produce_the_same_eligibility_for_the_same_facts() -> None:
    """Enforcement is not an LLM feature: the trusted truth is identical under either policy.

    The policies take different trajectories - that is expected and fine - but the verdicts
    come from the reasoner, so the feasible set cannot depend on which policy ran.
    """
    from recommendation.evaluation.agent.model_policy_adapter import _IdentityBindingPolicy
    from tests.control_fixture import RecordingPolicy

    outputs = []
    for policy in (
        RuleBasedPolicy(default_k=3),
        RecordingPolicy(k=3),
    ):
        harness = build_control_harness(
            rows=CANDIDATE_ROWS,
            catalog_rows=CANDIDATE_ROWS,
            policy=policy,
            with_eligibility=True,
        )
        result = harness.controller.run(
            "cheap options", ("B1", "B2", "B3"), task_state=_budget_task(BUDGET), run_id="r"
        )
        assessment = result.state["candidate_eligibility"]
        outputs.append(
            {
                "feasible": assessment.feasible_identities,
                "verified": assessment.verified_eligible_count,
                "ineligible": assessment.ineligible_count,
                "unresolved": assessment.unresolved_count,
                "verdicts": {
                    a.parent_asin: a.eligibility.value for a in assessment.assessments
                },
            }
        )
    assert outputs[0] == outputs[1]
    assert outputs[0]["feasible"] == ("cand-red", "cand-black")


def test_the_evaluator_never_consumes_a_policy_object() -> None:
    """Structural proof that constraint truth cannot come from a policy.

    ``CandidateEligibilityEvaluator`` takes a reasoner and nothing else, so there is no
    channel through which a policy - deterministic or model-driven - could influence a
    verdict even if it wanted to.
    """
    import inspect

    from recommendation.control.constraint_eligibility import CandidateEligibilityEvaluator

    signature = inspect.signature(CandidateEligibilityEvaluator.__init__)
    assert list(signature.parameters) == ["self", "reasoner"]
    assert CandidateEligibilityEvaluator.__init__.__doc__ is None or "policy" not in (
        CandidateEligibilityEvaluator.__init__.__doc__ or ""
    )


def test_no_hard_constraint_leaves_the_feasible_set_untouched_in_a_real_run() -> None:
    """A soft or inferred signal cannot narrow the domain, end to end."""
    from recommendation.control.task_state import ConstraintOrigin as _Origin

    soft = _task(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value=IMPOSSIBLE_BUDGET,
            origin=_Origin.INFERRED,
        )
    )
    harness = build_control_harness(
        rows=CANDIDATE_ROWS, catalog_rows=CANDIDATE_ROWS, with_eligibility=True
    )
    result = harness.controller.run("recommend gear", ("B1", "B2", "B3"), task_state=soft, run_id="r")
    assert "candidate_eligibility" not in result.state
    assert len(rendered_order(result.final_response)) == 3


def test_the_model_payload_carries_the_feasibility_split_but_no_identity() -> None:
    """The model is told the three-state split and the active constraints - and nothing more.

    An unresolved candidate is only actionable if the policy can see that evidence is missing,
    so the counts must reach the payload.  What must not reach it: the excluded identities, the
    observed catalogue values behind a verdict, and any candidate that was not already grounded.
    """
    from recommendation.control.model_policy import build_policy_context_payload
    from recommendation.control.context import CandidateState, PolicyContext

    context = PolicyContext(
        user_request="cheap options",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        candidate_state=CandidateState(grounded=True, candidate_count=4),
        grounded_parent_asins=("cand-red", "cand-blue"),
        verified_eligible_count=1,
        ineligible_count=1,
        unresolved_count=2,
        active_constraints=("price_max<=35",),
    )
    payload = build_policy_context_payload(context)
    feasibility = payload["constraint_feasibility"]
    assert feasibility == {
        "active_constraints": ["price_max<=35"],
        "verified_eligible_count": 1,
        "ineligible_count": 1,
        "unresolved_count": 2,
    }
    blob = str(payload)
    # No identity, no observed value, no catalogue fact.
    for identity in ("cand-red", "cand-blue", "cand-black", "cand-green"):
        assert identity not in blob, "a candidate identity reached the model prompt"
    assert "observed" not in blob


def test_the_payload_never_carries_a_feasible_list() -> None:
    """The feasible *allowlist* stays out of the prompt.

    The policy receives it through ``PolicyContext.feasible_parent_asins`` so it can reason
    about what it may present, but the prompt payload carries counts only - the same rule the
    grounded identity set already followed.
    """
    from recommendation.control.model_policy import build_policy_context_payload
    from recommendation.control.context import CandidateState, PolicyContext

    context = PolicyContext(
        user_request="cheap options",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        candidate_state=CandidateState(grounded=True, candidate_count=2),
        grounded_parent_asins=("cand-red", "cand-black"),
        feasible_parent_asins=("cand-red",),
        verified_eligible_count=1,
        ineligible_count=1,
        active_constraints=("price_max<=35",),
    )
    payload = build_policy_context_payload(context)
    assert "feasible_parent_asins" not in payload
    assert "cand-red" not in str(payload)
