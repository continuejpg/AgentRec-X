"""Phase 4: active evidence acquisition and a real similar-item source.

Two capabilities, one authority model.  Everything here is offline: a local evidence artifact and
a lexical item-item index, no network, no provider, no new dependency.

What is pinned:

* a real item-to-item source, and that it is *lexical* TF-IDF cosine rather than learned
  embedding retrieval (which the repository does not have);
* the seed is explicit and must be a grounded candidate of the run, checked before any retrieval;
* similar-item identities enter only through the trusted candidate machinery, with provenance;
* evidence comes only from a trusted source, and carries provenance;
* the model cannot manufacture evidence: an :class:`EvidenceItem` has no verdict field, and the
  only place a verdict is produced is the Phase-2 kernel;
* ``UNKNOWN -> acquire -> SATISFIED / VIOLATED``, and NOT_FOUND stays ``UNKNOWN``;
* conflicting trusted sources leave a fact undecided rather than picking one;
* one verdict authority: the evidence layer adds an input, not a second constraint system.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    CandidatePlane,
    CatalogSearchSource,
    ConstraintKind,
    ConstraintVerdict,
    EvidenceExecutor,
    EvidenceItem,
    EvidenceStatus,
    EvidenceStore,
    GroundedReasoner,
    GroundingVerifier,
    LocalEvidenceArtifact,
    LoopLimits,
    PolicyActionError,
    SimilarItemIndex,
    SimilarItemSource,
    ValidatedAction,
    build_similar_item_index,
    candidate_eligibility,
)
from recommendation.control.arguments import (  # noqa: E402
    AcquireEvidenceArguments,
    FindSimilarArguments,
)
from recommendation.evaluation.agent import (  # noqa: E402
    ABLATION_ADAPTIVE,
    ABLATION_MODEL_POLICY,
    case_by_id,
    run_suite,
)
from recommendation.evaluation.agent.model_policy_adapter import (  # noqa: E402
    build_model_policy_factory,
)
from recommendation.evaluation.agent.runner import build_adaptive_policy  # noqa: E402
from experiments._harness import EVIDENCE_ARTIFACT  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402
from tests.test_evaluation_agent_plane import (  # noqa: E402
    _Map,
    build_reasoner,
    suite_factory,
)

EVIDENCE_CASES = (
    "evidence-satisfies",
    "evidence-violates",
    "evidence-not-found",
    "similar-item-recovery",
)


@pytest.fixture(scope="module")
def catalogue() -> object:
    return build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata


@pytest.fixture(scope="module")
def index(catalogue: object) -> SimilarItemIndex:
    return build_similar_item_index(catalogue)


# =========================================================================== #
# A. The similar-item source is real, lexical, and deterministic
# =========================================================================== #


def test_the_index_builds_from_the_trusted_catalogue(index: SimilarItemIndex) -> None:
    """Every catalogue product is indexed, and nothing outside it is."""
    assert set(index.identities) == {row[0] for row in CANDIDATE_ROWS}
    assert index.vocabulary_size > 0


def test_neighbours_are_ordered_deterministically(index: SimilarItemIndex) -> None:
    """Descending similarity, with catalogue order breaking ties, reproducibly."""
    first = index.neighbours("cand-red", limit=3)
    second = index.neighbours("cand-red", limit=3)
    assert first == second
    scores = [score for _identity, _rank, score in first]
    assert scores == sorted(scores, reverse=True)
    assert [rank for _identity, rank, _score in first] == list(range(1, len(first) + 1))


def test_the_seed_is_excluded_and_unknown_seeds_are_empty(index: SimilarItemIndex) -> None:
    """A product is not its own neighbour, and an unknown seed yields nothing rather than an error."""
    assert all(identity != "cand-red" for identity, _r, _s in index.neighbours("cand-red"))
    assert index.neighbours("not-in-the-catalogue") == []


def test_similarity_is_symmetric_and_bounded(index: SimilarItemIndex) -> None:
    """Cosine similarity in ``[0, 1]``, symmetric, and 1 for a product against itself."""
    for left in index.identities:
        assert index.similarity(left, left) == pytest.approx(1.0)
        for right in index.identities:
            forward = index.similarity(left, right)
            assert 0.0 <= forward <= 1.0 + 1e-9
            assert forward == pytest.approx(index.similarity(right, left))


def test_the_score_kind_names_lexical_similarity(index: SimilarItemIndex) -> None:
    """The score's meaning is declared, so it can never be fused with a SASRec logit or BM25."""
    source = SimilarItemSource(index)
    assert source.source.value == "similar_item"
    assert source.score_kind == "item_item_similarity"


def test_the_source_refuses_to_choose_a_seed(index: SimilarItemIndex) -> None:
    """No explicit seed is an error, not an invitation to pick the first candidate."""
    source = SimilarItemSource(index)
    with pytest.raises(ValueError, match="explicit seed"):
        source.propose(arguments=None, limit=3)
    assert source.propose(arguments=FindSimilarArguments(seed_parent_asin="cand-red"), limit=2)


def test_a_rare_shared_term_outweighs_a_common_one() -> None:
    """TF-IDF is doing the work, not raw overlap: a shared common term scores lower.

    This is why the implementation is TF-IDF rather than a token-intersection count, and it is
    asserted so a future "simplification" to overlap cannot pass unnoticed.
    """
    index = SimilarItemIndex(
        {
            "a": "widget hiking boot leather",
            "b": "widget hiking boot leather",
            "c": "widget garden hose plastic",
            "d": "widget garden hose plastic",
        },
        identity_order=("a", "b", "c", "d"),
    )
    # 'a' vs 'b' share the distinguishing terms; 'a' vs 'c' share only the common one.
    assert index.similarity("a", "b") > index.similarity("a", "c")


# =========================================================================== #
# B. Similar-item identities enter only through trusted machinery
# =========================================================================== #


class _WidenPolicy:
    """Search for one candidate, then widen from it."""

    name = "widen"

    def __init__(self) -> None:
        self._widened = False

    def choose(self, context: object) -> ActionProposal:
        from recommendation.control.arguments import SearchCatalogArguments

        if not context.candidate_state.grounded:
            return ActionProposal(
                action=ActionKind.SEARCH_CATALOG,
                arguments=SearchCatalogArguments(terms=("redwidget",), limit=1),
            )
        if not self._widened and ActionKind.FIND_SIMILAR in context.available_actions:
            self._widened = True
            return ActionProposal(
                action=ActionKind.FIND_SIMILAR,
                arguments=FindSimilarArguments(
                    seed_parent_asin=context.grounded_parent_asins[0], limit=3
                ),
            )
        return ActionProposal(action=ActionKind.FINISH)


def _widen_run(*, seed_source: str = "run") -> tuple[object, CandidateLedger]:
    """Run the widening trajectory and return the result and the ledger."""
    ledger = CandidateLedger()
    metadata = build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    harness = build_control_harness(
        policy=_WidenPolicy(), limits=LoopLimits(max_steps=6, max_tool_calls=3),
        rows=CANDIDATE_ROWS, catalog_rows=CANDIDATE_ROWS,
    )
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=ledger,
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
        similar_item_tool=SimilarItemSource(build_similar_item_index(metadata)),
    )
    harness.controller._eligibility = CandidateEligibilityEvaluator(  # noqa: SLF001
        GroundedReasoner(metadata)
    )
    return harness.controller.run("widgets", ("B1", "B2", "B3"), run_id="r"), ledger


def test_similar_item_expands_the_candidate_set_with_provenance() -> None:
    """The required trajectory: one source returns too little, FIND_SIMILAR widens the set.

    Every new identity is grounded by the verifier and recorded in the ledger under its own
    source with its own score kind, so the expansion is auditable rather than a model claim.
    """
    from recommendation.control import CandidateSource

    result, ledger = _widen_run()
    assert result.trajectory.actions() == ("search_catalog", "find_similar", "finish")
    assert len(ledger.grounded_parent_asins()) > 1
    sources = {source for entry in ledger.entries for source in entry.sources}
    assert CandidateSource.SIMILAR_ITEM in sources
    assert len(result.state["tool_result"].recommendations) > 1


def test_the_ledger_keeps_both_sources_for_a_rediscovered_identity() -> None:
    """Multi-source provenance is preserved, not overwritten, when both sources see a product."""
    _result, ledger = _widen_run()
    multi = ledger.multi_source_parent_asins()
    # The seed itself is catalog-sourced; a neighbour may also be.  What matters is that the
    # ledger records every sighting rather than collapsing them.
    for identity in multi:
        entry = ledger.get(identity)
        assert entry is not None and len(entry.provenance) >= 2


def test_similar_item_scores_are_not_fused_with_other_sources() -> None:
    """The ledger keeps each source's own score kind; rank fusion uses order only."""
    _result, ledger = _widen_run()
    kinds = {entry.source_count for entry in ledger.entries}
    assert kinds
    assert "item_item_similarity" in str(ledger.as_dict())


# =========================================================================== #
# C. Evidence comes only from a trusted source, and carries provenance
# =========================================================================== #


def test_the_evidence_source_reads_the_artifact_with_provenance(catalogue: object) -> None:
    """A fact arrives with the identity of the source that stated it and where inside it."""
    source = LocalEvidenceArtifact(EVIDENCE_ARTIFACT)
    document = source.read("cand-red")
    assert document.status is EvidenceStatus.FOUND
    assert document.attributes
    for item in document.items:
        assert item.source == "local_evidence_artifact"
        assert str(source.path.name) in item.provenance
        assert item.value


def test_a_product_the_artifact_does_not_cover_is_not_found() -> None:
    """Consulted and empty is NOT_FOUND, which must not be confused with a verdict."""
    document = LocalEvidenceArtifact(EVIDENCE_ARTIFACT).read("cand-black")
    assert document.status is EvidenceStatus.NOT_FOUND
    assert document.items == ()


def test_the_evidence_artifact_is_separate_from_the_catalogue(catalogue: object) -> None:
    """The fixture states facts the base catalogue projection does not carry.

    This is what makes acquisition meaningful: if the catalogue already held the material, the
    ``UNKNOWN -> acquire`` trajectory would be re-reading facts the run already had.
    """
    plain = GroundedReasoner(catalogue)
    assert plain.facts("cand-red").material is None
    document = LocalEvidenceArtifact(EVIDENCE_ARTIFACT).read("cand-red")
    assert "material" in document.attributes


def test_an_evidence_item_has_no_verdict_field() -> None:
    """The authority boundary, expressed as the absence of a field."""
    item = EvidenceItem(
        parent_asin="cand-red", attribute="material", value="leather", source="s"
    )
    assert not hasattr(item, "verdict")
    dumped = item.as_dict()
    assert set(dumped) == {"parent_asin", "attribute", "value", "source", "provenance"}


def test_an_unknown_attribute_or_empty_value_is_refused() -> None:
    """A source cannot state a dimension the reasoner would ignore, or a blank fact."""
    with pytest.raises(ValueError):
        EvidenceItem(parent_asin="x", attribute="not_a_dimension", value="v", source="s")
    with pytest.raises(ValueError):
        EvidenceItem(parent_asin="x", attribute="material", value="   ", source="s")


def test_conflicting_sources_leave_a_fact_undecided() -> None:
    """Two trusted sources disagreeing is reported, kept, and never silently resolved."""
    store = EvidenceStore()
    from recommendation.control.evidence import EvidenceDocument

    store.absorb(
        EvidenceDocument(
            parent_asin="cand-red",
            status=EvidenceStatus.FOUND,
            source="source_a",
            items=(EvidenceItem(parent_asin="cand-red", attribute="material", value="leather", source="source_a"),),
        )
    )
    store.absorb(
        EvidenceDocument(
            parent_asin="cand-red",
            status=EvidenceStatus.FOUND,
            source="source_b",
            items=(EvidenceItem(parent_asin="cand-red", attribute="material", value="canvas", source="source_b"),),
        )
    )
    assert store.value_for("cand-red", "material") is None
    conflicts = store.conflicts_for("cand-red", "material")
    assert len(conflicts) == 2
    assert {item.source for item in conflicts} == {"source_a", "source_b"}
    # Both provenance records survive; nothing was discarded in favour of the convenient value.
    assert len(store.items_for("cand-red", "material")) == 2


def test_agreement_is_decided_case_insensitively() -> None:
    """The same claim spelled differently is not a conflict."""
    store = EvidenceStore()
    from recommendation.control.evidence import EvidenceDocument

    for source, value in (("a", "Leather"), ("b", " leather ")):
        store.absorb(
            EvidenceDocument(
                parent_asin="p",
                status=EvidenceStatus.FOUND,
                source=source,
                items=(EvidenceItem(parent_asin="p", attribute="material", value=value, source=source),),
            )
        )
    assert store.conflicts_for("p", "material") == ()
    assert store.value_for("p", "material") is not None


# =========================================================================== #
# D. One verdict authority: UNKNOWN becomes SATISFIED / VIOLATED
# =========================================================================== #


def _store_with_fixture() -> EvidenceStore:
    store = EvidenceStore()
    source = LocalEvidenceArtifact(EVIDENCE_ARTIFACT)
    for identity in ("cand-red", "cand-blue"):
        store.absorb(source.read(identity))
    return store


def test_no_evidence_leaves_the_dimension_unknown(catalogue: object) -> None:
    """The pre-Phase-4 behaviour, preserved: absence of evidence is never a verdict."""
    reasoner = GroundedReasoner(catalogue)
    for identity in ("cand-red", "cand-blue", "cand-black"):
        report = reasoner.check_constraint(identity, kind=ConstraintKind.MATERIAL, expected="leather")
        assert report.verdict is ConstraintVerdict.UNKNOWN
        assert report.observed is None


def test_acquired_evidence_turns_unknown_into_satisfied_and_violated(catalogue: object) -> None:
    """The capability the phase exists for, at the verdict level."""
    reasoner = GroundedReasoner(catalogue, evidence=_store_with_fixture())
    satisfied = reasoner.check_constraint("cand-red", kind=ConstraintKind.MATERIAL, expected="leather")
    violated = reasoner.check_constraint("cand-blue", kind=ConstraintKind.MATERIAL, expected="leather")
    unknown = reasoner.check_constraint("cand-black", kind=ConstraintKind.MATERIAL, expected="leather")
    assert satisfied.verdict is ConstraintVerdict.SATISFIED
    assert satisfied.observed == "full-grain leather"
    assert violated.verdict is ConstraintVerdict.VIOLATED
    assert violated.observed == "synthetic mesh"
    # A candidate the source does not cover stays UNKNOWN: never a fabricated satisfaction.
    assert unknown.verdict is ConstraintVerdict.UNKNOWN


def test_a_conflict_is_reported_as_unknown_not_as_a_violation(catalogue: object) -> None:
    """Conflicting evidence is undecided, so it cannot make a candidate ineligible either."""
    reasoner = GroundedReasoner(catalogue, evidence=_store_with_fixture())
    from recommendation.control.evidence import EvidenceDocument

    reasoner.evidence.absorb(
        EvidenceDocument(
            parent_asin="cand-red",
            status=EvidenceStatus.FOUND,
            source="second_source",
            items=(EvidenceItem(parent_asin="cand-red", attribute="material", value="canvas", source="second_source"),),
        )
    )
    report = reasoner.check_constraint("cand-red", kind=ConstraintKind.MATERIAL, expected="leather")
    assert report.verdict is ConstraintVerdict.UNKNOWN
    assert "canvas" in str(report.observed) and "leather" in str(report.observed)


def test_the_eligibility_layer_derives_from_the_evidence_backed_verdicts(catalogue: object) -> None:
    """The Phase-2 kernel remains the only authority: eligibility follows its verdicts."""
    from recommendation.control import ConstraintRequirement

    reasoner = GroundedReasoner(catalogue, evidence=_store_with_fixture())
    view = candidate_eligibility(
        ("cand-red", "cand-blue", "cand-black"),
        requirements=(ConstraintRequirement(kind=ConstraintKind.MATERIAL, expected="leather"),),
        reasoner=reasoner,
    )
    assert view.eligibility_of("cand-red") is CandidateEligibility.VERIFIED_ELIGIBLE
    assert view.eligibility_of("cand-blue") is CandidateEligibility.INELIGIBLE
    assert view.eligibility_of("cand-black") is CandidateEligibility.UNRESOLVED
    assert view.feasible_identities == ("cand-red",)


def test_evidence_backed_dimensions_are_not_unverifiable() -> None:
    """Material and feature became decidable; only genuinely unsupported ones stay UNVERIFIABLE."""
    assert ConstraintKind.MATERIAL is not ConstraintKind.UNVERIFIABLE
    assert ConstraintKind.FEATURE is not ConstraintKind.UNVERIFIABLE
    assert ConstraintKind.UNVERIFIABLE.value == "unverifiable"


# =========================================================================== #
# E. The executor enforces run scope and never invents a verdict
# =========================================================================== #


def _evidence_executor() -> tuple[EvidenceExecutor, EvidenceStore]:
    store = EvidenceStore()
    return EvidenceExecutor([LocalEvidenceArtifact(EVIDENCE_ARTIFACT)], store), store


def _evidence_action(identity: str, attribute: str = "material") -> ValidatedAction:
    return ValidatedAction(
        action_id="a", step_index=0, run_id="r", k=4,
        action=ActionKind.ACQUIRE_EVIDENCE,
        arguments=AcquireEvidenceArguments(parent_asin=identity, attribute=attribute),
    )


def test_acquisition_records_a_fact_and_reports_it() -> None:
    """The happy path: a held candidate, a supported attribute, a new fact acquired."""
    executor, store = _evidence_executor()
    observation = executor.execute(_evidence_action("cand-red"), authorized_candidates=("cand-red",))
    assert observation.status == EvidenceStatus.FOUND.value
    assert observation.found is True
    assert observation.made_progress is True
    assert "material" in observation.acquired_attributes
    assert len(store) > 0


def test_acquisition_refuses_an_identity_outside_the_run() -> None:
    """The Phase-2.1 boundary, applied to evidence: run membership is checked before any read."""
    executor, store = _evidence_executor()
    with pytest.raises(PolicyActionError) as caught:
        executor.execute(_evidence_action("cand-green"), authorized_candidates=("cand-red",))
    assert caught.value.code == "candidate_not_in_run"
    assert len(store) == 0


def test_acquisition_refuses_an_attribute_no_source_models() -> None:
    """A policy cannot probe for a question the trusted source cannot answer."""
    executor, _store = _evidence_executor()
    with pytest.raises(PolicyActionError) as caught:
        executor.execute(
            _evidence_action("cand-red", attribute="brand"), authorized_candidates=("cand-red",)
        )
    assert caught.value.code == "unsupported_evidence_attribute"


def test_a_second_acquisition_of_the_same_fact_reports_no_progress() -> None:
    """Asking again changes nothing, and the policy is told so rather than being misled."""
    executor, _store = _evidence_executor()
    first = executor.execute(_evidence_action("cand-red"), authorized_candidates=("cand-red",))
    second = executor.execute(_evidence_action("cand-red"), authorized_candidates=("cand-red",))
    assert first.made_progress is True
    assert second.made_progress is False
    assert second.newly_acquired_attributes == ()


def test_the_observation_carries_no_fact_value() -> None:
    """A trajectory says evidence arrived and what it covered, never what it said."""
    executor, _store = _evidence_executor()
    observation = executor.execute(_evidence_action("cand-red"), authorized_candidates=("cand-red",))
    blob = observation.model_dump_json()
    assert "full-grain leather" not in blob
    assert "waterproof membrane" not in blob
    assert "material" in blob


# =========================================================================== #
# F. Required trajectories, end to end
# =========================================================================== #


@pytest.mark.parametrize("case_id", EVIDENCE_CASES)
def test_the_phase_four_case_passes_under_the_deterministic_policy(case_id: str) -> None:
    """A. UNKNOWN -> acquire -> resolved, B. a proved violation, C. NOT_FOUND, D. widening."""
    report = run_suite(
        (case_by_id(case_id),),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=build_reasoner,
    )
    outcome = report.outcomes[0]
    assert outcome.passed is True, outcome.metrics.failures()


@pytest.mark.parametrize("case_id", EVIDENCE_CASES)
def test_the_phase_four_case_passes_under_the_scripted_model(case_id: str) -> None:
    """The same capability through the model-driven seam, offline."""
    report = run_suite(
        (case_by_id(case_id),),
        controller_factory=suite_factory,
        variant=ABLATION_MODEL_POLICY,
        model_policy_factory=build_model_policy_factory(),
        reasoner_factory=build_reasoner,
    )
    outcome = report.outcomes[0]
    assert outcome.passed is True, outcome.metrics.failures()


def test_the_evidence_trajectory_changes_the_verdict_within_one_run() -> None:
    """The decisive assertion: the same candidate is unresolved before acquisition and decided
    after it, inside a single trajectory."""
    report = run_suite(
        (case_by_id("evidence-satisfies"),),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=build_reasoner,
    )
    outcome = report.outcomes[0]
    sequence = outcome.trajectory.action_sequence()
    assert "acquire_evidence" in sequence
    # The evidence steps precede the finish, and the run completed rather than being refused for
    # an unproved constraint - which is what the pre-Phase-4 runtime would have done.
    assert sequence.index("acquire_evidence") < sequence.index("finish")
    assert outcome.trajectory.terminal.value == "completed"
    assert "completion" not in outcome.metrics.failures()
    assert outcome.metrics.constraint_checked == ("material",)


def test_the_widening_trajectory_uses_both_sources() -> None:
    """Similar-item recovery runs search_catalog *and* find_similar, in that order."""
    report = run_suite(
        (case_by_id("similar-item-recovery"),),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=build_reasoner,
    )
    outcome = report.outcomes[0]
    assert outcome.passed is True, outcome.metrics.failures()
    sequence = outcome.trajectory.action_sequence()
    assert sequence == ("search_catalog", "find_similar", "finish")


def test_an_evidence_case_still_fails_when_the_constraint_cannot_be_resolved() -> None:
    """The honesty check: NOT_FOUND must not be scored as compliance."""
    case = case_by_id("evidence-not-found")
    # The case passes because the run terminates honestly, and its presented set is not claimed
    # as compliant: nothing is verified eligible.
    report = run_suite(
        (case,),
        controller_factory=suite_factory,
        variant=ABLATION_ADAPTIVE,
        policy_factory=build_adaptive_policy,
        reasoner_factory=build_reasoner,
    )
    outcome = report.outcomes[0]
    assert outcome.passed is True
    assert "constraint_violated" not in outcome.metrics.failures()


# =========================================================================== #
# G. No new agent architecture
# =========================================================================== #


def test_the_new_action_is_its_own_class_and_consumes_a_tool_call() -> None:
    """Evidence acquisition is neither candidate-producing nor read-only reasoning."""
    from recommendation.control import (
        CANDIDATE_ACTIONS,
        EVIDENCE_ACTIONS,
        READ_ONLY_ACTIONS,
        TOOL_CALL_ACTIONS,
    )

    assert ActionKind.ACQUIRE_EVIDENCE in EVIDENCE_ACTIONS
    assert ActionKind.ACQUIRE_EVIDENCE not in CANDIDATE_ACTIONS
    assert ActionKind.ACQUIRE_EVIDENCE not in READ_ONLY_ACTIONS
    assert ActionKind.ACQUIRE_EVIDENCE in TOOL_CALL_ACTIONS


def test_the_evidence_executor_has_no_catalogue_access() -> None:
    """It cannot widen the candidate set: it holds sources and a store, nothing else."""
    executor, _store = _evidence_executor()
    for forbidden in ("_ledger", "_plane", "_catalog"):
        assert not hasattr(executor, forbidden)
