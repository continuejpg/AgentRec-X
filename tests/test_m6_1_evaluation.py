"""Tests for the frozen M6.1 paired evaluator.

The evaluator is exercised over hand-built frozen ranking tasks (the same synthetic
catalogue the policy tests use), with an injected deterministic feature builder stub and
with documented preference-conditioned labels where a quality claim is under test.

What is pinned here (M6.1 required categories):
* the protocol object matches the preregistration and has a stable digest;
* contract metrics are reported separately from quality metrics, with zero eligibility /
  identity / determinism violations;
* the label boundary is structural: the feature builder receives only the snapshot and the
  ``for_ranker`` projection, never the task labels;
* labels absent -> ``NOT_EXECUTED_NO_LABELS`` (no quality claim, no paired deltas);
* labels present with one common protocol -> ``MEASURED`` metrics with a fixed-seed paired
  bootstrap CI;
* repeated evaluation with the same tasks and protocol is byte-identical (verified through
  the report digest);
* raw scores are provenance only: changing them cannot change the report.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.ranking.errors import RankingProtocolError  # noqa: E402
from recommendation.ranking.evaluation import (  # noqa: E402
    DEFAULT_PROTOCOL,
    EvaluationProtocol,
    FrozenPairedEvaluator,
    default_feature_builder,
)
from recommendation.ranking.policies import (  # noqa: E402
    rank_all_arms,
    rank_arm,
    shared_eligibility_projection,
)
from recommendation.ranking.schemas import (  # noqa: E402
    ArmEvaluation,
    CandidateFeatureSet,
    EligibilityState,
    EvaluationReport,
    FrozenRankingTask,
    RankingArm,
    TaskLabel,
)
from tests.test_m6_1_policies import (  # noqa: E402
    build_snapshot,
    default_feature_spec,
    make_feature_set,
    scenario,
)

#: A documented synthetic label protocol for the quality-path tests.
LABEL_PROTOCOL = "synthetic_preference_labels_v1"


class StubFeatureBuilder:
    """Deterministic stand-in for ``BeliefAwareFeatureBuilder`` used by the evaluator.

    It records exactly the keyword arguments it is called with, which is how the
    label-boundary test proves the evaluator never passes the task labels to a builder.
    """

    def __init__(self, specs: dict[str, dict[str, Any]] | None = None) -> None:
        self.specs = specs or {}
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> CandidateFeatureSet:
        self.calls.append(dict(kwargs))
        snapshot = kwargs["snapshot"]
        spec = self.specs.get(snapshot.snapshot_id, default_feature_spec())
        return make_feature_set(snapshot, spec)


def labels_for(*, target: str = "cand-d", relevance: float = 2.0) -> tuple[TaskLabel, ...]:
    """One documented positive label plus a lower-relevance supporting label."""
    return (
        TaskLabel(candidate_id=target, relevance=relevance, label_source=LABEL_PROTOCOL),
        TaskLabel(candidate_id="cand-a", relevance=1.0, label_source=LABEL_PROTOCOL),
    )


def labeled_task(
    *,
    task_id: str = "task-1",
    protocol: str | None = LABEL_PROTOCOL,
    labels: Sequence[TaskLabel] | None = None,
) -> FrozenRankingTask:
    _, _, task = scenario(
        task_id=task_id,
        labels=labels if labels is not None else labels_for(),
        label_protocol=protocol,
    )
    return task


def unlabeled_task(*, task_id: str = "task-1") -> FrozenRankingTask:
    _, _, task = scenario(task_id=task_id)
    return task


# --------------------------------------------------------------------------- #
# protocol contract
# --------------------------------------------------------------------------- #


def test_default_protocol_matches_the_preregistration() -> None:
    protocol = DEFAULT_PROTOCOL
    assert protocol.protocol_name == "m6.1.paired-ranking.v1"
    assert protocol.dataset_source == "synthetic_trusted_fixture"
    assert protocol.arms == (
        RankingArm.A0_ORIGINAL,
        RankingArm.A1_EXPLICIT_RERANK,
        RankingArm.A2_BELIEF_TIEBREAK,
    )
    assert protocol.k_values == (1, 3, 5)
    assert protocol.bootstrap_seed == 20260201
    assert protocol.bootstrap_draws == 2000
    assert protocol.missing_evidence_policy == "unknown_neutral"
    assert protocol.quality_status_default == "NOT_EXECUTED_NO_LABELS"
    assert len(protocol.promotion_criteria) == 5
    assert protocol.label_protocol is None
    assert protocol.primary_question.startswith("Can a UnifiedDecisionState")
    assert len(protocol.digest) == 64
    assert set(protocol.digest) <= set("0123456789abcdef")


def test_protocol_digest_is_stable_and_seed_sensitive() -> None:
    first = EvaluationProtocol()
    second = EvaluationProtocol()
    assert first.digest == second.digest == DEFAULT_PROTOCOL.digest
    amended = EvaluationProtocol(bootstrap_seed=7)
    assert amended.digest != DEFAULT_PROTOCOL.digest
    # Arms are frozen: an amended arm list is refused rather than silently accepted.
    with pytest.raises(ValueError):
        EvaluationProtocol(arms=(RankingArm.A0_ORIGINAL,))
    with pytest.raises(ValueError):
        EvaluationProtocol(k_values=(5, 3))
    with pytest.raises(ValueError):
        EvaluationProtocol(bootstrap_draws=0)


# --------------------------------------------------------------------------- #
# contract metrics
# --------------------------------------------------------------------------- #


def test_contract_metrics_are_separate_from_quality_and_clean() -> None:
    tasks = (unlabeled_task(task_id="task-1"), unlabeled_task(task_id="task-2"))
    builder = StubFeatureBuilder()
    report = FrozenPairedEvaluator(feature_builder=builder).evaluate(tasks)
    assert isinstance(report, EvaluationReport)
    assert report.protocol_name == DEFAULT_PROTOCOL.protocol_name
    assert report.protocol_digest == DEFAULT_PROTOCOL.digest
    assert report.dataset_source == "synthetic_trusted_fixture"
    assert report.task_count == 2
    assert report.candidate_count_total == 12
    assert len(report.arms) == 3
    assert [arm.arm for arm in report.arms] == list(DEFAULT_PROTOCOL.arms)
    assert report.quality_status == "NOT_EXECUTED_NO_LABELS"
    assert report.paired_deltas == ()
    for arm in report.arms:
        contract = arm.contract
        assert contract.candidate_count == 12
        assert contract.candidate_identity_preserved is True
        assert contract.candidate_set_equal_to_snapshot is True
        assert contract.candidate_set_equal_to_expected_pool is True
        assert contract.ordering_deterministic is True
        assert contract.eligibility_invariant_violations == 0
        assert contract.explicit_matches_total > 0
        assert contract.inference_covered_candidates >= 0
    by_arm = {arm.arm: arm.contract for arm in report.arms}
    assert by_arm[RankingArm.A0_ORIGINAL].change_rate_vs_a0 == 0.0
    assert by_arm[RankingArm.A1_EXPLICIT_RERANK].change_rate_vs_a0 > 0.0
    assert by_arm[RankingArm.A2_BELIEF_TIEBREAK].change_rate_vs_a0 > 0.0
    assert by_arm[RankingArm.A2_BELIEF_TIEBREAK].change_rate_vs_a1 > 0.0
    assert by_arm[RankingArm.A0_ORIGINAL].top_k_overlap_vs_a0["1"] == 1.0
    for contract in by_arm.values():
        for overlap in contract.top_k_overlap_vs_a0.values():
            assert 0.0 <= overlap <= 1.0
    assert len(report.report_digest) == 64
    # Deliberately: change rates are diagnostics, never quality evidence.
    assert all(delta.status == "NOT_EXECUTED_NO_LABELS" for delta in report.paired_deltas) or (
        report.paired_deltas == ()
    )
    assert any("never effectiveness evidence" in note for note in report.notes)


def test_eligibility_exclusions_are_reported_and_never_violated() -> None:
    assessments = {
        "cand-a": EligibilityState.VERIFIED_ELIGIBLE,
        "cand-d": EligibilityState.VERIFIED_ELIGIBLE,
        "cand-b": EligibilityState.INELIGIBLE,
        "cand-c": EligibilityState.UNRESOLVED,
        # cand-e / cand-f remain NOT_ASSESSED and are therefore not presentable either.
    }
    _, _, task = scenario(assessments=assessments)
    report = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate([task])
    for arm in report.arms:
        assert arm.contract.candidate_count == 2
        # The trusted eligibility decision legitimately removed candidates from the
        # rankable pool: the expected-pool contract check must still hold, and the
        # snapshot comparison must be literally false for that documented reason.
        assert arm.contract.candidate_set_equal_to_snapshot is False
        assert arm.contract.candidate_set_equal_to_expected_pool is True
        assert arm.contract.candidate_identity_preserved is True
        assert arm.contract.eligibility_invariant_violations == 0
    assert any("eligibility projection applied once" in note for note in report.notes)
    assert any("excluded by hard-constraint eligibility" in note for note in report.notes)
    # The report explains why set=snapshot is False and that every arm ranks the same pool.
    assert any("candidate_set_equal_to_snapshot is False" in note for note in report.notes)
    assert any("cross-arm candidate-set equality is asserted" in note for note in report.notes)


def test_feature_coverage_and_unsupported_features_are_reported() -> None:
    base = unlabeled_task(task_id="task-1")
    _, features, _ = scenario(task_id="task-1")
    coverage = features.coverage.model_copy(
        update={"unsupported_inferred_features": ("waterproof",)}
    )
    features = features.model_copy(update={"coverage": coverage})

    class CoverageBuilder(StubFeatureBuilder):
        def build(self, **kwargs: Any) -> CandidateFeatureSet:
            self.calls.append(dict(kwargs))
            return features

    report = FrozenPairedEvaluator(feature_builder=CoverageBuilder()).evaluate([base])
    assert any("feature coverage" in note for note in report.notes)
    assert any("waterproof" in note for note in report.notes)


def test_repeated_evaluation_is_byte_identical() -> None:
    tasks = (unlabeled_task(task_id="task-1"), unlabeled_task(task_id="task-2"))
    first = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    second = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    assert first.report_digest == second.report_digest
    assert first.as_dict() == second.as_dict()
    # A third instance, fresh builder: still identical (no shared mutable state).
    third = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    assert third.report_digest == first.report_digest
    json.dumps(first.as_dict(), sort_keys=True)


# --------------------------------------------------------------------------- #
# label boundary (no ground-truth leakage)
# --------------------------------------------------------------------------- #


def test_feature_builder_receives_no_labels() -> None:
    builder = StubFeatureBuilder()
    task = labeled_task()
    FrozenPairedEvaluator(feature_builder=builder).evaluate([task])
    assert len(builder.calls) == 1
    call = builder.calls[0]
    assert set(call) == {"snapshot", "ranker_projection"}
    assert "labels" not in call and "label_protocol" not in call
    assert call["snapshot"] is task.snapshot
    assert call["ranker_projection"] is task.ranker_projection
    # The frozen feature contract has no label field at all (extra="forbid").
    assert "labels" not in CandidateFeatureSet.model_fields
    for feature in make_feature_set(task.snapshot, default_feature_spec()).features:
        assert "label" not in feature.model_dump()
        assert "label" not in feature.model_dump_json()


def test_policy_inputs_have_no_label_parameter() -> None:
    assert list(inspect.signature(rank_arm).parameters) == ["arm", "feature_set", "snapshot"]
    assert list(inspect.signature(rank_all_arms).parameters) == ["feature_set", "snapshot"]
    assert list(inspect.signature(shared_eligibility_projection).parameters) == [
        "feature_set"
    ]


def test_real_feature_builder_signature_has_no_label_parameter() -> None:
    """The accepted builder's public seam cannot receive ground-truth labels."""
    features_module = pytest.importorskip("recommendation.ranking.features")
    builder_class = getattr(features_module, "BeliefAwareFeatureBuilder", None)
    assert builder_class is not None, "the accepted feature builder class is missing"
    parameters = list(inspect.signature(builder_class.build).parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "self",
        "snapshot",
        "ranker_projection",
    ]
    for parameter in parameters[1:]:
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert not any("label" in parameter.name for parameter in parameters)
    # The default builder used by the evaluator is that accepted class.
    assert isinstance(default_feature_builder(), builder_class)


# --------------------------------------------------------------------------- #
# quality: absent, present, and protocol consistency
# --------------------------------------------------------------------------- #


def test_labels_absent_reports_not_executed_no_labels() -> None:
    tasks = (unlabeled_task(task_id="task-1"), unlabeled_task(task_id="task-2"))
    report = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    assert report.quality_status == "NOT_EXECUTED_NO_LABELS"
    assert report.paired_deltas == ()
    for arm in report.arms:
        assert arm.quality.status == "NOT_EXECUTED_NO_LABELS"
        assert arm.quality.mrr is None
        assert arm.quality.recall_at_k == {}
        assert arm.quality.ndcg_at_k == {}
        assert "historical purchase relevance is not preference-conditioned" in (
            arm.quality.note
        )
    assert any("NOT_EXECUTED_NO_LABELS" in note for note in report.notes)
    assert not any("quality_status=MEASURED" in note for note in report.notes)


def test_labels_present_measures_quality_with_deterministic_paired_ci() -> None:
    tasks = (labeled_task(task_id="task-1"), labeled_task(task_id="task-2"))
    first = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    second = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    assert first.quality_status == "MEASURED"
    assert first.report_digest == second.report_digest
    assert first.as_dict() == second.as_dict()
    for arm in first.arms:
        quality = arm.quality
        assert quality.status == "MEASURED"
        assert quality.label_protocol == LABEL_PROTOCOL
        assert quality.task_count_with_labels == 2
        assert quality.mrr is not None and 0.0 <= quality.mrr <= 1.0
        assert set(quality.recall_at_k) == {"1", "3", "5"}
        assert set(quality.ndcg_at_k) == {"1", "3", "5"}
    # Binary-label target cand-d: A1 ranks it 2nd, A2 ranks it 1st, so the paired MRR
    # delta A2 - A1 is exactly +0.5 on both tasks.
    deltas = {(delta.comparison, delta.metric): delta for delta in first.paired_deltas}
    a2_minus_a1_mrr = deltas[("a2_belief_tiebreak - a1_explicit_rerank", "mrr")]
    assert a2_minus_a1_mrr.status == "MEASURED"
    assert a2_minus_a1_mrr.mean_delta == pytest.approx(0.5)
    assert a2_minus_a1_mrr.sample_size == 2
    assert a2_minus_a1_mrr.ci_low is not None and a2_minus_a1_mrr.ci_high is not None
    assert a2_minus_a1_mrr.ci_low <= a2_minus_a1_mrr.mean_delta <= a2_minus_a1_mrr.ci_high
    assert f"seed={DEFAULT_PROTOCOL.bootstrap_seed}" in (a2_minus_a1_mrr.note or "")
    assert f"draws={DEFAULT_PROTOCOL.bootstrap_draws}" in (a2_minus_a1_mrr.note or "")
    # A2 also beats A0 on MRR (A0 ranks the target 4th -> delta 0.75).
    a2_minus_a0_mrr = deltas[("a2_belief_tiebreak - a0_original", "mrr")]
    assert a2_minus_a0_mrr.status == "MEASURED"
    assert a2_minus_a0_mrr.mean_delta == pytest.approx(0.75)
    # All expected comparisons and metrics are present.
    metric_names = {"mrr", "recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5"}
    assert set(delta.metric for delta in first.paired_deltas) == metric_names
    assert len(first.paired_deltas) == 2 * len(metric_names)


def test_pinned_label_protocol_mismatch_is_refused() -> None:
    protocol = EvaluationProtocol(label_protocol="different_frozen_protocol_v9")
    evaluator = FrozenPairedEvaluator(
        feature_builder=StubFeatureBuilder(), protocol=protocol
    )
    with pytest.raises(RankingProtocolError):
        evaluator.evaluate([labeled_task()])


def test_protocol_quality_status_default_is_kept_for_unlabelled_tasks() -> None:
    protocol = EvaluationProtocol(label_protocol=LABEL_PROTOCOL)
    report = FrozenPairedEvaluator(
        feature_builder=StubFeatureBuilder(), protocol=protocol
    ).evaluate([unlabeled_task()])
    assert report.quality_status == "NOT_EXECUTED_NO_LABELS"
    assert report.paired_deltas == ()


# --------------------------------------------------------------------------- #
# raw scores are provenance only
# --------------------------------------------------------------------------- #


def test_changing_raw_scores_cannot_change_the_report() -> None:
    base_tasks = (unlabeled_task(task_id="task-1"), unlabeled_task(task_id="task-2"))
    baseline = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(
        base_tasks
    )
    loud_scores = {"cand-a": 999.0, "cand-b": 1.0, "cand-c": 42.0, "cand-d": 0.0}
    loud_tasks = tuple(
        scenario(task_id=f"task-{index + 1}", scores=loud_scores)[2]
        for index in range(2)
    )
    perturbed = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(
        loud_tasks
    )
    assert perturbed.report_digest == baseline.report_digest
    assert perturbed.as_dict() == baseline.as_dict()


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def test_invalid_task_inputs_are_refused() -> None:
    evaluator = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder())
    with pytest.raises(RankingProtocolError):
        evaluator.evaluate([])
    duplicate = (unlabeled_task(task_id="task-1"), unlabeled_task(task_id="task-1"))
    with pytest.raises(RankingProtocolError):
        evaluator.evaluate(duplicate)
    with pytest.raises(RankingProtocolError):
        evaluator.evaluate(["not-a-task"])  # type: ignore[list-item]


def test_broken_builder_output_is_refused() -> None:
    class WrongTypeBuilder:
        def build(self, **kwargs: Any) -> Any:
            return {"not": "a feature set"}

    with pytest.raises(RankingProtocolError):
        FrozenPairedEvaluator(feature_builder=WrongTypeBuilder()).evaluate(
            [unlabeled_task()]
        )

    class MissingBuild:
        pass

    with pytest.raises(RankingProtocolError):
        FrozenPairedEvaluator(feature_builder=MissingBuild()).evaluate(
            [unlabeled_task()]
        )


def test_report_is_json_serialisable_and_digest_changes_with_protocol() -> None:
    tasks = (unlabeled_task(),)
    report = FrozenPairedEvaluator(feature_builder=StubFeatureBuilder()).evaluate(tasks)
    payload = report.as_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["protocol_digest"] == DEFAULT_PROTOCOL.digest
    assert round_tripped["report_digest"] == report.report_digest
    amended = FrozenPairedEvaluator(
        feature_builder=StubFeatureBuilder(),
        protocol=EvaluationProtocol(bootstrap_seed=99),
    ).evaluate(tasks)
    assert amended.protocol_digest != report.protocol_digest
    assert amended.report_digest != report.report_digest
    for arm in report.arms:
        assert isinstance(arm, ArmEvaluation)
