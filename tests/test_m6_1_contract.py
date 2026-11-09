"""Lead-owned M6.1 contract tests (offline ranking foundation).

These pin the boundary the milestone depends on:

* the ranking package imports without the model stack and is not imported by any
  production/serving module;
* candidate snapshots reject invalid/duplicate/stale identities at the contract level;
* snapshot digests use exactly the M5 algorithm (`candidate_pool_snapshot` /
  `ranking_snapshot`), so M5 references can be verified without a second digest scheme;
* evidence/eligibility enums line up with the accepted components whenever they are
  importable;
* ground-truth labels cannot structurally reach the feature builder.
"""

from __future__ import annotations

import inspect
import ast
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.ranking.schemas import (  # noqa: E402
    CandidateRecord,
    CandidateSnapshot,
    EligibilityState,
    EvaluationReport,
    EvidenceState,
    FrozenRankingTask,
    RankingArm,
    SnapshotExpectation,
)
from recommendation.catalog.metadata import normalize_product_record  # noqa: E402


def _record(
    candidate_id: str = "cand-a",
    *,
    item_id: int = 101,
    rank: int = 1,
    metadata: bool = True,
) -> CandidateRecord:
    product = (
        normalize_product_record(
            {
                "parent_asin": candidate_id,
                "title": f"Product {candidate_id}",
                "price": 30.0,
                "details": {"Color": "red"},
            }
        )
        if metadata
        else None
    )
    return CandidateRecord(
        candidate_id=candidate_id,
        item_id=item_id,
        original_rank=rank,
        score=1.0,
        metadata_status="found" if product is not None else "missing",
        metadata=product,
    )


def _snapshot(*records: CandidateRecord) -> CandidateSnapshot:
    return CandidateSnapshot(
        snapshot_id="snap-1",
        user_key="user-1",
        task_id="task-1",
        turn_key="task-1:000001",
        candidates=tuple(records),
    )


def test_ranking_package_imports_without_the_model_or_agent_stack() -> None:
    """Prove import isolation in a fresh interpreter.

    The assertion runs in a subprocess because other test modules in the same pytest
    process may legitimately import the accepted recommendation stack first; the claim is
    about the ranking package itself, not about unrelated modules imported by pytest.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import recommendation.ranking  # noqa: F401\n"
        "assert 'torch' not in sys.modules, 'ranking package imported torch'\n"
        "assert not any(m.startswith('recommendation.demo') for m in sys.modules), "
        "'ranking package imported the demo package'\n"
        "assert not any(m.startswith('recommendation.agent') for m in sys.modules), "
        "'ranking package imported the agent package'\n"
        "print('import-isolation-ok')\n"
    )
    completed = subprocess.run(
        [_sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_no_production_module_imports_the_offline_ranking_package() -> None:
    """Serving V1 / Agent / demo code must never import the offline lab package.

    M7 introduced exactly one explicitly gated consumer:
    ``recommendation/demo/grounded_actions.py`` uses the M6.1 public interface only inside
    the experimental refresh function, after the independent ``AGENTRECX_M7_GROUNDED_ACTIONS``
    switch is on.  The default demo import path must still not import it (asserted below).

    ``recommendation/inference/ranking.py`` is a different, pre-existing module and is
    legitimately imported relatively as ``from .ranking``; only references to the top-level
    ``recommendation.ranking`` package (or ``from ..ranking`` inside subpackages) count.
    """
    allowed = {
        Path("recommendation") / "demo" / "grounded_actions.py",
    }
    offenders: list[str] = []
    for path in (REPO_ROOT / "recommendation").rglob("*.py"):
        if "ranking" in path.parts:
            continue
        relative = path.relative_to(REPO_ROOT)
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "recommendation.ranking" in text
            or "from ..ranking" in text
            or "from recommendation import ranking" in text
        ):
            offenders.append(str(relative))
    assert offenders == [], offenders


def test_m7_ranking_consumer_is_lazy_and_gated() -> None:
    """The M7 bridge may reference M6.1, but only lazily inside the refresh function."""
    path = REPO_ROOT / "recommendation" / "demo" / "grounded_actions.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ranking_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "recommendation.ranking"
        ):
            ranking_imports.append(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("recommendation.ranking"):
                    ranking_imports.append(node)
    assert ranking_imports
    top_level = {id(node) for node in tree.body}
    assert all(id(node) not in top_level for node in ranking_imports)

    import subprocess

    code = (
        "import sys\n"
        "import recommendation.demo.agent_service  # noqa: F401\n"
        "assert 'recommendation.ranking' not in sys.modules, "
        "'the default demo import pulled in the offline ranking package'\n"
        "print('default-demo-import-ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, check=False
    )
    assert completed.returncode == 0


def test_candidate_record_rejects_pad_and_metadata_mismatch() -> None:
    with pytest.raises(ValidationError):
        CandidateRecord(candidate_id="x", item_id=0, original_rank=1, score=1.0)
    foreign = normalize_product_record(
        {"parent_asin": "someone-else", "title": "Other", "price": 1.0}
    )
    with pytest.raises(ValidationError):
        CandidateRecord(
            candidate_id="cand-a",
            item_id=101,
            original_rank=1,
            score=1.0,
            metadata_status="found",
            metadata=foreign,
        )


def test_candidate_snapshot_rejects_duplicates_and_non_contiguous_ranks() -> None:
    with pytest.raises(ValidationError):
        _snapshot(_record("cand-a"), _record("cand-a", item_id=102, rank=2))
    with pytest.raises(ValidationError):
        _snapshot(_record("cand-a", item_id=101), _record("cand-b", item_id=101, rank=2))
    with pytest.raises(ValidationError):
        _snapshot(_record("cand-a", rank=1), _record("cand-b", item_id=102, rank=3))


def test_snapshot_digests_use_the_m5_algorithm() -> None:
    from recommendation.state.shopping_adapter import (
        candidate_pool_snapshot,
        ranking_snapshot,
    )

    snapshot = _snapshot(
        _record("cand-a", item_id=101, rank=1),
        _record("cand-b", item_id=102, rank=2),
    )
    assert snapshot.candidate_pool_digest_value() == candidate_pool_snapshot(
        snapshot.candidate_ids
    ).digest
    assert snapshot.ranking_digest_value() == ranking_snapshot(
        snapshot.candidate_ids
    ).digest
    # Order-sensitive ranking digest: reversing the order changes it.
    reversed_ids = tuple(reversed(snapshot.candidate_ids))
    assert ranking_snapshot(reversed_ids).digest != snapshot.ranking_digest_value()


def test_snapshot_expectation_identity_model_is_validated() -> None:
    snapshot = _snapshot(_record("cand-a"))
    expectation = SnapshotExpectation(
        user_key="user-1",
        task_id="task-1",
        turn_key="task-1:000001",
        candidate_count=1,
        candidate_pool_digest=snapshot.candidate_pool_digest_value(),
        ranking_snapshot_digest=snapshot.ranking_digest_value(),
    )
    assert expectation.candidate_count == 1
    with pytest.raises(ValidationError):
        SnapshotExpectation(
            user_key="user-1",
            task_id="task-1",
            turn_key="task-1:000001",
            candidate_pool_digest="not-a-digest",
        )


def test_frozen_task_requires_snapshot_identity_consistency() -> None:
    from recommendation.ranking.schemas import CandidateRankingFeatures, CandidateFeatureSet
    from recommendation.state.decision_state import empty_state
    from recommendation.state.state_projector import for_ranker

    projection = for_ranker(empty_state(user_key="user-1", task_id="task-1"))
    snapshot = _snapshot(_record("cand-a"))
    task = FrozenRankingTask(
        task_id="task-1",
        user_key="user-1",
        turn_key="task-1:000001",
        snapshot=snapshot,
        ranker_projection=projection,
    )
    assert task.snapshot is snapshot
    with pytest.raises(ValidationError):
        FrozenRankingTask(
            task_id="task-2",
            user_key="user-1",
            turn_key="task-1:000001",
            snapshot=snapshot,
            ranker_projection=projection,
        )
    # The feature-set contract also refuses a feature set that does not cover the snapshot.
    with pytest.raises(ValidationError):
        CandidateFeatureSet(
            task_id="task-1",
            user_key="user-1",
            turn_key="task-1:000001",
            snapshot_id=snapshot.snapshot_id,
            snapshot_candidate_count=1,
            candidate_pool_digest=snapshot.candidate_pool_digest_value(),
            ranking_digest=snapshot.ranking_digest_value(),
            features=(),
        )
    assert CandidateRankingFeatures  # imported for the contract check above


def test_evidence_and_eligibility_enums_match_the_accepted_components() -> None:
    from recommendation.preference_matching.schemas import EvidenceStatus

    assert {state.value for state in EvidenceState} == {
        state.value for state in EvidenceStatus
    }
    constraint_eligibility = pytest.importorskip(
        "recommendation.control.constraint_eligibility",
        reason="accepted eligibility enum needs the model stack",
    )
    accepted = {state.value for state in constraint_eligibility.CandidateEligibility}
    ours = {
        state.value
        for state in EligibilityState
        if state is not EligibilityState.NOT_ASSESSED
    }
    assert ours == accepted


def test_arm_names_are_frozen() -> None:
    assert {arm.value for arm in RankingArm} == {
        "a0_original",
        "a1_explicit_rerank",
        "a2_belief_tiebreak",
    }


def test_evaluation_report_digest_is_deterministic_and_body_bound() -> None:
    from recommendation.ranking.schemas import ArmContractMetrics, ArmEvaluation

    report = EvaluationReport(
        protocol_name="m6.1.paired-ranking.v1",
        protocol_digest="a" * 64,
        dataset_source="synthetic_trusted_fixture",
        task_count=1,
        arms=(
            ArmEvaluation(
                arm=RankingArm.A0_ORIGINAL,
                contract=ArmContractMetrics(candidate_count=1),
            ),
        ),
        quality_status="NOT_EXECUTED_NO_LABELS",
    )
    first = report.with_digest()
    second = report.with_digest()
    assert first.report_digest == second.report_digest
    assert len(first.report_digest) == 64
    changed = report.model_copy(update={"task_count": 2}).with_digest()
    assert changed.report_digest != first.report_digest


def test_feature_builder_cannot_accept_ground_truth_labels() -> None:
    """The builder API must take only the snapshot and the M5 ranker projection."""

    pytest.importorskip("recommendation.ranking.features", reason="builder not implemented yet")
    from recommendation.ranking.features import BeliefAwareFeatureBuilder

    parameters = set(inspect.signature(BeliefAwareFeatureBuilder.build).parameters)
    assert "snapshot" in parameters
    assert "ranker_projection" in parameters
    assert "task" not in parameters
    assert "labels" not in parameters
