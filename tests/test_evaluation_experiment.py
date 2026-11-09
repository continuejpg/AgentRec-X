"""Tests for the reusable experiment evidence harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.evaluation.experiment import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    ExperimentBundle,
    ExperimentCaseResult,
    ExperimentError,
    ExperimentManifest,
    PopularityIndex,
    build_bundle,
    hash_file,
    hash_json,
    paired_bootstrap_delta,
    summarize_cases,
)


def _manifest(**overrides: object) -> ExperimentManifest:
    values: dict[str, object] = {
        "name": "unit-test",
        "protocol_version": "agentrecx.eval_protocol.v1",
        "cohort_id": "cohort-1",
        "cohort_sha256": "a" * 64,
        "candidate_pool_id": "pool-1",
        "candidate_pool_sha256": "b" * 64,
        "driver": "tests.driver",
        "driver_sha256": "c" * 64,
        "code_git_head": "deadbeef",
        "data_sha256": {"sequences": "d" * 64},
        "model_sha256": {"sasrec": "e" * 64},
        "test_frozen": True,
        "config": {"k": 10},
    }
    values.update(overrides)
    return ExperimentManifest(**values)  # type: ignore[arg-type]


def _case(case_id: str, target: int, **metrics: float) -> ExperimentCaseResult:
    return ExperimentCaseResult(case_id=case_id, target_item_id=target, metrics=metrics)


def test_hash_helpers_are_deterministic(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"agentrecx")
    assert hash_file(file_path) == hash_file(file_path)
    assert len(hash_file(file_path)) == 64
    assert hash_json({"b": 2, "a": 1}) == hash_json({"a": 1, "b": 2})
    assert hash_json({"a": 1}) != hash_json({"a": 2})


def test_manifest_is_frozen_and_json_serialisable() -> None:
    manifest = _manifest()
    assert manifest.test_frozen is True
    encoded = json.dumps(manifest.as_dict(), sort_keys=True)
    assert "cohort_sha256" in encoded
    assert manifest.as_dict()["data_sha256"] == {"sequences": "d" * 64}


def test_manifest_refuses_mutable_and_incomplete_runs() -> None:
    with pytest.raises(ExperimentError, match="test_frozen"):
        _manifest(test_frozen=False)
    with pytest.raises(ExperimentError, match="cohort_sha256"):
        _manifest(cohort_sha256="not-a-hash")
    with pytest.raises(ExperimentError, match="non-empty"):
        _manifest(data_sha256={})


def test_popularity_slices_are_deterministic_and_do_not_invent_unknowns() -> None:
    popularity = PopularityIndex.from_counts({1: 70, 2: 10, 3: 10, 4: 5, 5: 5, 6: 0})
    assert popularity.total_interactions == 100
    assert popularity.head_items == frozenset({1, 2})
    assert popularity.body_items == frozenset({3, 4})
    assert popularity.tail_items == frozenset({5, 6})
    assert popularity.slice_for(1) == "head"
    assert popularity.slice_for(3) == "body"
    assert popularity.slice_for(5) == "tail"
    assert popularity.slice_for(999) == "unknown"
    assert PopularityIndex.from_counts({}).total_interactions == 0


def test_summary_reports_overall_and_slices_separately() -> None:
    cases = [
        _case("a", 1, recall10=1.0, ndcg10=0.8),
        _case("b", 2, recall10=0.0, ndcg10=0.0),
        _case("c", 3, recall10=1.0, ndcg10=0.5),
    ]
    popularity = PopularityIndex.from_counts({1: 80, 2: 10, 3: 10})
    summary = summarize_cases(cases, popularity=popularity)
    assert summary.n_cases == 3
    assert summary.metrics["recall10"] == pytest.approx(2.0 / 3.0)
    assert summary.by_slice["head"]["recall10"] == pytest.approx(1.0)
    assert summary.by_slice["body"]["recall10"] == pytest.approx(0.5)
    assert summary.slice_counts == {"body": 2, "head": 1}
    assert "score" not in summary.as_dict()


def test_summary_skips_metrics_not_shared_by_all_cases() -> None:
    cases = [
        _case("a", 1, recall10=1.0, extra=2.0),
        _case("b", 2, recall10=0.0),
    ]
    summary = summarize_cases(cases)
    assert summary.metrics == {"recall10": 0.5}


def test_paired_bootstrap_is_case_paired_and_reproducible() -> None:
    baseline = [_case("a", 1, recall=0.0), _case("b", 2, recall=1.0)]
    candidate = [_case("a", 1, recall=1.0), _case("b", 2, recall=1.0)]
    first = paired_bootstrap_delta(
        baseline, candidate, metric="recall", draws=500, seed=DEFAULT_BOOTSTRAP_SEED
    )
    second = paired_bootstrap_delta(
        baseline, candidate, metric="recall", draws=500, seed=DEFAULT_BOOTSTRAP_SEED
    )
    assert first == second
    assert first.n == 2
    assert first.mean == pytest.approx(0.5)
    assert first.lower <= first.mean <= first.upper
    assert first.confidence == 0.95


def test_paired_bootstrap_rejects_unaligned_or_incomplete_arms() -> None:
    baseline = [_case("a", 1, recall=0.0)]
    with pytest.raises(ExperimentError, match="identical case ids"):
        paired_bootstrap_delta(
            baseline, [_case("b", 2, recall=1.0)], metric="recall", draws=10
        )
    with pytest.raises(ExperimentError, match="missing"):
        paired_bootstrap_delta(
            baseline, [_case("a", 1, ndcg=1.0)], metric="recall", draws=10
        )


def test_bundle_requires_aligned_arms_and_reports_paired_deltas() -> None:
    manifest = _manifest()
    baseline = [_case("a", 1, recall=0.0), _case("b", 2, recall=1.0)]
    candidate = [_case("a", 1, recall=1.0), _case("b", 2, recall=1.0)]
    bundle = build_bundle(manifest, {"baseline": baseline, "candidate": candidate})
    assert bundle["manifest"]["test_frozen"] is True
    assert set(bundle["arms"]) == {"baseline", "candidate"}
    assert bundle["paired_deltas"]["baseline"] == "baseline"
    assert bundle["paired_deltas"]["arms"]["candidate"]["recall"]["mean"] == pytest.approx(0.5)

    with pytest.raises(ExperimentError, match="identical case ids"):
        ExperimentBundle(
            manifest=manifest,
            arms={"a": tuple(baseline), "b": (_case("z", 9, recall=0.0),)},
        )


def test_empty_case_set_is_an_error_not_a_zero_summary() -> None:
    with pytest.raises(ExperimentError, match="empty"):
        summarize_cases([])
