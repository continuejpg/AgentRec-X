"""Integration sanity check of the evaluation protocol on the real Amazon artifact.

This milestone must **not** run a recommender and must not report
recommendation-quality metrics, so this module only checks structural facts about
the cohort derived from the Milestone 1.5 preprocessing artifacts:

* artifact/user/eligibility counts
* validation and test case counts
* train-history length statistics
* every target inside ``1..num_items`` and PAD absent
* history invariants (test history extends train history by exactly the
  validation item; no temporal leakage)
* deterministic split/reload

The 100k Sports and Outdoors prefix is an engineering integration sample, not a
representative benchmark: nothing here may be read as a statement about
recommendation difficulty or expected model performance.

If the artifacts are absent (fresh clone) the tests are skipped rather than
failing, because ``data/`` is git-ignored and intentionally not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config  # noqa: E402
from recommendation.evaluation import (  # noqa: E402
    FullRankingEvaluator,
    build_cohort,
    build_cohort_from_artifacts,
    cohort_summary,
)

SEQUENCES_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
MAPPINGS_PATH = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"

artifacts_available = pytest.mark.skipif(
    not (SEQUENCES_PATH.exists() and MAPPINGS_PATH.exists()),
    reason="Milestone 1.5 preprocessing artifacts not present (data/ is git-ignored)",
)


def _raw_payload() -> dict:
    with open(SEQUENCES_PATH, "rt", encoding="utf-8") as handle:
        return json.load(handle)


#: Module-level cache: the split is deterministic, so deriving it once is enough
#: and keeps the real-artifact tests fast (the cohort has ~8k users).
_COHORT_CACHE: dict[str, tuple[list, object]] = {}


def _cohort() -> tuple[list, object]:
    """Return the cached ``(cases, report)`` for the real artifact."""
    if "value" not in _COHORT_CACHE:
        _COHORT_CACHE["value"] = build_cohort_from_artifacts(
            str(SEQUENCES_PATH), str(MAPPINGS_PATH)
        )
    return _COHORT_CACHE["value"]  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Pure-synthetic end-to-end driver check (always runs)
# --------------------------------------------------------------------------- #


def test_end_to_end_driver_on_synthetic_cohort() -> None:
    """The driver end-to-end on a tiny cohort, with a deterministic oracle model.

    This is the only place a scorer is used, and it is a synthetic oracle - it
    demonstrates that the plumbing produces 1.0 metrics for a perfect ranker and
    that validation/test modes both run.  It is not a model result.
    """
    sequences = {
        "u1": [1, 2, 3],
        "u2": [4, 5, 6, 7],
        "u3": [1, 2],  # excluded: too short
    }
    num_items = 10
    cases, report = build_cohort(sequences, num_items=num_items, user_int_ids={"u1": 1, "u2": 2, "u3": 3})

    assert report.num_users_total == 3
    assert report.num_users_eligible == 2
    assert report.num_users_excluded == 1

    evaluator = FullRankingEvaluator(num_items=num_items, k_values=(5, 10, 20))

    calls: list[tuple[str, tuple[int, ...], int]] = []

    def oracle_scorer(history: tuple[int, ...], target: int) -> list[float]:
        """Rank the target first; ties never matter because it is strictly best."""
        calls.append(("score", history, target))
        vector = [0.0] * (num_items + 1)
        vector[target] = 1.0
        return vector

    for mode in ("validation", "test"):
        outcome = evaluator.evaluate(cases, oracle_scorer, mode=mode)
        assert outcome.report.num_cases == 2
        for k in (5, 10, 20):
            assert outcome.report.metrics["HR"][k] == 1.0
            assert outcome.report.metrics["Recall"][k] == 1.0
            assert outcome.report.metrics["NDCG"][k] == 1.0

    # Validation mode: history is train_history; test mode: train + validation item.
    assert ((1,), 2) in [(h, t) for _, h, t in calls]
    assert ((1, 2), 3) in [(h, t) for _, h, t in calls]
    assert ((4, 5, 6), 7) in [(h, t) for _, h, t in calls]


def test_end_to_end_driver_rejects_short_cohort() -> None:
    """A cohort whose users are all too short fails loudly rather than NaN-ing."""
    from recommendation.evaluation.metrics import EvaluationError

    cases, report = build_cohort({"u1": [1, 2]}, num_items=5)
    assert cases == []
    evaluator = FullRankingEvaluator(num_items=5)
    try:
        evaluator.evaluate(cases, lambda history, target: [0.0] * 6, mode="test")
    except EvaluationError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("expected EvaluationError for an all-too-short cohort")


# --------------------------------------------------------------------------- #
# Real artifact integration (skipped when data/ is absent)
# --------------------------------------------------------------------------- #


@artifacts_available
def test_real_artifact_raw_counts_match_cohort_accounting() -> None:
    """Artifact user count equals eligible + excluded users."""
    payload = _raw_payload()
    cases, report = _cohort()

    assert report.num_users_total == payload["num_users"]
    assert report.num_users_eligible == len(cases)
    assert report.num_users_eligible + report.num_users_excluded == report.num_users_total
    assert report.num_validation_cases == len(cases)
    assert report.num_test_cases == len(cases)
    assert report.catalog_size == payload["num_items"]


@artifacts_available
def test_real_artifact_eligibility_is_exactly_length_at_least_three() -> None:
    """Eligibility is precisely ``length >= 3``; no other rule is applied."""
    payload = _raw_payload()
    cases, report = _cohort()

    expected_eligible = sum(1 for rec in payload["sequences"] if rec["length"] >= 3)
    expected_excluded = sum(1 for rec in payload["sequences"] if rec["length"] < 3)

    assert report.num_users_eligible == expected_eligible
    assert report.num_users_excluded == expected_excluded
    assert {c.user_id for c in cases} == {
        rec["user_id"] for rec in payload["sequences"] if rec["length"] >= 3
    }


@artifacts_available
def test_real_artifact_split_shapes_are_exact() -> None:
    """Every case splits exactly as the leave-two-out contract requires."""
    payload = _raw_payload()
    lengths = {rec["user_id"]: rec for rec in payload["sequences"]}
    cases, _ = _cohort()

    for case in cases:
        rec = lengths[case.user_id]
        assert case.sequence_length == rec["length"] >= 3
        assert len(case.train_history) == rec["length"] - 2
        assert case.validation_target == rec["item_ids"][-2]
        assert case.test_target == rec["item_ids"][-1]
        assert case.train_history == tuple(rec["item_ids"][:-2])


@artifacts_available
def test_real_artifact_targets_and_histories_respect_invariants() -> None:
    """Targets are in-catalog, PAD-free, and histories show no temporal leakage."""
    cases, report = _cohort()
    summary = cohort_summary(cases, report)

    assert summary["targets"]["pad_present"] is False
    assert summary["targets"]["all_within_catalog"] is True
    assert summary["targets"]["min"] >= config.FIRST_REAL_ID
    assert summary["targets"]["max"] <= report.catalog_size
    assert summary["history"]["pad_present"] is False
    assert summary["history"]["validation_history_is_train_history"] is True
    assert summary["history"]["test_history_extends_train_by_one"] is True
    assert summary["history"]["test_history_prefix_matches_train"] is True
    assert summary["history"]["test_history_last_is_validation_target"] is True
    assert summary["train_history"]["count"] == len(cases)
    assert summary["train_history"]["min"] >= 1


@artifacts_available
def test_real_artifact_train_history_precedes_both_targets() -> None:
    """No temporal leakage: the train history is exactly the prefix before the targets.

    Note what is *not* asserted: an item appearing in both the train history and a
    target is perfectly legitimate - real users re-purchase the same product (the
    integration sample contains several such users). The protocol handles that by
    keeping the target eligible (see
    ``test_target_retained_even_when_it_appears_earlier_in_history``); ordering, not
    item uniqueness, is what prevents leakage.
    """
    payload = _raw_payload()
    by_user = {rec["user_id"]: rec for rec in payload["sequences"]}
    cases, _ = _cohort()

    for case in cases:
        rec = by_user[case.user_id]
        # Positional (ordering) guarantee: the two targets are the final two
        # interactions, so the train history contains only earlier events, and the
        # validation target precedes the test target.
        assert case.train_history == tuple(rec["item_ids"][:-2])
        assert case.validation_target == rec["item_ids"][-2]
        assert case.test_target == rec["item_ids"][-1]
        # Item-level repetition is *not* part of the contract: a user may
        # re-purchase the same product, so an item holding out as the test target
        # can also occur earlier (the sample contains such users). What must never
        # happen is a target being *newer* than the history it is predicted from,
        # and that is what the positional assertions above rule out.
        assert case.test_history[:-1] == case.validation_history
        assert case.test_history[-1] == case.validation_target


@artifacts_available
def test_real_artifact_repeated_targets_stay_eligible_in_the_candidate_pool() -> None:
    """Users who re-purchase an item keep that item rankable as the target.

    This is the real-data counterpart of the repeated-target regression test: the
    target must remain in the candidate pool even when it also occurs in the
    history.  Checking every one of the ~8k users x 2 targets with
    :func:`valid_candidates` would materialise ~190M list entries, so the
    membership check runs on a deterministic sample while the cheap pool-size
    invariant is checked across the whole cohort.
    """
    from recommendation.evaluation.metrics import valid_candidates

    cases, report = _cohort()
    evaluator = FullRankingEvaluator(num_items=report.catalog_size, k_values=(5,))

    repeated_cases: list[tuple[object, tuple[int, ...], int]] = []
    for case in cases:
        for history, target in (
            (case.validation_history, case.validation_target),
            (case.test_history, case.test_target),
        ):
            seen = set(history)
            masked = seen - {target}
            candidates = evaluator.candidate_count(history, target)

            # Full-cohort (cheap) invariant: the pool is the catalog minus the
            # history items other than the target, so the target is never masked.
            assert candidates == report.catalog_size - len(masked)
            if target in seen:
                # Without target retention the pool would be exactly one smaller.
                assert evaluator.candidate_count(history) == candidates - 1
                repeated_cases.append((case, history, target))
            else:
                assert evaluator.candidate_count(history) == candidates

    # The sample really does contain repeated targets, otherwise this test would
    # pass vacuously.
    assert repeated_cases, "expected at least one repeated-target case"

    # Deterministic sample of repeated-target cases: the target is really in the
    # candidate list, and only the other history items are masked out.
    step = max(1, len(repeated_cases) // 50)
    for case, history, target in repeated_cases[::step]:
        candidates = valid_candidates(report.catalog_size, history, target)
        assert target in candidates
        for item in set(history) - {target}:
            assert item not in candidates


@artifacts_available
def test_real_artifact_split_is_deterministic_across_reloads() -> None:
    """Re-loading the same artifacts yields an identical cohort."""
    first, report_a = build_cohort_from_artifacts(str(SEQUENCES_PATH), str(MAPPINGS_PATH))
    second, report_b = build_cohort_from_artifacts(str(SEQUENCES_PATH), str(MAPPINGS_PATH))

    assert report_a.as_dict() == report_b.as_dict()
    assert [c.user_id for c in first] == [c.user_id for c in second]
    assert [(c.train_history, c.validation_target, c.test_target) for c in first] == [
        (c.train_history, c.validation_target, c.test_target) for c in second
    ]


@artifacts_available
def test_real_artifact_catalog_matches_mappings() -> None:
    """The evaluation catalog size equals the mapping artifact's item count."""
    with open(MAPPINGS_PATH, "rt", encoding="utf-8") as handle:
        mappings = json.load(handle)
    _, report = build_cohort_from_artifacts(str(SEQUENCES_PATH), str(MAPPINGS_PATH))

    assert report.catalog_size == mappings["num_items"]
    assert config.PAD_ID not in build_catalog_safe(report.catalog_size)


def build_catalog_safe(num_items: int) -> tuple[int, ...]:
    """Local import indirection so this module has no extra top-level imports."""
    from recommendation.evaluation.split import build_catalog

    return build_catalog(num_items)
