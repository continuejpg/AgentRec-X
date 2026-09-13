"""Tests for the temporal leave-two-out split (Milestone 2A).

These are the hand-computable contract tests for
:mod:`recommendation.evaluation.split`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config  # noqa: E402
from recommendation.evaluation import split as split_mod  # noqa: E402
from recommendation.evaluation.split import (  # noqa: E402
    MIN_SEQUENCE_LENGTH,
    PROTOCOL_NAME,
    SplitError,
    build_catalog,
    load_cohort_from_artifacts,
    split_cohort,
    split_sequence,
    train_history_statistics,
    validate_item_ids,
)

# --------------------------------------------------------------------------- #
# Existence / import of the package
# --------------------------------------------------------------------------- #


def test_evaluation_package_exports_protocol_constants() -> None:
    """The protocol name and minimum length are frozen, public constants."""
    assert PROTOCOL_NAME == "temporal_leave_two_out"
    assert MIN_SEQUENCE_LENGTH == 3
    assert config.PAD_ID == 0
    assert config.FIRST_REAL_ID == 1


# --------------------------------------------------------------------------- #
# Exact leave-two-out behaviour
# --------------------------------------------------------------------------- #


def test_split_length_three_sequence_exactly() -> None:
    """n = 3 is the minimum eligible case; history is exactly the first item."""
    case = split_sequence("u", [7, 8, 9], num_items=10, user_int_id=1)

    assert case is not None
    assert case.train_history == (7,)
    assert case.validation_target == 8
    assert case.test_target == 9
    assert case.sequence_length == 3
    assert case.validation_history == (7,)
    assert case.test_history == (7, 8)


def test_split_longer_sequence_exactly() -> None:
    """For n = 6 the train history is the first four items."""
    case = split_sequence("u", [1, 2, 3, 4, 5, 6], num_items=10)

    assert case.train_history == (1, 2, 3, 4)
    assert case.validation_target == 5
    assert case.test_target == 6


def test_split_longer_sequence_does_not_truncate_history() -> None:
    """A long sequence keeps its whole prefix rather than a fixed window."""
    items = list(range(1, 21))
    case = split_sequence("u", items, num_items=20)

    assert case.train_history == tuple(range(1, 19))
    assert case.validation_target == 19
    assert case.test_target == 20
    assert len(case.train_history) == 18


def test_validation_history_is_train_history() -> None:
    """Validation is evaluated with exactly the train history."""
    case = split_sequence("u", [3, 1, 4, 1, 5], num_items=10)

    assert case.validation_history == case.train_history == (3, 1, 4)
    assert case.validation_seen == frozenset({3, 1, 4})


def test_test_history_includes_validation_interaction() -> None:
    """The test history is train history plus the validation interaction."""
    case = split_sequence("u", [3, 1, 4, 1, 5], num_items=10)

    assert case.test_history == case.train_history + (case.validation_target,)
    assert case.test_history == (3, 1, 4, 1)
    assert case.test_history[-1] == case.validation_target == 1
    assert case.test_seen == frozenset({3, 1, 4})


def test_no_temporal_leakage_targets_are_last_two() -> None:
    """Both targets are the final two positions; no target is inside the history."""
    case = split_sequence("u", [10, 20, 30, 40, 50], num_items=50)

    assert case.validation_target not in case.train_history
    assert case.test_target not in case.train_history
    assert case.test_target != case.validation_target
    # the test target is the very last interaction
    assert case.test_target == 50
    assert case.validation_target == 40


def test_pad_is_not_reachable_as_a_target() -> None:
    """PAD (0) can never become a target, even for a minimal sequence."""
    case = split_sequence("u", [0 + 1, 2, 3], num_items=10)
    assert case.validation_target == 2
    assert case.test_target == 3


# --------------------------------------------------------------------------- #
# Cohort eligibility
# --------------------------------------------------------------------------- #


def test_users_shorter_than_minimum_are_excluded() -> None:
    """Only sequences with length >= 3 enter the cohort."""
    sequences = {
        "u1": [1, 2],           # excluded
        "u2": [1],              # excluded
        "u3": [],               # excluded
        "u4": [1, 2, 3],        # eligible
        "u5": [4, 5, 6, 7],     # eligible
    }
    cases, report = split_cohort(sequences, num_items=10)

    assert [c.user_id for c in cases] == ["u4", "u5"]
    assert report.num_users_total == 5
    assert report.num_users_eligible == 2
    assert report.num_users_excluded == 3
    assert report.num_validation_cases == 2
    assert report.num_test_cases == 2


def test_exactly_minimum_length_is_eligible() -> None:
    """Length == MIN_SEQUENCE_LENGTH is eligible (boundary is inclusive)."""
    cases, report = split_cohort({"u": [1, 2, 3]}, num_items=3)
    assert len(cases) == 1
    assert report.num_users_excluded == 0


def test_empty_cohort_is_reported_not_hidden() -> None:
    """An all-too-short cohort yields zero cases and an explicit report."""
    cases, report = split_cohort({"u1": [1], "u2": [2]}, num_items=2)
    assert cases == []
    assert report.num_users_eligible == 0
    assert report.num_users_excluded == 2


def test_cohort_accepts_iterable_of_pairs() -> None:
    """Both a mapping and an iterable of pairs are accepted."""
    cases, report = split_cohort([("u2", [1, 2, 3]), ("u1", [4, 5, 6])], num_items=6)
    assert [c.user_id for c in cases] == ["u1", "u2"]
    assert report.num_users_total == 2


def test_cohort_order_is_deterministic_by_int_id_then_user_id() -> None:
    """Case order follows (user_int_id, user_id), not insertion order."""
    seqs = {"zzz": [1, 2, 3], "aaa": [1, 2, 3], "mmm": [1, 2, 3]}
    ids = {"zzz": 3, "aaa": 1, "mmm": 2}
    forward, _ = split_cohort(seqs, 3, user_int_ids=ids)
    backward, _ = split_cohort(dict(reversed(list(seqs.items()))), 3, user_int_ids=ids)
    assert [c.user_id for c in forward] == ["aaa", "mmm", "zzz"]
    assert [c.user_id for c in forward] == [c.user_id for c in backward]


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


def test_source_sequence_is_not_mutated() -> None:
    """Splitting must not modify the caller's sequence object."""
    items = [9, 8, 7, 6, 5]
    snapshot = list(items)

    case = split_sequence("u", items, num_items=10, user_int_id=1)
    cases, _ = split_cohort({"u": items}, num_items=10)

    assert items == snapshot
    assert isinstance(case.train_history, tuple)
    assert isinstance(cases[0].train_history, tuple)
    # mutating the result must not be possible / must not touch the source
    assert case.train_history == (9, 8, 7)


def test_split_cohort_does_not_mutate_a_mapping_of_lists() -> None:
    """Nested lists inside a mapping are left untouched."""
    sequences = {"u1": [1, 2, 3], "u2": [4, 5, 6]}
    snapshot = {k: list(v) for k, v in sequences.items()}
    split_cohort(sequences, num_items=6)
    assert sequences == snapshot


# --------------------------------------------------------------------------- #
# Catalog and id validation
# --------------------------------------------------------------------------- #


def test_catalog_is_exactly_one_to_num_items() -> None:
    """The catalog never includes PAD and is contiguous from 1."""
    assert build_catalog(1) == (1,)
    assert build_catalog(5) == (1, 2, 3, 4, 5)
    assert config.PAD_ID not in build_catalog(100)


def test_catalog_rejects_invalid_num_items() -> None:
    """A non-positive or non-integer catalogue size is an error."""
    for bad in (0, -1, True, 2.5, "5", None):
        try:
            build_catalog(bad)  # type: ignore[arg-type]
        except SplitError:
            continue
        raise AssertionError(f"expected SplitError for num_items={bad!r}")


def test_pad_in_sequence_is_rejected() -> None:
    """PAD inside a sequence is an explicit error, not silently accepted."""
    try:
        split_sequence("u", [1, 0, 3], num_items=10)
    except SplitError as exc:
        assert "PAD" in str(exc)
        return
    raise AssertionError("expected SplitError for PAD inside a sequence")


def test_out_of_catalog_item_is_rejected() -> None:
    """Ids above num_items, negatives and non-integers are rejected."""
    for bad_items in ([1, 2, 11], [1, 2, -3], [1, 2, "3"], [1, 2, 3.0], [1, 2, True]):
        try:
            split_sequence("u", bad_items, num_items=10)  # type: ignore[arg-type]
        except SplitError:
            continue
        raise AssertionError(f"expected SplitError for items={bad_items!r}")


def test_validate_item_ids_accepts_valid_range() -> None:
    """The validator is silent for a legitimate catalog range."""
    validate_item_ids([1, 5, 10], num_items=10)
    validate_item_ids([], num_items=10)


# --------------------------------------------------------------------------- #
# Statistics / report
# --------------------------------------------------------------------------- #


def test_train_history_statistics() -> None:
    """Length statistics are computed over train histories only."""
    cases, _ = split_cohort({"u1": [1, 2, 3], "u2": [1, 2, 3, 4, 5]}, num_items=5)
    stats = train_history_statistics(cases)

    assert stats["count"] == 2
    assert stats["min"] == 1          # length-3 sequence -> 1 train item
    assert stats["max"] == 3          # length-5 sequence -> 3 train items
    assert stats["total"] == 4
    assert stats["mean"] == 2.0


def test_train_history_statistics_on_empty_cohort() -> None:
    """Empty cohorts give explicit nulls rather than raising."""
    stats = train_history_statistics([])
    assert stats["count"] == 0 and stats["min"] is None and stats["total"] == 0


def test_split_report_is_serialisable() -> None:
    """The split report round-trips through JSON."""
    _, report = split_cohort({"u": [1, 2, 3]}, num_items=3)
    payload = json.dumps(report.as_dict())
    restored = json.loads(payload)
    assert restored["protocol"] == PROTOCOL_NAME
    assert restored["num_users_eligible"] == 1
    assert restored["catalog_size"] == 3


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #


def _write_artifacts(tmp_path: Path, sequences: list[dict], num_items: int) -> tuple[Path, Path]:
    seq_path = tmp_path / "X_sequences.json"
    map_path = tmp_path / "X_mappings.json"
    seq_path.write_text(
        json.dumps(
            {
                "format": "agentrecx.sequences.v1",
                "num_users": len(sequences),
                "num_items": num_items,
                "num_interactions": sum(len(s["item_ids"]) for s in sequences),
                "sequences": sequences,
            }
        ),
        encoding="utf-8",
    )
    map_path.write_text(
        json.dumps({"num_users": len(sequences), "num_items": num_items, "item2id": {}, "user2id": {}}),
        encoding="utf-8",
    )
    return seq_path, map_path


def test_load_cohort_from_artifacts(tmp_path: Path) -> None:
    """A preprocessing artifact is read and split, honouring num_items."""
    sequences = [
        {"user_id": "u1", "user_int_id": 1, "item_ids": [1, 2]},
        {"user_id": "u2", "user_int_id": 2, "item_ids": [1, 2, 3]},
        {"user_id": "u3", "user_int_id": 3, "item_ids": [3, 2, 1, 4]},
    ]
    seq_path, map_path = _write_artifacts(tmp_path, sequences, num_items=4)

    cases, report = load_cohort_from_artifacts(seq_path, map_path)

    assert [c.user_id for c in cases] == ["u2", "u3"]
    assert report.catalog_size == 4
    assert report.num_users_total == 3
    assert report.num_users_excluded == 1
    assert cases[0].train_history == (1,)


def test_load_cohort_rejects_num_items_disagreement(tmp_path: Path) -> None:
    """A sequences/mappings catalogue-size mismatch is an explicit error."""
    seq_path, map_path = _write_artifacts(
        tmp_path, [{"user_id": "u1", "user_int_id": 1, "item_ids": [1, 2, 3]}], num_items=5
    )
    map_path.write_text(json.dumps({"num_items": 7}), encoding="utf-8")

    try:
        load_cohort_from_artifacts(seq_path, map_path)
    except SplitError as exc:
        assert "num_items disagreement" in str(exc)
        return
    raise AssertionError("expected SplitError for num_items disagreement")


def test_load_sequences_artifact_rejects_foreign_json(tmp_path: Path) -> None:
    """A JSON file that is not a sequences artifact is rejected."""
    path = tmp_path / "not_an_artifact.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    try:
        split_mod.load_sequences_artifact(path)
    except SplitError:
        return
    raise AssertionError("expected SplitError for a non-artifact JSON file")
