"""Tests for the Milestone 1 Amazon Reviews 2023 preprocessing pipeline.

Run with pytest::

    python -m pytest tests/ -v

If pytest is not installed, the same tests can be executed with the plain
standalone runner at the bottom of this file::

    python tests/test_preprocess.py

The tests import only the standard library plus their own ``sample_data``
fixture module, so both entry points behave identically.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

# Make ``recommendation`` and ``tests.sample_data`` importable no matter which
# directory the tests are launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation import config  # noqa: E402
from recommendation.io_utils import (  # noqa: E402
    Interaction,
    file_fingerprint,
    load_interactions,
    read_json,
)
from recommendation.preprocess import (  # noqa: E402
    build_id_mapping,
    compute_statistics,
    group_by_user,
    iterative_k_core_filter,
    preprocess_interactions,
    run_preprocessing,
    sort_interactions,
)
from tests import sample_data  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def write_sample(tmp_path: Path, include_special_lines: bool = True) -> Path:
    """Write the synthetic dataset into ``tmp_path`` and return its path."""
    path = Path(tmp_path) / "synthetic_reviews.jsonl"
    return sample_data.write_sample(path, include_special_lines=include_special_lines)


def load_sample(tmp_path: Path, include_special_lines: bool = True):
    """Load the synthetic dataset, returning ``(interactions, report)``."""
    return load_interactions(write_sample(tmp_path, include_special_lines))


def interaction(user: str, item: str, ts: int, rating: float = 5.0) -> Interaction:
    """Shorthand constructor for tests."""
    return Interaction(user_id=user, item_id=item, timestamp=ts, rating=rating)


# --------------------------------------------------------------------------- #
# 1. Timestamp ordering
# --------------------------------------------------------------------------- #


def test_chronological_ordering_matches_expected_sequences(tmp_path: Path) -> None:
    """Unordered input must produce each user's history in chronological order."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    produced = {
        user: [row.item_id for row in rows] for user, rows in result.sequences.items()
    }
    assert produced == sample_data.EXPECTED_FINAL_SEQUENCES, produced


def test_chronological_ordering_timestamps_are_nondecreasing(tmp_path: Path) -> None:
    """Timestamps must be non-decreasing within every emitted sequence."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    for user, rows in result.sequences.items():
        timestamps = [row.timestamp for row in rows]
        assert timestamps == sorted(timestamps), f"{user} not chronological: {timestamps}"

    # The raw file really is unsorted, otherwise this test proves nothing.
    raw_timestamps = [row.timestamp for row in interactions]
    assert raw_timestamps != sorted(raw_timestamps), "sample data should be shuffled"


def test_sort_interactions_is_deterministic_on_ties() -> None:
    """Equal timestamps fall back to (item_id, rating) so order is total."""
    rows = [
        interaction("u", "i9", 1000, 5.0),
        interaction("u", "i2", 1000, 3.0),
        interaction("u", "i2", 1000, 4.0),
    ]
    first = sort_interactions(rows)
    second = sort_interactions(list(reversed(rows)))
    assert [(r.item_id, r.rating) for r in first] == [
        ("i2", 3.0),
        ("i2", 4.0),
        ("i9", 5.0),
    ]
    assert [r.item_id for r in first] == [r.item_id for r in second]


# --------------------------------------------------------------------------- #
# 2. User frequency filtering
# --------------------------------------------------------------------------- #


def test_users_below_min_frequency_are_removed(tmp_path: Path) -> None:
    """``u_low`` (1 interaction) and ``u_edge`` (3) are below the default threshold."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    assert set(result.sequences) == set(sample_data.EXPECTED_FINAL_SEQUENCES)
    for user in sample_data.EXPECTED_REMOVED_USERS:
        assert user not in result.sequences
        assert user not in result.user2id
    assert result.user2id.keys() == set(sample_data.EXPECTED_FINAL_SEQUENCES)


def test_user_at_exactly_threshold_survives() -> None:
    """A user with exactly ``min_user_interactions`` rows must be kept."""
    rows = [
        interaction("u_exact", "i1", 10),
        interaction("u_exact", "i2", 20),
        interaction("u_exact", "i3", 30),
        interaction("u_exact", "i4", 40),
        interaction("u_exact", "i5", 50),
    ]
    result = iterative_k_core_filter(rows, min_user_interactions=5, min_item_interactions=1)
    assert result.users == {"u_exact"}
    assert len(result.interactions) == 5


# --------------------------------------------------------------------------- #
# 3. Item frequency filtering
# --------------------------------------------------------------------------- #


def test_items_below_min_frequency_are_removed(tmp_path: Path) -> None:
    """Low-frequency items must vanish.

    ``i6``, ``i7`` and ``i8`` each end up with 3 interactions, below the
    threshold of 5, and none of them touches the core items.
    """
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    for item in sample_data.EXPECTED_REMOVED_ITEMS:
        assert item not in result.item2id, f"{item} should have been filtered out"
    assert set(result.item2id) == set(sample_data.EXPECTED_FINAL_ITEM_IDS)
    assert result.statistics["after_filtering"]["items"] == sample_data.EXPECTED_FINAL_ITEMS


def test_item_filtering_is_configurable() -> None:
    """Lowering both thresholds must retain more entities, never fewer."""
    rows = [
        interaction("u1", "i1", 10),
        interaction("u2", "i2", 20),
        interaction("u3", "i3", 30),
    ]
    strict = iterative_k_core_filter(rows, min_user_interactions=2, min_item_interactions=2)
    relaxed = iterative_k_core_filter(rows, min_user_interactions=1, min_item_interactions=1)
    assert strict.users == set()
    assert len(relaxed.users) == 3 and len(relaxed.items) == 3


# --------------------------------------------------------------------------- #
# 4. Iterative k-core
# --------------------------------------------------------------------------- #


def test_kcore_requires_a_second_iteration(tmp_path: Path) -> None:
    """Removing items must be able to push a user below threshold.

    ``u6`` starts with exactly 5 interactions, so a naive "filter users once,
    then filter items once" implementation would keep it - and would therefore
    also keep ``i6``/``i7`` alive.  All five of its interactions are on those two
    sparse items, which die in the item pass, dropping ``u6`` to 3; only a
    *second* user pass catches that.
    """
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    assert "u6" not in result.sequences
    assert "u6" not in result.user2id
    assert "i6" not in result.item2id and "i7" not in result.item2id

    # The history must show the cascade rather than a single pass.
    assert result.filtering.history == sample_data.EXPECTED_K_CORE_HISTORY, result.filtering.history
    assert result.filtering.rounds == sample_data.EXPECTED_K_CORE_ROUNDS
    assert result.filtering.converged

    # Round 1 removes the two users that are below threshold outright...
    assert result.filtering.history[0][1] == 2
    # ...and round 2 removes exactly the one user starved by the item pass.
    assert result.filtering.history[1][1] == 1
    assert result.filtering.history[1][2] == 0


def test_kcore_fixed_point_minimal_case() -> None:
    """Minimal hand-checkable k-core case, small enough to verify by eye.

    The core is five users and five items, fully dense: ``u1``..``u5`` each
    review ``i1``..``i5`` exactly once, so every user and every item sits at
    exactly 5 interactions - the threshold - and must all survive (a threshold
    is inclusive).

    On top of that sits a sparse user ``uX`` with only 3 interactions on its own
    three private items.  ``uX`` is below threshold so the user pass drops it,
    which empties ``ix1``..``ix3`` and removes them in the same round's item
    pass.  The test then asserts the fixed-point property explicitly: no
    surviving user or item is below either threshold.
    """
    rows = []
    timestamp = 0
    for user in ("u1", "u2", "u3", "u4", "u5"):
        for item in ("i1", "i2", "i3", "i4", "i5"):
            timestamp += 1
            rows.append(interaction(user, item, timestamp))
    # Sparse user, below min_user_interactions, with private items that can only
    # ever have 1 interaction each.
    for item in ("ix1", "ix2", "ix3"):
        timestamp += 1
        rows.append(interaction("uX", item, timestamp))

    result = iterative_k_core_filter(rows, min_user_interactions=5, min_item_interactions=5)

    assert result.users == {"u1", "u2", "u3", "u4", "u5"}
    assert "uX" not in result.users
    assert result.items == {"i1", "i2", "i3", "i4", "i5"}
    assert len(result.interactions) == 25
    assert result.converged

    # The fixed point must genuinely satisfy the k-core property: nothing left
    # behind may be below either threshold.
    remaining_users = Counter(row.user_id for row in result.interactions)
    remaining_items = Counter(row.item_id for row in result.interactions)
    assert all(count >= 5 for count in remaining_users.values()), remaining_users
    assert all(count >= 5 for count in remaining_items.values()), remaining_items


def test_kcore_output_is_a_true_fixed_point_on_cascade_data(tmp_path: Path) -> None:
    """On data that really cascades, a single pass is provably insufficient.

    The synthetic dataset's ``u6`` survives round 1's *user* pass at exactly 5
    interactions and is only removed after round 1's *item* pass starves it.
    Re-running the filter on the survivors must be a no-op.
    """
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    # u6 is present going into the loop at exactly the threshold...
    raw_counts = Counter(row.user_id for row in interactions)
    assert raw_counts["u6"] == 5
    # ...and gone from the output, which required the extra iteration.
    assert "u6" not in result.filtering.users
    assert result.filtering.rounds > 1

    # Feeding the survivors back in must change nothing.
    again = iterative_k_core_filter(result.filtering.interactions, 5, 5)
    assert again.users == result.filtering.users
    assert again.items == result.filtering.items
    assert again.history == [(1, 0, 0)], again.history


def test_kcore_is_idempotent(tmp_path: Path) -> None:
    """Re-running the filter on its own output must change nothing."""
    interactions, _ = load_sample(tmp_path)
    first = iterative_k_core_filter(interactions, 5, 5)
    second = iterative_k_core_filter(first.interactions, 5, 5)

    assert second.users == first.users
    assert second.items == first.items
    assert len(second.interactions) == len(first.interactions)
    assert second.history == [(1, 0, 0)], second.history


def test_kcore_rejects_invalid_thresholds() -> None:
    """A threshold below 1 has no meaning and must be rejected loudly."""
    for user_min, item_min in ((0, 5), (5, 0), (-1, 5)):
        try:
            iterative_k_core_filter([], min_user_interactions=user_min, min_item_interactions=item_min)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for thresholds {(user_min, item_min)}")


# --------------------------------------------------------------------------- #
# 5. Deterministic ID mapping
# --------------------------------------------------------------------------- #


def test_mappings_are_deterministic_across_runs(tmp_path: Path) -> None:
    """Two independent runs over the same input produce identical mappings."""
    first_interactions, _ = load_sample(tmp_path)
    second_interactions, _ = load_sample(tmp_path)

    first = preprocess_interactions(first_interactions)
    second = preprocess_interactions(second_interactions)

    assert first.user2id == second.user2id
    assert first.item2id == second.item2id
    assert first.user2id == build_id_mapping(sample_data.EXPECTED_FINAL_SEQUENCES)
    assert first.item2id == build_id_mapping(first.filtering.items)


def test_mapping_is_independent_of_input_order() -> None:
    """Shuffling the input must not change integer ids."""
    rows = [
        interaction("uB", "iZ", 1),
        interaction("uA", "iY", 2),
        interaction("uA", "iX", 3),
    ]
    forward = build_id_mapping([row.user_id for row in rows])
    backward = build_id_mapping([row.user_id for row in reversed(rows)])
    assert forward == backward == {"uA": 1, "uB": 2}

    item_forward = build_id_mapping([row.item_id for row in rows])
    item_backward = build_id_mapping([row.item_id for row in reversed(rows)])
    assert item_forward == item_backward == {"iX": 1, "iY": 2, "iZ": 3}


def test_mappings_are_contiguous_and_start_at_one(tmp_path: Path) -> None:
    """Real ids must be exactly 1..N with no gaps."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    assert sorted(result.item2id.values()) == list(
        range(config.FIRST_REAL_ID, config.FIRST_REAL_ID + len(result.item2id))
    )
    assert sorted(result.user2id.values()) == list(
        range(config.FIRST_REAL_ID, config.FIRST_REAL_ID + len(result.user2id))
    )


# --------------------------------------------------------------------------- #
# 6. Padding invariant
# --------------------------------------------------------------------------- #


def test_padding_id_is_never_assigned(tmp_path: Path) -> None:
    """``0`` is reserved for padding in both tables."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    assert config.PAD_ID == 0
    assert config.PAD_ID not in set(result.item2id.values())
    assert config.PAD_ID not in set(result.user2id.values())
    assert min(result.item2id.values()) == config.FIRST_REAL_ID
    assert min(result.user2id.values()) == config.FIRST_REAL_ID

    # And in the persisted document too, including the inverse arrays.
    mappings = json.loads(json.dumps(_mappings_payload(result)))
    assert 0 not in mappings["item2id"].values()
    assert mappings["id2item"][0] is None
    assert mappings["id2user"][0] is None


# --------------------------------------------------------------------------- #
# 7. Output correctness
# --------------------------------------------------------------------------- #


def test_generated_sequences_only_contain_valid_item_ids(tmp_path: Path) -> None:
    """Every emitted id must exist in ``item2id`` and be >= 1."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)

    valid_ids = set(result.item2id.values())
    for user, rows in result.sequences.items():
        for row in rows:
            assert row.item_id in result.item2id, f"{user}: {row.item_id} unmapped"
            assert result.item2id[row.item_id] in valid_ids
            assert result.item2id[row.item_id] >= config.FIRST_REAL_ID

    # Sequence lengths must respect both thresholds.
    for user, rows in result.sequences.items():
        assert len(rows) >= 5, f"{user} below min_user_interactions"
    item_counts: dict[str, int] = {}
    for rows in result.sequences.values():
        for row in rows:
            item_counts[row.item_id] = item_counts.get(row.item_id, 0) + 1
    assert all(count >= 5 for count in item_counts.values()), item_counts


def test_persisted_artifacts_are_self_consistent(tmp_path: Path) -> None:
    """The three written artifacts must agree with each other."""
    raw_path = write_sample(tmp_path)
    out_dir = Path(tmp_path) / "processed"
    result = run_preprocessing(
        raw_path,
        category="Synthetic",
        processed_dir=out_dir,
        verbose=False,
    )

    sequences = read_json(result.output_paths["sequences"])
    mappings = read_json(result.output_paths["mappings"])
    metadata = read_json(result.output_paths["metadata"])

    assert sequences["num_users"] == len(mappings["user2id"])
    assert sequences["num_items"] == len(mappings["item2id"])
    assert sequences["num_interactions"] == metadata["statistics"]["after_filtering"]["interactions"]

    for record in sequences["sequences"]:
        assert record["length"] == len(record["item_ids"])
        assert record["length"] == len(record["unix_ms"]) == len(record["ratings"])
        assert record["unix_ms"] == sorted(record["unix_ms"])
        assert record["user_int_id"] == mappings["user2id"][record["user_id"]]
        assert all(item_id >= 1 for item_id in record["item_ids"])
        for int_id, parent_asin in zip(record["item_ids"], record["parent_asins"]):
            assert mappings["id2item"][int_id] == parent_asin

    assert metadata["statistics"]["after_filtering"]["users"] == sample_data.EXPECTED_FINAL_USERS
    assert metadata["statistics"]["after_filtering"]["items"] == sample_data.EXPECTED_FINAL_ITEMS
    assert metadata["k_core"]["converged"] is True


def test_statistics_report_required_fields(tmp_path: Path) -> None:
    """The statistics block must carry every required before/after field."""
    interactions, _ = load_sample(tmp_path)
    result = preprocess_interactions(interactions)
    stats = result.statistics

    assert stats["before_filtering"]["interactions"] == len(interactions)
    assert stats["before_filtering"]["users"] == sample_data.RAW_USER_COUNT
    assert stats["before_filtering"]["items"] == sample_data.RAW_ITEM_COUNT
    assert stats["after_filtering"]["interactions"] == sample_data.EXPECTED_FINAL_INTERACTIONS
    assert stats["after_filtering"]["users"] == sample_data.EXPECTED_FINAL_USERS
    assert stats["after_filtering"]["items"] == sample_data.EXPECTED_FINAL_ITEMS
    assert stats["after_filtering"]["average_sequence_length"] == round(
        sample_data.EXPECTED_FINAL_INTERACTIONS / sample_data.EXPECTED_FINAL_USERS, 4
    )
    assert stats["after_filtering"]["min_sequence_length"] == 5
    assert stats["after_filtering"]["max_sequence_length"] == 5
    assert stats["after_filtering"]["median_sequence_length"] == 5.0
    assert stats["after_filtering"]["sequence_length_distribution"]["count"] == 5
    assert 0.0 <= stats["after_filtering"]["sparsity"]["sparsity"] <= 1.0


def test_compute_statistics_on_empty_input() -> None:
    """Degenerate input must not raise."""
    stats = compute_statistics({}, [])
    assert stats["after_filtering"]["users"] == 0
    assert stats["after_filtering"]["min_sequence_length"] is None
    assert stats["after_filtering"]["average_sequence_length"] == 0.0


# --------------------------------------------------------------------------- #
# 8. Input validation / safety
# --------------------------------------------------------------------------- #


def test_malformed_records_are_counted_not_crashed(tmp_path: Path) -> None:
    """A parse error, a missing field and a bad rating must all be reported."""
    interactions, report = load_sample(tmp_path)

    assert report.parse_errors == 1
    assert report.rejected["missing_fields"] == 1
    assert report.rejected["bad_rating"] == 1
    assert report.accepted == sample_data.EXPECTED_ACCEPTED_RECORDS
    assert len(interactions) == sample_data.EXPECTED_ACCEPTED_RECORDS
    assert report.total_records == sample_data.RAW_RECORD_COUNT + len(sample_data.SPECIAL_LINES)

    reasons = report.as_dict()["rejected_by_reason"]
    assert reasons["missing_fields"] == 1 and reasons["bad_rating"] == 1


def test_duplicate_policy_keeps_the_configured_occurrence(tmp_path: Path) -> None:
    """The duplicated triples must resolve according to the configured policy."""
    interactions, report = load_sample(tmp_path)
    assert report.duplicate_records_dropped == sample_data.EXPECTED_DUPLICATES_DROPPED

    def find(rows, user: str, item: str):
        return [row for row in rows if row.user_id == user and row.item_id == item]

    # u1 x i3: rows are 5.0 then 3.0 -> "last" keeps 3.0.
    u1_i3 = find(interactions, "u1", "i3")
    assert len(u1_i3) == 1
    assert u1_i3[0].rating == 3.0, "policy 'last' must keep the later rating"

    # u2 x i1: rows are 4.0 then 3.0 -> "last" keeps 3.0.
    u2_i1 = find(interactions, "u2", "i1")
    assert len(u2_i1) == 1
    assert u2_i1[0].rating == 3.0

    first_path = Path(tmp_path) / "first.jsonl"
    sample_data.write_sample(first_path)
    first_rows, first_report = load_interactions(first_path, deduplicate="first")
    assert first_report.duplicate_records_dropped == sample_data.EXPECTED_DUPLICATES_DROPPED
    assert find(first_rows, "u1", "i3")[0].rating == 5.0, "policy 'first' keeps the first row"
    assert find(first_rows, "u2", "i1")[0].rating == 4.0

    keep_rows, keep_report = load_interactions(first_path, deduplicate="keep")
    assert keep_report.duplicate_records_dropped == 0
    for user, item in sample_data.DUPLICATE_PAIRS:
        assert len(find(keep_rows, user, item)) == 2, f"{user}/{item} duplicate lost"


def test_invalid_deduplicate_policy_is_rejected(tmp_path: Path) -> None:
    """Unknown policies must fail loudly rather than silently degrade."""
    path = write_sample(tmp_path, include_special_lines=False)
    try:
        load_interactions(path, deduplicate="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown deduplicate policy")


def test_raw_input_is_not_modified(tmp_path: Path) -> None:
    """Preprocessing must leave the raw file byte-identical."""
    raw_path = write_sample(tmp_path)
    before = file_fingerprint(raw_path)
    before_bytes = raw_path.read_bytes()

    run_preprocessing(
        raw_path,
        category="Synthetic",
        processed_dir=Path(tmp_path) / "processed",
        verbose=False,
    )

    after = file_fingerprint(raw_path)
    assert before == after, f"raw file changed: {before} -> {after}"
    assert raw_path.read_bytes() == before_bytes


def test_load_interactions_does_not_mutate_records(tmp_path: Path) -> None:
    """Normalisation must copy, never edit in place."""
    original = {"user_id": "u", "parent_asin": " i1 ", "timestamp": "123", "rating": 4}
    snapshot = dict(original)
    interactions, _ = load_interactions(_write_lines(tmp_path, [original]))

    assert original == snapshot
    assert interactions[0].item_id == "i1"  # whitespace stripped in the copy only


def test_missing_input_file_raises() -> None:
    """A bad path must raise FileNotFoundError."""
    try:
        load_interactions("/nonexistent/definitely/not/here.jsonl")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_group_by_user_preserves_order() -> None:
    """Grouping must not reorder rows within a user."""
    rows = [interaction("u1", "i1", 1), interaction("u2", "i2", 2), interaction("u1", "i3", 3)]
    grouped = group_by_user(rows)
    assert [row.item_id for row in grouped["u1"]] == ["i1", "i3"]
    assert [row.item_id for row in grouped["u2"]] == ["i2"]


# --------------------------------------------------------------------------- #
# Internal helpers used by the tests above
# --------------------------------------------------------------------------- #


def _mappings_payload(result):
    """Import indirection so the padding test can inspect the persisted shape."""
    from recommendation.preprocess import build_mappings_document

    return build_mappings_document(result)


def _write_lines(tmp_path: Path, records: list[dict]) -> Path:
    """Write literal records (objects or raw strings) as a JSONL file."""
    path = Path(tmp_path) / "literal.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record if isinstance(record, str) else json.dumps(record))
            handle.write("\n")
    return path


# --------------------------------------------------------------------------- #
# Standalone runner (used when pytest is unavailable)
# --------------------------------------------------------------------------- #


def _collect_tests() -> list:
    """Return every ``test_*`` function defined in this module, in file order."""
    return [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]


def main() -> int:
    """Run all tests without pytest.  Returns a process exit code."""
    tests = _collect_tests()
    passed = 0
    failures: list[tuple[str, str]] = []

    for test in tests:
        name = test.__name__
        try:
            parameters = test.__code__.co_varnames[: test.__code__.co_argcount]
            if parameters == ("tmp_path",):
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
        except Exception:  # noqa: BLE001 - test runner must report everything
            failures.append((name, traceback.format_exc()))
            print(f"FAIL {name}")
        else:
            passed += 1
            print(f"PASS {name}")

    print()
    print(f"{passed} passed, {len(failures)} failed, {len(tests)} total")
    for name, detail in failures:
        print()
        print(f"===== {name} =====")
        print(detail)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
