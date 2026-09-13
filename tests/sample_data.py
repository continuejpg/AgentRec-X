"""Deterministic tiny synthetic dataset for Milestone 1.

The dataset is hand-authored so every requested behaviour is exercised and can
be checked by eye.  Timestamps are written as unix *seconds* for readability and
multiplied by 1000 on output, matching the millisecond timestamps of the real
Amazon Reviews 2023 release.

Shape of the dataset
--------------------
Five "core" users (``u1``..``u5``) each interact with the same five items
(``i1``..``i5``).  Because every core user interacts with every core item, each
core item ends up with exactly 5 interactions and each core user with 5 distinct
interactions **after duplicate collapsing** - exactly at the default threshold
of 5, so both sides survive.  A separate sparse region (users ``u6``..``u8``,
items ``i6``..``i8``) touches no core item, so pruning it cannot accidentally
drag the core below threshold.

Deliberate properties
---------------------
1. **Unsorted timestamps** - rows are emitted shuffled and each user's history
   is internally out of order.
2. **Users below the minimum frequency** - ``u7`` (1 interaction) and ``u8``
   (3 interactions).
3. **Items below the minimum frequency** - ``i6``, ``i7`` and ``i8`` each end up
   with 3 interactions, all below the threshold of 5.
4. **Iterative k-core behaviour** - ``u6`` has exactly 5 interactions, so the
   first user pass keeps it.  Every one of those 5 is on ``i6``/``i7``, which the
   item pass deletes, dropping ``u6`` to 3.  Only a *second* user pass catches
   that; a single filter-users-then-filter-items implementation wrongly keeps
   ``u6`` (and so would keep ``i6``/``i7`` alive).
5. **Multiple users and items** - 8 users and 8 distinct items before filtering;
   5 users and 5 items survive.
6. **Deterministic mappings** - integer ids are assigned from sorted surviving
   string ids, so repeated runs (and any input order) give identical mappings.
7. **Padding reservation** - the 5 surviving items map to 1..5; id 0 stays
   unused because ids start at :data:`recommendation.config.FIRST_REAL_ID`.
8. **Valid chronological output** - each user's expected order is stated in
   :data:`EXPECTED_SEQUENCES` and asserted by the tests.
9. **Malformed input** - one record missing ``parent_asin``, one record with a
   non-numeric rating, and one unparseable JSON line, so the loader must report
   them instead of crashing.
10. **Duplicate interactions** - two ``(user, item, timestamp)`` triples appear
    twice with different ratings, pinning the deduplication policy.  The default
    ``"last"`` policy keeps the later row.

K-core trace with the defaults (``min_user_interactions = 5``,
``min_item_interactions = 5``)::

    raw: 8 users, 8 items, 35 rows -> 33 distinct interactions (2 duplicates)
    item frequencies: i1=6 i2=5 i3=5 i4=6 i5=5 | i6=3 i7=1 i8=2
    round 1: user pass removes u7 (1 interaction), u8 (2)
             item pass removes i6, i7, i8 (all below threshold)
             (that drops u6 from 5 interactions to 2)
    round 2: user pass removes u6 (2)
             item pass removes nothing -> fixed point reached
    round 3: nothing removed -> converged
    final: 5 users, 5 items, 25 interactions

The cascade is visible in the round count: ``u6`` survives round 1's user pass
at exactly 5 interactions and is only removed in round 2, after round 1's *item*
pass starved it.  A single "filter users, then filter items" pass would leave
``u6`` in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Canonical dataset definition
# --------------------------------------------------------------------------- #

#: ``user -> [(item, timestamp_seconds, rating), ...]``, intentionally unordered.
#:
#: Rows whose ``(user, item, timestamp)`` triple repeats are intentional; see
#: :data:`DUPLICATE_PAIRS`.  For each such triple the first row listed here is
#: overwritten by the second under the default ``"last"`` dedup policy.
USER_INTERACTIONS: dict[str, list[tuple[str, int, float]]] = {
    # ---- core: five users, the same five items, 5 interactions each --------
    "u1": [
        ("i1", 1700000100, 5.0),
        ("i2", 1700000200, 5.0),
        ("i3", 1700000300, 5.0),
        ("i3", 1700000300, 3.0),  # duplicate triple -> rating becomes 3.0
        ("i4", 1700000400, 5.0),
        ("i5", 1700000500, 5.0),
    ],
    "u2": [
        ("i1", 1700000110, 4.0),
        ("i1", 1700000110, 3.0),  # duplicate triple -> rating becomes 3.0
        ("i2", 1700000210, 4.0),
        ("i3", 1700000310, 4.0),
        ("i4", 1700000410, 4.0),
        ("i5", 1700000510, 4.0),
    ],
    "u3": [
        ("i1", 1700000120, 5.0),
        ("i2", 1700000220, 5.0),
        ("i3", 1700000320, 5.0),
        ("i4", 1700000420, 5.0),
        ("i5", 1700000520, 5.0),
    ],
    "u4": [
        ("i1", 1700000130, 4.0),
        ("i2", 1700000230, 4.0),
        ("i3", 1700000330, 4.0),
        ("i4", 1700000430, 4.0),
        ("i5", 1700000530, 4.0),
    ],
    "u5": [
        ("i1", 1700000140, 3.0),
        ("i2", 1700000240, 3.0),
        ("i3", 1700000340, 3.0),
        ("i4", 1700000440, 3.0),
        ("i5", 1700000540, 3.0),
    ],
    # ---- the sparse region, pruned by the cascade --------------------------
    # u6 has exactly 5 interactions, so the first user pass keeps it.  But it
    # is anchored to i1/i4 (which survive) *and* to i6/i7/i8 (which do not).
    # Losing the three dying items drops u6 to 2 interactions, which only a
    # second user pass can catch.
    "u6": [
        ("i1", 1700000150, 4.0),
        ("i4", 1700000450, 4.0),
        ("i8", 1700000830, 4.0),
        ("i6", 1700000600, 4.0),
        ("i7", 1700000700, 4.0),
    ],
    "u7": [  # 1 interaction: below min_user_interactions
        ("i6", 1700000610, 5.0),
    ],
    "u8": [  # 2 interactions: below min_user_interactions
        ("i6", 1700000620, 5.0),
        ("i8", 1700000810, 4.0),
    ],
}

#: ``(user, item)`` pairs that appear twice at the same timestamp.  Both rows are
#: listed adjacently in :data:`USER_INTERACTIONS`; the second wins under the
#: default ``"last"`` dedup policy.
DUPLICATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("u1", "i3"),
    ("u2", "i1"),
)

#: Extra rows appended verbatim to the file to exercise malformed-input paths.
SPECIAL_LINES: list[str] = [
    # Not valid JSON at all.
    '{"user_id": "u1", "parent_asin": "i1", "timestamp": 1700000100, "rating":',
    # Valid JSON, missing the required parent_asin field.
    '{"user_id": "u_missing", "timestamp": 1700001700, "rating": 5.0}',
    # Valid JSON, rating is not numeric.
    '{"user_id": "u_bad", "parent_asin": "i1", "timestamp": 1700001800, "rating": "not-a-number"}',
]

#: Counters for the canonical dataset (before the special malformed lines).
RAW_USER_COUNT = len(USER_INTERACTIONS)
RAW_ITEM_COUNT = len({item for rows in USER_INTERACTIONS.values() for item, _, _ in rows})
#: Rows declared in :data:`USER_INTERACTIONS` (already including the repeats).
DECLARED_ROWS = sum(len(rows) for rows in USER_INTERACTIONS.values())
#: Rows actually written for the canonical dataset (one per declared row).
RAW_RECORD_COUNT = DECLARED_ROWS
#: Distinct interactions after duplicate collapsing.
RAW_DISTINCT_INTERACTIONS = RAW_RECORD_COUNT - len(DUPLICATE_PAIRS)


def _verify_duplicate_declarations() -> None:
    """Fail loudly at import time if :data:`DUPLICATE_PAIRS` drifts from the data.

    Every declared duplicate pair must appear exactly twice for the same
    ``(user, item, timestamp)`` triple - once as the retained row and once as the
    repeat - otherwise ``EXPECTED_DUPLICATES_DROPPED`` would be a lie.
    """
    seen: Counter = Counter()
    for user_id, rows in USER_INTERACTIONS.items():
        for item_id, ts_seconds, _ in rows:
            seen[(user_id, item_id, ts_seconds)] += 1

    declared = {(user, item) for user, item in DUPLICATE_PAIRS}
    actual = {
        (user, item)
        for (user, item, _ts), count in seen.items()
        if count > 1
    }
    if declared != actual:
        raise AssertionError(
            "DUPLICATE_PAIRS does not match USER_INTERACTIONS: "
            f"declared={sorted(declared)} actual={sorted(actual)}"
        )
    for (user, item, _ts), count in seen.items():
        if count > 1 and count != 2:
            raise AssertionError(f"{user}/{item} appears {count} times, expected 2")


_verify_duplicate_declarations()


def build_records() -> list[dict[str, Any]]:
    """Build the raw JSON records (full Amazon Reviews 2023 shape).

    Records carry the fields the pipeline ignores (``title``, ``text``,
    ``images``, ``asin``, ``helpful_vote``, ``verified_purchase``) so the loader
    is exercised against the real schema rather than a stripped one.

    Rows are *not* emitted in timestamp order - :data:`USER_INTERACTIONS` is
    deliberately unordered per user, which is enough to force the pipeline to
    sort.  Records are emitted in that fixed order rather than randomly
    shuffled, because the dedup policy is defined over file order (``"last"``
    keeps the last occurrence): a random shuffle would make which duplicate wins
    non-deterministic.  For every pair in :data:`DUPLICATE_PAIRS` the first row
    listed is the one replaced by the second.
    """
    records: list[dict[str, Any]] = []
    index = 0
    for user_id, rows in USER_INTERACTIONS.items():
        for item_id, ts_seconds, rating in rows:
            index += 1
            records.append(
                {
                    "rating": rating,
                    "title": f"synthetic review {index}",
                    "text": f"synthetic review body {index} for {user_id}/{item_id}",
                    "images": [],
                    "asin": f"A{index:06d}",
                    "parent_asin": item_id,
                    "user_id": user_id,
                    "timestamp": ts_seconds * 1000,
                    "helpful_vote": index % 3,
                    "verified_purchase": index % 2 == 0,
                }
            )
    return records


#: Expected surviving item sequence per user, oldest first, derived by hand from
#: :data:`USER_INTERACTIONS` with the duplicate triples collapsed.  Ties on
#: timestamp are broken by ``(parent_asin, rating)`` - see
#: :func:`recommendation.preprocess.sort_interactions`.
EXPECTED_SEQUENCES: dict[str, list[str]] = {
    "u1": ["i1", "i2", "i3", "i4", "i5"],
    "u2": ["i1", "i2", "i3", "i4", "i5"],
    "u3": ["i1", "i2", "i3", "i4", "i5"],
    "u4": ["i1", "i2", "i3", "i4", "i5"],
    "u5": ["i1", "i2", "i3", "i4", "i5"],
    "u6": [],  # removed: only interacts with items the cascade deletes
    "u7": [],  # removed: below min_user_interactions
    "u8": [],  # removed: below min_user_interactions
}

#: Users the k-core fixed point must remove with the default thresholds.
EXPECTED_REMOVED_USERS = ("u6", "u7", "u8")
#: Items the k-core fixed point must remove with the default thresholds.
EXPECTED_REMOVED_ITEMS = ("i6", "i7", "i8")
#: Users and items that must survive.
EXPECTED_FINAL_USER_IDS = ("u1", "u2", "u3", "u4", "u5")
EXPECTED_FINAL_ITEM_IDS = ("i1", "i2", "i3", "i4", "i5")

#: Expectations for the default thresholds.
EXPECTED_FINAL_SEQUENCES = {
    user: items for user, items in EXPECTED_SEQUENCES.items() if items
}
EXPECTED_FINAL_USERS = len(EXPECTED_FINAL_SEQUENCES)
EXPECTED_FINAL_ITEMS = len(EXPECTED_FINAL_ITEM_IDS)
EXPECTED_FINAL_INTERACTIONS = sum(len(row) for row in EXPECTED_FINAL_SEQUENCES.values())

#: Raw (pre-k-core, post-dedup) counts.
EXPECTED_ACCEPTED_RECORDS = RAW_DISTINCT_INTERACTIONS
EXPECTED_DUPLICATES_DROPPED = len(DUPLICATE_PAIRS)

#: Round-by-round k-core history expected for the default thresholds:
#: ``(round, users_removed, items_removed)``.
#:
#: The sparse region is built so the counts are unambiguous: ``i6`` has 3
#: interactions (u6, u7, u8), ``i7`` has 3 (all u6), ``i8`` has 2 (u6, u8).
#: Round 1's user pass drops ``u7`` (1) and ``u8`` (2); its item pass then drops
#: ``i6``, ``i7`` and ``i8``, which starves ``u6`` (5 -> 1 interaction).  Round 2
#: removes ``u6``; round 3 confirms the fixed point.
#:
#: Note the ordering inside a round: the user pass runs first, so a user whose
#: items are removed by that same round's item pass is only caught in the *next*
#: round.  That is exactly why the loop must iterate instead of running one user
#: pass and one item pass.
EXPECTED_K_CORE_HISTORY = [(1, 2, 3), (2, 1, 0), (3, 0, 0)]
#: Number of passes performed, including the final pass that removes nothing.
EXPECTED_K_CORE_ROUNDS = 3


def write_sample(path: str | Path, include_special_lines: bool = True) -> Path:
    """Write the synthetic dataset to ``path`` as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [json.dumps(record, ensure_ascii=False) for record in build_records()]
    if include_special_lines:
        lines.extend(SPECIAL_LINES)

    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating the synthetic dataset."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.sample_data",
        description="Generate the tiny synthetic Amazon Reviews 2023 style dataset.",
    )
    parser.add_argument(
        "--out",
        default="data/raw/synthetic_reviews.jsonl",
        help="output path (default: data/raw/synthetic_reviews.jsonl)",
    )
    parser.add_argument(
        "--no-special-lines",
        action="store_true",
        help="omit the deliberately malformed lines",
    )
    args = parser.parse_args(argv)

    path = write_sample(args.out, include_special_lines=not args.no_special_lines)
    print(f"wrote {path}")
    print(f"  well-formed rows (duplicates included) : {RAW_RECORD_COUNT}")
    print(f"  distinct interactions after dedup      : {RAW_DISTINCT_INTERACTIONS}")
    if not args.no_special_lines:
        print(f"  malformed/extra lines                  : {len(SPECIAL_LINES)}")
    print(f"  distinct users                         : {RAW_USER_COUNT}")
    print(f"  distinct items                         : {RAW_ITEM_COUNT}")
    print(f"  expected after k-core (min=5)          : "
          f"{EXPECTED_FINAL_USERS} users, {EXPECTED_FINAL_ITEMS} items, "
          f"{EXPECTED_FINAL_INTERACTIONS} interactions")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
