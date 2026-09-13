"""Core preprocessing for AgentRec-X Milestone 1.

Converts validated Amazon Reviews 2023 interactions into chronological,
k-core-filtered, deterministically indexed user sequences.

Pipeline stages
---------------
1. **Load & validate** (:mod:`recommendation.io_utils`) - raw JSONL records are
   validated and normalised; malformed rows are counted and skipped, never
   silently coerced, and the raw objects are never mutated.
2. **Chronological ordering** - within every user, interactions are sorted
   ascending by timestamp (ties broken by the raw item id, then rating, so the
   order is a total order and therefore reproducible).  Sequences always read
   past -> future, which is what keeps leave-one-out evaluation free of temporal
   leakage.
3. **Iterative k-core filtering** - users and items below the configured
   minimum interaction counts are removed repeatedly until the surviving sets
   stop changing.  This is a true fixed point: removing items can push a user
   below ``min_user_interactions``, which can in turn orphan further items.
4. **Deterministic ID mapping** - integer ids are assigned in ascending order of
   the raw (string) identifier, so the same input always yields the same
   mapping.  Item (and user) id ``0`` is reserved for padding.
5. **Persistence & statistics** - sequences, mappings and metadata are written
   as JSON (see ``recommendation/README.md`` for the documented format).

Everything here runs on CPU with the standard library plus numpy-free Python.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from recommendation import config
from recommendation.io_utils import (
    Interaction,
    NormalizationReport,
    load_interactions,
    write_json,
)

# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #


def sort_interactions(interactions: Iterable[Interaction]) -> list[Interaction]:
    """Return ``interactions`` sorted chronologically.

    The sort key is ``(timestamp, item_id, rating)``.  ``timestamp`` alone can
    tie when a user reviews several products in the same second (the Amazon
    Reviews 2023 release stores second-or-finer precision), so the remaining
    fields act as deterministic tie-breakers rather than leaving the order up to
    the sort algorithm.
    """
    return sorted(interactions, key=lambda row: (row.timestamp, row.item_id, row.rating))


def group_by_user(interactions: Iterable[Interaction]) -> dict[str, list[Interaction]]:
    """Group interactions per user, preserving input order within each user."""
    grouped: dict[str, list[Interaction]] = {}
    for interaction in interactions:
        grouped.setdefault(interaction.user_id, []).append(interaction)
    return grouped


def build_id_mapping(keys: Iterable[str]) -> dict[str, int]:
    """Build a deterministic ``raw_id -> integer_id`` mapping.

    Integer ids start at :data:`config.FIRST_REAL_ID` (1); id
    :data:`config.PAD_ID` (0) is deliberately left unused so downstream
    embedding tables can treat index 0 as PAD.  Keys are sorted
    lexicographically, which makes the mapping independent of input order,
    dict iteration order and hash seed.
    """
    ordered_keys = sorted(set(keys))
    return {
        key: index
        for index, key in enumerate(ordered_keys, start=config.FIRST_REAL_ID)
    }


# --------------------------------------------------------------------------- #
# Iterative k-core filtering
# --------------------------------------------------------------------------- #


@dataclass
class FilterResult:
    """Outcome of :func:`iterative_k_core_filter`."""

    #: Surviving interactions, in their original (unsorted) order.
    interactions: list[Interaction]
    #: Raw user ids that survived.
    users: set[str]
    #: Raw item ids that survived.
    items: set[str]
    #: Number of filtering passes performed (1 means already stable).
    rounds: int
    #: ``(round_index, removed_users, removed_items)`` for each pass.
    history: list[tuple[int, int, int]]

    @property
    def converged(self) -> bool:
        """True when no entities were removed in the final pass."""
        return bool(self.history) and self.history[-1][1] == 0 and self.history[-1][2] == 0


def iterative_k_core_filter(
    interactions: Sequence[Interaction],
    min_user_interactions: int = config.MIN_USER_INTERACTIONS,
    min_item_interactions: int = config.MIN_ITEM_INTERACTIONS,
    max_rounds: int = config.MAX_K_CORE_ROUNDS,
) -> FilterResult:
    """Remove low-frequency users and items until the graph reaches a fixed point.

    A single "filter users once, then items once" pass is *not* sufficient: an
    item removed in the item step can drop a user below
    ``min_user_interactions``, and that user's removal can drop further items
    below ``min_item_interactions``.  This function therefore alternates
    user-filtering and item-filtering passes until a full pass changes nothing.

    Counting is incremental - item removals decrement the affected users'
    counters - so each pass costs ``O(#interactions)`` and convergence typically
    takes a couple of passes.

    Parameters
    ----------
    interactions:
        Validated interactions (see :class:`~recommendation.io_utils.Interaction`).
    min_user_interactions, min_item_interactions:
        Inclusive minimum counts; a user/item with *exactly* the threshold
        survives.  Must be ``>= 1`` (a threshold of 1 keeps everything).
    max_rounds:
        Safety limit on the number of passes.

    Returns
    -------
    FilterResult
        Survivors plus a per-round history so the behaviour is auditable.
    """
    if min_user_interactions < 1:
        raise ValueError("min_user_interactions must be >= 1")
    if min_item_interactions < 1:
        raise ValueError("min_item_interactions must be >= 1")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    # (user, item) -> number of surviving interactions for that pair.  Multiple
    # reviews of the same item still count as multiple interactions, which is
    # the usual k-core convention for implicit-feedback benchmarks.
    pair_counts: Counter[tuple[str, str]] = Counter(
        (row.user_id, row.item_id) for row in interactions
    )

    # user -> {item: pair_count}; item -> {user: pair_count}
    user_items: dict[str, dict[str, int]] = {}
    item_users: dict[str, dict[str, int]] = {}
    for (user_id, item_id), count in pair_counts.items():
        user_items.setdefault(user_id, {})[item_id] = count
        item_users.setdefault(item_id, {})[user_id] = count

    # Frequency counters kept in sync with the adjacency structures above.
    user_counts: dict[str, int] = {u: sum(items.values()) for u, items in user_items.items()}
    item_counts: dict[str, int] = {i: sum(users.values()) for i, users in item_users.items()}

    history: list[tuple[int, int, int]] = []
    rounds = 0

    while rounds < max_rounds:
        rounds += 1

        # --- pass: drop users below the threshold ------------------------- #
        doomed_users = [u for u, count in user_counts.items() if count < min_user_interactions]
        for user_id in doomed_users:
            for item_id, count in user_items.pop(user_id).items():
                # Detach the user from the item *first*: if this empties the
                # item we would otherwise lose the adjacency entry while stale
                # per-user counts remained, which would corrupt later passes.
                item_users.get(item_id, {}).pop(user_id, None)
                item_counts[item_id] -= count
                if item_counts[item_id] <= 0:
                    del item_counts[item_id]
            del user_counts[user_id]

        # --- pass: drop items below the threshold ------------------------- #
        doomed_items = [i for i, count in item_counts.items() if count < min_item_interactions]
        for item_id in doomed_items:
            for user_id, count in item_users.pop(item_id).items():
                user_items.get(user_id, {}).pop(item_id, None)
                user_counts[user_id] -= count
                if user_counts[user_id] <= 0:
                    del user_counts[user_id]
            del item_counts[item_id]

        history.append((rounds, len(doomed_users), len(doomed_items)))

        if not doomed_users and not doomed_items:
            break  # fixed point reached
    else:
        raise RuntimeError(
            f"k-core filtering did not converge within {max_rounds} rounds; "
            "this indicates a bug because every non-empty round strictly shrinks the graph"
        )

    surviving_users = set(user_counts)
    surviving_items = set(item_counts)
    survivors = [
        row
        for row in interactions
        if row.user_id in surviving_users and row.item_id in surviving_items
    ]

    return FilterResult(
        interactions=survivors,
        users=surviving_users,
        items=surviving_items,
        rounds=rounds,
        history=history,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _sequence_length_stats(lengths: Sequence[int]) -> dict[str, Any]:
    """Summarise a list of sequence lengths."""
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(lengths)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 4),
        "median": float(median(ordered)),
    }


def _sparsity(num_interactions: int, num_users: int, num_items: int) -> dict[str, float]:
    """Return density/sparsity of the user-item matrix."""
    possible = num_users * num_items
    if possible == 0:
        return {"observable_cells": 0, "density": 0.0, "sparsity": 1.0}
    density = num_interactions / possible
    return {
        "observable_cells": possible,
        "density": density,
        "sparsity": 1.0 - density,
    }


def compute_statistics(
    grouped_sequences: dict[str, list[Interaction]],
    original_interactions: Sequence[Interaction],
) -> dict[str, Any]:
    """Compute before/after-filtering statistics for a pipeline run.

    ``grouped_sequences`` must already be filtered *and* sorted, so the sequence
    lengths it reports describe exactly what was written to disk.
    """
    lengths = [len(rows) for rows in grouped_sequences.values()]

    remaining = sum(lengths)
    remaining_users = len(grouped_sequences)
    remaining_items = len(
        {row.item_id for rows in grouped_sequences.values() for row in rows}
    )

    # "Before filtering" = every validated interaction, i.e. the input to the
    # k-core stage.  Duplicate (user, item, timestamp) rows have already been
    # removed by the loader and counted separately in the normalization report.
    before_users = len({row.user_id for row in original_interactions})
    before_items = len({row.item_id for row in original_interactions})

    return {
        "before_filtering": {
            "interactions": len(original_interactions),
            "users": before_users,
            "items": before_items,
            "sparsity": _sparsity(len(original_interactions), before_users, before_items),
        },
        "after_filtering": {
            "interactions": remaining,
            "users": remaining_users,
            "items": remaining_items,
            "average_sequence_length": (
                round(remaining / remaining_users, 4) if remaining_users else 0.0
            ),
            "min_sequence_length": min(lengths) if lengths else None,
            "max_sequence_length": max(lengths) if lengths else None,
            "median_sequence_length": float(median(lengths)) if lengths else None,
            "sequence_length_distribution": _sequence_length_stats(lengths),
            "sparsity": _sparsity(remaining, remaining_users, remaining_items),
        },
        "retained_fraction": {
            "interactions": (
                round(remaining / len(original_interactions), 6)
                if original_interactions
                else 0.0
            ),
            "users": round(remaining_users / before_users, 6) if before_users else 0.0,
            "items": round(remaining_items / before_items, 6) if before_items else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class PreprocessResult:
    """Everything produced by :func:`run_preprocessing`."""

    category: str
    min_user_interactions: int
    min_item_interactions: int
    normalization: NormalizationReport
    filtering: FilterResult
    user2id: dict[str, int]
    item2id: dict[str, int]
    sequences: dict[str, list[Interaction]]
    statistics: dict[str, Any]
    output_paths: dict[str, Path]


def preprocess_interactions(
    interactions: Sequence[Interaction],
    category: str = config.DEFAULT_CATEGORY,
    min_user_interactions: int = config.MIN_USER_INTERACTIONS,
    min_item_interactions: int = config.MIN_ITEM_INTERACTIONS,
    normalization: NormalizationReport | None = None,
) -> PreprocessResult:
    """Run stages 2-5 (ordering, k-core, mapping, statistics) in memory.

    Kept separate from :func:`run_preprocessing` so tests can exercise the
    algorithm without touching the filesystem.
    """
    # Chronological ordering, per user.
    ordered = sort_interactions(interactions)
    grouped = group_by_user(ordered)

    # Iterative k-core filtering on the ordered data.  Order is preserved
    # through filtering, so the surviving sequences stay chronological.
    filtering = iterative_k_core_filter(
        [row for rows in grouped.values() for row in rows],
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    surviving_sequences: dict[str, list[Interaction]] = {}
    for user_id, rows in grouped.items():
        kept = [
            row
            for row in rows
            if row.user_id in filtering.users and row.item_id in filtering.items
        ]
        if kept:
            surviving_sequences[user_id] = kept

    statistics = compute_statistics(surviving_sequences, list(interactions))

    return PreprocessResult(
        category=category,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        normalization=normalization or NormalizationReport(),
        filtering=filtering,
        user2id=build_id_mapping(surviving_sequences),
        item2id=build_id_mapping(filtering.items),
        sequences=surviving_sequences,
        statistics=statistics,
        output_paths={},
    )


def build_mappings_document(result: PreprocessResult) -> dict[str, Any]:
    """Build the ``*_mappings.json`` payload.

    Both directions are stored: the forward maps are what models need, and the
    inverse arrays make it trivial to turn a predicted integer back into an
    Amazon ``parent_asin`` when inspecting recommendations.
    """
    user2id = result.user2id
    item2id = result.item2id
    id2user = [None] * (max(user2id.values(), default=0) + 1)
    id2item = [None] * (max(item2id.values(), default=0) + 1)
    for raw_id, int_id in user2id.items():
        id2user[int_id] = raw_id
    for raw_id, int_id in item2id.items():
        id2item[int_id] = raw_id

    return {
        "padding": {
            "pad_id": config.PAD_ID,
            "first_real_id": config.FIRST_REAL_ID,
            "note": (
                "Integer id 0 is reserved for padding in both tables; real ids "
                "start at 1, so 0 never appears in user2id/item2id values."
            ),
        },
        "num_users": len(user2id),
        "num_items": len(item2id),
        # JSON object keys must be strings.  Amazon ids are already strings for
        # users; parent_asin values are strings too, so no lossy conversion
        # happens here.
        "user2id": dict(sorted(user2id.items(), key=lambda kv: kv[1])),
        "item2id": dict(sorted(item2id.items(), key=lambda kv: kv[1])),
        "id2user": id2user,
        "id2item": id2item,
    }


def build_sequences_document(result: PreprocessResult) -> dict[str, Any]:
    """Build the ``*_sequences.json`` payload.

    One record per user, with parallel arrays aligned by position so that
    ``item_ids[k]`` happened at ``unix_ms[k]`` with ``ratings[k]``.
    """
    records = []
    for raw_user_id, rows in sorted(
        result.sequences.items(), key=lambda kv: result.user2id[kv[0]]
    ):
        records.append(
            {
                "user_id": raw_user_id,
                "user_int_id": result.user2id[raw_user_id],
                "item_ids": [result.item2id[row.item_id] for row in rows],
                "parent_asins": [row.item_id for row in rows],
                "unix_ms": [row.timestamp for row in rows],
                "ratings": [row.rating for row in rows],
                "length": len(rows),
            }
        )

    return {
        "format": "agentrecx.sequences.v1",
        "category": result.category,
        "description": (
            "One entry per user. Parallel arrays are aligned by index and sorted "
            "chronologically (oldest first). item_ids are 1-based integers into "
            "item2id (0 is padding and never occurs here)."
        ),
        "num_users": len(records),
        "num_items": len(result.item2id),
        "num_interactions": sum(record["length"] for record in records),
        "sequences": records,
    }


def build_metadata_document(
    result: PreprocessResult,
    raw_path: str | Path | None = None,
    raw_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``*_metadata.json`` payload (config + stats + provenance)."""
    return {
        "format": "agentrecx.preprocessing.v1",
        "milestone": "M1-amazon-preprocessing",
        "category": result.category,
        "schema": {
            "dataset": "Amazon Reviews 2023",
            "interaction_file": "review_categories/<Category>.jsonl.gz",
            "field_mapping": {
                "user_id": config.FIELD_USER,
                "item_id": config.FIELD_ITEM,
                "timestamp": config.FIELD_TIMESTAMP,
                "rating": config.FIELD_RATING,
            },
            "item_id_rationale": (
                "parent_asin groups colour/size/style variants of one product and "
                "matches the parent-level item ids used by earlier Amazon benchmarks."
            ),
            "unused_fields": ["title", "text", "images", "asin", "helpful_vote", "verified_purchase"],
        },
        "config": {
            "min_user_interactions": result.min_user_interactions,
            "min_item_interactions": result.min_item_interactions,
            "pad_id": config.PAD_ID,
            "first_real_id": config.FIRST_REAL_ID,
            "deduplicate_policy": config.DEDUPLICATE_POLICY,
            "max_k_core_rounds": config.MAX_K_CORE_ROUNDS,
            "ordering": "timestamp ascending per user; ties by (parent_asin, rating)",
        },
        "provenance": {
            "raw_input": str(raw_path) if raw_path is not None else None,
            "raw_input_fingerprint": raw_fingerprint,
            "python_version": sys.version.split()[0],
        },
        "normalization": result.normalization.as_dict(),
        "k_core": {
            "rounds": result.filtering.rounds,
            "converged": result.filtering.converged,
            "history": [
                {"round": index, "removed_users": users, "removed_items": items}
                for index, users, items in result.filtering.history
            ],
        },
        "statistics": result.statistics,
    }


def run_preprocessing(
    raw_path: str | Path,
    category: str = config.DEFAULT_CATEGORY,
    min_user_interactions: int = config.MIN_USER_INTERACTIONS,
    min_item_interactions: int = config.MIN_ITEM_INTERACTIONS,
    processed_dir: Path | None = None,
    deduplicate: str = config.DEDUPLICATE_POLICY,
    verbose: bool = True,
) -> PreprocessResult:
    """Run the full pipeline for one raw file and persist all artifacts."""
    from recommendation.io_utils import file_fingerprint

    raw_path = Path(raw_path)
    fingerprint_before = file_fingerprint(raw_path)

    interactions, normalization = load_interactions(raw_path, deduplicate=deduplicate)
    if verbose:
        print(f"[load] {raw_path}")
        print(f"[load] validated interactions: {len(interactions)}")
        report = normalization.as_dict()
        print(
            f"[load] raw records: {report['total_records']} | "
            f"accepted: {report['accepted_records']} | "
            f"parse errors: {report['parse_errors']} | "
            f"rejected: {report['rejected_total']} "
            f"({report['rejected_by_reason'] or 'none'}) | "
            f"duplicates dropped: {report['duplicate_records_dropped']}"
        )

    result = preprocess_interactions(
        interactions,
        category=category,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        normalization=normalization,
    )

    if verbose:
        print(
            f"[k-core] rounds: {result.filtering.rounds} | "
            f"history: {result.filtering.history} | "
            f"converged: {result.filtering.converged}"
        )

    output_paths = config.artifact_paths(category, processed_dir)
    write_json(output_paths["sequences"], build_sequences_document(result))
    write_json(output_paths["mappings"], build_mappings_document(result))
    write_json(
        output_paths["metadata"],
        build_metadata_document(result, raw_path=raw_path, raw_fingerprint=fingerprint_before),
    )

    fingerprint_after = file_fingerprint(raw_path)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError(
            f"raw input changed during preprocessing: {fingerprint_before} -> {fingerprint_after}"
        )

    result.output_paths = output_paths
    return result


def format_statistics(result: PreprocessResult) -> str:
    """Render :func:`compute_statistics` output as a readable report."""
    before = result.statistics["before_filtering"]
    after = result.statistics["after_filtering"]
    lines = [
        "Dataset statistics",
        "------------------",
        "Before filtering:",
        f"  interactions : {before['interactions']}",
        f"  users        : {before['users']}",
        f"  items        : {before['items']}",
        f"  sparsity     : {before['sparsity']['sparsity']:.6f}",
        "After filtering:",
        f"  interactions : {after['interactions']}",
        f"  users        : {after['users']}",
        f"  items        : {after['items']}",
        f"  avg seq len  : {after['average_sequence_length']}",
        f"  min seq len  : {after['min_sequence_length']}",
        f"  max seq len  : {after['max_sequence_length']}",
        f"  median seq   : {after['median_sequence_length']}",
        f"  sparsity     : {after['sparsity']['sparsity']:.6f}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line entry point
# --------------------------------------------------------------------------- #


def build_arg_parser() -> "argparse.ArgumentParser":
    """Create the CLI argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m recommendation.preprocess",
        description=(
            "Preprocess Amazon Reviews 2023 interactions into chronological, "
            "k-core-filtered sequential recommendation data."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="raw interaction file (.jsonl, .jsonl.gz or .parquet)",
    )
    parser.add_argument(
        "--category",
        "-c",
        default=config.DEFAULT_CATEGORY,
        help=f"category name used for artifact file names (default: {config.DEFAULT_CATEGORY})",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=config.MIN_USER_INTERACTIONS,
        help=f"k-core threshold for users (default: {config.MIN_USER_INTERACTIONS})",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=config.MIN_ITEM_INTERACTIONS,
        help=f"k-core threshold for items (default: {config.MIN_ITEM_INTERACTIONS})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=f"directory for processed artifacts (default: {config.PROCESSED_DIR})",
    )
    parser.add_argument(
        "--deduplicate",
        choices=list(config.VALID_DEDUPLICATE_POLICIES),
        default=config.DEDUPLICATE_POLICY,
        help=(
            "how to treat repeated (user_id, parent_asin, timestamp) records "
            f"(default: {config.DEDUPLICATE_POLICY})"
        ),
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)

    result = run_preprocessing(
        raw_path=args.input,
        category=args.category,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        processed_dir=Path(args.output_dir) if args.output_dir else None,
        deduplicate=args.deduplicate,
        verbose=not args.quiet,
    )

    print()
    print(format_statistics(result))
    print()
    print("Artifacts written:")
    for name, path in result.output_paths.items():
        size = path.stat().st_size if path.exists() else 0
        print(f"  {name:11s}: {path} ({size} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

