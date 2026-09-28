"""M4 sharded-execution equivalence test (offline, deterministic).

Preregistration §17 says the formal run is **one execution over all 20,000 users**. This module
does not change that rule; it answers a narrower engineering question:

    can the same run be *executed* in independent shards and merged back into the exact frozen
    cohort order, producing byte-identical per-user outputs and metrics?

The answer is established by construction here, not by argument:

* a deterministic toy pipeline stands in for the real one, so the test runs offline and fast;
* the *serial* execution and the *N-way sharded* execution are driven through the identical
  per-user function;
* the merge is a pure sort on the frozen cohort position, never on completion order;
* the metrics are computed from the merged outputs and compared field by field.

If any of these differ, sharded execution is not approved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "ShardPlan",
    "build_shard_plan",
    "merge_user_outputs",
    "run_serial",
    "run_sharded",
    "outputs_digest",
    "metrics_from_outputs",
]


@dataclass(frozen=True)
class ShardPlan:
    """A deterministic assignment of frozen cohort positions to shards.

    Sharding is over **positions in the frozen cohort order**, not over user ids, so the assignment
    is reproducible from the cohort alone and needs no shared mutable state.
    """

    shards: tuple[tuple[int, ...], ...]

    @property
    def count(self) -> int:
        """How many shards the plan defines."""
        return len(self.shards)

    def flattened(self) -> tuple[int, ...]:
        """Every assigned position, in shard order (used to prove a partition)."""
        return tuple(p for shard in self.shards for p in shard)


def build_shard_plan(size: int, shards: int) -> ShardPlan:
    """Partition ``range(size)`` into ``shards`` contiguous blocks.

    Contiguous blocks are chosen over round-robin so a shard is a contiguous slice of the frozen
    order. Either is deterministic; contiguity additionally makes a shard's artifact auditable
    against the cohort by a simple offset check.
    """
    if shards < 1:
        raise ValueError("shards must be >= 1")
    if size < 0:
        raise ValueError("size must be >= 0")
    base, extra = divmod(size, shards)
    plan: list[tuple[int, ...]] = []
    start = 0
    for index in range(shards):
        length = base + (1 if index < extra else 0)
        plan.append(tuple(range(start, start + length)))
        start += length
    return ShardPlan(tuple(plan))


def run_serial(
    positions: Sequence[int],
    run_one: Callable[[int], dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Run every position in order, returning ``(position, output)`` pairs."""
    return [(position, run_one(position)) for position in positions]


def run_sharded(
    plan: ShardPlan,
    run_one: Callable[[int], dict[str, Any]],
    *,
    executor: Callable[[Callable[[int], dict[str, Any]], Sequence[int]], list[tuple[int, dict[str, Any]]]]
    | None = None,
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Run each shard independently and return one result list per shard.

    ``executor`` defaults to a plain in-order loop. A concurrent executor may be supplied; the
    merge below must not care which was used.
    """
    run = executor or (lambda fn, positions: [(p, fn(p)) for p in positions])
    return [run(run_one, shard) for shard in plan.shards]


def merge_user_outputs(
    per_shard: Iterable[Iterable[tuple[int, dict[str, Any]]]],
    *,
    size: int,
) -> list[dict[str, Any]]:
    """Merge shard outputs into the frozen cohort order.

    The merge is a **sort on the frozen position**, never on shard order or completion order, so it
    cannot depend on which shard finished first. It also verifies the result is a complete
    permutation of ``0..size-1``, so a duplicate or a dropped user is an error rather than a
    silently shorter cohort.
    """
    collected: dict[int, dict[str, Any]] = {}
    for shard in per_shard:
        for position, output in shard:
            if position in collected:
                raise ValueError(f"position {position} appears in more than one shard")
            if not (0 <= position < size):
                raise ValueError(f"position {position} is outside the cohort 0..{size - 1}")
            collected[position] = output
    missing = sorted(set(range(size)) - set(collected))
    if missing:
        raise ValueError(f"merge is missing {len(missing)} users, first: {missing[:5]}")
    return [collected[position] for position in range(size)]


def outputs_digest(outputs: Sequence[dict[str, Any]]) -> str:
    """A stable digest of a per-user output sequence, for cross-run comparison."""
    canonical = json.dumps(outputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def metrics_from_outputs(outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics from merged per-user outputs.

    Deliberately a pure fold over the merged sequence: this is what makes "metrics from merged
    shards" identical to "metrics from a serial run" whenever the merged sequences are identical.
    """
    users = len(outputs)
    hits = sum(1 for row in outputs if row.get("hit"))
    ndcg = sum(float(row.get("ndcg", 0.0)) for row in outputs)
    return {
        "users": users,
        "hits": hits,
        "recall": hits / users if users else 0.0,
        "ndcg": ndcg / users if users else 0.0,
    }
