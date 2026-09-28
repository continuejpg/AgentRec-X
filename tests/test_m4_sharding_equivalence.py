"""Serial vs sharded execution equivalence for the M4 fixed arm (offline, deterministic).

The formal 20,000-user LLM run is one execution under preregistration §17. This test does not
change that; it establishes the *engineering* property that the same per-user work can be executed
in independent shards and merged back into the exact frozen cohort order.

It runs the **real** M4 harness (``M4HarnessFactory`` + ``FixedTraversalPolicy`` + the frozen source
adapters) over a synthetic head table, so the property is checked against the code that will run,
not against a toy. The fixed arm is used because it is deterministic and needs no provider, which
makes an exact comparison possible offline.

Checked here:

* the merged user set is a complete permutation of the frozen cohort positions;
* the merged user order equals the frozen cohort order;
* per-user rankings are identical between serial and sharded execution;
* the per-user action/trajectory records are identical;
* the evaluator-facing inputs (histories/targets/scores) are identical;
* aggregate metrics computed from the merged outputs are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pytest

from recommendation.control.arguments import CandidateSource

from experiments.m4_harness import (
    M4HarnessFactory,
    M4HeadTable,
    M4IdentityMap,
)
from experiments.m4_policies import FixedTraversalPolicy
from experiments.m4_sharding import (
    build_shard_plan,
    merge_user_outputs,
    metrics_from_outputs,
    outputs_digest,
    run_serial,
    run_sharded,
)

# --------------------------------------------------------------------------- #
# synthetic frozen cohort + head table
# --------------------------------------------------------------------------- #

NUM_ITEMS = 40
USERS = 24
SOURCES = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)


def _identities() -> list[str]:
    return [f"B{index:04d}" for index in range(1, NUM_ITEMS + 1)]


def _identity_map() -> M4IdentityMap:
    ids = _identities()
    return M4IdentityMap({value: i + 1 for i, value in enumerate(ids)}, [None, *ids])


def _head_for(source: CandidateSource, user_int_id: int) -> tuple[tuple[str, int, float], ...]:
    """A deterministic synthetic head that differs per source and per user.

    Different sources must return *different* lists, otherwise dropping one from the ledger could
    not change the fused order and the test would not exercise the merge meaningfully.
    """
    ids = _identities()
    offset = {s: i for i, s in enumerate(SOURCES)}[source] * 3
    rows = []
    for rank in range(20):
        index = (user_int_id * 2 + offset + rank) % len(ids)
        rows.append((ids[index], rank + 1, 1.0 / (rank + 1)))
    return tuple(rows)


def _head_table() -> M4HeadTable:
    return M4HeadTable(
        heads={
            source: {uid: _head_for(source, uid) for uid in range(1, USERS + 1)}
            for source in SOURCES
        },
        score_kinds={
            CandidateSource.CATALOG_SEARCH: "bm25",
            CandidateSource.HISTORY: "sasrec_logit",
            CandidateSource.SIMILAR_ITEM: "item_item_similarity",
            CandidateSource.TWO_TOWER: "two_tower_cosine",
        },
    )


@dataclass(frozen=True)
class _Case:
    """A minimal ``EvaluationCase``-shaped object."""

    user_int_id: int
    user_id: str
    test_history: tuple[int, ...]
    test_target: int


def _cases() -> list[_Case]:
    """A frozen cohort in canonical order, with targets inside the catalogue."""
    out: list[_Case] = []
    for index in range(USERS):
        uid = index + 1
        history = ((index % NUM_ITEMS) + 1, ((index + 7) % NUM_ITEMS) + 1)
        out.append(
            _Case(
                user_int_id=uid,
                user_id=f"user{uid:03d}",
                test_history=history,
                test_target=((index + 19) % NUM_ITEMS) + 1,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# the per-user unit of work, shared by both execution modes
# --------------------------------------------------------------------------- #


def _run_one_factory(cases: Sequence[_Case], item2id: dict[str, int]):
    """Return ``run_one(position) -> output`` for the frozen cohort."""
    identity = _identity_map()
    factory = M4HarnessFactory(
        identity_map=identity,
        table=_head_table(),
        policy_factory=FixedTraversalPolicy,
    )

    def run_one(position: int) -> dict[str, Any]:
        case = cases[position]
        harness = factory(case)
        result = harness.controller.run("recommend something", harness.trusted_history)
        ranking = list(harness.ledger.ranked_parent_asins())
        ranked_ids = [item2id[value] for value in ranking if value in item2id]
        control = result.control
        return {
            "user_int_id": case.user_int_id,
            "ranking": ranked_ids,
            "action_sequence": [
                str(getattr(step.action_proposal.get("action"), "value", step.action_proposal.get("action")))
                for step in (getattr(result.trajectory, "steps", ()) or ())
                if isinstance(getattr(step, "action_proposal", None), dict)
            ],
            "sources_queried": sorted(
                tool.source.value for tool in harness.tools.values() if tool.consumed
            ),
            "target": case.test_target,
            "hit": case.test_target in ranked_ids[:10],
            "ndcg": (
                1.0 / np.log2(ranked_ids.index(case.test_target) + 2)
                if case.test_target in ranked_ids[:10]
                else 0.0
            ),
            "status": str(getattr(control.status, "value", control.status)),
            "tool_calls": int(control.tool_call_count),
        }

    return run_one


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shards", [1, 2, 3, 5, 7])
def test_sharded_execution_matches_serial_execution(shards: int) -> None:
    """Serial and N-way sharded execution must produce identical outputs and metrics."""
    cases = _cases()
    item2id = {f"B{index:04d}": index for index in range(1, NUM_ITEMS + 1)}
    run_one = _run_one_factory(cases, item2id)
    positions = tuple(range(len(cases)))

    serial_pairs = run_serial(positions, run_one)
    serial_outputs = [output for _position, output in serial_pairs]

    plan = build_shard_plan(len(cases), shards)
    per_shard = run_sharded(plan, run_one)
    merged = merge_user_outputs(per_shard, size=len(cases))

    # 1. user set + order after merge
    assert [row["user_int_id"] for row in merged] == [
        case.user_int_id for case in cases
    ], "merged order is not the frozen cohort order"

    # 2. per-user outputs, trajectory records and evaluator inputs
    assert outputs_digest(merged) == outputs_digest(serial_outputs)

    # 3. metrics from merged outputs
    assert metrics_from_outputs(merged) == metrics_from_outputs(serial_outputs)

    # 4. the traversal was attempted across the frozen source set
    #
    # NOTE: this asserts the *attempt*, not success. The frozen plane's own
    # `_CatalogSearchSourceTool` requires `arguments.terms` (candidate_plane.py:216) while
    # `SELECT_SOURCE` carries only `source`/`limit`, so `SELECT_SOURCE{catalog_search}` returns
    # zero candidates. That is a repository property, not a sharding property, and it is recorded
    # separately in `test_the_catalog_search_source_cannot_be_served_through_select_source`.
    for row in merged:
        assert row["tool_calls"] == len(SOURCES)
        assert row["sources_queried"], "no source was queried at all"


def test_the_shard_plan_is_a_complete_partition() -> None:
    """No user may be duplicated or dropped by the assignment itself."""
    for size, shards in ((24, 3), (20, 7), (1, 3), (0, 2), (10, 10)):
        plan = build_shard_plan(size, shards)
        assert sorted(plan.flattened()) == list(range(size))
        assert len(plan.flattened()) == size


def test_merge_rejects_a_duplicate_user() -> None:
    """A duplicated position is an error, not a silently shorter cohort."""
    with pytest.raises(ValueError, match="more than one shard"):
        merge_user_outputs([[(0, {"a": 1})], [(0, {"a": 2})]], size=1)


def test_merge_rejects_a_dropped_user() -> None:
    """A missing position is an error, not a silently shorter cohort."""
    with pytest.raises(ValueError, match="missing"):
        merge_user_outputs([[(0, {"a": 1})]], size=2)


def test_merge_rejects_an_out_of_range_position() -> None:
    """A shard cannot smuggle in a user outside the frozen cohort."""
    with pytest.raises(ValueError, match="outside the cohort"):
        merge_user_outputs([[(5, {"a": 1})]], size=2)


def test_merge_order_does_not_depend_on_shard_completion_order() -> None:
    """The merge sorts on the frozen position, so completion order cannot matter."""
    outputs = [{"n": index} for index in range(6)]
    plan = build_shard_plan(6, 3)

    def shard_rows(shard: Sequence[int]) -> list[tuple[int, dict[str, Any]]]:
        return [(position, outputs[position]) for position in shard]

    forward = [shard_rows(shard) for shard in plan.shards]
    backward = list(reversed(forward))
    assert merge_user_outputs(forward, size=6) == merge_user_outputs(backward, size=6)


def test_a_concurrent_executor_yields_the_same_merge() -> None:
    """A thread-pool executor must not change the merged result."""
    import concurrent.futures

    cases = _cases()
    item2id = {f"B{index:04d}": index for index in range(1, NUM_ITEMS + 1)}
    run_one = _run_one_factory(cases, item2id)
    positions = tuple(range(len(cases)))

    serial_outputs = [output for _p, output in run_serial(positions, run_one)]
    plan = build_shard_plan(len(cases), 4)

    def pool_executor(fn, shard_positions):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn, position): position for position in shard_positions}
            results = [(futures[f], f.result()) for f in concurrent.futures.as_completed(futures)]
        return sorted(results, key=lambda pair: pair[0])

    per_shard = run_sharded(plan, run_one, executor=pool_executor)
    merged = merge_user_outputs(per_shard, size=len(cases))
    assert outputs_digest(merged) == outputs_digest(serial_outputs)


def test_the_catalog_search_source_cannot_be_served_through_select_source() -> None:
    """Records the repository property that blocks a uniform four-source ``SELECT_SOURCE``.

    ``CandidatePlane`` wraps the two dedicated tools in its own adapters
    (``candidate_plane.py:271-276``). ``_HistorySourceTool`` reads the run's trusted history, but
    ``_CatalogSearchSourceTool.propose`` reads ``arguments.terms`` (``:216``) and returns ``[]``
    when there are none. ``SELECT_SOURCE`` carries ``SelectSourceArguments`` -- a ``source`` and a
    ``limit``, never ``terms`` -- so the catalogue search source is **unreachable** through
    ``SELECT_SOURCE`` while remaining reachable through its dedicated ``SEARCH_CATALOG`` action.

    This test pins the behaviour so that a change is noticed rather than assumed. It does not
    approve the asymmetry: M4's adaptive arm cannot select ``CATALOG_SEARCH`` uniformly with the
    other three sources, which is a blocker for the frozen design, not a detail.
    """
    from recommendation.control.arguments import CandidateSource, SelectSourceArguments

    cases = _cases()
    item2id = {f"B{index:04d}": index for index in range(1, NUM_ITEMS + 1)}
    factory = M4HarnessFactory(
        identity_map=_identity_map(), table=_head_table(), policy_factory=FixedTraversalPolicy
    )
    harness = factory(cases[0])
    plane = harness.controller._candidate_plane

    catalog_tool = plane._tools[CandidateSource.CATALOG_SEARCH]
    rows = catalog_tool.propose(
        arguments=SelectSourceArguments(source=CandidateSource.CATALOG_SEARCH, limit=5), limit=5
    )
    assert rows == [], "catalog_search unexpectedly served rows through SELECT_SOURCE"

    # The other three are served: two pass through unwrapped, and history reads trusted history.
    for source in (CandidateSource.SIMILAR_ITEM, CandidateSource.TWO_TOWER):
        tool = plane._tools[source]
        served = tool.propose(
            arguments=SelectSourceArguments(source=source, limit=5),
            limit=5,
            history=harness.trusted_history,
        )
        assert served, f"{source.value} should be servable through SELECT_SOURCE"
    _ = item2id
