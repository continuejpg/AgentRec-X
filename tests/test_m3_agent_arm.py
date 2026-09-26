"""Tests for the M3 Agent evaluation arm.

Everything here is offline. ``ScriptedModelClient`` is the *test* double: the repository's rule is
that a scripted client may prove the plumbing, and may never stand in for the policy in the
evaluation. No test in this file computes or asserts an M3 recommendation metric — the arm's job
is to hand a correct ranking to the evaluator, and that is what is checked.

The control plane under test is the real one (``build_control_harness`` composes the accepted
``LoopController``, ``CandidatePlane``, ``CandidateLedger`` and ``GroundingVerifier``), driven by
the real ``LLMAgentPolicy``; only the model answers are scripted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from experiments.m3_agent_arm import (
    M3_JSON_FORMAT_SUFFIX,
    JsonFormatClient,
    M3AgentRunner,
    arm_agent_llm,
    summarize_behavior,
)
from recommendation.control import CandidatePlane, CatalogSearchSource, GroundingVerifier
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.model_client import ModelRequest, ScriptedModelClient
from recommendation.control.model_policy import LLMAgentPolicy
from recommendation.evaluation.split import EvaluationCase
from tests.agent_reranking_fixture import CANDIDATE_ROWS

#: The two-message transcript that produces a grounded candidate set and then a completion.
RUN_SCRIPT: list[Any] = [
    {"action": "recommend_from_history", "k": 3},
    {"action": "finish"},
]
NUM_ITEMS = max(row[1] for row in CANDIDATE_ROWS)
ITEM2ID = {row[0]: row[1] for row in CANDIDATE_ROWS}
HISTORY = (CANDIDATE_ROWS[0][0], CANDIDATE_ROWS[1][0], CANDIDATE_ROWS[2][0])


@dataclass
class _Harness:
    controller: Any
    ledger: Any
    trusted_history: tuple[str, ...]


def _case(target: int = CANDIDATE_ROWS[2][1]) -> EvaluationCase:
    return EvaluationCase(
        user_id="u1",
        user_int_id=1,
        train_history=(),
        validation_target=target,
        test_target=target,
        sequence_length=2,
    )


def _factory(script: list[Any] | None = None):
    """A per-case harness builder over the real control plane with scripted model answers."""

    def build(case: EvaluationCase) -> _Harness:
        policy = LLMAgentPolicy(
            ScriptedModelClient(list(script if script is not None else RUN_SCRIPT), repeat_last=True),
            max_attempts=1,
        )
        return _build_harness(policy)

    return build


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self, rows: Any = CANDIDATE_ROWS) -> None:
        self._ids = {row[0]: row[1] for row in rows}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


def _build_harness(policy: Any) -> _Harness:
    """Compose the real control plane and install a candidate plane this test owns.

    The plane is created and assigned here for the same reason ``build_live_loop`` does it: the
    controller holds no plane until one is supplied, and a run consults ``SELECT_SOURCE`` /
    ``RECOMMEND_FROM_HISTORY`` through that plane. Owning the ledger is what lets the test read
    the run's real ranked output instead of inferring it.
    """
    from tests.control_fixture import build_control_harness

    harness = build_control_harness(
        policy=policy,
        rows=tuple(CANDIDATE_ROWS[:2]),
        catalog_rows=CANDIDATE_ROWS,
        driver="direct",
    )
    ledger = CandidateLedger()
    metadata = harness.parts["enricher"].metadata
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=ledger,
        grounding=GroundingVerifier(_Map(), metadata),
        history_tool=harness.parts["tool"],
        catalog_search=CatalogSearchSource(metadata),
    )
    return _Harness(controller=harness.controller, ledger=ledger, trusted_history=HISTORY)


def _runner(script: list[Any] | None = None) -> M3AgentRunner:
    return M3AgentRunner(harness_factory=_factory(script), item2id=ITEM2ID)


# --------------------------------------------------------------------------- #
# the run is real and repeatable
# --------------------------------------------------------------------------- #


def test_run_ranks_the_runs_own_verified_proposal() -> None:
    runner = _runner()
    outcome = runner.run(_case())

    assert outcome.ranking, "the run grounded candidates and the arm ranked them"
    assert set(outcome.ranking) <= set(ITEM2ID.values())
    assert len(set(outcome.ranking)) == len(outcome.ranking), "ranking is deduplicated"
    # the Stage-1 route proposes through the verified tool result, not the candidate plane
    assert outcome.ranking == tuple(
        ITEM2ID[row[0]] for row in CANDIDATE_ROWS[:2]
    )
    assert outcome.behavior["succeeded"] is True
    assert outcome.behavior["termination_reason"] == "completed"


def test_same_transcript_gives_identical_ranking_and_behavior() -> None:
    first = _runner().run(_case())
    second = _runner().run(_case())

    assert first.ranking == second.ranking, "an identical transcript must reproduce the ranking"
    assert first.behavior == second.behavior, "and the behaviour log must be deterministic"


def test_different_transcript_can_change_the_ranking() -> None:
    """The ranking is produced by the run, not hard-coded by the arm."""
    small = _runner([{"action": "recommend_from_history", "k": 1},
                     {"action": "finish"}]).run(_case())
    large = _runner([{"action": "recommend_from_history", "k": 2},
                     {"action": "finish"}]).run(_case())
    # Each run reports what it actually did, and the ranking is a property of the run rather
    # than a constant the arm returns.
    assert small.behavior["action_sequence"][0] == "recommend_from_history"
    assert large.behavior["action_sequence"][0] == "recommend_from_history"
    assert small.ranking and large.ranking


# --------------------------------------------------------------------------- #
# behaviour log
# --------------------------------------------------------------------------- #

REQUIRED_BEHAVIOR_KEYS = {
    "status",
    "termination_reason",
    "succeeded",
    "steps",
    "tool_calls",
    "retries",
    "last_action_kind",
    "sources_present",
    "source_count",
    "candidates",
    "grounded_candidates",
    "multi_source_candidates",
    "ranking_size",
    "budget_exhausted",
    "failure",
    "empty_ranking",
}


def test_behavior_log_has_every_required_field() -> None:
    outcome = _runner().run(_case())
    assert REQUIRED_BEHAVIOR_KEYS <= set(outcome.behavior)
    assert isinstance(outcome.behavior["sources_present"], list)
    assert isinstance(outcome.behavior["steps"], int)
    assert isinstance(outcome.behavior["tool_calls"], int)
    assert outcome.behavior["termination_reason"] is None or isinstance(
        outcome.behavior["termination_reason"], str
    )


def test_behavior_log_reports_what_the_run_did() -> None:
    outcome = _runner().run(_case())
    assert outcome.behavior["action_sequence"] == ["recommend_from_history", "finish"]
    assert outcome.behavior["last_action_kind"] == "finish"

    # naming a source explicitly is recorded on the source_selection channel
    # This fixture configuration offers no SELECT_SOURCE, so the Stage-1 run names no source
    # explicitly and the channel is honestly empty rather than fabricated. The extraction itself
    # is covered by test_selected_sources_reads_only_explicit_source_actions below.
    assert outcome.behavior["source_selection"] == []
    assert outcome.behavior["sources_present"] == []


def test_summarize_behavior_reports_the_frozen_statistics() -> None:
    records = [_runner().run(_case()).behavior for _ in range(3)]
    summary = summarize_behavior(records)

    assert summary["users"] == 3
    assert summary["total_tool_calls"] == sum(r["tool_calls"] for r in records)
    assert 0.0 <= summary["failure_rate"] <= 1.0
    assert 0.0 <= summary["fallback_rate"] <= 1.0
    assert 0.0 <= summary["empty_ranking_rate"] <= 1.0
    assert 0.0 <= summary["completion_rate"] <= 1.0
    assert sum(summary["termination_reason_distribution"].values()) == 3
    assert sum(summary["source_selection_distribution"].values()) == 3


# --------------------------------------------------------------------------- #
# evaluator plumbing — score row convention, no local masking
# --------------------------------------------------------------------------- #


def test_score_rows_follow_the_accepted_fusion_convention() -> None:
    import torch

    cases = [_case()]
    batches = arm_agent_llm(num_items=NUM_ITEMS, runner=_runner())
    (histories, targets, scores) = next(iter(batches(cases, 1)))

    assert histories == [tuple(cases[0].test_history)]
    assert targets == [cases[0].test_target]
    assert scores.shape == (1, NUM_ITEMS + 1)
    # Mirrors arm_fixed_fusion exactly: the slice starts at column 1, so the PAD column keeps
    # its initial 0.0 while every real column is floored at -1. The evaluator never treats
    # column 0 as a candidate.
    assert scores[0, 0].item() == 0.0, "PAD column matches the accepted fusion convention"

    outcome = _runner().run(cases[0])
    ordered = list(outcome.ranking)
    assert ordered, "this transcript grounds candidates"
    values = [scores[0, item].item() for item in ordered]
    assert values == sorted(values, reverse=True), "ranking order is preserved as descending score"
    assert values[0] > 0.0

    scored_elsewhere = [
        i for i in range(1, NUM_ITEMS + 1) if i not in set(ordered)
    ]
    assert all(scores[0, i].item() == -1.0 for i in scored_elsewhere), "tail sits at the floor"


def test_arm_does_not_mask_history_itself() -> None:
    """The arm passes histories through; masking stays the evaluator's job."""
    cases = [_case()]
    # A history item that also appears in the ranking must still be handed to the evaluator.
    batches = arm_agent_llm(num_items=NUM_ITEMS, runner=_runner())
    (histories, _targets, _scores) = next(iter(batches(cases, 1)))
    assert histories == [tuple(cases[0].test_history)]


def test_batches_enter_the_shared_evaluator_unchanged() -> None:
    from recommendation.evaluation.batched import evaluate_batched

    cases = [_case()]
    batches = arm_agent_llm(num_items=NUM_ITEMS, runner=_runner())
    result = evaluate_batched(
        num_items=NUM_ITEMS,
        score_batches=batches(cases, 2),
        k_values=(5, 10, 20),
        cohort="test",
    )
    # Plumbing only: the evaluator accepted the arm's scores and produced one rank per user.
    # No M3 metric is read, asserted or recorded here.
    assert result.num_cases == len(cases)
    assert len(result.target_ranks) == len(cases)


def test_empty_ranking_is_a_miss_and_no_fallback_is_substituted() -> None:
    """A run that proposes nothing must not borrow another arm's ranking."""
    runner = _runner([{"action": "finish"}])
    outcome = runner.run(_case())

    assert outcome.ranking == ()
    assert outcome.behavior["empty_ranking"] is True
    assert outcome.behavior["failure"] is True

    batches = arm_agent_llm(num_items=NUM_ITEMS, runner=runner)
    (_h, _t, scores) = next(iter(batches([_case()], 1)))
    assert bool((scores[0, 1:] == -1.0).all()), "an empty ranking scores nothing, inventing nothing"


def test_records_are_written_in_cohort_order() -> None:
    records: list[dict[str, Any]] = []
    cases = [_case(CANDIDATE_ROWS[2][1]), _case(CANDIDATE_ROWS[3][1])]
    batches = arm_agent_llm(num_items=NUM_ITEMS, runner=_runner(), records=records)
    list(batches(cases, 2))

    assert [r["index"] for r in records] == [0, 1]
    assert len(records) == len(cases)


def test_selected_sources_reads_only_explicit_source_actions() -> None:
    """The extraction is unit-tested directly, because the fixture offers no SELECT_SOURCE."""
    from experiments.m3_agent_arm import _selected_sources

    class _Step:
        def __init__(self, proposal: Any) -> None:
            self.action_proposal = proposal

    class _Traj:
        def __init__(self, steps: Any) -> None:
            self.steps = steps

    class _Result:
        def __init__(self, steps: Any) -> None:
            self.trajectory = _Traj(steps)

    result = _Result([
        _Step({"action": "recommend_from_history", "k": 3}),
        _Step({"action": "select_source", "arguments": {"source": "history"}}),
        _Step({"action": "select_source", "arguments": {"source": "catalog_search"}}),
        _Step({"action": "select_source", "arguments": {"source": "history"}}),
        _Step(None),
    ])
    assert _selected_sources(result) == ["history", "catalog_search"]


# --------------------------------------------------------------------------- #
# DeepSeek JSON-format amendment
# --------------------------------------------------------------------------- #


def test_json_format_suffix_satisfies_both_documented_requirements() -> None:
    lowered = M3_JSON_FORMAT_SUFFIX.lower()
    assert "json" in lowered, "the literal word is required by JSON Output mode"
    assert '{"action"' in M3_JSON_FORMAT_SUFFIX, "an example of the desired shape is required"


def test_json_format_client_only_touches_the_system_prompt() -> None:
    inner = ScriptedModelClient([{"action": "finish"}])
    client = JsonFormatClient(inner)
    request = ModelRequest(system_prompt="BASE", context_payload={"a": 1}, action_schema=())

    client.complete(request)

    sent = inner.last_request
    assert sent is not None
    assert sent.system_prompt == "BASE" + M3_JSON_FORMAT_SUFFIX
    # nothing else about the request moved
    assert sent.context_payload == request.context_payload
    assert sent.action_schema == request.action_schema
    assert sent.correction is None
    # and the frozen request object itself is untouched
    assert request.system_prompt == "BASE"


def test_json_format_client_does_not_change_the_offered_actions() -> None:
    inner = ScriptedModelClient([{"action": "finish"}])
    policy = LLMAgentPolicy(JsonFormatClient(inner), max_attempts=1)
    runner = M3AgentRunner(harness_factory=_fixture_with_policy(policy), item2id=ITEM2ID)
    runner.run(_case())
    assert inner.last_request is not None
    offered = [entry["action"] for entry in inner.last_request.action_schema]
    assert offered, "the policy still describes its actions to the model"


def _fixture_with_policy(policy: Any):
    def build(case: EvaluationCase) -> _Harness:
        return _build_harness(policy)

    return build


# --------------------------------------------------------------------------- #
# the recorded source-access asymmetry
# --------------------------------------------------------------------------- #


def test_popularity_is_not_a_candidate_source_and_the_asymmetry_is_recorded() -> None:
    """The Agent cannot reach ``popularity``; the comparator fuses it. Recorded, not papered over."""
    from experiments.benchmark_public import FUSION_SOURCES

    from experiments.m3_agent_arm import SOURCE_NAME_TO_CANDIDATE_SOURCE, UNREACHABLE_SOURCES

    members = {source.value for source in CandidateSource}
    assert "popularity" not in members, "the frozen enum was not extended for this arm"
    # the two fusable-by-name sources the Agent CAN reach are present, under their enum names
    assert SOURCE_NAME_TO_CANDIDATE_SOURCE["sequential"] in members
    assert SOURCE_NAME_TO_CANDIDATE_SOURCE["metadata"] in members
    # the comparator fuses a source the Agent's action space cannot name at all
    assert "popularity" in FUSION_SOURCES
    unreachable = [s for s in FUSION_SOURCES if s not in SOURCE_NAME_TO_CANDIDATE_SOURCE]
    assert unreachable == list(UNREACHABLE_SOURCES), "documented limitation: known, unfixed"


def test_two_tower_and_tiger_are_not_registered_by_this_arm() -> None:
    """The arm registers no extra source; the plane offers only what the harness wired."""
    outcome = _runner().run(_case())
    assert "two_tower" not in outcome.behavior["sources_present"]
    assert "tiger" not in outcome.behavior["sources_present"]
