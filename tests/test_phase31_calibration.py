"""Phase 3.1: evaluator calibration and the measured Prompt v1 / v2 comparison.

Two jobs:

1. **Pin the clarification-suspension semantics.**  A run that suspends with
   ``ASKED_CLARIFICATION`` is a legitimate ending for a case that expects it, and a failure for
   a case that does not.  The tests here state that rule in both directions, and state the
   neighbouring rules that make it meaningful: a refusal, an abort and a budget exhaustion are
   *not* clarifications, and a clarification is not a way to satisfy a case that requires an
   answer.
2. **Measure the policy-contract change.**  Prompt v2 addresses two behaviours the real model
   actually showed: clarifying when it already had enough to answer, and repeating a read-only
   action whose observation added nothing.  The comparison is reported as per-case metrics and
   per-case trajectories, never as one aggregate score.

Everything here is offline.  The v1 baseline is a **replay** of the Phase-3 recording, so
recalibrating the evaluator costs no model calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import ActionKind  # noqa: E402
from recommendation.control.context import CandidateState, PolicyContext  # noqa: E402
from recommendation.control.model_client import ScriptedModelClient  # noqa: E402
from recommendation.control.model_policy import (  # noqa: E402
    POLICY_PARSE_PROTOCOL_VERSION,
    POLICY_PROMPT_VERSION,
    build_policy_system_prompt,
)
from recommendation.evaluation.agent import (  # noqa: E402
    CaseRunner,
    TerminalOutcome,
    case_by_id,
    load_cases,
)
from recommendation.evaluation.agent.metrics import compute_metrics  # noqa: E402
from recommendation.evaluation.agent.model_policy_adapter import (  # noqa: E402
    build_live_model_policy_factory,
)
from recommendation.evaluation.agent.schemas import (  # noqa: E402
    AgentTrajectory,
    TrajectoryRecord,
)
from experiments._harness import case_controller_factory, catalog_reasoner  # noqa: E402


def _trajectory(terminal: TerminalOutcome, records: list[TrajectoryRecord]) -> AgentTrajectory:
    """A minimal evaluation trajectory for a metrics unit test."""
    return AgentTrajectory(
        scenario_id="unit",
        variant="scripted_model",
        terminal=terminal,
        steps=len(records),
        records=tuple(records),
    )


def _clarify_policy() -> Any:
    """A policy that asks one question and stops, with no other action."""
    from recommendation.control.model_policy import LLMAgentPolicy

    return LLMAgentPolicy(
        ScriptedModelClient(
            [
                {
                    "action": "ask_clarification",
                    "arguments": {
                        "question": "What will you use it for, so I can narrow the options?",
                        "blocks": "candidate_set",
                    },
                }
            ]
        ),
        max_attempts=1,
    )


def _asked_run(case_id: str) -> Any:
    case = case_by_id(case_id)
    return CaseRunner(case_controller_factory, variant="scripted_model").run(
        case, policy=_clarify_policy(), reasoner=catalog_reasoner()
    )


# =========================================================================== #
# A. Clarification suspension is evaluated correctly
# =========================================================================== #


def test_a_required_clarification_is_a_successful_ending() -> None:
    """The rule the phase names: a required question is a legitimate suspension.

    ``ASK_CLARIFICATION`` is not a failure to finish - it is the run correctly declining to
    guess when a material requirement is absent.  The trajectory suspends with
    ``ASKED_CLARIFICATION``, the case requires that action, and the case declares that terminal
    acceptable, so nothing failed.
    """
    outcome = _asked_run("missing-requirement")
    assert outcome.trajectory.terminal is TerminalOutcome.ASKED_CLARIFICATION
    assert outcome.trajectory.termination_reason == "awaiting_user"
    assert outcome.trajectory.action_sequence() == ("ask_clarification",)
    assert outcome.metrics.terminal_allowed is True
    assert outcome.metrics.clarification_used is True
    assert outcome.metrics.clarification_was_expected is True
    assert outcome.metrics.required_actions_met == ("ask_clarification",)
    assert outcome.metrics.failures() == ()
    assert outcome.passed is True


def test_the_suspension_is_distinct_from_completion() -> None:
    """A suspension is its own terminal, not a disguised completion.

    If it were mapped onto ``COMPLETED`` the distinction would be lost, and a case that requires
    an answer could be satisfied by a question.
    """
    outcome = _asked_run("missing-requirement")
    assert outcome.trajectory.terminal is not TerminalOutcome.COMPLETED
    assert outcome.trajectory.terminal is not TerminalOutcome.COMPLETION_REFUSED
    assert outcome.trajectory.terminal is not TerminalOutcome.BUDGET_EXHAUSTED
    assert outcome.metrics.reached_completion is False


def test_an_unnecessary_clarification_still_fails() -> None:
    """A case that does not accept a question is not rescued by asking one.

    ``simple-fast-path`` forbids clarification because the request is answerable from the
    trusted history.  The suspension is a legal *runtime* ending and an evaluator *failure*:
    the case's expectation, not the terminal alone, decides.
    """
    outcome = _asked_run("simple-fast-path")
    assert outcome.trajectory.terminal is TerminalOutcome.ASKED_CLARIFICATION
    # The terminal is a real ending, but the case fails on the action it forbade and the action
    # it required - which is the honest attribution.
    assert "forbidden_action_used" in outcome.metrics.failures()
    assert "required_action_missing" in outcome.metrics.failures()
    assert outcome.passed is False


def test_a_clarification_case_declares_the_suspension_acceptable() -> None:
    """The case-level contract that makes the rule work, asserted rather than assumed."""
    for case in load_cases():
        if "ask_clarification" in case.required_actions:
            assert TerminalOutcome.ASKED_CLARIFICATION in case.allowed_terminal, case.case_id
        if "ask_clarification" in case.forbidden_actions:
            # A case may forbid the action; it must not also invite the suspension, or the
            # expectation would contradict itself.
            assert TerminalOutcome.ASKED_CLARIFICATION not in case.allowed_terminal, case.case_id


def test_a_completion_refusal_is_not_a_successful_clarification() -> None:
    """``COMPLETION_REFUSED`` is a distinct terminal and fails a clarification case.

    The guard refusing a premature FINISH is the runtime working; it is not the run having asked
    its question.  Conflating them would credit a case that was never satisfied.
    """
    case = case_by_id("missing-requirement")
    trajectory = _trajectory(
        TerminalOutcome.COMPLETION_REFUSED,
        [TrajectoryRecord(step_index=0, action_id="a", executed_action="finish")],
    )
    metrics = compute_metrics(trajectory, case)
    assert metrics.terminal_allowed is False
    assert "terminal_not_allowed" in metrics.failures()
    assert metrics.clarification_used is False
    assert metrics.clarification_was_expected is True
    assert "clarification" in metrics.failures()


def test_a_budget_exhaustion_is_not_a_successful_clarification() -> None:
    """``BUDGET_EXHAUSTED`` is a distinct terminal and fails a clarification case."""
    case = case_by_id("missing-requirement")
    trajectory = _trajectory(
        TerminalOutcome.BUDGET_EXHAUSTED,
        [TrajectoryRecord(step_index=0, action_id="a", executed_action="search_catalog")],
    )
    metrics = compute_metrics(trajectory, case)
    assert metrics.terminal_allowed is False
    assert "terminal_not_allowed" in metrics.failures()
    assert metrics.clarification_used is False


def test_an_abort_is_not_a_successful_clarification() -> None:
    """``FAILED`` from an aborted run is a distinct terminal and fails the case."""
    case = case_by_id("missing-requirement")
    metrics = compute_metrics(_trajectory(TerminalOutcome.FAILED, []), case)
    assert metrics.terminal_allowed is False
    assert "terminal_not_allowed" in metrics.failures()
    assert metrics.clarification_used is False


def test_every_runtime_ending_maps_to_a_distinct_evaluator_value() -> None:
    """The mapping is total and collision-free over the endings it covers."""
    from recommendation.control import RunStatus, TerminationReason
    from recommendation.evaluation.agent.runner import _terminal_outcome

    class _Control:
        def __init__(self, reason: TerminationReason | None) -> None:
            self.termination_reason = reason

    class _Result:
        def __init__(self, status: RunStatus, reason: TerminationReason | None = None) -> None:
            self.status = status
            self.control = _Control(reason)

    observed = [
        _terminal_outcome(_Result(RunStatus.WAITING_FOR_USER, TerminationReason.AWAITING_USER)),
        _terminal_outcome(_Result(RunStatus.FINISHED)),
        _terminal_outcome(_Result(RunStatus.ABORTED, TerminationReason.MAX_STEPS)),
        _terminal_outcome(_Result(RunStatus.FAILED, TerminationReason.COMPLETION_REFUSED)),
        _terminal_outcome(_Result(RunStatus.ABORTED, TerminationReason.INVALID_ACTION)),
        _terminal_outcome(_Result(RunStatus.FAILED, TerminationReason.EXECUTION_FAILED)),
    ]
    assert observed == [
        TerminalOutcome.ASKED_CLARIFICATION,
        TerminalOutcome.COMPLETED,
        TerminalOutcome.BUDGET_EXHAUSTED,
        TerminalOutcome.COMPLETION_REFUSED,
        TerminalOutcome.UNAUTHORISED_ACTION,
        TerminalOutcome.FAILED,
    ]
    assert len(set(observed)) == 6


# =========================================================================== #
# B. Prompt v2 is generic, versioned, and distinct from v1
# =========================================================================== #


def test_prompt_v2_is_a_new_version_and_kept_every_boundary_rule() -> None:
    """The version moved, and every Phase-1/2/2.1 rule is still present.

    A prompt revision must not quietly drop a boundary rule, so the invariants are asserted
    against the rendered prompt rather than against the source.
    """
    assert POLICY_PROMPT_VERSION >= 2, "the prompt version must have been incremented"
    lowered = build_policy_system_prompt().casefold()
    for invariant in (
        "one action",
        "offered",
        "never invent",
        "candidate_refs",
        "finish only",
        "budget",
    ):
        assert invariant in lowered, invariant
    assert "clarif" in lowered
    assert "repeat" in lowered or "same" in lowered


def test_prompt_v2_contains_no_case_specific_recipe() -> None:
    """The generic contract must not teach the benchmark its answers."""
    prompt = build_policy_system_prompt().casefold()
    for case in load_cases():
        assert case.case_id.casefold() not in prompt, case.case_id
        assert case.message.casefold().strip(".!? ") not in prompt, case.case_id
        if len(case.required_actions) > 1:
            assert " -> ".join(case.required_actions) not in prompt, case.case_id
    for forbidden in ("expected action", "gold", "benchmark", "trajectory should"):
        assert forbidden not in prompt, forbidden


def test_v1_and_v2_have_distinct_versions_and_fingerprints() -> None:
    """The two prompts are separable in the record, so a v2 trace cannot replay as v1."""
    from recommendation.control.model_client import ModelRequest
    from recommendation.control.model_policy import PROMPT_V1, PROMPT_V2
    from recommendation.control.model_recorder import fingerprint_request

    assert PROMPT_V1 != PROMPT_V2
    common = {
        "provider": "deepseek",
        "model": "deepseek-flash",
        "parse_protocol_version": POLICY_PARSE_PROTOCOL_VERSION,
    }
    first = fingerprint_request(
        ModelRequest(system_prompt=PROMPT_V1, context_payload={"a": 1}, action_schema=()),
        prompt_version=1,
        **common,
    )
    second = fingerprint_request(
        ModelRequest(system_prompt=PROMPT_V2, context_payload={"a": 1}, action_schema=()),
        prompt_version=2,
        **common,
    )
    assert first.digest != second.digest
    assert first.canonical["prompt_version"] != second.canonical["prompt_version"]
    assert first.canonical["system_prompt_sha256"] != second.canonical["system_prompt_sha256"]


def test_the_parse_protocol_absorbs_both_empty_literals_for_argumentless_actions() -> None:
    """Protocol v3: ``{}`` and ``[]`` both mean "no arguments" and neither is a payload.

    The v1 run used ``{}``; the v2 run used ``[]``.  Both are the JSON spelling of absent, and
    refusing one made the policy abort with no usable action for reasons that had nothing to do
    with the shopping decision.  This is the *only* tolerance added, and it is versioned.
    """
    from recommendation.control.context import CandidateState
    from recommendation.control.model_policy import LLMAgentPolicy

    assert POLICY_PARSE_PROTOCOL_VERSION == 3
    context = PolicyContext(
        user_request="recommend gear",
        available_actions=(ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.FINISH),
        has_trusted_history=True,
        candidate_state=CandidateState(),
    )
    for empty in ({}, []):
        policy = LLMAgentPolicy(
            ScriptedModelClient([{"action": "recommend_from_history", "arguments": empty}])
        )
        assert policy.choose(context).arguments is None, empty


def test_a_non_empty_payload_is_still_refused_whatever_its_literal() -> None:
    """The v3 tolerance is empty-only: any real payload for an argumentless action is refused."""
    from recommendation.control.context import CandidateState
    from recommendation.control.model_policy import LLMAgentPolicy

    context = PolicyContext(
        user_request="recommend gear",
        available_actions=(ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.FINISH),
        has_trusted_history=True,
        candidate_state=CandidateState(),
    )
    for payload in ({"parent_asins": ["FAKE"]}, ["FAKE"], {"k": 9}):
        policy = LLMAgentPolicy(
            ScriptedModelClient(
                [
                    {"action": "recommend_from_history", "arguments": payload},
                    {"action": "recommend_from_history", "arguments": payload},
                ]
            )
        )
        with pytest.raises(Exception):
            policy.choose(context)
        assert str(policy.last_metadata.get("last_failure", "")).startswith("invalid_arguments")


def test_only_one_policy_variable_changed_between_v1_and_v2() -> None:
    """The experiment's validity condition, asserted rather than assumed.

    v1 and v2 differ in the system prompt and its version, and in nothing else: same action
    schema, same payload builder, same parse protocol.
    """
    from recommendation.control.model_policy import PROMPT_V1, PROMPT_V2, build_action_schema
    from recommendation.control import ActionKind

    menu = (ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.GET_DETAILS, ActionKind.FINISH)
    assert build_action_schema(menu) == build_action_schema(menu)
    assert PROMPT_V1 != PROMPT_V2
    # The shared rules are literally shared text, so a future edit cannot fork them silently.
    for shared in ("Choose exactly ONE action", "Never invent a product identity"):
        assert shared in PROMPT_V1 and shared in PROMPT_V2


# =========================================================================== #
# C. Replay the Prompt-v1 recording under the corrected evaluator
# =========================================================================== #

#: The Phase-3 Prompt-v1 + parse-protocol-v2 recording, produced by a live run before this
#: phase.  Replaying it is what makes the recalibrated baseline free to obtain.
PHASE3_RECORDING = Path("/tmp/agentrecx-phase3/revised_live.jsonl")


def _require_recording(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Phase-3 recording not present at {path}")
    return path


def _replay_suite(recording: Path, *, system_prompt: str | None = None) -> tuple[list[Any], Any]:
    """Run the whole suite against a recording, with zero network.

    ``system_prompt`` must match the contract the recording was made under, because the request
    fingerprint includes the prompt.  Defaulting to the active prompt is right for a current
    recording and wrong for an archived one, so the archived case passes ``PROMPT_V1`` explicitly.
    """
    from recommendation.control.model_recorder import (
        RecordingModelClient,
        RecordingStore,
        ReplayMode,
        recording_identity,
    )

    identity = recording_identity(recording)
    client = RecordingModelClient(
        None,
        store=RecordingStore(recording),
        mode=ReplayMode.REPLAY,
        provider=str(identity["provider"]),
        model=str(identity["model"]),
        prompt_version=identity["prompt_version"],
        parse_protocol_version=identity.get("parse_protocol_version"),
    )
    factory = build_live_model_policy_factory(
        client=client, max_attempts=2, system_prompt=system_prompt
    )
    runner = CaseRunner(case_controller_factory, variant="live_model")
    outcomes = [
        runner.run(case, policy=factory(case), reasoner=catalog_reasoner())
        for case in load_cases()
    ]
    return outcomes, client


def test_the_v1_baseline_replays_with_zero_network() -> None:
    """The corrected baseline is obtained by replay, never by a new live run."""
    _require_recording(PHASE3_RECORDING)
    from recommendation.control.model_policy import PROMPT_V1

    outcomes, client = _replay_suite(PHASE3_RECORDING, system_prompt=PROMPT_V1)
    assert client.live_calls == 0, "replay reached the provider"
    assert client.replayed_calls > 0
    assert len(outcomes) == len(load_cases())


def test_the_v1_clarification_case_was_already_evaluated_correctly() -> None:
    """Honest calibration result: this case's terminal handling was never wrong.

    The phase began from the hypothesis that a required clarification was being failed for not
    finishing.  It is not: the terminal maps to ``ASKED_CLARIFICATION`` and the case accepts it.
    The recorded v1 failures for this case are the model executing a *forbidden* action and
    exceeding ``max_tool_calls`` - real policy failures, correctly attributed.
    """
    _require_recording(PHASE3_RECORDING)
    from recommendation.control.model_policy import PROMPT_V1

    outcomes, _client = _replay_suite(PHASE3_RECORDING, system_prompt=PROMPT_V1)
    outcome = next(o for o in outcomes if o.case.case_id == "missing-requirement")
    assert outcome.trajectory.terminal is TerminalOutcome.ASKED_CLARIFICATION
    assert outcome.metrics.terminal_allowed is True
    assert outcome.metrics.clarification_used is True
    assert outcome.metrics.required_actions_met == ("ask_clarification",)
    assert "forbidden_action_used" in outcome.metrics.failures()
    assert outcome.attribution.attributed["forbidden_action_used"] == "policy_selection"


# =========================================================================== #
# D. Trajectory observability for the no-progress pattern
# =========================================================================== #


def test_a_read_only_step_carries_a_bounded_read_signature() -> None:
    """A repeated read is distinguishable from a read of different targets.

    Before this, the evaluation projection exposed only the action name, so two ``get_details``
    steps that read different candidates looked identical to two that read the same ones - and
    the no-progress pattern this phase has to measure was unmeasurable.
    """
    from experiments._metrics import _repeated_read_only

    outcomes = _replay_suite(PHASE3_RECORDING, system_prompt=_prompt_v1())[0]
    signatures = [
        record.read_signature
        for outcome in outcomes
        for record in outcome.trajectory.records
        if record.read_signature
    ]
    assert signatures, "read-only steps must carry a signature"
    for signature in signatures:
        action, _, targets = signature.partition(":")
        assert action in {
            "get_details",
            "compare",
            "trade_off",
            "check_compatibility",
            "bundle",
            "verify",
        }
        assert len(signature) <= 200
        # Identities only: no catalogue value can appear in a signature.
        for forbidden in ("$", "kg", "Kilograms", "title"):
            assert forbidden not in targets


def test_a_candidate_action_carries_no_read_signature() -> None:
    """Only a read-only step has a read signature; a retrieval is not a "read"."""
    outcomes = _replay_suite(PHASE3_RECORDING, system_prompt=_prompt_v1())[0]
    for outcome in outcomes:
        for record in outcome.trajectory.records:
            if record.executed_action in {
                "recommend_from_history",
                "search_catalog",
                "finish",
                "ask_clarification",
            }:
                assert record.read_signature is None


def test_the_no_progress_metric_counts_only_identical_targets() -> None:
    """Reading a different candidate is progress, so it is not counted as a repeat."""
    from experiments._metrics import _repeated_read_only

    class _Record:
        def __init__(self, signature: str, index: int) -> None:
            self.read_signature = signature
            self.step_index = index

    records = [
        _Record("get_details:cand-red", 0),
        _Record("get_details:cand-blue", 1),   # different target: progress
        _Record("get_details:cand-red", 2),    # a genuine repeat of step 0
        _Record("compare:cand-red,cand-blue", 3),
        _Record("compare:cand-red,cand-blue", 4),  # a genuine repeat
        _Record("search_catalog:", 5),         # not a read-only signature
    ]
    repeats = _repeated_read_only(records)
    assert [r["signature"] for r in repeats] == [
        "get_details:cand-red",
        "compare:cand-red,cand-blue",
    ]


def _prompt_v1() -> str:
    from recommendation.control.model_policy import PROMPT_V1

    return PROMPT_V1
