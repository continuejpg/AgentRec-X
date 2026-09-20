"""Phase 1 tests: the model-driven AgentPolicy.

`tests/test_agent_control_plane.py` proves the loop; `tests/test_evaluation_agent_plane.py`
proves the evaluation plane.  This file proves the **policy layer**: that a model can choose the
next action, and that nothing about the trust model weakened to let it.

The distinction the whole file rests on:

    the model decides **what action to propose**
    trusted code decides whether it is legal, how it runs, which identities exist, which facts
    are true, whether memory may be committed, and whether FINISH is permitted

Every test here is offline: a :class:`~recommendation.control.model_client.ScriptedModelClient`
stands in for a provider, so the suite needs no network and no API key.  Families:

**A. Structured output** - a valid answer becomes a valid proposal; malformed answers do not.
**B. Authority** - the model cannot name an unoffered action, invent an argument, or produce a
   candidate identity.
**C. Memory and FINISH boundaries** - a proposal cannot commit memory or end a run.
**D. Observation-dependent replanning** - the second policy call sees the updated context.
**E. Bounded failure** - a model that never answers properly terminates inside existing budgets.
**F. Provider independence** - the seam is injectable and the tests use no external service.
**G. Integration** - the full trajectory the phase asks for, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ARGUMENTS_BY_ACTION,
    ActionKind,
    ActionProposal,
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    GroundedReasoner,
    GroundingVerifier,
    LoopLimits,
    PolicyActionError,
    PolicyContext,
    ReasoningExecutor,
    RunStatus,
    TerminationReason,
)
from recommendation.control.context import CandidateState  # noqa: E402
from recommendation.control.model_client import (  # noqa: E402
    ModelCallError,
    ModelRequest,
    ModelResponse,
    ScriptedModelClient,
    StructuredModelClient,
)
from recommendation.control.model_policy import (  # noqa: E402
    LLM_POLICY_NAME,
    LLMAgentPolicy,
    build_action_schema,
    build_policy_context_payload,
    build_policy_system_prompt,
)
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

#: The action menu a typical mid-run context offers.
MENU: tuple[ActionKind, ...] = (
    ActionKind.RECOMMEND_FROM_HISTORY,
    ActionKind.SEARCH_CATALOG,
    ActionKind.ASK_CLARIFICATION,
    ActionKind.GET_DETAILS,
    ActionKind.COMPARE,
    ActionKind.FINISH,
)


def context(
    *,
    available: tuple[ActionKind, ...] = MENU,
    request: str = "I need waterproof trail shoes under $120.",
    grounded: bool = False,
    count: int = 0,
    remaining_steps: int = 5,
    remaining_tool_calls: int = 3,
    last_observation: object | None = None,
    awaiting_user: bool = False,
    candidates: tuple[str, ...] = (),
) -> PolicyContext:
    """Build a bounded policy context for a unit test.

    ``candidates`` are the run's legal reasoning targets.  They are passed as plain identities
    and projected into references by the context, exactly as the loop does, so a test exercises
    the real projection rather than hand-building reference objects.
    """
    from recommendation.control.context import CandidateReference

    return PolicyContext(
        user_request=request,
        available_actions=available,
        has_trusted_history=True,
        candidate_state=CandidateState(grounded=grounded, candidate_count=count),
        grounded_parent_asins=candidates,
        grounded_candidates=tuple(
            CandidateReference(parent_asin=identity) for identity in candidates
        ),
        last_observation=last_observation,  # type: ignore[arg-type]
        remaining_steps=remaining_steps,
        remaining_tool_calls=remaining_tool_calls,
        awaiting_user=awaiting_user,
    )


def policy_for(*answers: object, max_attempts: int = 2) -> tuple[LLMAgentPolicy, ScriptedModelClient]:
    """Build a policy over a scripted model, repeating the last answer."""
    client = ScriptedModelClient(list(answers), repeat_last=True)
    return LLMAgentPolicy(client, max_attempts=max_attempts), client


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self) -> None:
        self._ids = {row[0]: row[1] for row in CANDIDATE_ROWS}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


def loop_harness(
    *answers: object,
    rows: object = CANDIDATE_ROWS,
    engine_error: Exception | None = None,
    limits: LoopLimits | None = None,
    max_attempts: int = 2,
) -> tuple[object, LLMAgentPolicy, ScriptedModelClient, CandidateLedger]:
    """Compose a real control plane whose policy is model-driven.

    The candidate plane and the reasoning executor are wired with the fixture catalogue so a
    model-proposed ``SEARCH_CATALOG`` or ``COMPARE`` is executable rather than merely offered.
    """
    policy, client = policy_for(*answers, max_attempts=max_attempts)
    harness = build_control_harness(
        policy=policy,
        limits=limits or LoopLimits(max_steps=6, max_tool_calls=3),
        rows=rows,
        catalog_rows=CANDIDATE_ROWS,
        engine_error=engine_error,
    )
    metadata = harness.parts["enricher"].metadata
    ledger = CandidateLedger()
    harness.controller._reasoning = ReasoningExecutor(GroundedReasoner(metadata))  # noqa: SLF001
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=ledger,
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
    )
    return harness, policy, client, ledger


# =========================================================================== #
# A. Structured output
# =========================================================================== #


def test_a_valid_model_answer_becomes_a_valid_proposal() -> None:
    """The happy path: one JSON object in, one ActionProposal out."""
    policy, client = policy_for(
        {"action": "search_catalog", "arguments": {"terms": ["waterproof", "trail"], "limit": 4}}
    )
    proposal = policy.choose(context())
    assert isinstance(proposal, ActionProposal)
    assert proposal.action is ActionKind.SEARCH_CATALOG
    assert proposal.arguments.terms == ("waterproof", "trail")
    assert proposal.arguments.limit == 4
    assert client.call_count == 1


def test_a_model_response_object_is_accepted_as_well_as_a_mapping() -> None:
    """The seam accepts a ModelResponse from a real adapter, not only a dict."""
    policy, _ = policy_for(ModelResponse(text='{"action": "finish"}', model_id="real-adapter"))
    proposal = policy.choose(context())
    assert proposal.action is ActionKind.FINISH
    assert policy.last_metadata["model_id"] == "real-adapter"


def test_an_optional_rationale_is_recorded_but_not_authoritative() -> None:
    """A rationale is diagnostic text, truncated, and never a control signal."""
    policy, _ = policy_for({"action": "finish", "rationale": "x" * 500})
    proposal = policy.choose(context())
    assert proposal.rationale is not None
    assert len(proposal.rationale) <= 280
    assert len(policy.last_metadata["rationale"]) <= 200


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("please just recommend something", "not_json"),
        ("[1, 2, 3]", "not_a_json_object"),
        ('{"arguments": {"terms": ["x"]}}', "missing_action"),
        ('{"action": 7}', "missing_action"),
        ("", "empty_response"),
        ("   ", "empty_response"),
    ],
    ids=["prose", "array", "no_action", "non_string_action", "empty", "whitespace"],
)
def test_malformed_model_output_is_rejected_with_a_reason(answer: str, reason: str) -> None:
    """Every malformed class is refused, and the reason is named for the trajectory."""
    policy, client = policy_for(answer, max_attempts=2)
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    # An empty answer never reaches parsing, so ``parse_ok`` is legitimately absent there.
    assert policy.last_metadata.get("parse_ok", False) is False
    assert policy.last_metadata["last_failure"] == reason
    assert client.call_count == 2, "the bounded failure path is exercised"


def test_a_single_bad_answer_is_corrected_within_one_policy_call() -> None:
    """One formatting mistake should not lose the step: the policy asks again with a hint."""
    policy, client = policy_for(
        "not json", {"action": "finish", "rationale": "corrected"}
    )
    proposal = policy.choose(context())
    assert proposal.action is ActionKind.FINISH
    assert client.call_count == 2
    assert policy.last_metadata["recovered_after"] == ("not_json",)
    # The correction is sent to the model, so a retry is informed rather than blind.
    assert client.requests[-1].correction == "not_json"


# =========================================================================== #
# B. Authority
# =========================================================================== #


def test_an_unknown_action_is_rejected() -> None:
    """An action that does not exist is refused, not coerced."""
    policy, _ = policy_for({"action": "launch_missiles"})
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    assert policy.last_metadata["last_failure"].startswith("unknown_action")


def test_an_action_the_system_did_not_offer_is_rejected() -> None:
    """Availability belongs to the controller; the model cannot grant itself an action."""
    policy, _ = policy_for({"action": "compare", "arguments": {"parent_asins": ["a", "b"]}})
    with pytest.raises(PolicyActionError):
        policy.choose(context(available=(ActionKind.SEARCH_CATALOG, ActionKind.FINISH)))
    assert policy.last_metadata["last_failure"] == "action_not_offered:compare"


def test_invalid_arguments_are_rejected() -> None:
    """The action's own declared argument model is the contract, and it is enforced."""
    policy, _ = policy_for(
        {"action": "search_catalog", "arguments": {"query": "shoes"}}  # wrong field name
    )
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    assert policy.last_metadata["last_failure"] == "invalid_arguments:search_catalog"


def test_a_foreign_argument_cannot_be_smuggled_in() -> None:
    """A model cannot attach an argument belonging to another action."""
    policy, _ = policy_for(
        {"action": "search_catalog", "arguments": {"terms": ["x"], "sql": "DROP TABLE"}}
    )
    with pytest.raises(PolicyActionError):
        policy.choose(context())


def test_a_candidate_identity_cannot_be_smuggled_into_a_candidate_action() -> None:
    """The identity channel is closed: a candidate action accepts no product identity."""
    policy, _ = policy_for(
        {
            "action": "recommend_from_history",
            "k": 3,
            "arguments": {"parent_asins": ["FAKE-1"]},
        }
    )
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    assert policy.last_metadata["last_failure"] == "invalid_arguments:recommend_from_history"


def test_only_read_only_actions_may_name_products_at_all() -> None:
    """The structural rule, asserted against the real argument registry."""
    from recommendation.control import READ_ONLY_ACTIONS

    naming = {
        action
        for action, model in ARGUMENTS_BY_ACTION.items()
        if "parent_asins" in model.model_fields
    }
    assert naming, "the reasoning actions must be able to name products"
    assert naming <= set(READ_ONLY_ACTIONS)
    for action in (ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.SEARCH_CATALOG, ActionKind.FINISH):
        assert "parent_asins" not in ARGUMENTS_BY_ACTION[action].model_fields


def test_an_invented_identity_is_refused_rather_than_looked_up() -> None:
    """A loop-level attack: only trusted sources may create candidate identity.

    The model names a product that does not exist, in a read-only action where naming products
    *is* legal.  Phase 2.1 tightened what happens next.  Before, the executor asked the
    catalogue, found nothing, and reported the identity as ``unknown`` in an otherwise
    successful observation - which meant a run-relevant fact question was answered by looking
    up an identity no trusted source had ever produced.  Now the run-membership check runs
    first and refuses the whole action, so the identity never reaches the catalogue at all.
    """
    harness, _policy, _client, ledger = loop_harness(
        {"action": "recommend_from_history", "k": 2},
        {"action": "get_details", "arguments": {"parent_asins": ["FAKE-9", "cand-red"]}},
        {"action": "finish"},
    )
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001
    assert "FAKE-9" not in ledger
    assert "FAKE-9" not in ledger.grounded_parent_asins()

    # The action was refused, and the refusal is a bounded, payload-free failure observation.
    refusals = [
        step for step in result.trajectory.steps
        if step.observation and step.observation.get("status") == "failed"
    ]
    assert refusals, "an identity outside the run must produce a failed observation"
    assert refusals[0].observation["verification_status"] == "refused"
    # No detail observation was produced at all: the catalogue was never consulted, so the
    # valid member of the pair was not silently answered either.
    details = [
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "details"
    ]
    assert details == [], "a partially-authorized action must not be partially answered"


def test_the_policy_is_given_no_catalogue_or_history_channel() -> None:
    """What the policy client receives contains no history, no store and no catalogue."""
    policy, client = policy_for({"action": "finish"})
    policy.choose(context(request="I like red shoes. My history: B000000001"))
    request = client.last_request
    assert request is not None
    blob = str(request.context_payload) + request.system_prompt
    for forbidden in ("trusted_user_history", "user_key", "memory_id", "parent_asin"):
        assert forbidden not in blob
    # The request has no field through which any of them could travel.
    assert set(ModelRequest.model_fields) >= {"system_prompt", "context_payload", "action_schema"}
    for absent in ("history", "catalog", "ledger", "memory"):
        assert absent not in ModelRequest.model_fields


def test_the_prompt_lists_only_offered_actions_and_their_real_arguments() -> None:
    """The action schema is generated, so it cannot promise an action that does not exist."""
    schema = build_action_schema((ActionKind.SEARCH_CATALOG, ActionKind.FINISH))
    names = [entry["action"] for entry in schema]
    assert names == ["search_catalog", "finish"]
    search = next(entry for entry in schema if entry["action"] == "search_catalog")
    assert search["required_arguments"] == ["terms"]
    assert {arg["name"] for arg in search["arguments"]} == {"terms", "limit"}
    assert search["produces_candidates"] is True
    assert next(e for e in schema if e["action"] == "finish")["required_arguments"] == []


# =========================================================================== #
# C. Memory and FINISH boundaries
# =========================================================================== #


def test_the_policy_cannot_commit_persistent_memory() -> None:
    """A model proposal reaches memory only through the existing proposal validator."""
    policy, _ = policy_for({"action": "propose_memory_write", "arguments": {"value": "red"}})
    # ``propose_memory_write`` is argumentless in the contract, so smuggling a value is refused
    # before any validator sees it.
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    # And the policy holds no store with which it could commit anything.
    for forbidden in ("memory_service", "store", "commit", "user_key"):
        assert not hasattr(policy, forbidden)


def test_finish_is_only_a_proposal_and_the_policy_cannot_end_the_run() -> None:
    """A FINISH proposal carries no completion signal."""
    policy, _ = policy_for({"action": "finish"})
    proposal = policy.choose(context())
    assert proposal.action is ActionKind.FINISH
    for forbidden in ("finished", "status", "termination_reason", "completed"):
        assert forbidden not in ActionProposal.model_fields
    assert not hasattr(policy, "finish_run")


def test_a_refused_finish_lets_the_model_choose_again() -> None:
    """The boundary that matters: CompletionGuard refuses, and the run recovers.

    The history source fails, so the model's FINISH is refused with ``last_execution_failed``.
    The model reads that refusal in its next context and switches to a source that works.
    """
    harness, _policy, client, _ledger = loop_harness(
        {"action": "recommend_from_history", "k": 3},
        {"action": "finish", "rationale": "probably fine"},
        {"action": "search_catalog", "arguments": {"terms": ["redwidget"], "limit": 3}},
        {"action": "finish"},
        engine_error=RuntimeError("history source unavailable"),
    )
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001

    assert result.status is RunStatus.FINISHED
    assert result.trajectory.actions() == (
        "recommend_from_history",
        "finish",
        "search_catalog",
        "finish",
    )
    # Step 1's FINISH was refused by the guard, not by the model.
    refused = result.trajectory.steps[1].verification_result
    assert refused is not None and refused.verified is False
    assert refused.code == "last_execution_failed"
    # Step 3's FINISH was accepted.
    accepted = result.trajectory.steps[3].verification_result
    assert accepted is not None and accepted.verified is True
    assert accepted.code == "completion_accepted"
    assert client.call_count == 4, "one model call per policy decision"


# =========================================================================== #
# D. Observation-dependent replanning
# =========================================================================== #


def test_the_second_policy_call_receives_the_updated_context() -> None:
    """The core property: same task, different Observation, different model action.

    The scripted model answers ``recommend_from_history`` first and ``search_catalog`` second.
    The *only* thing that changed between the calls is the observation, so this proves the loop
    feeds observations back into the policy rather than replaying a fixed sequence.
    """
    harness, _policy, client, _ledger = loop_harness(
        {"action": "recommend_from_history", "k": 3},
        {"action": "search_catalog", "arguments": {"terms": ["redwidget"], "limit": 3}},
        {"action": "finish"},
        rows=(),  # the history source returns nothing
    )
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001

    assert result.trajectory.actions() == ("recommend_from_history", "search_catalog", "finish")
    assert client.call_count == 3

    # The first call saw no candidates and no observation.
    first = client.requests[0].context_payload
    assert first["last_observation"] is None
    assert first["candidates"]["grounded"] is False

    # The second call saw the empty result of the first - that is what changed its action.
    second = client.requests[1].context_payload
    assert second["last_observation"]["kind"] == "recommendation"
    assert second["last_observation"]["status"] == "empty"
    assert second["last_observation"]["has_candidates"] is False
    assert second["budget"]["remaining_steps"] < first["budget"]["remaining_steps"]


def test_a_grounded_candidate_set_changes_what_the_model_is_told() -> None:
    """The context payload reflects candidate state, so a policy can reason about next steps."""
    policy, client = policy_for({"action": "finish"}, {"action": "finish"})
    policy.choose(context(grounded=True, count=4, remaining_steps=2, remaining_tool_calls=1))
    payload = client.last_request.context_payload
    assert payload["candidates"] == {
        "grounded": True,
        "count": 4,
        "verification_status": "unverified",
        # Phase 2.1: the bounded reference list travels with the counts.  Empty here, because
        # this context offers no legal target - the counts and the refs are independent.
        "candidate_refs": [],
    }
    assert payload["budget"]["remaining_steps"] == 2
    assert payload["budget"]["remaining_tool_calls"] == 1


def test_the_payload_exposes_summaries_not_payloads() -> None:
    """A reasoning observation is summarised by kind and status, not by its facts."""
    from recommendation.control import DetailObservation

    observation = DetailObservation(
        action_id="a",
        step_index=0,
        action=ActionKind.GET_DETAILS,
        facts={"cand-red": {"title": "RedWidget", "weight_text": "0.800 kg"}},
        requested=("cand-red",),
    )
    payload = build_policy_context_payload(context(last_observation=observation))
    blob = str(payload)
    assert payload["last_observation"]["kind"] == "details"
    # The catalogue facts the model was shown the result of are NOT echoed into the prompt.
    assert "RedWidget" not in blob
    assert "0.800" not in blob
    assert "facts" not in payload["last_observation"]


# =========================================================================== #
# E. Bounded failure
# =========================================================================== #


def test_a_model_that_never_answers_properly_terminates_within_budget() -> None:
    """Repeated malformed output cannot loop: the policy aborts, and the run ends."""
    harness, _policy, client, _ledger = loop_harness("definitely not json", max_attempts=2)
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001

    assert result.status is RunStatus.ABORTED
    assert result.control.termination_reason is TerminationReason.NO_AVAILABLE_ACTION
    # Two attempts in one policy call, then the abort: bounded, not a retry storm.
    assert client.call_count == 2
    assert result.control.step_count == 0


def test_a_provider_failure_is_normalised_into_the_bounded_path() -> None:
    """A transport failure aborts the run cleanly rather than crashing the loop."""
    harness, _policy, client, _ledger = loop_harness(RuntimeError("provider exploded"))
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001

    assert result.status is RunStatus.ABORTED
    assert result.control.termination_reason is TerminationReason.NO_AVAILABLE_ACTION
    assert client.call_count == 1
    assert result.trajectory.steps[0].policy_metadata["last_failure"] == "provider_error"


def test_a_model_client_that_raises_model_call_error_is_reported_by_code() -> None:
    """The client seam distinguishes a transport failure from an unusable answer."""
    policy, _ = policy_for("x")
    # A scripted Exception entry is raised as ModelCallError by the client.
    client = ScriptedModelClient([ValueError("nope")])
    policy = LLMAgentPolicy(client)
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    assert policy.last_metadata["last_failure"] == "provider_error"


def test_max_attempts_bounds_one_decision_and_is_validated() -> None:
    """The per-decision attempt budget is configuration, and it is checked."""
    with pytest.raises(PolicyActionError):
        LLMAgentPolicy(ScriptedModelClient([{"action": "finish"}]), max_attempts=0)
    policy, client = policy_for("bad", max_attempts=3)
    with pytest.raises(PolicyActionError):
        policy.choose(context())
    assert client.call_count == 3


def test_an_empty_offered_menu_is_refused_without_calling_the_model() -> None:
    """No legal action means no question to ask, so the model is not consulted."""
    policy, client = policy_for({"action": "finish"})
    with pytest.raises(PolicyActionError):
        policy.choose(context(available=()))
    assert client.call_count == 0


# =========================================================================== #
# F. Provider independence
# =========================================================================== #


def test_a_scripted_client_satisfies_the_structured_model_protocol() -> None:
    """The test double is a real implementation of the seam, not a special case."""
    assert isinstance(ScriptedModelClient([{"action": "finish"}]), StructuredModelClient)


def test_a_bad_model_object_is_rejected_at_construction() -> None:
    """A collaborator without the seam method is a configuration error."""
    for bad in (object(), None, type("NoComplete", (), {})()):
        with pytest.raises(PolicyActionError):
            LLMAgentPolicy(bad)  # type: ignore[arg-type]


def test_no_provider_sdk_is_imported_by_the_control_plane() -> None:
    """The dependency policy holds: the core imports no LLM SDK."""
    for module in (
        "model_client.py",
        "model_policy.py",
    ):
        source = (REPO_ROOT / "recommendation" / "control" / module).read_text(encoding="utf-8")
        for forbidden in ("import openai", "import anthropic", "google.generativeai", "litellm"):
            assert forbidden not in source, f"{module} must not import a provider SDK"


def test_the_policy_identity_is_recorded_in_the_trajectory() -> None:
    """Observability: a trajectory says a model decided, and how it went."""
    harness, _policy, _client, _ledger = loop_harness(
        {"action": "recommend_from_history", "k": 3},
        {"action": "finish"},
    )
    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001
    for step in result.trajectory.steps:
        metadata = step.policy_metadata
        assert metadata["policy"] == LLM_POLICY_NAME
        assert metadata["parse_ok"] is True
        assert metadata["model_calls"] >= 1
        assert metadata["proposed_action"]
    # And the trajectory stays payload-free: no prompt, no chain-of-thought.
    blob = str([step.policy_metadata for step in result.trajectory.steps])
    assert "system_prompt" not in blob
    assert build_policy_system_prompt()[:20] not in blob


# =========================================================================== #
# G. Integration
# ===========================================================================


class _TargetSelectingModel:
    """A model double that chooses its reasoning *targets* from the offered references.

    Phase 2.1's whole point is that the model, not a binder, picks which grounded candidate a
    reasoning action names.  A fixed script cannot demonstrate that - it encodes the answer
    ahead of time and would keep "working" if the binder overrode it.  This double instead
    reads ``candidates.candidate_refs`` from each request and builds its arguments from what it
    was actually offered, so the executed action proves the model's own selection reached the
    executor.
    """

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.selected: list[tuple[str, ...]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: Any) -> Any:
        import json

        from recommendation.control.model_client import ModelResponse

        self.requests.append(request)
        offered = {entry["action"] for entry in request.action_schema}
        refs = tuple(
            str(entry.get("parent_asin", ""))
            for entry in request.context_payload.get("candidates", {}).get("candidate_refs", [])
            if entry.get("parent_asin")
        )
        kind = (request.context_payload.get("last_observation") or {}).get("kind")

        if not refs:
            action: dict[str, Any] = {
                "action": "search_catalog",
                # Each fixture title is one lowercase token, so one query grounds one candidate.
                "arguments": {"terms": ["redwidget"], "limit": 3},
                "rationale": "nothing grounded yet",
            }
        elif kind != "details":
            # Read the candidate the model itself chooses first - not a target chosen for it.
            self.selected.append((refs[0],))
            action = {
                "action": "get_details",
                "arguments": {"parent_asins": [refs[0]]},
                "rationale": f"read the grounded facts for {refs[0]}",
            }
        else:
            action = {"action": "finish", "rationale": "the candidate is answered"}
        return ModelResponse(text=json.dumps(action, sort_keys=True), model_id="target-selecting")


def test_full_model_driven_trajectory_offline() -> None:
    """The end-to-end trajectory: retrieve, let the model pick a target, reason, finish.

    A model chooses every action *and* every reasoning target from its own context; the trusted
    pipeline executes them and the completion guard authorises the end. No network, no API key,
    no real model.
    """
    model = _TargetSelectingModel()
    policy = LLMAgentPolicy(model)
    harness = build_control_harness(
        policy=policy,
        limits=LoopLimits(max_steps=6, max_tool_calls=3),
        rows=CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
    )
    metadata = harness.parts["enricher"].metadata
    ledger = CandidateLedger()
    harness.controller._reasoning = ReasoningExecutor(GroundedReasoner(metadata))  # noqa: SLF001
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=ledger,
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
    )

    result = harness.controller.run("Recommend gear.", ("B1", "B2", "B3"), run_id="r")  # noqa: SLF001

    assert result.status is RunStatus.FINISHED
    assert result.control.termination_reason is TerminationReason.COMPLETED
    assert result.trajectory.actions() == (
        "search_catalog",
        "get_details",
        "finish",
    )
    assert result.route == "recommend"
    assert model.call_count == 3

    # Every action was chosen from the model's own context, one decision per step.
    for step in result.trajectory.steps:
        assert step.policy_metadata["policy"] == LLM_POLICY_NAME
    # Only the retrieval consumes a tool call; reasoning reads facts and costs none.
    assert result.control.tool_call_count == 1
    # The target the model selected was the one it was offered - and the executor accepted it
    # because it is a grounded candidate of this run.
    assert model.selected, "the model chose a reasoning target"
    grounded = set(ledger.grounded_parent_asins())
    for chosen in model.selected:
        assert set(chosen) <= grounded, "the model targeted only offered candidates"
    assert result.control.step_count == 3
    # And the response names the source that actually produced the candidates.
    assert "catalogue text search" in result.final_response


def test_the_model_policy_and_a_deterministic_policy_are_interchangeable() -> None:
    """Both policies drive the same controller with no runtime switch or special case."""
    from recommendation.control import RuleBasedPolicy

    model_harness, _policy, _client, _ledger = loop_harness(
        {"action": "recommend_from_history", "k": 4},
        {"action": "finish"},
    )
    model_result = model_harness.controller.run(  # noqa: SLF001
        "Recommend gear.", ("B1", "B2", "B3"), run_id="r"
    )

    deterministic = build_control_harness(
        policy=RuleBasedPolicy(default_k=4),
        limits=LoopLimits(max_steps=6, max_tool_calls=3),
        rows=CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
    )
    deterministic_result = deterministic.controller.run(
        "Recommend gear.", ("B1", "B2", "B3"), run_id="r"
    )

    # Same controller, same executor, same action sequence, same response.
    assert model_result.trajectory.actions() == deterministic_result.trajectory.actions()
    assert model_result.route == deterministic_result.route
    assert model_result.final_response == deterministic_result.final_response
    # The trajectories differ only in which policy is recorded as having decided.  Every step is
    # attributable - an unnamed policy would make a run's provenance unverifiable - but only the
    # model policy reports how it decided.  Neither record is a decision input.
    assert all(s.policy_metadata.get("policy") == LLM_POLICY_NAME for s in model_result.trajectory.steps)
    assert all(
        s.policy_metadata.get("policy") == RuleBasedPolicy().name
        for s in deterministic_result.trajectory.steps
    )
    assert all(s.policy_metadata.get("model_calls") is not None for s in model_result.trajectory.steps)
    assert all("model_calls" not in s.policy_metadata for s in deterministic_result.trajectory.steps)
