"""E-01A-2: an LLM-selected plan reaches the service's own controller, isolated from legacy memory.

Two verified layers here:

1. the policy's query isolation (``use_memory=False`` for LLM-selected runs, legacy default kept);
2. the service entry point that turns a **validated** selection into an executed plan - it must
   ignore the request body's ``sources``, re-check availability at execution time, and refuse a
   direct selection.

The full tool-execution leg (real ``CandidatePlane`` + ``CandidateLedger`` with doubles) is **not**
covered by this file; see the E-01A-2 report, which states that gap explicitly.
"""

from __future__ import annotations

from typing import Any

import pytest

from recommendation.control import ActionKind
from recommendation.control.context import PolicyContext
from recommendation.demo.agent_policy import DemoSourcePlanPolicy
from recommendation.demo.agent_schemas import AgentRecommendRequest
from recommendation.demo.agent_service import DemoAgentError, DemoAgentService
from recommendation.demo.tool_selection import SourceSelection
from recommendation.memory.query_adapter import project_active_preferences
from recommendation.memory.schemas import PreferenceMemoryEntry, PreferenceMemorySnapshot


def entry(value: str) -> PreferenceMemoryEntry:
    return PreferenceMemoryEntry(
        memory_id="m1", user_key="u", kind="category", value=value, polarity="prefer",
        source_text="the user said so", source_turn_id="t1", logical_seq=1, created_at=1.0,
    )


def context(*, request: str, snapshot: PreferenceMemorySnapshot | None) -> PolicyContext:
    """The **real** policy context the controller builds.

    ``choose()`` reads ``step_index`` and calls ``action_available(FINISH)``, so a duck-typed
    stand-in is not a faithful context - the first version of this test proved that by failing
    with ``AttributeError: 'types.SimpleNamespace' object has no attribute 'step_index'``.

    The preference field carries the same value-level **projection** the controller builds
    (``project_active_preferences``), not the M9 snapshot: the policy must be exercised against
    what production actually hands it, or this test would prove a boundary that does not exist.
    """
    return PolicyContext(
        user_request=request,
        available_actions=(ActionKind.SEARCH_CATALOG, ActionKind.FINISH),
        has_trusted_history=True,
        active_preferences=project_active_preferences(snapshot),
        step_index=0,
    )


def search_terms(policy: DemoSourcePlanPolicy, ctx) -> tuple[str, ...]:
    proposal = policy.choose(ctx)
    assert proposal is not None and proposal.action.value == "search_catalog", proposal
    return tuple(proposal.arguments.terms)


def test_an_llm_selected_policy_does_not_append_stored_preferences() -> None:
    snapshot = PreferenceMemorySnapshot(
        user_key="u", entries=(entry("trail running shoes"),)
    )
    isolated = DemoSourcePlanPolicy(("catalog_search",), use_memory=False)
    terms = search_terms(isolated, context(request="basketball", snapshot=snapshot))
    assert terms == ("basketball",), "the current request must not be diluted by history"


def test_the_legacy_policy_still_uses_memory_by_default() -> None:
    snapshot = PreferenceMemorySnapshot(
        user_key="u", entries=(entry("trail running shoes"),)
    )
    legacy = DemoSourcePlanPolicy(("catalog_search",))  # default: use_memory=True
    terms = search_terms(legacy, context(request="basketball", snapshot=snapshot))
    assert terms == ("basketball", "trail running shoes"), "M-02A behaviour is unchanged"


def test_the_switch_only_affects_the_search_arguments() -> None:
    """Non-search actions are unaffected: the same plan proposes the same first source.

    The context offers an action per planned source (derived from ``SOURCE_PLANS``, so no action
    name is hard-coded here) plus FINISH.  If the switch leaked outside the search branch, the
    first proposal would change - it must not.
    """
    from recommendation.demo.agent_policy import SOURCE_PLANS

    plan = ("history", "two_tower", "catalog_search")
    snapshot = PreferenceMemorySnapshot(user_key="u", entries=(entry("trail running shoes"),))
    offered = tuple(SOURCE_PLANS[source] for source in plan) + (ActionKind.FINISH,)

    def first_proposal(use_memory: bool):
        policy = DemoSourcePlanPolicy(plan, use_memory=use_memory)
        ctx = PolicyContext(
            user_request="find me a tent",
            available_actions=offered,
            has_trusted_history=True,
            active_preferences=project_active_preferences(snapshot),
            step_index=0,
        )
        return policy.choose(ctx)

    isolated = first_proposal(False)
    legacy = first_proposal(True)
    assert isolated.action is legacy.action is SOURCE_PLANS["history"], (
        "the memory switch must not change which non-search action is proposed"
    )


# ---------------------------------------------------------------- service entry point


class _StubService(DemoAgentService):
    """Executes the real ``run_selection`` logic with the heavy collaborators replaced."""

    def __init__(self, available: tuple[str, ...]) -> None:
        self._available = available
        self.executed: list[tuple[tuple[str, ...], bool]] = []
        self.snapshots: list[object] = []
        self.replan_clients: list[object] = []

    @property
    def available_sources(self) -> tuple[str, ...]:  # type: ignore[override]
        return self._available

    def _execute_plan(  # type: ignore[override]
        self, *, plan, payload, use_memory, preference_snapshot=None, replan_client=None
    ):
        # ``preference_snapshot`` (E-02A) is the caller's trusted memory read, which this stub
        # never supplies; it is accepted so the double keeps mirroring the real entry point, and it
        # is recorded so a test can assert that a selection carries no memory of its own.
        # ``replan_client`` (E-03B) is the same: the caller's model client, which a selection
        # without bounded replanning simply does not pass.
        self.executed.append((tuple(plan), bool(use_memory)))
        self.snapshots.append(preference_snapshot)
        self.replan_clients.append(replan_client)
        return {"plan": tuple(plan), "use_memory": use_memory, "payload_sources": payload.sources}


def payload(*sources: str) -> AgentRecommendRequest:
    return AgentRecommendRequest(
        message="find me a tent", history=["B0BX5QFWQN", "B0BKFGDBBM"], k=5, sources=list(sources)
    )


def test_a_selected_plan_is_executed_with_memory_isolation() -> None:
    service = _StubService(("history", "catalog_search"))
    selection = SourceSelection(route="recommend", selected_sources=("catalog_search",))
    result = service.run_selection(selection, payload("history"))
    assert service.executed == [(("catalog_search",), False)], (
        "the LLM's selection runs, not the request body's sources, and with memory isolated"
    )
    assert service.snapshots == [None], (
        "a selection carries no memory of its own: the ranking snapshot is the caller's to supply"
    )
    # ``AgentRecommendRequest.sources`` is normalised to a tuple by the schema.
    assert tuple(result["payload_sources"]) == ("history",), "the body is never rewritten"


def test_a_multi_source_selection_keeps_its_order() -> None:
    service = _StubService(("history", "catalog_search"))
    selection = SourceSelection(route="recommend", selected_sources=("catalog_search", "history"))
    service.run_selection(selection, payload())
    assert service.executed == [(("catalog_search", "history"), False)]


def test_a_source_that_became_unavailable_is_refused_before_execution() -> None:
    service = _StubService(("history",))  # TIGER never registered/ready here
    selection = SourceSelection(route="recommend", selected_sources=("tiger",))
    with pytest.raises(DemoAgentError) as excinfo:
        service.run_selection(selection, payload())
    assert excinfo.value.code == "unsupported_source_plan"
    assert service.executed == [], "no tool may run when the plan is not available"


def test_a_direct_selection_never_starts_a_run() -> None:
    service = _StubService(("history",))
    with pytest.raises(DemoAgentError) as excinfo:
        service.run_selection(SourceSelection(route="direct"), payload())
    assert excinfo.value.code == "not_a_recommendation"
    assert service.executed == []


def test_an_empty_selection_cannot_be_normalised_into_a_plan() -> None:
    service = _StubService(("history",))
    with pytest.raises(DemoAgentError) as excinfo:
        service.run_selection(SourceSelection(route="recommend", selected_sources=()), payload())
    assert excinfo.value.code == "unsupported_source_plan"


# --------------------------------------------------------------------------- #
# Real Controller -> ActionValidator -> CandidatePlane -> CandidateLedger execution
#
# The collaborators come from the accepted control fixture (a FixedEngine plus the real tool,
# enricher, matcher and reranker); the Controller, plane, validator, grounding and ledger are the
# production implementations.  No recommendation model is loaded and no provider is called.
# --------------------------------------------------------------------------- #


def _identity_engine(engine: Any) -> Any:
    """Give the fixture engine the trusted identity surface the plane's grounding requires.

    ``GroundingVerifier`` needs ``has_parent_asin`` / ``parent_asin_to_item_id`` /
    ``item_id_to_parent_asin`` (see ``control/grounding.py:163``).  Production engines provide
    them; the lightweight ``FixedEngine`` does not, so the test supplies a consistent bijection
    built from the same rows the engine recommends from.  Everything else delegates unchanged.
    """
    from tests.control_fixture import CANDIDATE_ROWS

    pairs: dict[str, int] = {}
    for row in CANDIDATE_ROWS:
        text = next((v for v in _row_values(row) if isinstance(v, str) and v), None)
        number = next((v for v in _row_values(row) if isinstance(v, int) and not isinstance(v, bool)), None)
        if text is not None and number is not None:
            pairs.setdefault(text, number)
    assert pairs, "the fixture rows must expose an identity and an item id"

    class _IdentityEngine:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def has_parent_asin(self, value: str) -> bool:
            return str(value) in pairs

        def parent_asin_to_item_id(self, value: str) -> int:
            return pairs[str(value)]

        def item_id_to_parent_asin(self, item_id: int) -> str:
            return {number: text for text, number in pairs.items()}[int(item_id)]

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _IdentityEngine(engine)


def _steps(response: Any) -> list[dict[str, Any]]:
    """The executed tool calls, as plain dicts (whatever the schema's field naming is)."""
    trajectory = getattr(response, "trajectory", None)
    steps = (getattr(trajectory, "tool_calls", None)
             or getattr(trajectory, "steps", None)
             or getattr(response, "tool_calls", None)
             or ())
    out: list[dict[str, Any]] = []
    for step in steps:
        if hasattr(step, "model_dump"):
            out.append(step.model_dump())
        elif isinstance(step, dict):
            out.append(step)
        else:
            out.append({"action": getattr(step, "action", None)})
    return out


def _row_values(row: Any) -> list[Any]:
    if isinstance(row, dict):
        return list(row.values())
    if isinstance(row, (tuple, list)):
        return list(row)
    values = [getattr(row, name, None) for name in dir(row) if not name.startswith("_")]
    values += list(getattr(row, "__dict__", {}).values())
    return values


def _service(*, engine_error: Exception | None = None):
    from tests.control_fixture import build_full_capability

    capability, engine, parts = build_full_capability(engine_error=engine_error)
    engine = _identity_engine(engine)
    service = DemoAgentService(
        tool=parts["tool"],
        enricher=parts["enricher"],
        metadata=parts["enricher"].metadata,
        matcher=parts["matcher"],
        reranker=parts["reranker"],
        engine=engine,
    )
    return service, parts, engine


def _row_identities(row: Any) -> list[str]:
    """Every string field a fixture row exposes, whatever its shape (dict, tuple or object)."""
    return [value for value in _row_values(row) if isinstance(value, str) and value]


def _fixture_history() -> list[str]:
    from tests.control_fixture import CANDIDATE_ROWS

    identities: list[str] = []
    for row in CANDIDATE_ROWS:
        for value in _row_identities(row):
            if value not in identities:
                identities.append(value)
                break
    assert identities, f"the fixture must expose candidate identities, got {CANDIDATE_ROWS[:1]!r}"
    return identities[:3]


def _all_fixture_identities() -> list[str]:
    from tests.control_fixture import CANDIDATE_ROWS

    values: list[str] = []
    for row in CANDIDATE_ROWS:
        for value in _row_identities(row):
            if value not in values:
                values.append(value)
    return values


def payload_for(message: str, *sources: str) -> AgentRecommendRequest:
    return AgentRecommendRequest(message=message, history=_fixture_history(), k=3,
                                 sources=list(sources) or ["history"])


def test_the_request_body_cannot_override_the_llm_selection() -> None:
    service, _parts, _engine = _service()
    selection = SourceSelection(route="recommend", selected_sources=("catalog_search",))
    response = service.run_selection(selection, payload_for("waterproof jacket", "history"))
    actions = [step.get("action") for step in _steps(response)]
    # The selected source runs through its own capability action (``search_catalog`` for the BM25
    # tool); the request body asked for ``history`` instead and must not be what ran.
    assert "search_catalog" in actions, actions
    assert "recommend_from_history" not in actions, (
        f"the request body's source must not be consulted: {actions}"
    )
    # Whether the run succeeds depends on whether the catalogue matches the query; what must hold
    # either way is that a non-success never carries fabricated candidates.
    if response.terminal.succeeded:
        assert response.recommendations, "a successful run must return real candidates"
    else:
        assert response.recommendations == (), (
            f"a refused completion must not present candidates: {response.terminal}"
        )


def test_a_two_source_selection_runs_both_sources() -> None:
    service, _parts, _engine = _service()
    selection = SourceSelection(route="recommend", selected_sources=("history", "catalog_search"))
    response = service.run_selection(selection, payload_for("waterproof jacket and tent"))
    actions = [step.get("action") for step in _steps(response)]
    # Evidence from the real run: the two selected sources execute through their own capability
    # actions, in plan order, and the run closes with a guarded ``finish``.
    source_actions = [name for name in actions
                      if name in ("recommend_from_history", "search_catalog", "select_source",
                                  "find_similar")]
    assert "recommend_from_history" in source_actions, actions
    assert "search_catalog" in source_actions, actions
    assert actions[-1] == "finish", actions
    assert response.terminal.succeeded is True
    assert response.recommendations, "the fused run must return real candidates"


def test_a_failing_tool_produces_no_fabricated_recommendations() -> None:
    service, _parts, _engine = _service(engine_error=RuntimeError("engine exploded"))
    selection = SourceSelection(route="recommend", selected_sources=("history",))
    response = service.run_selection(selection, payload_for("recommend gear"))
    assert response.recommendations == (), "a failed tool must not yield candidates"
    assert response.terminal.succeeded is False


def test_the_legacy_run_entry_still_works() -> None:
    """Regression for the defect the execution tests surfaced: ``_build_controller`` referenced a
    ``use_memory`` it did not receive, which broke the legacy ``run()`` path as well."""
    service, _parts, _engine = _service()
    response = service.run(payload_for("recommend gear", "history"))
    assert response.terminal.succeeded is True
    assert response.recommendations, "the legacy path must still return real candidates"
