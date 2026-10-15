"""E-01A-STATE-FIX: one request's controller state must not reach the next request.

The bug this file pins: ``DemoSourcePlanPolicy`` remembered the sources it had already proposed in
a mutable set that survived from one run to the next, and ``DemoAgentService`` cached the compiled
controller - so one policy, one candidate plane and one ``CandidateLedger`` served many requests.
A second request then behaved differently from a first one with identical input: once its first
source had grounded candidates, the next planned source was judged "already served" and skipped,
and its grounded block reported candidates retrieved by an earlier request.

Every test here runs the **real** composition - ``DemoAgentService`` (the production entry point),
``LoopController``, ``ActionValidator``, ``CandidatePlane``, ``GroundingVerifier``,
``CandidateLedger`` and the real ``DemoSourcePlanPolicy`` - over the light deterministic fixture
tools from ``tests.control_fixture``.  Nothing about the agent path is mocked, and the assertions
are on executed actions, their counts, the terminal state and the presented candidates - never on
a private field alone.
"""

from __future__ import annotations

import threading
from typing import Any

from recommendation.control import LoopLimits, RecommendFromHistoryCapability
from recommendation.demo.agent_policy import DemoSourcePlanPolicy
from recommendation.demo.agent_schemas import AgentRecommendRequest
from recommendation.demo.agent_service import DemoAgentService
from recommendation.demo.tool_selection import SourceSelection
from tests.agent_reranking_fixture import CANDIDATE_ROWS, INITIAL_ORDER
from tests.control_fixture import build_full_capability

#: A trusted history the fixture catalogue does **not** contain, so the history source returns the
#: fixture candidates unmasked (the accepted Tool masks candidates the history already consumed).
#: Using the candidate identities *as* the history would mask every candidate the Tool is about to
#: return, which is a different scenario: the run would have nothing grounded and the plan could
#: not advance at all.  These are the same shape of unknown identifier the demo's own session
#: profiles use, so this is the ordinary path, not a special case.
HISTORY = ["B0000000011", "B0000000012", "B0000000013"]

#: The tool budget every service in this file is built with, so a run's retries stay comparable.
TOOL_BUDGET = 4


class _IdentityEngine:
    """The fixture engine plus the trusted identity surface the grounding verifier requires.

    ``GroundingVerifier`` refuses an identity map that cannot answer, and the plane grounds every
    candidate through it, so the double has to provide the same bijection the real serving engine
    does.  Everything else delegates unchanged.
    """

    def __init__(self, inner: Any, rows: Any = CANDIDATE_ROWS) -> None:
        self._inner = inner
        self._by_asin = {str(row[0]): int(row[1]) for row in rows}
        self._by_id = {int(row[1]): str(row[0]) for row in rows}

    def has_parent_asin(self, value: str) -> bool:
        return str(value) in self._by_asin

    def parent_asin_to_item_id(self, value: str) -> int:
        return self._by_asin[str(value)]

    def item_id_to_parent_asin(self, value: int) -> str:
        return self._by_id.get(int(value), str(value))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _build(*, limits: LoopLimits | None = None) -> tuple[DemoAgentService, Any]:
    """The production service plus the fixture engine, so a test can break retrieval on purpose.

    The engine's ``error`` attribute is the fixture's own public switch: setting it makes the next
    ``recommend()`` raise, which is how the failure paths below are produced without mocking the
    agent stack.
    """
    _capability, engine, parts = build_full_capability()
    service = DemoAgentService(
        tool=parts["tool"],
        enricher=parts["enricher"],
        metadata=parts["enricher"].metadata,
        matcher=parts["matcher"],
        reranker=parts["reranker"],
        engine=_IdentityEngine(engine),
        limits=limits or LoopLimits(max_steps=6, max_tool_calls=TOOL_BUDGET, max_retries=1),
    )
    return service, engine


def _service(*, limits: LoopLimits | None = None) -> DemoAgentService:
    return _build(limits=limits)[0]


def _payload(*sources: str, k: int = 3, message: str = "find me a RedWidget") -> AgentRecommendRequest:
    return AgentRecommendRequest(
        message=message, history=HISTORY, k=k, sources=list(sources) or ["history"]
    )


def _selection(*sources: str, k: int = 3) -> SourceSelection:
    return SourceSelection(route="recommend", selected_sources=tuple(sources) or ("history",))


def _steps(response: Any) -> list[dict[str, Any]]:
    """The run's executed steps, as plain dicts."""
    steps = getattr(response.trajectory, "steps", ()) or ()
    return [step.model_dump() for step in steps]


def _actions(response: Any) -> list[str]:
    return [step["action"] for step in _steps(response)]


def _executed_counts(response: Any) -> dict[str, int]:
    """Executed actions and how often, counted from the trajectory (validated steps only)."""
    counts: dict[str, int] = {}
    for step in _steps(response):
        if step["validated"] and step["action"]:
            counts[step["action"]] = counts.get(step["action"], 0) + 1
    return counts


def _presented(response: Any) -> list[str]:
    return [str(item.parent_asin) for item in (response.recommendations or ())]


def _grounded_identities(response: Any) -> list[str]:
    return [str(candidate.parent_asin) for candidate in (response.grounded.candidates or ())]


# --------------------------------------------------------------------------- #
# A. the same service, the same plan, run twice
# --------------------------------------------------------------------------- #


def test_the_same_request_twice_executes_the_same_sources_both_times() -> None:
    """The regression: run 2 must consult the planned catalog source just as run 1 did."""
    service = _service()
    payload = _payload("history", "catalog_search")

    first = service.run(payload)
    second = service.run(payload)

    for label, response in (("first", first), ("second", second)):
        counts = _executed_counts(response)
        assert counts == {"recommend_from_history": 1, "search_catalog": 1, "finish": 1}, (
            f"the {label} run must consult both planned sources exactly once, got {counts}"
        )
        assert "select_source" not in counts, "an unplanned source must not be consulted"

    assert _presented(first) == _presented(second), (
        f"identical input must present the same candidates: {_presented(first)} vs "
        f"{_presented(second)}"
    )
    assert first.route == second.route == "recommend"
    assert first.terminal.succeeded is True and second.terminal.succeeded is True


def test_a_second_request_reports_only_its_own_grounded_candidates() -> None:
    """Provenance is per request: run 2 must not report what run 1 retrieved.

    The fixture catalogue matches ``RedWidget`` to one identity and ``BlueWidget`` to another, so a
    ledger that outlived its request is directly observable in the ``grounded`` block.
    """
    service = _service()
    selection = _selection("catalog_search")

    red = service.run_selection(selection, _payload(message="RedWidget"))
    blue = service.run_selection(selection, _payload(message="BlueWidget"))

    assert _presented(red) == ["cand-red"], _presented(red)
    assert _presented(blue) == ["cand-blue"], _presented(blue)
    assert _grounded_identities(red) == ["cand-red"]
    assert _grounded_identities(blue) == ["cand-blue"], (
        "the second request must not report the first request's candidate as its own"
    )
    assert blue.grounded.grounded_count == 1


def test_a_failed_run_does_not_change_the_next_run() -> None:
    """A failed run leaves nothing behind: the next identical run is a normal successful run."""
    service, engine = _build()
    payload = _payload("history", "catalog_search")
    healthy = service.run(payload)

    engine.error = RuntimeError("engine exploded")
    try:
        broken = service.run(payload)
    finally:
        engine.error = None

    assert broken.terminal.succeeded is False
    assert _presented(broken) == [], "a failed run must not invent candidates"

    recovered = service.run(payload)
    assert recovered.terminal.succeeded is True
    assert _executed_counts(recovered) == _executed_counts(healthy), (
        "the run after a failure must execute exactly what the healthy run did"
    )
    assert _presented(recovered) == _presented(healthy)


# --------------------------------------------------------------------------- #
# B. alternating plans through one service
# --------------------------------------------------------------------------- #


def test_alternating_plans_do_not_contaminate_each_other() -> None:
    """Each request consults exactly its own plan - no more, no less - in any order."""
    service = _service()
    expected = {
        "catalog_only": {"search_catalog": 1, "finish": 1},
        "history_only": {"recommend_from_history": 1, "finish": 1},
        "both": {"recommend_from_history": 1, "search_catalog": 1, "finish": 1},
    }
    sequence = [
        ("catalog_only", _payload("catalog_search", message="RedWidget")),
        ("both", _payload("history", "catalog_search")),
        ("history_only", _payload("history")),
        ("both", _payload("history", "catalog_search")),
        ("catalog_only", _payload("catalog_search", message="RedWidget")),
        ("history_only", _payload("history")),
    ]
    seen: dict[str, list[str]] = {name: [] for name in expected}
    for index, (name, payload) in enumerate(sequence):
        response = service.run(payload)
        counts = _executed_counts(response)
        assert counts == expected[name], f"request {index} ({name}) executed {counts}"

        # The catalog-only request is the one that reveals a leaked "already served" set: with a
        # leaked plan the policy skips the source and the run refuses instead of recommending.
        if name == "catalog_only":
            assert response.route == "recommend" and _presented(response) == ["cand-red"]
        if name == "history_only":
            assert response.route == "recommend"
            assert _presented(response) == list(INITIAL_ORDER[:3])
        seen[name].append(",".join(_presented(response)))

    for name, results in seen.items():
        assert len(set(results)) == 1, f"{name} presented different candidates per run: {results}"


# --------------------------------------------------------------------------- #
# C. both entry points (legacy request-declared plan, and the llm_agent selection path)
# --------------------------------------------------------------------------- #


def test_the_legacy_entry_point_is_stable_across_requests() -> None:
    service = _service()
    payload = _payload("catalog_search", message="RedWidget")
    responses = [service.run(payload) for _ in range(3)]
    assert [_presented(response) for response in responses] == [["cand-red"]] * 3
    assert [_executed_counts(response) for response in responses] == [
        {"search_catalog": 1, "finish": 1}] * 3


def test_the_llm_selected_entry_point_is_stable_across_requests() -> None:
    """The new path shares the service, so it must be isolated from itself *and* from the legacy one."""
    service = _service()
    selection = _selection("history", "catalog_search")

    legacy = service.run(_payload("history", "catalog_search"))
    first = service.run_selection(selection, _payload())
    second = service.run_selection(selection, _payload())

    for label, response in (("selected #1", first), ("selected #2", second)):
        assert _executed_counts(response) == {
            "recommend_from_history": 1, "search_catalog": 1, "finish": 1}, label
    assert _presented(first) == _presented(second) == _presented(legacy)
    assert service.controllers_built == 3, (
        "one controller per request: a selected run must never reuse the legacy request's"
    )


# --------------------------------------------------------------------------- #
# D. the trajectory is the evidence, request after request
# --------------------------------------------------------------------------- #


def test_each_run_reports_its_own_actions_and_terminal_state() -> None:
    service = _service()
    first = service.run(_payload("history", "catalog_search"))
    second = service.run(_payload("history", "catalog_search"))

    assert _actions(first) == ["recommend_from_history", "search_catalog", "finish"]
    assert _actions(second) == _actions(first), (
        "the second run's trajectory must show both sources, not an early finish"
    )
    for response in (first, second):
        assert response.terminal.termination_reason == "completed"
        assert response.terminal.tool_calls == 2
        assert response.trajectory.step_count == 3
        assert response.trajectory.refusals == ()
        assert _presented(response), "a completed run presents candidates"


def test_a_plan_is_not_restarted_within_one_run() -> None:
    """One run consults each planned source once - the fix must not make it loop.

    The in-run guard is what stops the policy repeating its first source until the budget runs out;
    it has to keep working now that the state is cleared per run instead of never.
    """
    service = _service()
    for payload in (_payload("catalog_search", message="RedWidget"),
                    _payload("history"),
                    _payload("history", "catalog_search")):
        response = service.run(payload)
        counts = _executed_counts(response)
        for action, count in counts.items():
            assert count == 1, f"{action} executed {count} times in one run: {counts}"
        assert counts["finish"] == 1
        assert response.terminal.tool_calls == len(
            [name for name in counts if name != "finish"])


# --------------------------------------------------------------------------- #
# E. budgets and failure semantics are unchanged
# --------------------------------------------------------------------------- #


def test_budgets_still_bound_a_run_and_the_terminal_state_stays_truthful() -> None:
    """The two budgets keep their accepted semantics, and a bounded run leaves nothing behind.

    The controller owns the budget, not the menu: once the tool budget is spent it withdraws the
    tool actions (``_LoopEngine.available_actions``) and a grounded run finishes with what it has,
    while a run with nothing grounded ends as a refusal.  Neither may exceed its limits, and
    neither may change what the next request does.
    """
    # 1. One tool call is not enough for a two-source plan: the second source is never consulted,
    #    because the controller withdrew it rather than executing it without budget.
    tight = _service(limits=LoopLimits(max_steps=6, max_tool_calls=1, max_retries=0))
    bounded = tight.run(_payload("history", "catalog_search"))
    assert bounded.terminal.tool_calls <= 1
    assert bounded.terminal.steps <= 6
    assert _executed_counts(bounded).get("search_catalog", 0) == 0, (
        "a source the budget no longer covers must not be consulted"
    )
    assert bounded.terminal.succeeded is True, (
        "a grounded run may still finish within its budget; the budget bounds work, not success"
    )

    # 2. The step budget stops a run deterministically, with the controller's own reason.
    starved = _service(limits=LoopLimits(max_steps=1, max_tool_calls=4, max_retries=0))
    stopped = starved.run(_payload("history", "catalog_search"))
    assert stopped.terminal.steps <= 1
    assert stopped.terminal.succeeded is False
    assert stopped.terminal.termination_reason == "max_steps"

    # 3. A tool that cannot produce a grounded candidate ends the run inside its budget.
    failing, engine = _build(limits=LoopLimits(max_steps=6, max_tool_calls=2, max_retries=0))
    engine.error = RuntimeError("engine exploded")
    try:
        failed = failing.run(_payload("history", "catalog_search"))
    finally:
        engine.error = None
    assert failed.terminal.succeeded is False
    assert failed.terminal.tool_calls <= 2
    assert failed.terminal.termination_reason in (
        "completion_refused", "max_tool_calls", "max_steps", "execution_failed"
    )
    assert _presented(failed) == [], "a failed run must not present candidates"

    # 4. None of those bounded runs changed what a fresh request does.
    relaxed = _service().run(_payload("history", "catalog_search"))
    assert relaxed.terminal.succeeded is True
    assert _executed_counts(relaxed) == {
        "recommend_from_history": 1, "search_catalog": 1, "finish": 1}


def test_in_run_retries_still_happen_within_the_budget() -> None:
    """A source the controller legitimately calls again *inside* one run is still allowed.

    ``max_retries``/``max_tool_calls`` decide this, not the policy's run-scoped memory: with a
    source that cannot produce a grounded candidate the run retries and then refuses, exactly as it
    did before the fix.
    """
    service, engine = _build()
    engine.error = RuntimeError("engine exploded")
    try:
        response = service.run(_payload("history"))
    finally:
        engine.error = None

    counts = _executed_counts(response)
    assert counts.get("recommend_from_history", 0) > 1, (
        f"the run must be allowed to call the failing source again within its budget: {counts}"
    )
    assert response.terminal.tool_calls <= TOOL_BUDGET
    assert response.terminal.succeeded is False
    assert _presented(response) == []


# --------------------------------------------------------------------------- #
# F. concurrency: requests sharing a service must share nothing mutable
# --------------------------------------------------------------------------- #


def test_concurrent_requests_do_not_share_run_state() -> None:
    """Eight threads, one service, one plan: every request must execute its whole plan.

    With one cached controller and one shared policy, concurrent runs interleave on the same
    mutable served-set.  Composing a controller per request removes the shared object entirely, so
    this asserts the property the sequential tests cannot: isolation under real overlap.
    """
    service = _service()
    payload = _payload("history", "catalog_search")
    # Warm the shared, immutable collaborators (catalogue lookup, tool wiring) so the threads
    # exercise the run path rather than first-call initialisation.
    service.run(payload)

    results: list[Any] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker() -> None:
        try:
            start.wait(timeout=30)
            for _ in range(2):
                response = service.run(payload)
                with lock:
                    results.append(response)
        except BaseException as exc:  # noqa: BLE001 - reported as a test failure below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors, f"concurrent runs raised: {errors!r}"
    assert len(results) == 16

    reference = _presented(results[0])
    for index, response in enumerate(results):
        counts = _executed_counts(response)
        assert counts == {"recommend_from_history": 1, "search_catalog": 1, "finish": 1}, (
            f"run {index} under concurrency executed {counts}"
        )
        assert _presented(response) == reference, f"run {index} presented differently"


# --------------------------------------------------------------------------- #
# G. the policy is run-local even when a caller reuses the instance itself
# --------------------------------------------------------------------------- #


def test_the_policy_itself_is_run_local() -> None:
    """The class contract, independent of how a caller composes controllers.

    A caller that hands one policy instance to one controller and runs it twice must get the same
    behaviour as a caller that builds a controller per run - that is what the step-0 reset
    promises, and it is asserted here against the real loop rather than against a private field.
    """
    from recommendation.control import (
        CandidateLedger,
        CandidatePlane,
        CatalogSearchSource,
        GroundingVerifier,
        LoopController,
    )

    _capability, engine, parts = build_full_capability()
    identified = _IdentityEngine(engine)
    policy = DemoSourcePlanPolicy(("history", "catalog_search"), default_k=3, use_memory=False)
    controller = LoopController(
        policy,
        RecommendFromHistoryCapability(
            parts["tool"],
            product_enricher=parts["enricher"],
            preference_matcher=parts["matcher"],
            preference_reranker=parts["reranker"],
        ),
        limits=LoopLimits(max_steps=6, max_tool_calls=TOOL_BUDGET, max_retries=1),
        candidate_plane=CandidatePlane(
            ledger=CandidateLedger(),
            grounding=GroundingVerifier(identified, parts["enricher"].metadata),
            history_tool=parts["tool"],
            catalog_search=CatalogSearchSource(parts["enricher"].metadata),
        ),
    )

    for label in ("first", "second"):
        result = controller.run("find me a RedWidget", HISTORY)
        actions = [step.validated_action["action"] for step in result.trajectory.steps
                   if step.validated_action]
        assert actions == ["recommend_from_history", "search_catalog", "finish"], (
            f"the {label} run on a reused policy executed {actions}"
        )


def test_the_run_reset_keeps_a_plan_advancing_and_restartable() -> None:
    """Step 0 restarts the plan; later steps advance it and never restart it."""
    from recommendation.control import ActionKind
    from recommendation.control.context import CandidateState, PolicyContext

    policy = DemoSourcePlanPolicy(("history", "catalog_search"), default_k=3)
    offered = (ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.SEARCH_CATALOG, ActionKind.FINISH)

    def context(step_index: int, *, grounded: bool) -> PolicyContext:
        return PolicyContext(
            user_request="find me a RedWidget",
            available_actions=offered,
            has_trusted_history=True,
            candidate_state=CandidateState(grounded=grounded, candidate_count=3 if grounded else 0),
            step_index=step_index,
        )

    # Run 1: first source, then the second source once candidates are grounded, then FINISH.
    assert policy.choose(context(0, grounded=False)).action is ActionKind.RECOMMEND_FROM_HISTORY
    assert policy.choose(context(1, grounded=True)).action is ActionKind.SEARCH_CATALOG
    assert policy.choose(context(2, grounded=True)).action is ActionKind.FINISH
    # A refused FINISH does not restart the plan: it keeps proposing FINISH until a budget stops it.
    assert policy.choose(context(3, grounded=True)).action is ActionKind.FINISH

    # Run 2 on the same instance starts over instead of resuming run 1's finished plan.
    assert policy.choose(context(0, grounded=False)).action is ActionKind.RECOMMEND_FROM_HISTORY
    assert policy.choose(context(1, grounded=True)).action is ActionKind.SEARCH_CATALOG
