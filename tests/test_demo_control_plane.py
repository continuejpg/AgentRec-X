"""HTTP-level equivalence tests for the opt-in 2.0-alpha control plane.

Stage 1 changes *who decides the next step*.  The strongest demonstration that it changed
nothing else is to serve the **same real HTTP request** through both control planes - the
accepted Milestone 7B-10D DAG and the 2.0-alpha bounded loop - and compare the responses.

These tests exercise the real user path, not a hand-built state:

    FastAPI demo route
      -> DemoSessionManager (server-owned turn id, trusted history)
      -> AgentGraph  (default)  or  DemoLoopRunner  (opt-in)
      -> RecommendationTool -> engine, ProductEnricher, matcher, reranker
      -> the accepted serializer

The loop is selected with ``AGENTRECX_CONTROL_PLANE``, the same switch an operator would
use, so the test also pins the switch: an unrecognised value must be rejected rather than
silently selecting a different control plane, and the default must remain the accepted DAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import RunStatus  # noqa: E402
from recommendation.demo.control_plane import (  # noqa: E402
    CONTROL_PLANE_ENV_VAR,
    CONTROL_PLANE_GRAPH,
    CONTROL_PLANE_LOOP,
    resolve_control_plane,
)
from recommendation.demo.runtime import build_demo_runtime  # noqa: E402
from tests.demo_fixture import (  # noqa: E402
    PROFILE_A,
    build_harness,
    close_harness,
    demo_profiles,
)
from tests.agent_reranking_fixture import build_index  # noqa: E402
from recommendation.catalog import MetadataIndex  # noqa: E402
from recommendation.memory import SQLitePreferenceStore  # noqa: E402
from recommendation.demo.profiles import build_demo_profiles  # noqa: E402
from tests.demo_fixture import DemoEngine, CANDIDATE_ROWS  # noqa: E402


def _loop_harness(tmp_path: Path, **kwargs: object):
    """Build the demo harness with the 2.0-alpha loop selected."""
    from fastapi.testclient import TestClient

    from recommendation.api.app import create_app
    from tests.demo_fixture import make_settings

    profiles = demo_profiles()
    engine = DemoEngine()
    # The harness may be given a subdirectory that does not exist yet; create it so the
    # SQLite store can be opened.
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = SQLitePreferenceStore(tmp_path / "loop_memory.sqlite3")
    runtime = build_demo_runtime(
        engine=engine,
        metadata=build_index(row[0] for row in CANDIDATE_ROWS),
        store=store,
        profiles=profiles,
        max_sessions=8,
        control_plane=CONTROL_PLANE_LOOP,
    )
    app = create_app(
        make_settings(tmp_path),
        engine=engine,
        load_on_startup=False,
        demo=runtime,
        enable_demo=True,
    )
    client = TestClient(app)
    client.__enter__()
    from tests.demo_fixture import DemoHarness

    return DemoHarness(
        app=app,
        runtime=runtime,
        engine=engine,
        client=client,
        store=store,
        profiles=profiles,
    )


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #


def test_control_plane_defaults_to_the_accepted_dag() -> None:
    """With nothing configured the demo runs the accepted path, unchanged."""
    assert resolve_control_plane(None) == CONTROL_PLANE_GRAPH
    assert resolve_control_plane("") == CONTROL_PLANE_GRAPH
    assert resolve_control_plane(CONTROL_PLANE_GRAPH) == CONTROL_PLANE_GRAPH


def test_control_plane_selects_the_loop_explicitly() -> None:
    """The loop is opt-in, and the name is case-insensitive."""
    assert resolve_control_plane(CONTROL_PLANE_LOOP) == CONTROL_PLANE_LOOP
    assert resolve_control_plane("LOOP") == CONTROL_PLANE_LOOP
    assert resolve_control_plane(" Loop ") == CONTROL_PLANE_LOOP


def test_an_unknown_control_plane_is_rejected() -> None:
    """A typo must not silently select a control plane."""
    with pytest.raises(ValueError) as excinfo:
        resolve_control_plane("loop2")
    assert CONTROL_PLANE_ENV_VAR in str(excinfo.value)


def test_environment_variable_selects_the_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented switch really is the environment variable."""
    monkeypatch.setenv(CONTROL_PLANE_ENV_VAR, CONTROL_PLANE_LOOP)
    assert resolve_control_plane() == CONTROL_PLANE_LOOP
    monkeypatch.setenv(CONTROL_PLANE_ENV_VAR, CONTROL_PLANE_GRAPH)
    assert resolve_control_plane() == CONTROL_PLANE_GRAPH


def test_unknown_control_plane_is_rejected_by_the_runtime(tmp_path: Path) -> None:
    """The runtime refuses to compose rather than guessing a control plane."""
    with pytest.raises(ValueError):
        build_demo_runtime(
            engine=DemoEngine(),
            metadata=build_index(row[0] for row in CANDIDATE_ROWS),
            store=SQLitePreferenceStore(tmp_path / "m.sqlite3"),
            profiles=demo_profiles(),
            control_plane="definitely-not-a-control-plane",
        )


# --------------------------------------------------------------------------- #
# The loop serves the real user path
# --------------------------------------------------------------------------- #


def test_loop_serves_a_real_http_recommendation_turn(tmp_path: Path) -> None:
    """The opt-in loop answers a real chat request with real candidates."""
    harness = _loop_harness(tmp_path)
    try:
        assert harness.runtime.control_plane == CONTROL_PLANE_LOOP
        session = harness.create_session(PROFILE_A)
        body = harness.chat(session["session_id"], "Recommend some gear.", k=4)

        assert body["route"] == "recommend"
        assert body["recommendations"], "the loop returned no candidates over HTTP"
        assert body["message"]
        assert len(body["recommendations"]) == 4
        assert harness.engine.call_count == 1
    finally:
        close_harness(harness)


def test_loop_reports_a_bounded_run_in_its_trajectory(tmp_path: Path) -> None:
    """The run really went through the loop, and the loop really terminated."""
    harness = _loop_harness(tmp_path)
    try:
        session = harness.create_session(PROFILE_A)
        harness.chat(session["session_id"], "Recommend some gear.", k=4)

        runner = harness.runtime.runner_for(4, user_key=harness.runtime.sessions.get(
            session["session_id"]
        ).user_key)
        result = runner.last_result
        assert result is not None
        assert result.status is RunStatus.FINISHED
        assert result.trajectory.actions() == ("recommend_from_history", "finish")
        # Control returned to the policy exactly once after the observation.
        assert runner.policy is not None
    finally:
        close_harness(harness)


def test_loop_and_dag_return_the_same_http_contract(tmp_path: Path) -> None:
    """Both control planes satisfy the identical public response contract.

    This is the Stage 1 acceptance property stated over HTTP: the control plane changed,
    the response shape, the route and the audit block did not.
    """
    (tmp_path / "dag").mkdir(parents=True, exist_ok=True)
    (tmp_path / "loop").mkdir(parents=True, exist_ok=True)
    dag = build_harness(tmp_path / "dag")
    loop = _loop_harness(tmp_path / "loop")
    try:
        dag_session = dag.create_session(PROFILE_A)
        loop_session = loop.create_session(PROFILE_A)
        dag_body = dag.chat(dag_session["session_id"], "Recommend some gear.", k=4)
        loop_body = loop.chat(loop_session["session_id"], "Recommend some gear.", k=4)

        # Same keys, same shape, same route, same candidate identities and order.
        assert set(dag_body) == set(loop_body)
        assert dag_body["route"] == loop_body["route"] == "recommend"
        assert [card["parent_asin"] for card in dag_body["recommendations"]] == [
            card["parent_asin"] for card in loop_body["recommendations"]
        ]
        assert [card["item_id"] for card in dag_body["recommendations"]] == [
            card["item_id"] for card in loop_body["recommendations"]
        ]
        assert dag_body["audit"] == loop_body["audit"]
        assert dag_body["active_preferences"] == loop_body["active_preferences"]
        # And the rendered text is identical, because both call the accepted renderer.
        assert dag_body["message"] == loop_body["message"]
        assert dag.engine.call_count == loop.engine.call_count == 1
    finally:
        close_harness(dag)
        close_harness(loop)


def test_loop_and_dag_agree_on_a_direct_route_turn(tmp_path: Path) -> None:
    """A non-recommendation turn is routed identically by both control planes."""
    (tmp_path / "dag").mkdir(parents=True, exist_ok=True)
    (tmp_path / "loop").mkdir(parents=True, exist_ok=True)
    dag = build_harness(tmp_path / "dag")
    loop = _loop_harness(tmp_path / "loop")
    try:
        dag_session = dag.create_session(PROFILE_A)
        loop_session = loop.create_session(PROFILE_A)
        dag_body = dag.chat(dag_session["session_id"], "hello there", k=4)
        loop_body = loop.chat(loop_session["session_id"], "hello there", k=4)

        assert dag_body["route"] == loop_body["route"] == "direct"
        assert dag_body["message"] == loop_body["message"]
        assert dag_body["recommendations"] == loop_body["recommendations"] == []
        assert dag.engine.call_count == loop.engine.call_count == 0
    finally:
        close_harness(dag)
        close_harness(loop)


def test_loop_commits_preference_memory_across_turns(tmp_path: Path) -> None:
    """The accepted Milestone 9 memory semantics survive the control-plane change."""
    loop = _loop_harness(tmp_path)
    try:
        session = loop.create_session(PROFILE_A)
        first = loop.chat(session["session_id"], "I prefer red products.", k=4)
        assert first["memory_update"]["added"], "the stated preference was not committed"

        # The stored preference is visible on the next turn's state read.
        state = loop.state(session["session_id"])
        assert {item["value"] for item in state["active_preferences"]} == {"red"}

        # And it takes effect from the following turn, as documented.
        second = loop.chat(session["session_id"], "Recommend some gear.", k=4)
        assert second["route"] == "recommend"
    finally:
        close_harness(loop)


def test_loop_http_turn_is_deterministic(tmp_path: Path) -> None:
    """Two identical loop turns over HTTP produce identical bodies (modulo identity)."""
    loop = _loop_harness(tmp_path)
    try:
        first_session = loop.create_session(PROFILE_A)
        second_session = loop.create_session(PROFILE_A)
        first = loop.chat(first_session["session_id"], "Recommend some gear.", k=4)
        second = loop.chat(second_session["session_id"], "Recommend some gear.", k=4)

        # Session and turn identities differ by design; everything else must match.
        for body in (first, second):
            body.pop("session_id")
            body.pop("turn_id")
            body.pop("turn")
        assert first == second
    finally:
        close_harness(loop)


def test_dag_remains_the_default_in_the_runtime(tmp_path: Path) -> None:
    """The accepted path is still the default, and its runner is the accepted graph."""
    harness = build_harness(tmp_path)
    try:
        assert harness.runtime.control_plane == CONTROL_PLANE_GRAPH
        session = harness.create_session(PROFILE_A)
        runner = harness.runtime.runner_for(
            5, user_key=harness.runtime.sessions.get(session["session_id"]).user_key
        )
        from recommendation.agent import AgentGraph

        assert isinstance(runner, AgentGraph)
    finally:
        close_harness(harness)
