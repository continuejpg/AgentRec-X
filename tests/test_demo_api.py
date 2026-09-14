"""Milestone 11 HTTP contract tests for the demo API.

Offline and deterministic: the app is built over a synthetic engine and catalogue, so no
349 MB checkpoint, no 300 MB metadata artifact, no network and no provider API is used.

What is pinned here:

* the demo endpoints exist with their documented request/response schemas;
* request validation is strict and a client cannot inject server-owned fields;
* unknown/expired sessions are explicit errors, never silently recreated;
* backend failures map onto documented statuses and never return fabricated content;
* the three accepted Milestone 6 endpoints keep their exact behaviour.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import (  # noqa: E402
    HISTORY,
    PROFILE_A,
    build_harness,
    close_harness,
    make_settings,
)
from recommendation.api.app import create_app  # noqa: E402
from recommendation.demo import DEMO_API_VERSION  # noqa: E402


@pytest.fixture()
def harness(tmp_path: Path):
    """A demo-enabled app over synthetic artifacts, torn down afterwards."""
    built = build_harness(tmp_path)
    try:
        yield built
    finally:
        close_harness(built)


# --------------------------------------------------------------------------- #
# Session endpoints
# --------------------------------------------------------------------------- #


def test_create_session_returns_an_opaque_isolated_session(harness) -> None:
    body = harness.create_session()
    assert body["api_version"] == DEMO_API_VERSION
    assert body["turn"] == 0
    assert body["profile"]["profile_id"] == PROFILE_A
    assert body["profile"]["history_length"] == len(HISTORY)
    # The raw trusted history is never part of the public session view.
    assert "trusted_user_history" not in json.dumps(body)
    for item in HISTORY:
        assert item not in json.dumps(body)


def test_create_session_rejects_an_unknown_profile(harness) -> None:
    response = harness.client.post("/v1/demo/sessions", json={"profile_id": "nope"})
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_profile"


def test_create_session_rejects_extra_fields(harness) -> None:
    """A browser may not inject server-owned state through the session request."""
    for injected in (
        {"trusted_user_history": list(HISTORY)},
        {"user_key": "attacker"},
        {"session_id": "attacker"},
        {"history": list(HISTORY)},
    ):
        response = harness.client.post(
            "/v1/demo/sessions", json={"profile_id": PROFILE_A, **injected}
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"] == "invalid_request"


def test_create_session_rejects_a_blank_profile(harness) -> None:
    assert harness.client.post("/v1/demo/sessions", json={"profile_id": "  "}).status_code == 422


def test_session_ids_are_distinct(harness) -> None:
    ids = {harness.create_session()["session_id"] for _ in range(5)}
    assert len(ids) == 5


def test_profiles_endpoint_lists_server_owned_profiles(harness) -> None:
    body = harness.client.get("/v1/demo/profiles").json()
    assert {item["profile_id"] for item in body["profiles"]} == set(harness.profiles)
    assert "trusted_user_history" not in json.dumps(body)


def test_session_state_reports_metadata_not_history(harness) -> None:
    created = harness.create_session()
    body = harness.state(created["session_id"])
    assert body["session_id"] == created["session_id"]
    assert body["turn"] == 0
    assert body["active_preferences"] == []
    assert body["active_preference_count"] == 0
    assert "user_key" not in json.dumps(body)


def test_demo_health_reports_readiness(harness) -> None:
    body = harness.client.get("/v1/demo/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["metadata_loaded"] is True
    assert body["demo_ready"] is True
    assert body["profiles"] == len(harness.profiles)
    assert body["max_sessions"] == harness.manager.max_sessions
    # No path, no secret.
    assert "/root/" not in json.dumps(body)


# --------------------------------------------------------------------------- #
# Unknown session behaviour
# --------------------------------------------------------------------------- #


def test_unknown_session_is_an_explicit_404(harness) -> None:
    missing = "00000000-0000-4000-8000-000000000000"
    assert harness.client.get(f"/v1/demo/sessions/{missing}").status_code == 404
    chat = harness.client.post(
        f"/v1/demo/sessions/{missing}/chat", json={"message": "Recommend products."}
    )
    assert chat.status_code == 404
    assert chat.json()["error"] == "session_not_found"
    assert harness.client.delete(f"/v1/demo/sessions/{missing}").status_code == 404


def test_unknown_session_is_never_silently_created(harness) -> None:
    """A 404 must not leave a new session behind: isolation depends on it."""
    before = len(harness.manager)
    harness.client.post(
        "/v1/demo/sessions/00000000-0000-4000-8000-000000000000/chat",
        json={"message": "Recommend products."},
    )
    assert len(harness.manager) == before
    assert harness.engine.call_count == 0


def test_reset_session_makes_the_id_stop_working(harness) -> None:
    created = harness.create_session()
    response = harness.client.delete(f"/v1/demo/sessions/{created['session_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["reset"] is True
    assert body["session_id"] == created["session_id"]
    assert harness.client.get(f"/v1/demo/sessions/{created['session_id']}").status_code == 404


def test_reset_only_affects_the_target_session(harness) -> None:
    first = harness.create_session()
    second = harness.create_session()
    harness.chat(second["session_id"], "I prefer black.")

    harness.client.delete(f"/v1/demo/sessions/{first['session_id']}")

    assert harness.client.get(f"/v1/demo/sessions/{first['session_id']}").status_code == 404
    assert harness.active_values(second["session_id"]) == {("color", "prefer", "black")}


# --------------------------------------------------------------------------- #
# Chat request validation
# --------------------------------------------------------------------------- #


def test_chat_rejects_a_blank_or_missing_message(harness) -> None:
    session = harness.create_session()["session_id"]
    assert harness.client.post(
        f"/v1/demo/sessions/{session}/chat", json={}
    ).status_code == 422
    assert harness.client.post(
        f"/v1/demo/sessions/{session}/chat", json={"message": "   "}
    ).status_code == 422
    assert harness.client.post(
        f"/v1/demo/sessions/{session}/chat", json={"message": ""}
    ).status_code == 422


def test_chat_rejects_an_over_long_message(harness) -> None:
    session = harness.create_session()["session_id"]
    response = harness.client.post(
        f"/v1/demo/sessions/{session}/chat", json={"message": "x" * 2001}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_k", ["5", 5.0, True, 0, -1, 101, None, [5]])
def test_chat_rejects_a_non_strict_or_out_of_range_k(harness, bad_k) -> None:
    session = harness.create_session()["session_id"]
    response = harness.client.post(
        f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend products.", "k": bad_k}
    )
    assert response.status_code == 422, f"k={bad_k!r} was accepted"


def test_chat_rejects_server_owned_fields(harness) -> None:
    """The trust boundary: none of these may be supplied by a client."""
    session = harness.create_session()["session_id"]
    for injected in (
        {"trusted_user_history": list(HISTORY)},
        {"history": list(HISTORY)},
        {"parent_asins": list(HISTORY)},
        {"preference_snapshot": {}},
        {"reranking": {}},
        {"user_key": "attacker"},
        {"session_id": "attacker"},
        {"turn_id": "attacker:000001"},
        {"route": "recommend"},
    ):
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": "Recommend products.", **injected},
        )
        assert response.status_code == 422, f"injected {injected} was accepted"
        assert response.json()["error"] == "invalid_request"


def test_chat_accepts_the_documented_bounds(harness) -> None:
    session = harness.create_session()["session_id"]
    for k in (1, 100):
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": "Recommend products.", "k": k},
        )
        assert response.status_code == 200, response.text


# --------------------------------------------------------------------------- #
# Chat response shape
# --------------------------------------------------------------------------- #


def test_chat_response_is_structured_not_prose_only(harness) -> None:
    session = harness.create_session()["session_id"]
    body = harness.chat(session, "Recommend some products.")

    assert body["route"] == "recommend"
    assert body["turn"] == 1
    assert body["turn_id"] == f"{session}:000001"
    assert body["message"]
    assert body["recommendations"], "the response must carry structured cards"
    card = body["recommendations"][0]
    for field in (
        "reranked_rank",
        "original_rank",
        "parent_asin",
        "item_id",
        "sasrec_score",
        "match_count",
        "violation_count",
        "unknown_count",
        "metadata_status",
        "evidence",
        "movement_summary",
    ):
        assert field in card, f"card is missing {field!r}"
    assert body["audit"]["reranking_applied"] is True


def test_chat_response_never_carries_trusted_history_or_internal_keys(harness) -> None:
    session = harness.create_session()["session_id"]
    text = json.dumps(harness.chat(session, "Recommend some products."))
    assert "trusted_user_history" not in text
    assert "user_key" not in text
    assert "memory_id" not in text
    for item in HISTORY:
        assert item not in text, "a trusted history item leaked into the response"


def test_direct_route_performs_no_recommendation_work(harness) -> None:
    session = harness.create_session()["session_id"]
    before = harness.engine.call_count
    body = harness.chat(session, "hello there")
    assert body["route"] == "direct"
    assert body["recommendations"] == []
    assert body["audit"]["candidate_count"] == 0
    assert body["audit"]["reranking_applied"] is False
    assert harness.engine.call_count == before, "the direct route called the engine"


def test_recommendation_route_calls_the_engine_once(harness) -> None:
    session = harness.create_session()["session_id"]
    before = harness.engine.call_count
    harness.chat(session, "Recommend some products.")
    assert harness.engine.call_count == before + 1


def test_turn_ids_are_server_owned_and_increment(harness) -> None:
    session = harness.create_session()["session_id"]
    first = harness.chat(session, "Recommend some products.")
    second = harness.chat(session, "I prefer black.")
    third = harness.chat(session, "Recommend again.")
    assert [first["turn_id"], second["turn_id"], third["turn_id"]] == [
        f"{session}:000001",
        f"{session}:000002",
        f"{session}:000003",
    ]
    assert [first["turn"], second["turn"], third["turn"]] == [1, 2, 3]


def test_empty_candidate_set_is_a_neutral_state(tmp_path: Path) -> None:
    """Candidate exhaustion is legal and never padded with an invented product."""
    empty = build_harness(tmp_path, rows=())
    try:
        session = empty.create_session()["session_id"]
        body = empty.chat(session, "Recommend some products.")
        assert body["recommendations"] == []
        assert body["audit"]["candidate_count"] == 0
        assert body["route"] == "recommend"
        assert "No unseen product is left" in body["message"]
        assert empty.engine.call_count == 1
        assert empty.client.get(f"/v1/demo/sessions/{session}").json()["turn"] == 1
    finally:
        close_harness(empty)


# --------------------------------------------------------------------------- #
# Failure mapping
# --------------------------------------------------------------------------- #


def test_recommendation_failure_maps_to_a_documented_status(harness) -> None:
    """A Tool failure returns an error body, never fabricated recommendations."""
    session = harness.create_session()["session_id"]
    harness.engine.error = RuntimeError("engine exploded")
    try:
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend some products."}
        )
    finally:
        harness.engine.error = None

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "recommendation_failed"
    assert "recommendations" not in body
    assert "exploded" not in json.dumps(body), "internal error text leaked"
    assert "/root/" not in json.dumps(body)


def test_enrichment_failure_maps_to_a_documented_status(harness) -> None:
    session = harness.create_session()["session_id"]

    class ExplodingEnricher:
        def enrich(self, result, query=""):
            raise RuntimeError("metadata layer unavailable")

    harness.runtime.enricher.__dict__  # sanity: the runtime attribute exists
    original = harness.runtime.graph_for(5, user_key=harness.manager.get(session).user_key)
    assert original is not None
    harness.runtime.enricher = ExplodingEnricher()
    harness.runtime._graphs.clear()  # noqa: SLF001 - force the next turn to rebuild
    try:
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend some products."}
        )
    finally:
        harness.runtime.enricher = original.product_enricher
        harness.runtime._graphs.clear()  # noqa: SLF001

    assert response.status_code == 502
    assert response.json()["error"] == "demo_backend_failed"
    assert "unavailable" not in json.dumps(response.json())


def test_demo_error_mapping_is_exhaustive() -> None:
    """Every documented failure class maps onto a documented status and code."""
    from recommendation.agent import AgentConfigurationError, MalformedDecision
    from recommendation.api.demo_routes import map_demo_exception
    from recommendation.demo import (
        DemoRuntimeError,
        SessionCapacityExceeded,
        UnknownProfile,
        UnknownSession,
    )
    from recommendation.reranking import RerankingError
    from recommendation.tools import (
        InvalidRecommendationRequest,
        MissingUserHistory,
        RecommendationUnavailable,
        UnknownHistoryItem,
    )

    cases = [
        (UnknownSession("x"), 404, "session_not_found"),
        (UnknownProfile("x"), 404, "unknown_profile"),
        (SessionCapacityExceeded("x"), 503, "session_capacity_exceeded"),
        (DemoRuntimeError("x"), 503, "demo_unavailable"),
        (MalformedDecision("x"), 502, "agent_failed"),
        (AgentConfigurationError("x"), 502, "agent_failed"),
        (RerankingError("x"), 502, "preference_stage_failed"),
        (MissingUserHistory("x"), 422, "invalid_history"),
        (UnknownHistoryItem("x"), 422, "invalid_history"),
        (InvalidRecommendationRequest("x"), 422, "invalid_request"),
        (RecommendationUnavailable("x"), 502, "recommendation_failed"),
        (ValueError("x"), 502, "demo_backend_failed"),
        (RuntimeError("x"), 502, "demo_backend_failed"),
    ]
    for exc, status, code in cases:
        mapped = map_demo_exception(exc)
        assert (mapped.status_code, mapped.code) == (status, code), exc
        assert "/root/" not in mapped.detail
        assert "Traceback" not in mapped.detail
        if status >= 500:
            # Every 5xx detail is authored by the mapping, never taken from the failure.
            assert str(exc) not in mapped.detail, f"{code} echoed the exception text"


# --------------------------------------------------------------------------- #
# Milestone 6 compatibility
# --------------------------------------------------------------------------- #


def test_m6_endpoints_are_unchanged_with_the_demo_enabled(harness) -> None:
    health = harness.client.get("/health")
    assert health.status_code == 200
    assert set(health.json()) == {"status", "model_loaded", "device", "detail"}

    model = harness.client.get("/v1/model")
    assert model.status_code == 200
    assert set(model.json()) == set(harness.engine.model_metadata())

    recommend = harness.client.post("/v1/recommend", json={"history": list(HISTORY), "k": 3})
    assert recommend.status_code == 200
    body = recommend.json()
    assert body["returned_k"] == 3
    assert "recommendations" in body
    # The M6 endpoint is the direct inference path: it is NOT routed through the agent.
    assert "route" not in body
    assert "active_preferences" not in body


def test_m6_recommend_does_not_touch_sessions_or_memory(harness) -> None:
    before = len(harness.manager)
    harness.client.post("/v1/recommend", json={"history": list(HISTORY), "k": 2})
    assert len(harness.manager) == before


def test_openapi_publishes_both_contracts(harness) -> None:
    paths = set(harness.client.get("/openapi.json").json()["paths"])
    assert {"/health", "/v1/model", "/v1/recommend"} <= paths
    assert {
        "/v1/demo/health",
        "/v1/demo/profiles",
        "/v1/demo/sessions",
        "/v1/demo/sessions/{session_id}",
        "/v1/demo/sessions/{session_id}/chat",
    } <= paths


def test_demo_can_be_disabled_for_the_accepted_m6_service(tmp_path: Path) -> None:
    """With the demo off, the app is the accepted Milestone 6 service."""
    from fastapi.testclient import TestClient

    harness = build_harness(tmp_path, demo=False)
    try:
        assert harness.client.get("/v1/demo/health").status_code == 404
        assert harness.client.post(
            "/v1/demo/sessions", json={"profile_id": PROFILE_A}
        ).status_code == 404
        assert harness.client.get("/health").status_code == 200
        assert harness.client.post(
            "/v1/recommend", json={"history": list(HISTORY), "k": 2}
        ).status_code == 200
        assert "demo_runtime" not in str(harness.client.app.state)
    finally:
        close_harness(harness)
    assert make_settings(tmp_path).device == "cpu"


def test_startup_refuses_to_serve_an_uncomposable_demo(tmp_path: Path) -> None:
    """A missing artifact must fail startup, not the first browser request."""
    from fastapi.testclient import TestClient

    from recommendation.demo import DemoRuntimeError

    def exploding_factory(**_kwargs):
        raise DemoRuntimeError("catalogue metadata artifact is missing")

    app = create_app(
        make_settings(tmp_path),
        load_on_startup=False,
        enable_demo=True,
        demo_factory=exploding_factory,
    )
    with pytest.raises(DemoRuntimeError):
        with TestClient(app):
            pass  # pragma: no cover - startup must raise before this


# --------------------------------------------------------------------------- #
# Architectural guards: no recommendation policy in the controller
# --------------------------------------------------------------------------- #

CONTROLLER = REPO_ROOT / "recommendation" / "api" / "demo_routes.py"
DEMO_DIR = REPO_ROOT / "recommendation" / "demo"

#: Implementation symbols that belong behind an accepted abstraction, never in the
#: web/controller layer.
FORBIDDEN_POLICY_SYMBOLS = (
    "sort_key_for",
    "rerank_candidates",
    "match_candidates",
    "PreferenceReranker",
    "PreferenceCandidateMatcher",
    "retrieve_evidence",
    "bm25",
    "BM25",
    "torch",
    "SASRecInferenceEngine",
    "scope_matches",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _identifiers(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            pass
    return names


def test_controller_contains_no_recommendation_policy() -> None:
    """The HTTP layer orchestrates; it never scores, sorts, matches or reranks."""
    names = _identifiers(_tree(CONTROLLER))
    offending = names & set(FORBIDDEN_POLICY_SYMBOLS)
    assert offending == set(), f"controller contains policy symbols: {sorted(offending)}"

    source = CONTROLLER.read_text(encoding="utf-8")
    assert "sorted(" not in source, "the controller must not order candidates"
    assert "violation_count" not in source, "the controller must not implement the policy key"


def test_controller_imports_only_accepted_seams() -> None:
    """No retrieval, matching, reranking or inference implementation is imported."""
    imported = set()
    for node in ast.walk(_tree(CONTROLLER)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for banned in (
        "recommendation.rag",
        "recommendation.preference_matching",
        "recommendation.memory",
        "recommendation.inference",
        "recommendation.reranking.reranker",
        "torch",
        "numpy",
    ):
        assert banned not in imported, f"the controller imports {banned}"


def test_demo_package_does_not_import_torch_or_a_provider_sdk() -> None:
    for path in sorted(DEMO_DIR.glob("*.py")):
        imported = set()
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for banned in ("torch", "openai", "anthropic", "requests", "httpx"):
            assert banned not in imported, f"{path.name} imports {banned}"


def test_demo_layer_never_serializes_graph_state_blindly() -> None:
    """Serialization is an explicit whitelist, not ``dict(state)``.

    Checked over the AST rather than raw text, so the module may *describe* the
    anti-pattern it avoids without tripping its own guard (the Milestone 10C lesson).
    """
    tree = _tree(DEMO_DIR / "serialization.py")

    blind_dict_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    ]
    assert blind_dict_calls == [], "serialization must not call dict(...) on graph state"

    dumps = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "model_dump"
    ]
    assert dumps == [], "serialization must not dump internal models into the response"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not any(keyword.arg is None for keyword in node.keywords), (
                "serialization must not splat an internal mapping into a response model"
            )
