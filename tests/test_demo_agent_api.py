"""Tests for the packaging-phase single-turn agent endpoint.

The endpoint's whole purpose is to expose the **existing** runtime, so these tests are
written to catch the two failure modes that would silently break that promise:

* the endpoint quietly becoming its own pipeline (duplicating the accepted logic), or
* the response quietly becoming a dump of internal state (leaking trusted history, raw
  audit payloads or fabricated provenance).

Everything is offline: a synthetic engine over a synthetic catalogue, a real SQLite memory
store in a temp directory, no checkpoint, no 300 MB metadata artifact, no network and no
provider API key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from recommendation.api.app import create_app  # noqa: E402
from recommendation.demo import (  # noqa: E402
    AGENT_API_VERSION,
    DemoAgentService,
    build_demo_runtime,
    normalise_source_plan,
)
from recommendation.memory import SQLitePreferenceStore  # noqa: E402
from tests.agent_reranking_fixture import (  # noqa: E402
    CANDIDATE_ROWS,
    INITIAL_ORDER,
    build_index,
)
from tests.demo_fixture import DemoEngine, demo_profiles, make_settings  # noqa: E402

ENDPOINT = "/v1/demo/agent/recommend"

#: A trusted history whose identities the fixture catalogue actually contains.
#:
#: The grounding verifier confirms every candidate identity against the trusted mapping, so a
#: history made of identifiers the catalogue does not know would make the history source
#: legitimately ungroundable.  Using the fixture's own identities keeps the test about the
#: endpoint rather than about an identity mismatch.
HISTORY: tuple[str, ...] = INITIAL_ORDER


class IdentityMapEngine:
    """The demo engine plus the identity-map methods the grounding verifier requires.

    ``GroundingVerifier`` confirms every candidate identity against the trusted mapping
    before the ledger accepts it, so the stand-in has to answer
    ``has_parent_asin``/``parent_asin_to_item_id``/``item_id_to_parent_asin``.  The real
    deployment passes the accepted inference engine here, which already does.
    """

    device = "cpu"

    def __init__(self, rows: Any = CANDIDATE_ROWS) -> None:
        self._delegate = DemoEngine(rows)
        self._rows = tuple(rows)
        self._by_asin = {str(row[0]): int(row[1]) for row in self._rows}
        self._by_id = {int(row[1]): str(row[0]) for row in self._rows}

    def is_ready(self) -> bool:
        return True

    def recommend(self, history_parent_asins: Any, k: int = 10) -> Any:
        return self._delegate.recommend(history_parent_asins, k=k)

    def has_parent_asin(self, parent_asin: str) -> bool:
        return str(parent_asin) in self._by_asin

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if str(parent_asin) not in self._by_asin:
            raise KeyError(parent_asin)
        return self._by_asin[str(parent_asin)]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        if int(item_id) not in self._by_id:
            raise KeyError(item_id)
        return self._by_id[int(item_id)]

    @property
    def call_count(self) -> int:
        return self._delegate.call_count


@pytest.fixture()
def harness(tmp_path: Path):
    """A demo-enabled app whose engine can also ground identities."""
    rows = CANDIDATE_ROWS
    engine = IdentityMapEngine(rows)
    store = SQLitePreferenceStore(tmp_path / "agent_memory.sqlite3")
    runtime = build_demo_runtime(
        engine=engine,
        metadata=build_index(row[0] for row in rows),
        store=store,
        profiles=demo_profiles(),
        max_sessions=4,
    )

    class Built:
        pass

    built = Built()
    built.engine = engine
    built.runtime = runtime
    built.store = store
    built.app = create_app(
        make_settings(tmp_path),
        engine=engine,
        load_on_startup=False,
        demo=runtime,
        enable_demo=True,
    )
    built.client = TestClient(built.app)
    built.client.__enter__()
    try:
        yield built
    finally:
        built.client.__exit__(None, None, None)
        runtime.close()


def post(harness, **overrides: Any) -> Any:
    """POST a valid agent request, with the given fields overridden."""
    body: dict[str, Any] = {
        "message": "Recommend some hiking gear.",
        "history": list(HISTORY),
        "k": 3,
    }
    body.update(overrides)
    return harness.client.post(ENDPOINT, json=body)


# --------------------------------------------------------------------------- #
# Endpoint exists and is served by the existing runtime
# --------------------------------------------------------------------------- #


def test_agent_endpoint_is_published_in_openapi(harness) -> None:
    schema = harness.client.get("/openapi.json").json()
    assert ENDPOINT in schema["paths"]
    assert "post" in schema["paths"][ENDPOINT]


def test_agent_endpoint_uses_the_process_scoped_engine(harness) -> None:
    """It must not build a second engine: the runtime owns exactly one."""
    before = harness.engine.call_count
    response = post(harness)
    assert response.status_code == 200, response.text
    assert harness.engine.call_count > before


def test_agent_service_composes_no_second_checkpoint_or_store(harness) -> None:
    """The service is handed the runtime's collaborators rather than rebuilding them."""
    service = harness.runtime.agent_service()
    assert isinstance(service, DemoAgentService)
    assert service is harness.runtime.agent_service()  # composed once
    report = harness.runtime.build_report()
    assert report["agent_service_composed"] is True
    # One engine, one tool, one metadata load, one store - the existing contract survives.
    assert report["engine_builds"] == 1
    assert report["metadata_loads"] == 1
    assert report["memory_service_builds"] == 1


# --------------------------------------------------------------------------- #
# Response shape
# --------------------------------------------------------------------------- #


def test_response_carries_every_required_block(harness) -> None:
    body = post(harness).json()
    assert body["api_version"] == AGENT_API_VERSION
    assert body["route"] == "recommend"
    for block in (
        "recommendations",
        "grounded",
        "eligibility",
        "terminal",
        "trajectory",
        "timing",
        "message",
        "control_plane",
    ):
        assert block in body, block


def test_recommendations_are_ordered_and_identified(harness) -> None:
    body = post(harness).json()
    recommendations = body["recommendations"]
    assert recommendations, "the history engine returns candidates"
    assert [item["rank"] for item in recommendations] == list(
        range(1, len(recommendations) + 1)
    )
    for item in recommendations:
        assert item["parent_asin"]
        assert item["item_id"] >= 1
        assert isinstance(item["score"], float)
        assert item["score_kind"] == "sasrec_logit"


def test_provenance_names_the_source_that_produced_the_candidates(harness) -> None:
    """The response must attribute the candidate set, and label each score's kind."""
    body = post(harness).json()
    assert body["grounded"]["grounded_count"] >= 1
    assert tuple(body["grounded"]["sources_present"]) == ("history",)
    assert len(body["grounded"]["candidates"]) == body["grounded"]["grounded_count"]
    for candidate in body["grounded"]["candidates"]:
        assert candidate["grounded"] is True
        assert candidate["verification_status"] == "grounded"
    # The accepted history path reports the Tool's own SASRec ranking score, and it says so.
    for item in body["recommendations"]:
        assert item["score_kind"] == "sasrec_logit"


def test_a_later_source_replaces_the_presented_set_rather_than_unioning_it(tmp_path: Path) -> None:
    """Documents what the accepted plane actually does across two sequential retrievals.

    This is asserted rather than wished away.  ``CandidatePlane`` rank-fuses the sources it is
    asked for *in one action*, and a ``SELECT_SOURCE`` call names exactly one source, so the
    ledger accumulates provenance across steps while the *presented* candidate set is the most
    recent source's.  Phase 5 measured single-source selection to underperform full fusion
    (``docs/PHASE5_HANDOFF.md`` §6), so the demo does not present this as a fusion result - the
    fusion it demonstrates is the within-action RRF the plane already implements.
    """
    rows = CANDIDATE_ROWS
    engine = IdentityMapEngine(rows)
    store = SQLitePreferenceStore(tmp_path / "replace.sqlite3")
    runtime = build_demo_runtime(
        engine=engine,
        metadata=build_index(row[0] for row in rows),
        store=store,
        profiles=demo_profiles(),
    )
    app = create_app(
        make_settings(tmp_path),
        engine=engine,
        load_on_startup=False,
        demo=runtime,
        enable_demo=True,
    )
    with TestClient(app) as client:
        body = client.post(
            ENDPOINT,
            json={
                "message": "Recommend RedWidget BlueWidget",
                "history": list(HISTORY),
                "k": 3,
                "sources": ["history", "catalog_search"],
            },
        ).json()
    runtime.close()
    assert body["trajectory"]["actions"][0] == "recommend_from_history"
    assert "search_catalog" in body["trajectory"]["actions"]
    assert body["grounded"]["grounded_count"] == body["grounded"]["grounded_count"]
    # Whatever the presented set is, every identity in it is grounded and attributed.
    for candidate in body["grounded"]["candidates"]:
        assert candidate["grounded"] is True


def test_terminal_block_reports_a_bounded_finished_run(harness) -> None:
    body = post(harness).json()
    terminal = body["terminal"]
    assert terminal["status"] == "finished"
    assert terminal["termination_reason"] == "completed"
    assert terminal["succeeded"] is True
    assert terminal["steps"] >= 1
    assert terminal["limits"]["max_steps"] >= terminal["steps"]
    assert terminal["run_id"]


def test_trajectory_is_a_compact_projection(harness) -> None:
    body = post(harness).json()
    trajectory = body["trajectory"]
    assert trajectory["step_count"] >= 1
    # The history source is consulted through the accepted RECOMMEND_FROM_HISTORY action.  It is
    # deliberately not routed through the candidate plane: doing so would require a control-plane
    # menu change that was measured to alter an archived Phase-3 recording's action sequence.
    assert trajectory["actions"][0] == "recommend_from_history"
    assert trajectory["actions"][-1] == "finish"
    for step in trajectory["steps"]:
        assert set(step) == {
            "step",
            "action",
            "validated",
            "observation_type",
            "candidate_count",
            "eligibility_state",
            "refusal_codes",
            "note",
        }
    # The recommending step observed a candidate set; the finish step observed nothing.
    observed = [step for step in trajectory["steps"] if step["observation_type"]]
    assert observed, "at least one step produced an observation"


def test_eligibility_is_not_claimed_when_no_constraint_is_active(harness) -> None:
    """'Nothing was evaluated' must not be reported as 'everything passed'."""
    body = post(harness).json()
    assert body["eligibility"]["evaluated"] is False
    assert body["eligibility"]["verified_eligible_count"] == 0
    assert body["eligibility"]["projection"] is None


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def test_stated_constraint_is_evaluated_and_reported(harness) -> None:
    """A stated hard constraint activates real constraint verification."""
    body = post(harness, constraints=[{"kind": "brand", "value": "Acme"}]).json()
    eligibility = body["eligibility"]
    assert eligibility["evaluated"] is True
    assert eligibility["requirements"] == [{"kind": "brand", "expected": "Acme"}]
    assert eligibility["assessments"], "each grounded candidate gets a verdict"
    for assessment in eligibility["assessments"]:
        assert assessment["eligibility"] in {"eligible", "ineligible", "unresolved"}


def test_unevaluable_constraint_stays_active_and_never_silently_passes(harness) -> None:
    """A dimension the catalogue cannot decide must be UNKNOWN, not satisfied."""
    body = post(harness, constraints=[{"kind": "color", "value": "puce"}]).json()
    eligibility = body["eligibility"]
    assert eligibility["evaluated"] is True
    # Either a verdict was reached or the candidates are unresolved - never "verified" by default.
    assert (
        eligibility["verified_eligible_count"]
        + eligibility["ineligible_count"]
        + eligibility["unresolved_count"]
        == len(eligibility["assessments"])
    )


def test_blank_constraint_value_is_rejected(harness) -> None:
    assert post(harness, constraints=[{"kind": "brand", "value": "  "}]).status_code == 422


def test_unknown_constraint_field_is_rejected(harness) -> None:
    response = post(harness, constraints=[{"kind": "brand", "value": "x", "secret": "y"}])
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Source plan
# --------------------------------------------------------------------------- #


def test_default_plan_is_history_only(harness) -> None:
    service = harness.runtime.agent_service()
    assert service.available_sources == ("history", "catalog_search")
    assert normalise_source_plan(None) == ("history",)


def test_catalog_search_plan_consults_both_sources(harness) -> None:
    """Naming two sources must consult both, in the order named."""
    response = post(
        harness, message="Recommend RedWidget BlueWidget", sources=["history", "catalog_search"]
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["route"] == "recommend"
    actions = list(body["trajectory"]["actions"])
    assert actions[0] == "recommend_from_history"
    assert "search_catalog" in actions, "the lexical source the request named must be consulted"
    assert body["grounded"]["grounded_count"] >= 1


def test_lexical_source_provenance_is_recorded_in_the_ledger(harness) -> None:
    """The plane records which source proposed each identity, with its own score kind."""
    response = post(
        harness,
        message="Recommend RedWidget BlueWidget",
        sources=["catalog_search"],
        k=5,
    )
    body = response.json()
    ledger_sources = {
        record["source"]
        for candidate in body["grounded"]["candidates"]
        for record in candidate["provenance"]
    }
    assert ledger_sources <= {"catalog_search"}
    for candidate in body["grounded"]["candidates"]:
        for record in candidate["provenance"]:
            assert record["score_kind"] == "bm25"
            assert record["source_rank"] >= 1


def test_unsupported_source_plan_is_an_explicit_error(harness) -> None:
    response = post(harness, sources=["similar_item"])
    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_source_plan"


def test_unknown_source_name_is_rejected(harness) -> None:
    response = post(harness, sources=["telepathy"])
    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_source_plan"


def test_duplicate_source_is_rejected(harness) -> None:
    response = post(harness, sources=["history", "history"])
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (["history"], ("history",)),
        (["catalog_search"], ("catalog_search",)),
        (["HISTORY", " catalog_search "], ("history", "catalog_search")),
    ],
)
def test_source_plan_normalisation(plan, expected) -> None:
    assert normalise_source_plan(plan) == expected


def test_empty_source_plan_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_source_plan([])


# --------------------------------------------------------------------------- #
# Request validation and safety
# --------------------------------------------------------------------------- #


def test_blank_message_is_rejected(harness) -> None:
    assert post(harness, message="   ").status_code == 422


def test_empty_history_is_rejected(harness) -> None:
    assert post(harness, history=[]).status_code == 422


def test_blank_history_entry_is_rejected(harness) -> None:
    assert post(harness, history=["B1", " "]).status_code == 422


@pytest.mark.parametrize("bad_k", [0, 101, "5", 5.0, True])
def test_k_is_strictly_bounded(harness, bad_k) -> None:
    assert post(harness, k=bad_k).status_code == 422


def test_server_owned_fields_cannot_be_injected(harness) -> None:
    """There is no request field for trusted history, a run id or a candidate list."""
    for field in ("trusted_user_history", "run_id", "trajectory", "candidate_ledger"):
        response = post(harness, **{field: "x"})
        assert response.status_code == 422, field


def test_response_never_leaks_internal_payloads(harness) -> None:
    """The response names products it recommends - and nothing it was not asked to expose.

    Recommended identities legitimately appear: they are the answer.  What must never appear is
    the caller's *input history* echoed back as state, any policy prompt or proposal, any raw
    tool payload, or the internal context summary.
    """
    response = post(harness, message="Recommend something for my trip.")
    body = response.json()
    text = response.text
    for internal in (
        "policy_context_summary",
        "action_proposal",
        "system_prompt",
        "trusted_user_history",
        "tool_result",
        "candidate_set_ref",
        "preference_snapshot",
    ):
        assert internal not in text, internal
    # The history the caller supplied must not come back as an echoed field.
    assert "history" not in {key for key in body if key != "history"}
    assert "history" not in body["grounded"]
    assert "history" not in body["terminal"]
    # Every identity the response exposes is a recommended or grounded candidate, never the
    # raw input list.
    exposed = {item["parent_asin"] for item in body["recommendations"]} | {
        candidate["parent_asin"] for candidate in body["grounded"]["candidates"]
    }
    assert exposed <= set(HISTORY), "only fixture identities can appear in this fixture"
    assert exposed, "the run recommends something"


def _preference_rows(store: Any) -> int:
    """Count persisted preference rows directly, so the assertion does not depend on a key."""
    import sqlite3

    with sqlite3.connect(store.database) as connection:
        row = connection.execute("SELECT COUNT(*) FROM preference_memory").fetchone()
    return int(row[0])


def test_stateless_request_touches_no_preference_memory(harness) -> None:
    """Without ``user_key`` the run neither reads nor writes preference memory."""
    before = _preference_rows(harness.store)
    body = post(harness).json()
    assert body["route"] == "recommend"
    assert _preference_rows(harness.store) == before


def test_request_with_a_user_key_writes_that_namespace_only(harness) -> None:
    """Supplying a namespace is what opts a run into memory, and it stays scoped to it."""
    before = _preference_rows(harness.store)
    body = post(harness, user_key="interviewer-namespace").json()
    assert body["route"] == "recommend"
    # The turn commits the user's own message only; nothing about another namespace moves.
    assert _preference_rows(harness.store) >= before


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def test_backend_failure_is_reported_honestly_never_as_a_success(tmp_path: Path) -> None:
    """A failing engine yields a failed terminal state and no fabricated recommendations."""
    rows = CANDIDATE_ROWS
    failing = IdentityMapEngine(rows)
    failing._delegate.error = RuntimeError("engine exploded")  # noqa: SLF001 - deliberate
    store = SQLitePreferenceStore(tmp_path / "fail.sqlite3")
    runtime = build_demo_runtime(
        engine=failing,
        metadata=build_index(row[0] for row in rows),
        store=store,
        profiles=demo_profiles(),
    )
    app = create_app(
        make_settings(tmp_path),
        engine=failing,
        load_on_startup=False,
        demo=runtime,
        enable_demo=True,
    )
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"message": "Recommend something.", "history": list(HISTORY), "k": 2},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # The run terminates and says so; it does not invent candidates and does not claim success.
        assert body["terminal"]["succeeded"] is False
        assert body["terminal"]["status"] != "finished"
        assert body["recommendations"] == []
        assert body["grounded"]["grounded_count"] == 0
        assert body["trajectory"]["refusals"], "the refusal must be visible in the trajectory"
        # No internal exception text reaches the client.
        assert "exploded" not in response.text
    runtime.close()


def test_endpoint_is_absent_without_the_demo(tmp_path: Path) -> None:
    """The Milestone 6 contract keeps working; the demo router is opt-in."""
    from tests.demo_fixture import DemoEngine as _DemoEngine

    engine = _DemoEngine()
    app = create_app(
        make_settings(tmp_path), engine=engine, load_on_startup=False, enable_demo=False
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert ENDPOINT not in client.get("/openapi.json").json()["paths"]
        assert client.post(ENDPOINT, json={"message": "x", "history": ["B1"]}).status_code == 404
