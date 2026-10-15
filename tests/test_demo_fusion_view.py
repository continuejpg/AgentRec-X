"""Contract tests for the four-source fusion / Agent dispatch demo views.

These are static-plus-schema tests: they prove the view is actually served, that it talks to the
real endpoints, that it never treats a *configured* TIGER source as a *ready* one, and that the
payloads it sends are valid against the API's own request models.  Browser rendering is **not**
covered here (no browser automation in this environment) - see the P4-DEMO-01 report.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from recommendation.api.schemas import RecommendResponse
from recommendation.demo.agent_schemas import AgentRecommendRequest

WEB = pathlib.Path(__file__).resolve().parents[1] / "recommendation" / "web"


def test_the_page_loads_the_fusion_view() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '/demo/fusion.js' in html, "the four-source view must be served with the demo page"


def test_the_view_calls_only_real_endpoints() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    for endpoint in ('"/health"', '"/v1/recommend"', '"/v1/demo/agent/recommend"'):
        assert endpoint in js, endpoint


def test_the_view_never_fabricates_results() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    for forbidden in ("Math.random", "mock", "faker", "lorem ipsum", "placeholder-item"):
        assert forbidden not in js, f"the view must not synthesise data ({forbidden})"


def test_tiger_is_shown_as_serving_only_when_the_worker_is_ready() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    # A configured source plus a ready worker, never the source list alone.
    assert 'state4 === "ready"' in js
    assert 'sources.indexOf("tiger") >= 0 && state4 === "ready"' in js
    # And the four states are all rendered.
    for state in ("disabled", "loading", "ready", "failed"):
        assert state in js, state


def test_the_view_escapes_untrusted_text() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js, "untrusted API text must be written with textContent only"
    assert "textContent" in js


def four_source_payload() -> dict:
    return {
        "recommendations": [{"rank": 1, "item_id": 119194, "parent_asin": "B08HWP19XK",
                             "score": 0.04918032786885246}],
        "requested_k": 1, "returned_k": 1, "history_length": 3,
        "effective_history_length": 3, "history_truncated": False, "eligible_candidates": 58,
        "timings_ms": {"source_tiger_ms": 1404.429, "fusion_ms": 0.238},
        "score_kind": "rrf_rank_fusion",
        "provenance": {"B08HWP19XK": {
            "item_id": 119194, "fused_rank": 1, "fused_score": 0.04918032786885246,
            "score_kind": "rrf_rank_fusion",
            "sources": ["sequential", "two_tower", "tiger"],
            "source_ranks": {"sequential": 1, "two_tower": 1, "tiger": 1},
            "source_score_kinds": {"sequential": "sasrec_logit", "two_tower": "two_tower_cosine",
                                   "tiger": "tiger_sid_sequence_logprob"},
            "tiger": {"rank": 1, "score": -5.069414145218616,
                      "score_kind": "tiger_sid_sequence_logprob", "item_id": 119194,
                      "semantic_id": [104, 44, 2, 0], "retrieval_mode": "CERTIFIED"}}},
    }


def test_the_real_serving_response_shape_renders() -> None:
    """The exact payload the fusion view consumes, validated by the API's own model."""
    response = RecommendResponse(**four_source_payload())
    entry = response.provenance["B08HWP19XK"]
    assert entry["sources"] == ["sequential", "two_tower", "tiger"]
    assert entry["tiger"]["retrieval_mode"] == "CERTIFIED"
    assert entry["tiger"]["score_kind"] == "tiger_sid_sequence_logprob"
    assert response.score_kind == "rrf_rank_fusion"


def test_a_candidate_without_tiger_reports_only_its_real_sources() -> None:
    payload = four_source_payload()
    entry = payload["provenance"]["B08HWP19XK"]
    entry["sources"] = ["sequential", "two_tower"]
    entry.pop("tiger")
    rendered = RecommendResponse(**payload).provenance["B08HWP19XK"]
    assert "tiger" not in rendered
    assert rendered["sources"] == ["sequential", "two_tower"]


def test_the_agent_payload_the_view_sends_is_valid() -> None:
    """The view builds this body; the API's request model must accept it."""
    body = {"message": "find me something for hiking",
            "history": ["B0BX5QFWQN", "B0BKFGDBBM", "B07BWFJVPH"], "k": 5,
            "sources": ["history", "tiger"]}
    request = AgentRecommendRequest(**body)
    assert request.message and request.history and request.sources
    assert request.user_key is None, "the view must not generate a user_key"


def test_the_agent_view_uses_agent_source_names_not_serving_names() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    agent_block = js.split("var AGENT_SOURCES")[1].split("];")[0]
    for name in ("history", "catalog_search", "two_tower", "tiger"):
        assert name in agent_block
    assert "catalog_lexical" not in agent_block, "serving names must not be sent to the Agent"


# --------------------------------------------------------------------------- #
# P4-DEMO-02: real titles, the ten-identity sample, and fallbacks
# --------------------------------------------------------------------------- #


class _Record:
    def __init__(self, title):
        self.title = title


def test_title_projection_returns_real_titles_and_null_for_unknown() -> None:
    from recommendation.api.app import title_projection

    catalogue = {"B0AAA": _Record("Hiking Pole"), "B0BBB": _Record(""), "B0CCC": _Record(None)}
    out = title_projection(["B0AAA", "B0BBB", "B0CCC", "B0DDD"], catalogue)
    assert out["titles"]["B0AAA"] == "Hiking Pole"
    assert out["titles"]["B0BBB"] is None, "an empty title must fall back to the ASIN"
    assert out["titles"]["B0CCC"] is None
    assert out["titles"]["B0DDD"] is None, "an unknown identity is not invented"
    assert out["count"] == 4 and out["with_title"] == 1


def test_title_projection_deduplicates_and_bounds_the_request() -> None:
    from recommendation.api.app import title_projection

    catalogue = {"B0AAA": _Record("T")}
    assert title_projection(["B0AAA", "B0AAA"], catalogue)["count"] == 1
    with pytest.raises(ValueError):
        title_projection([f"B0{i}" for i in range(51)], catalogue)


def test_title_projection_survives_a_broken_catalogue() -> None:
    from recommendation.api.app import title_projection

    class _Broken:
        def get(self, _key):
            raise RuntimeError("index gone")

    assert title_projection(["B0AAA"], _Broken())["titles"]["B0AAA"] is None
    assert title_projection(["B0AAA"], None)["titles"]["B0AAA"] is None


def test_the_view_uses_the_ten_verified_sample_identities() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    expected = ["B07C1D8DKL", "B01KI10HVM", "B0C4MNRN7R", "B077L2HYKH", "B016YLUXLE",
                "B08CTBYWFG", "B0BHNPVSQ3", "B071RB7QKX", "B000AO7JIW", "B075V9TG8V"]
    block = js.split("var SAMPLE_HISTORY = [")[1].split("];")[0]
    for asin in expected:
        assert asin in block, asin
    assert js.count("B07C1D8DKL") == 1, "the sample history appears exactly once"


def test_the_view_fetches_titles_only_for_returned_identities() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert 'var TITLES = "/v1/catalogue/titles";' in js
    assert "els.titleNodes" in js and "loadTitles()" in js
    assert "no title in the catalogue for this identity" in js
    assert "title lookup unavailable" in js, "a failed lookup must degrade, not blank the row"


# --------------------------------------------------------------------------- #
# P4-DEMO-03: the app must actually build (a real runtime check, not a source grep)
# --------------------------------------------------------------------------- #


class _EngineStub:
    """Just enough engine for /health and the routes: no model, no worker."""

    device = "cpu"
    catalogue: dict = {}

    def is_ready(self) -> bool:
        return False


def test_create_app_returns_a_working_asgi_app_with_the_required_routes() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from recommendation.api.app import create_app

    app = create_app(engine=_EngineStub(), load_on_startup=False, warmup=False)  # type: ignore[arg-type]
    assert app is not None, "create_app() must return the application (a None here 500s every route)"
    assert isinstance(app, FastAPI)

    paths = {getattr(route, "path", None) for route in app.routes}
    # The serving routes are unconditional.  The /v1/demo/* router is mounted only when a demo
    # runtime can be composed, which needs the real artifacts - so it is asserted live in the HTTP
    # acceptance run rather than here.
    for required in ("/health", "/v1/recommend", "/v1/catalogue/titles"):
        assert required in paths, f"{required} is not registered"

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert "tiger" in health.json(), "the health surface reports TIGER's state"
        titles = client.get("/v1/catalogue/titles", params={"asins": "B0AAA,B0BBB"})
        assert titles.status_code == 200
        body = titles.json()
        assert body["titles"] == {"B0AAA": None, "B0BBB": None}, "no catalogue -> null, never invented"
        assert body["count"] == 2 and body["with_title"] == 0
        too_many = client.get("/v1/catalogue/titles",
                              params={"asins": ",".join(f"B{i}" for i in range(51))})
        assert too_many.status_code == 422, "the projection is bounded"
        page = client.get("/demo/")
        assert page.status_code in (200, 404), "the static mount must not error"


# --------------------------------------------------------------------------- #
# P4-DEMO-04: presentation layers, loading-vs-failure wording, no false eligibility
# --------------------------------------------------------------------------- #


def test_the_unready_detail_names_a_loading_tiger_worker() -> None:
    from types import SimpleNamespace

    from recommendation.api.demo_routes import _unready_detail

    loading = SimpleNamespace(model_loaded=False,
                              engine=SimpleNamespace(tiger_health=lambda: {"state": "loading"}))
    assert "loading" in _unready_detail(loading) and "TIGER" in _unready_detail(loading)

    failed = SimpleNamespace(model_loaded=False, engine=SimpleNamespace(
        tiger_health=lambda: {"state": "failed", "error": "handshake timeout"}))
    assert "failed" in _unready_detail(failed) and "handshake timeout" in _unready_detail(failed)

    model = SimpleNamespace(model_loaded=False, engine=SimpleNamespace(tiger_health=lambda: None))
    assert "model is still loading" in _unready_detail(model)

    unknown = SimpleNamespace(model_loaded=True, engine=SimpleNamespace())
    assert "not fully constructed" in _unready_detail(unknown)


def test_the_agent_view_is_layered() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    for heading in ("1. Execution summary", "2. Action timeline", "3. Recommendations ("):
        assert heading in js, heading
    assert "executed by the controller's" in js and "deterministic policy" in js


def test_the_agent_view_never_claims_eligibility_without_evidence() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert '"Eligibility: not evaluated"' in js
    assert '"Eligibility: evaluated"' in js
    # "Verified" must never be asserted from an absent payload.
    assert "Eligibility: verified" not in js and "eligibility verified" not in js.lower()


def test_tool_call_count_is_derived_from_real_steps() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert 'step.action !== "finish"' in js, "finish is not a tool call"
    assert "trajectory.tool_calls || trajectory.steps" in js


def test_raw_evidence_is_collapsed_but_kept() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    for label in ("Show raw technical evidence", "Show technical details",
                  "Show full response (raw evidence)"):
        assert label in js, label
    assert js.count("raw.hidden = true") >= 1 and "rawBody.hidden = true" in js


def test_the_sample_message_matches_the_sample_history() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert "Recommend some cycling accessories based on my purchase history." in js
    assert "find me something for hiking" not in js
    # The message must not be presented as a source constraint unless the code proves it.
    assert "does not constrain" not in js


def test_the_fusion_and_tiger_scores_stay_distinguishable() -> None:
    js = (WEB / "fusion.js").read_text(encoding="utf-8")
    assert "sequence log-probability, not a probability" in js
    assert "NOT a probability" in js
    assert 'score_kind: " + scoreKind' in js
