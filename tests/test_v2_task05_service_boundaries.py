"""V2 task 05: system-level boundary tests over the real /v1/recommend route.

Runs the real application (lifespan loads the real models) under the SASRec-only profile and checks
the accepted schema boundaries: normal history, short history, over-long history (frozen truncation),
illegal input, several legal K values, and startup/shutdown.  No production rule, score or schema is
changed, and nothing here duplicates the frozen 100-case parity runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

pytestmark = pytest.mark.skipif(
    not (MAPPINGS.exists() and SEQUENCES.exists()),
    reason="the real processed artifacts are not present in this checkout",
)


@pytest.fixture(scope="module")
def client():
    os.environ["AGENTRECX_FUSION_PROFILE"] = "serving_v1_sequential_only"
    from fastapi.testclient import TestClient

    from recommendation.api.app import app

    # raise_server_exceptions=False so a server-side failure is observed as its real HTTP status
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def real_history():
    """A real parent_asin history taken from the accepted artifact (no user identity exported)."""
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    id_to_asin = {int(v): str(k) for k, v in mappings["item2id"].items()}
    from experiments.e04b2_gate_a import stream_sequences

    for element in stream_sequences(SEQUENCES):
        items = [int(v) for v in element.get("item_ids", ())]
        if len(items) >= 6:
            return [id_to_asin[i] for i in items]
    pytest.skip("no usable history in the artifact")


def _ids(body: dict) -> list[int]:
    return [int(card["item_id"]) for card in body.get("recommendations", [])]


def test_normal_history_returns_k_legal_cards(client, real_history) -> None:
    response = client.post("/v1/recommend", json={"history": real_history, "k": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["requested_k"] == 10 and body["returned_k"] <= 10
    ids = _ids(body)
    assert ids and len(ids) == body["returned_k"]
    assert len(set(ids)) == len(ids)
    assert all(card["rank"] == index for index, card in enumerate(body["recommendations"], start=1))
    assert all(isinstance(card["parent_asin"], str) and card["parent_asin"] for card in body["recommendations"])


def test_single_item_history_is_accepted(client, real_history) -> None:
    response = client.post("/v1/recommend", json={"history": real_history[-1:], "k": 5})
    assert response.status_code == 200
    assert len(_ids(response.json())) <= 5


def test_over_long_history_uses_the_frozen_truncation(client, real_history) -> None:
    long_history = (real_history * 20)[:120]          # far beyond max_seq_len = 50
    response = client.post("/v1/recommend", json={"history": long_history, "k": 20})
    assert response.status_code == 200
    body = response.json()
    assert body["history_length"] == len(long_history)   # the caller's length is echoed
    assert 0 < len(_ids(body)) <= 20


@pytest.mark.parametrize("k", [1, 5, 20, 50])
def test_legal_k_values_are_accepted(client, real_history, k: int) -> None:
    response = client.post("/v1/recommend", json={"history": real_history, "k": k})
    assert response.status_code == 200
    body = response.json()
    assert body["requested_k"] == k
    assert len(_ids(body)) <= k


@pytest.mark.parametrize("payload", [
    {"history": [], "k": 5},                       # empty history
    {"history": [""], "k": 5},                     # empty identity
    {"history": [123], "k": 5},                    # wrong type
    {"k": 5},                                      # missing history
])
def test_illegal_inputs_are_refused_without_a_500(client, payload) -> None:
    response = client.post("/v1/recommend", json=payload)
    assert response.status_code in (400, 404, 422), response.status_code
    assert response.status_code < 500


def test_unknown_identity_is_a_client_error(client, real_history) -> None:
    """Task 05 fix: a history identity outside the served catalogue is a 422, not a 500."""
    response = client.post("/v1/recommend", json={"history": ["NOT-A-REAL-ASIN-000"], "k": 5})
    assert response.status_code == 422, response.status_code
    body = response.json()
    assert body.get("error") == "unknown_item" and body.get("detail")
    # no stack trace, no internal path, no sensitive content leaks to the caller
    assert "Traceback" not in json.dumps(body) and "/mnt/" not in json.dumps(body)


def test_an_internal_failure_is_still_a_server_error(client, real_history, monkeypatch) -> None:
    """The narrow mapping must not swallow genuine engine failures."""
    import recommendation.api.app as app_module

    class _Boom:
        def recommend(self, *args, **kwargs):
            raise RuntimeError("simulated internal failure")

    state = None
    for candidate in (getattr(app_module, "state", None),):
        state = candidate
    if state is None or not hasattr(state, "engine"):
        pytest.skip("the engine handle is not reachable from this module")
    original = state.engine
    state.engine = _Boom()
    try:
        response = client.post("/v1/recommend", json={"history": real_history, "k": 5})
        assert response.status_code >= 500, response.status_code
    finally:
        state.engine = original


@pytest.mark.parametrize("k", [0, -1, 10_000])
def test_illegal_k_values_are_refused(client, real_history, k: int) -> None:
    response = client.post("/v1/recommend", json={"history": real_history, "k": k})
    assert response.status_code == 422, response.status_code


def test_response_matches_the_official_schema_keys(client, real_history) -> None:
    body = client.post("/v1/recommend", json={"history": real_history, "k": 3}).json()
    for key in ("recommendations", "requested_k", "returned_k", "history_length"):
        assert key in body, key
    for card in body["recommendations"]:
        assert set(card) >= {"rank", "item_id", "parent_asin", "score"}


def test_no_tiger_or_lexical_source_is_active_in_the_single_source_profile() -> None:
    from recommendation.inference.fusion import FUSION_PROFILES

    assert FUSION_PROFILES["serving_v1_sequential_only"] == ("sequential",)
    assert FUSION_PROFILES["serving_v1_two_tower_only"] == ("two_tower",)
    for name in ("serving_v1_sequential_only", "serving_v1_two_tower_only"):
        assert "tiger" not in FUSION_PROFILES[name]
        assert "catalog_lexical" not in FUSION_PROFILES[name]
