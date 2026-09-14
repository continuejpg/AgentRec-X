"""FastAPI service tests (Milestone 6).

Uses FastAPI's in-process ``TestClient`` against a tiny synthetic checkpoint, so the
suite never loads the 349 MB formal run or the 156,746-item catalog.

The API layer is only an adapter: these tests check the wire contract, error mapping
and lifecycle behaviour, not model quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch", reason="API tests require PyTorch")
fastapi = pytest.importorskip("fastapi", reason="API tests require FastAPI")
from fastapi.testclient import TestClient  # noqa: E402

from tests.sasrec_inference_fixture import (  # noqa: E402
    MAX_SEQ_LEN,
    NUM_ITEMS,
    asin_for_item,
    build_fixture,
)
from recommendation.api.app import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ServiceSettings,
    create_app,
)
from recommendation.inference import InferenceConfig, SASRecInferenceEngine  # noqa: E402

KNOWN = asin_for_item(1)
UNKNOWN = "B9999999999"


@pytest.fixture()
def client(tmp_path: Path):
    """A test client whose model is a tiny synthetic checkpoint.

    The engine is injected so startup does not touch the formal run, and the engine
    identity is exposed so a test can prove the model is not reloaded per request.
    """
    detail = build_fixture(tmp_path)
    settings = ServiceSettings(
        checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
        mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
        manifest_path=detail["manifest_path"],  # type: ignore[arg-type]
        device="cpu",
    )
    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
            mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            manifest_path=detail["manifest_path"],  # type: ignore[arg-type]
            device="cpu",
        )
    )
    app = create_app(settings, engine=engine, load_on_startup=False)
    with TestClient(app) as test_client:
        yield test_client, engine


# --------------------------------------------------------------------------- #
# 29. /health
# --------------------------------------------------------------------------- #


def test_health_reports_ready(client) -> None:
    """A loaded model reports ok with the serving device."""
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "model_loaded": True, "device": "cpu", "detail": None}


def test_health_reports_not_ready_when_model_missing(tmp_path: Path) -> None:
    """The process can be up while the model is NOT loaded, and must say so."""
    settings = ServiceSettings(
        checkpoint_path=tmp_path / "absent.pt",
        mappings_path=tmp_path / "absent_mappings.json",
        manifest_path=None,
        device="cpu",
    )
    app = create_app(settings, load_on_startup=True)
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert body["device"] is None
    assert body["detail"], "an unready service must explain why"


# --------------------------------------------------------------------------- #
# 30. /v1/model
# --------------------------------------------------------------------------- #


def test_model_info_endpoint(client) -> None:
    """Model metadata exposes the frozen architecture and provenance split."""
    test_client, engine = client
    response = test_client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "SASRec"
    assert body["num_items"] == NUM_ITEMS
    assert body["max_seq_len"] == MAX_SEQ_LEN
    assert body["hidden_size"] == engine.model_config.hidden_size
    assert body["device"] == "cpu"
    assert body["checkpoint_sha256"] == engine.checkpoint_sha256
    assert body["provenance"]["formal_run_git"]["commit"] == "0" * 40
    assert "serving_note" in body["provenance"]


def test_model_info_does_not_leak_paths(client) -> None:
    """No local filesystem path is exposed to clients."""
    test_client, _ = client
    text = json.dumps(test_client.get("/v1/model").json())
    assert "/root/" not in text
    assert ".pt" not in text
    assert str(Path.home()) not in text


def test_model_endpoint_errors_when_unready(tmp_path: Path) -> None:
    """An unready service returns a client-safe 500 for model metadata."""
    settings = ServiceSettings(
        checkpoint_path=tmp_path / "absent.pt",
        mappings_path=tmp_path / "absent.json",
        manifest_path=None,
    )
    app = create_app(settings, load_on_startup=True)
    with TestClient(app) as test_client:
        response = test_client.get("/v1/model")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "inference_failed"
    assert "absent.pt" not in body["detail"], "internal paths must not leak"


# --------------------------------------------------------------------------- #
# 31. /v1/recommend success
# --------------------------------------------------------------------------- #


def test_recommend_success(client) -> None:
    """A valid request returns the documented response contract."""
    test_client, engine = client
    response = test_client.post("/v1/recommend", json={"history": [KNOWN], "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {
        "recommendations", "requested_k", "returned_k",
        "history_length", "effective_history_length", "history_truncated",
        "eligible_candidates",
    }
    assert body["requested_k"] == 3
    assert body["returned_k"] == len(body["recommendations"]) == 3
    assert body["history_length"] == 1
    assert body["effective_history_length"] == 1
    assert body["history_truncated"] is False
    for index, item in enumerate(body["recommendations"], start=1):
        assert item["rank"] == index
        assert item["item_id"] != 0
        assert item["parent_asin"] == engine.item_id_to_parent_asin(item["item_id"])
        assert isinstance(item["score"], float)


def test_recommend_defaults_k_to_ten(client) -> None:
    """Omitting k uses the documented default."""
    test_client, _ = client
    response = test_client.post("/v1/recommend", json={"history": [KNOWN]})
    assert response.status_code == 200
    assert response.json()["requested_k"] == 10


def test_recommend_excludes_seen_items(client) -> None:
    """No returned item is one the caller already interacted with."""
    test_client, _ = client
    history = [asin_for_item(i) for i in range(1, 5)]
    body = test_client.post("/v1/recommend", json={"history": history, "k": 5}).json()
    returned = {item["parent_asin"] for item in body["recommendations"]}
    assert not (returned & set(history))


def test_recommend_never_returns_pad(client) -> None:
    """PAD (item_id 0) never appears in recommendations."""
    test_client, _ = client
    body = test_client.post("/v1/recommend", json={"history": [KNOWN], "k": 10}).json()
    assert all(item["item_id"] != 0 for item in body["recommendations"])


def test_recommend_has_no_duplicates(client) -> None:
    """Recommendation ids are distinct."""
    test_client, _ = client
    body = test_client.post("/v1/recommend", json={"history": [KNOWN], "k": 10}).json()
    ids = [item["item_id"] for item in body["recommendations"]]
    assert len(ids) == len(set(ids))


def test_recommend_accepts_duplicate_history_items(client) -> None:
    """Duplicate interactions are valid input and are not rejected."""
    test_client, _ = client
    response = test_client.post(
        "/v1/recommend", json={"history": [KNOWN, KNOWN, asin_for_item(2)], "k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["history_length"] == 3
    assert body["effective_history_length"] == 3


def test_recommend_reports_history_truncation(client) -> None:
    """A history longer than max_seq_len is reported as truncated."""
    test_client, _ = client
    history = [asin_for_item(i) for i in range(1, NUM_ITEMS + 1)]
    body = test_client.post("/v1/recommend", json={"history": history, "k": 10}).json()
    assert body["history_truncated"] is True
    assert body["history_length"] == NUM_ITEMS
    assert body["effective_history_length"] == MAX_SEQ_LEN
    assert body["returned_k"] == 0  # whole catalog seen


def test_recommend_fewer_than_k_candidates(client) -> None:
    """Requesting more than remain returns what is available."""
    test_client, _ = client
    history = [asin_for_item(i) for i in range(1, NUM_ITEMS - 1)]
    body = test_client.post("/v1/recommend", json={"history": history, "k": 50}).json()
    assert body["requested_k"] == 50
    assert body["returned_k"] == 2


def test_recommend_zero_candidates(client) -> None:
    """A fully-seen catalog returns a valid empty list with returned_k = 0."""
    test_client, _ = client
    history = [asin_for_item(i) for i in range(1, NUM_ITEMS + 1)]
    response = test_client.post("/v1/recommend", json={"history": history, "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["returned_k"] == 0


def test_recommend_repeated_request_is_deterministic(client) -> None:
    """Identical requests return identical bodies."""
    test_client, _ = client
    payload = {"history": [KNOWN, asin_for_item(3)], "k": 4}
    baseline = test_client.post("/v1/recommend", json=payload).json()["recommendations"]
    for _ in range(4):
        assert (
            test_client.post("/v1/recommend", json=payload).json()["recommendations"] == baseline
        )


# --------------------------------------------------------------------------- #
# 32-33. Validation and error behaviour
# --------------------------------------------------------------------------- #


def test_empty_history_is_a_client_error(client) -> None:
    """An empty history is rejected with 422, never 500."""
    test_client, _ = client
    response = test_client.post("/v1/recommend", json={"history": [], "k": 5})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert "history" in body["detail"]


def test_unknown_item_is_a_client_error(client) -> None:
    """An unknown parent_asin is a clear client error."""
    test_client, _ = client
    response = test_client.post("/v1/recommend", json={"history": [UNKNOWN], "k": 5})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "unknown_item"
    assert UNKNOWN in body["detail"]


def test_k_bounds_are_enforced(client) -> None:
    """k outside 1..100 is a client error."""
    test_client, _ = client
    for bad_k in (0, -1, 101, 1000):
        response = test_client.post("/v1/recommend", json={"history": [KNOWN], "k": bad_k})
        assert response.status_code == 422, bad_k
        assert response.json()["error"] == "invalid_request"


def test_wrong_field_types_are_client_errors(client) -> None:
    """Wrong types produce structured 422s rather than 500s."""
    test_client, _ = client
    cases = [
        {"history": "not-a-list", "k": 5},
        {"history": [123], "k": 5},
        {"history": [KNOWN], "k": "ten"},
        {"history": [KNOWN], "k": 1.5},
    ]
    for payload in cases:
        response = test_client.post("/v1/recommend", json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"] == "invalid_request"


def test_malformed_json_is_a_client_error(client) -> None:
    """Malformed JSON is a 422, not a 500."""
    test_client, _ = client
    response = test_client.post(
        "/v1/recommend", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_unexpected_fields_are_rejected(client) -> None:
    """Extra body fields are refused by the schema."""
    test_client, _ = client
    response = test_client.post(
        "/v1/recommend", json={"history": [KNOWN], "k": 5, "unexpected": 1}
    )
    assert response.status_code == 422


def test_errors_do_not_leak_internals(client) -> None:
    """Error bodies contain no stack traces, paths or secrets."""
    test_client, _ = client
    for payload in ({"history": [], "k": 5}, {"history": [UNKNOWN], "k": 5}, {"k": 0}):
        body = json.dumps(test_client.post("/v1/recommend", json=payload).json())
        assert "/root/" not in body
        assert "Traceback" not in body
        assert ".pt" not in body


# --------------------------------------------------------------------------- #
# 34. Lifecycle: the model is not reloaded per request
# --------------------------------------------------------------------------- #


def test_model_is_not_reloaded_per_request(client) -> None:
    """The injected engine instance persists across many requests."""
    test_client, engine = client
    weights_before = engine.model.item_embedding.weight.detach().clone()
    loaded_at = engine.loaded_at
    for _ in range(10):
        assert test_client.post("/v1/recommend", json={"history": [KNOWN], "k": 2}).status_code == 200
    assert test_client.app.state.service.engine is engine
    assert engine.loaded_at == loaded_at, "the engine was rebuilt between requests"
    assert engine.model.item_embedding.weight.detach().equal(weights_before)


def test_openapi_documents_the_three_endpoints(client) -> None:
    """The documented endpoints are all published."""
    test_client, _ = client
    schema = test_client.get("/openapi.json").json()
    assert {"/health", "/v1/model", "/v1/recommend"} <= set(schema["paths"])


def test_accepted_checkpoint_constant_is_a_sha256() -> None:
    """The formal checkpoint identity constant is a well-formed digest."""
    assert len(ACCEPTED_CHECKPOINT_SHA256) == 64
    assert all(c in "0123456789abcdef" for c in ACCEPTED_CHECKPOINT_SHA256)


def test_model_metadata_keys_match_schema_exactly(client) -> None:
    """Regression: the served metadata must not contain keys the schema forbids.

    The response models use ``extra="forbid"``, so an engine metadata field that is
    absent from the schema turns ``GET /v1/model`` into a 500.  This test pins the two
    definitions together so adding a metadata key cannot silently break the endpoint.
    """
    test_client, engine = client
    response = test_client.get("/v1/model")
    assert response.status_code == 200, response.text
    assert set(response.json()) == set(engine.model_metadata())
    assert response.json()["model_parameters_frozen"] is True
    assert response.json()["parameter_count"] > 0


def test_engine_metadata_has_no_unexpected_keys(client) -> None:
    """Every key the engine reports is part of the published response contract."""
    from recommendation.api.schemas import ModelInfoResponse

    _, engine = client
    allowed = set(ModelInfoResponse.model_fields)
    unexpected = set(engine.model_metadata()) - allowed
    assert not unexpected, f"engine metadata would break /v1/model: {unexpected}"
