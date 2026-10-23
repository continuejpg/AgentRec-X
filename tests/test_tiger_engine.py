"""Contract tests for the live TIGER serving engine (fake worker transport).

These tests never load a TIGER checkpoint and never start the backend interpreter: they drive
:class:`TigerServingEngine` through an injected :class:`FakeTransport` so the *engine-side*
contract can be tested on CPU with no backend assets.  The real-model path lives in
``tests/test_tiger_engine_real.py`` and is skipped when the accepted checkpoint is absent.

What is asserted here
---------------------
* a valid history completes a recommendation and identity is re-attached from the trusted map;
* PAD (item id 0), non-integers, negative ids and unknown parent ASINs are refused;
* an item id the trusted map cannot resolve is a hard failure - never substituted or dropped;
* a Semantic ID is never mistaken for a product identity;
* ``CERTIFIED`` is never silently relabelled ``APPROXIMATE`` (and vice versa);
* worker failure, worker timeout and missing assets are explicit, typed failures;
* results are deterministic and carry their provenance;
* the result satisfies the existing ``TigerSourceTool`` seam, including ``score_kind``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from recommendation.inference.tiger_engine import (
    APPROXIMATE,
    CERTIFIED,
    TigerAssetsMissing,
    TigerContractViolation,
    TigerEngineError,
    TigerServingEngine,
    TigerWorkerConfig,
    TigerWorkerError,
    TigerWorkerTimeout,
)
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_plane import CandidatePlane
from recommendation.control.policy import PolicyActionError
from recommendation.control.tiger_source import TIGER_SCORE_KIND, TigerSourceTool

# --------------------------------------------------------------------------- #
# Fixtures: a three-product trusted catalogue and a canned worker
# --------------------------------------------------------------------------- #

IDENTITY: dict[str, int] = {
    "B0TRUSTED01": 11,
    "B0TRUSTED02": 22,
    "B0TRUSTED03": 33,
    "B0OTHER0004": 44,
}

ITEM_PAYLOAD: list[dict[str, Any]] = [
    {"item_id": 33, "rank": 1, "score": -1.25, "semantic_id": [7, 12, 200],
     "token_path": [1007, 1512, 1700], "resolver": {"path_item": 33}},
    {"item_id": 11, "rank": 2, "score": -2.5, "semantic_id": [7, 13, 201],
     "token_path": [1007, 1513, 1701], "resolver": {"path_item": 11}},
    {"item_id": 22, "rank": 3, "score": -3.75, "semantic_id": [8, 12, 202],
     "token_path": [1008, 1512, 1702], "resolver": {"path_item": 22}},
]


def result_frame(items: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "op": "recommend_result",
        "request_id": 1,
        "contract": "agentrecx.tiger.serve.v1",
        "retrieval_mode": CERTIFIED,
        "requested_mode": CERTIFIED,
        "k": 3,
        "required_frontier": 6,
        "history_length": 3,
        "wall_seconds": 0.5,
        "checkpoint_sha256": "a" * 64,
        "sid_artifact_sha256": "b" * 64,
        "resolver": {
            "candidate_item_for_path": "smallest_item_id_wins",
            "collision_policy": "smallest_item_id_wins",
            "levels": 3,
            "dedup_levels": 2,
            "collisions_observed": 0,
        },
        "summary": {"status": CERTIFIED, "candidates": len(items or ITEM_PAYLOAD)},
        "items": ITEM_PAYLOAD if items is None else items,
    }
    frame.update(overrides)
    return frame


class FakeTransport:
    """A transport that replays canned frames and records every request frame."""

    def __init__(
        self,
        *,
        handshake: Mapping[str, Any] | None = None,
        responses: list[Any] | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.handshake = dict(
            handshake
            if handshake is not None
            else {
                "op": "ready",
                "contract": "agentrecx.tiger.serve.v1",
                "device": "cpu",
                "load_seconds": 12.5,
                "resolver": result_frame()["resolver"],
                "identity": {
                    "catalogue_items": 156746,
                    "checkpoint_sha256": "a" * 64,
                    "sid_artifact_sha256": "b" * 64,
                    "catalogue_sha256": "c" * 64,
                    "generator_layout_sha256": "d" * 64,
                    "exposure_sha256": "e" * 64,
                },
            }
        )
        self.responses = list(responses or [result_frame()])
        self.start_error = start_error
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def start(self, timeout: float) -> dict[str, Any]:
        if self.start_error is not None:
            raise self.start_error
        return self.handshake

    def request(self, frame: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.sent.append(json.loads(json.dumps(frame)))
        response = self.responses.pop(0) if self.responses else result_frame()
        if isinstance(response, Exception):
            raise response
        return dict(response)

    def close(self) -> None:
        self.closed = True

    def stderr_tail(self) -> str:
        return "fake stderr"


def build_engine(transport: FakeTransport, **kwargs: Any) -> TigerServingEngine:
    """An engine wired to the fake transport; assets are not checked on this path."""
    kwargs.setdefault("k_max", 20)
    return TigerServingEngine(
        IDENTITY,
        config=TigerWorkerConfig(k_max=kwargs.pop("k_max")),
        transport_factory=lambda _config: transport,
        verify_assets_on_start=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. A valid history completes inference
# --------------------------------------------------------------------------- #


def test_valid_history_completes_and_reattaches_trusted_identity() -> None:
    transport = FakeTransport()
    with build_engine(transport) as engine:
        result = engine.recommend((11, 33), k=3)

    assert result.retrieval_mode == CERTIFIED
    assert result.item_ids == (33, 11, 22)
    assert result.parent_asins == ("B0TRUSTED03", "B0TRUSTED01", "B0TRUSTED02")
    assert [item.rank for item in result.recommendations] == [1, 2, 3]
    assert [item.score for item in result.recommendations] == [-1.25, -2.5, -3.75]
    assert result.recommendations[0].semantic_id == (7, 12, 200)
    assert result.checkpoint_sha256 == "a" * 64
    assert result.sid_artifact_sha256 == "b" * 64

    # The request carried internals only: no canonical identity, no target, no scores.
    (sent,) = transport.sent
    assert sent["op"] == "recommend"
    assert sent["history"] == [11, 33]
    assert sent["k"] == 3
    assert all(isinstance(value, int) for value in sent["history"])
    flat = json.dumps(sent)
    assert "B0TRUSTED" not in flat
    assert "target" not in flat


def test_frontier_defaults_to_the_project_convention() -> None:
    """``required_frontier == k + |unique history|`` (step26_gates.py G16.1)."""
    transport = FakeTransport()
    with build_engine(transport) as engine:
        engine.recommend((11, 11, 22), k=20)
    assert transport.sent[0]["required_frontier"] == 20 + 2


def test_explicit_frontier_is_passed_through() -> None:
    transport = FakeTransport()
    with build_engine(transport) as engine:
        engine.recommend((11,), k=5, required_frontier=99)
    assert transport.sent[0]["required_frontier"] == 99


# --------------------------------------------------------------------------- #
# 2/3. Rejections: PAD, unknown ids, bad history
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "history",
    [
        (),
        (0,),                      # PAD is never an item
        (-3,),                     # negative id
        (11, "B0TRUSTED01"),       # a string is not an item id
        (11, True),                # bool is not an item id
        (11, 2.5),                 # float is not an item id
    ],
)
def test_bad_histories_are_refused_before_any_request(history: tuple[Any, ...]) -> None:
    transport = FakeTransport()
    engine = build_engine(transport)  # not started: validation must fail first
    with pytest.raises(TigerEngineError):
        engine.recommend(history, k=3)
    assert transport.sent == []


def test_history_that_cannot_be_encoded_is_a_source_local_failure() -> None:
    from recommendation.control.candidate_plane import CandidateSourceUnavailable

    transport = FakeTransport()
    tool = TigerSourceTool(build_engine(transport), max_k=5)
    with pytest.raises(CandidateSourceUnavailable) as excinfo:
        tool.propose(arguments=None, limit=5, history=("B0NOTINMAP",))
    assert "could not resolve the trusted history" in str(excinfo.value)
    assert tool.last_outcome["reason"] == "history_unmappable"
    assert transport.sent == []


def test_unmappable_item_id_from_the_worker_fails_closed() -> None:
    payload = [dict(ITEM_PAYLOAD[0], item_id=999999)]
    transport = FakeTransport(responses=[result_frame(items=payload)])
    with build_engine(transport) as engine:
        with pytest.raises(TigerContractViolation) as excinfo:
            engine.recommend((11,), k=1)
    assert "trusted identity map" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 4. A Semantic ID is not an identity
# --------------------------------------------------------------------------- #


def test_semantic_id_is_never_used_as_the_product_identity() -> None:
    # A Semantic ID whose digits collide with real item ids must not change identity.
    payload = [dict(ITEM_PAYLOAD[1], rank=1, semantic_id=[11, 22, 33])]
    transport = FakeTransport(responses=[result_frame(items=payload)])
    with build_engine(transport) as engine:
        result = engine.recommend((22,), k=1)
    (item,) = result.recommendations
    assert item.item_id == 11
    assert item.parent_asin == "B0TRUSTED01"
    assert item.semantic_id == (11, 22, 33)


def test_item_id_zero_from_the_worker_is_refused() -> None:
    payload = [dict(ITEM_PAYLOAD[0], item_id=0)]
    transport = FakeTransport(responses=[result_frame(items=payload)])
    with build_engine(transport) as engine:
        with pytest.raises(TigerContractViolation) as excinfo:
            engine.recommend((11,), k=1)
    assert "PAD" in str(excinfo.value) or "item_id" in str(excinfo.value)


def test_missing_semantic_id_is_refused() -> None:
    payload = [{k: v for k, v in ITEM_PAYLOAD[0].items() if k != "semantic_id"}]
    transport = FakeTransport(responses=[result_frame(items=payload)])
    with build_engine(transport) as engine:
        with pytest.raises(TigerContractViolation):
            engine.recommend((11,), k=1)


def test_non_contiguous_ranks_are_refused() -> None:
    payload = [dict(ITEM_PAYLOAD[0], rank=1), dict(ITEM_PAYLOAD[1], rank=5)]
    transport = FakeTransport(responses=[result_frame(items=payload)])
    with build_engine(transport) as engine:
        with pytest.raises(TigerContractViolation) as excinfo:
            engine.recommend((11,), k=2)
    assert "contiguous" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 5/6. Mode fidelity
# --------------------------------------------------------------------------- #


def test_certified_is_never_downgraded_silently() -> None:
    transport = FakeTransport(responses=[result_frame(retrieval_mode=APPROXIMATE)])
    with build_engine(transport, mode=CERTIFIED) as engine:
        with pytest.raises(TigerContractViolation) as excinfo:
            engine.recommend((11,), k=2)
    assert "never reported as certified" in str(excinfo.value)


def test_downgrade_is_allowed_only_when_explicitly_requested() -> None:
    transport = FakeTransport(responses=[result_frame(retrieval_mode=APPROXIMATE)])
    with build_engine(transport, mode=CERTIFIED, allow_mode_downgrade=True) as engine:
        result = engine.recommend((11,), k=2)
    assert result.retrieval_mode == APPROXIMATE
    assert result.requested_mode == CERTIFIED


def test_approximate_request_reports_approximate() -> None:
    transport = FakeTransport(responses=[result_frame(retrieval_mode=APPROXIMATE)])
    with build_engine(transport, mode=APPROXIMATE) as engine:
        result = engine.recommend((11,), k=2)
    assert result.retrieval_mode == APPROXIMATE
    assert transport.sent[0]["mode"] == APPROXIMATE


def test_unknown_mode_from_the_worker_is_refused() -> None:
    transport = FakeTransport(responses=[result_frame(retrieval_mode="BEST_EFFORT")])
    with build_engine(transport) as engine:
        with pytest.raises(TigerContractViolation):
            engine.recommend((11,), k=2)


# --------------------------------------------------------------------------- #
# 7. Failure modes
# --------------------------------------------------------------------------- #


def test_worker_error_frame_is_a_typed_failure() -> None:
    transport = FakeTransport(
        responses=[TigerWorkerError("retrieval failed", code="retrieval_failed")]
    )
    with build_engine(transport) as engine:
        with pytest.raises(TigerWorkerError) as excinfo:
            engine.recommend((11,), k=2)
    assert excinfo.value.code == "retrieval_failed"


def test_worker_timeout_is_a_typed_failure() -> None:
    transport = FakeTransport(responses=[TigerWorkerTimeout("no answer", code="timeout")])
    with build_engine(transport) as engine:
        with pytest.raises(TigerWorkerTimeout):
            engine.recommend((11,), k=2)


def test_missing_assets_are_reported_with_every_missing_path(tmp_path: Path) -> None:
    config = TigerWorkerConfig(
        checkpoint_dir=tmp_path / "checkpoint",
        sid_dir=tmp_path / "sid",
        catalogue_dir=tmp_path / "handoff",
    )
    engine = TigerServingEngine(IDENTITY, config=config, transport_factory=lambda _c: FakeTransport())
    missing = engine.missing_assets()
    assert any("tiger.pt" in entry for entry in missing)
    assert any("catalogue.json" in entry for entry in missing)
    with pytest.raises(TigerAssetsMissing):
        engine.start()


def test_missing_assets_short_circuit_the_worker_entirely(tmp_path: Path) -> None:
    transport = FakeTransport()
    config = TigerWorkerConfig(
        checkpoint_dir=tmp_path / "checkpoint",
        sid_dir=tmp_path / "sid",
        catalogue_dir=tmp_path / "handoff",
    )
    engine = TigerServingEngine(IDENTITY, config=config, transport_factory=lambda _c: transport)
    with pytest.raises(TigerAssetsMissing):
        engine.start()
    assert transport.sent == []


def test_pinned_asset_hashes_are_enforced() -> None:
    transport = FakeTransport()
    engine = TigerServingEngine(
        IDENTITY,
        config=TigerWorkerConfig(),
        transport_factory=lambda _c: transport,
        verify_assets_on_start=False,
        expected_checkpoint_sha256="f" * 64,
    )
    with pytest.raises(TigerContractViolation) as excinfo:
        engine.start()
    assert "pinned" in str(excinfo.value)
    assert transport.closed is True


def test_wrong_contract_handshake_is_refused() -> None:
    transport = FakeTransport(handshake={"op": "ready", "contract": "something.else"})
    engine = TigerServingEngine(
        IDENTITY, transport_factory=lambda _c: transport, verify_assets_on_start=False
    )
    with pytest.raises(TigerContractViolation):
        engine.start()


# --------------------------------------------------------------------------- #
# 8. Determinism and provenance
# --------------------------------------------------------------------------- #


def test_repeated_requests_are_deterministic_and_identify_their_provenance() -> None:
    transport = FakeTransport(responses=[result_frame(), result_frame(request_id=2)])
    with build_engine(transport) as engine:
        first = engine.recommend((11, 22), k=3)
        second = engine.recommend((11, 22), k=3)
        report = engine.asset_report()

    assert [item.as_dict() for item in first.recommendations] == [
        item.as_dict() for item in second.recommendations
    ]
    assert transport.sent[0]["history"] == transport.sent[1]["history"]
    assert report["checkpoint_sha256"] == "a" * 64
    assert report["sid_artifact_sha256"] == "b" * 64
    assert report["device"] == "cpu"
    assert report["paths"]["checkpoint_dir"].endswith("tiger_public_2026/checkpoint")


# --------------------------------------------------------------------------- #
# 9. The existing source seam
# --------------------------------------------------------------------------- #


def test_result_satisfies_tiger_source_tool() -> None:
    transport = FakeTransport()
    tool = TigerSourceTool(build_engine(transport), max_k=10)
    with tool.engine:
        triples = tool.propose(arguments=None, limit=3, history=("B0TRUSTED01", "B0TRUSTED03"))

    assert tool.source is CandidateSource.TIGER
    assert tool.score_kind == TIGER_SCORE_KIND
    # Two of the three certified rows are the caller's own history, and the source masks
    # seen items by design (``tiger_source.py`` step 2, "Deterministic seen-item filter");
    # ranks are renumbered after filtering.  The pre-mask expectation predated that step.
    assert triples == [("B0TRUSTED02", 1, -3.75)]
    assert tool.last_outcome["seen_filtered"] == 2
    proposal = tool.last_proposals[0]
    assert proposal.retrieval_mode == CERTIFIED
    assert proposal.checkpoint_sha256 == "a" * 64
    assert proposal.sid_artifact_sha256 == "b" * 64
    assert proposal.resolver["candidate_item_for_path"] == "smallest_item_id_wins"
    token = proposal.provenance_token()
    assert "mode=CERTIFIED" in token and "ckpt=" in token and "sid=" in token


def test_require_certified_refuses_an_approximate_engine() -> None:
    transport = FakeTransport(responses=[result_frame(retrieval_mode=APPROXIMATE)])
    tool = TigerSourceTool(build_engine(transport, allow_mode_downgrade=True), require_certified=True)
    with tool.engine:
        with pytest.raises(Exception) as excinfo:
            tool.propose(arguments=None, limit=3, history=("B0TRUSTED01",))
    assert "certified" in str(excinfo.value).lower()


def test_empty_retrieval_is_an_empty_head_not_an_error() -> None:
    transport = FakeTransport(responses=[result_frame(items=[])])
    tool = TigerSourceTool(build_engine(transport), max_k=5)
    with tool.engine:
        assert tool.propose(arguments=None, limit=5, history=("B0TRUSTED01",)) == []
    assert tool.last_outcome["state"] == "EMPTY"


def test_candidate_plane_registers_the_tiger_tool() -> None:
    """The plane's registration branch (candidate_plane.py:303-308).

    ``ledger``/``grounding`` are unused by that branch, so the test passes ``None`` for them to
    exercise registration without standing up a full run.
    """
    transport = FakeTransport()
    tool = TigerSourceTool(build_engine(transport), max_k=5)
    plane = CandidatePlane(ledger=None, grounding=None, tiger_tool=tool)  # type: ignore[arg-type]
    assert plane is not None

    class _WrongSource:
        source = CandidateSource.TWO_TOWER
        score_kind = "two_tower_cosine"

        def propose(self, **_kwargs: Any) -> list[tuple[str, int, float]]:
            return []

    with pytest.raises(PolicyActionError):
        CandidatePlane(ledger=None, grounding=None, tiger_tool=_WrongSource())  # type: ignore[arg-type]
