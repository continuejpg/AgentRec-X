"""Core SASRec inference engine tests (Milestone 6).

Framework independent: no HTTP here.  All tests use the tiny synthetic checkpoint
fixture, so the suite never loads the 349 MB formal run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="inference tests require PyTorch")

from tests.sasrec_inference_fixture import (  # noqa: E402
    MAX_SEQ_LEN,
    NUM_ITEMS,
    PARENT_ASINS,
    asin_for_item,
    build_fixture,
)
from recommendation.datasets.sasrec import PAD_ID  # noqa: E402
from recommendation.inference import (  # noqa: E402
    InferenceConfig,
    InferenceError,
    RequestValidationError,
    SASRecInferenceEngine,
    UnknownItemError,
    load_item_mapping,
    resolve_device,
)


@pytest.fixture()
def engine(tmp_path: Path) -> SASRecInferenceEngine:
    """A ready engine backed by the tiny synthetic checkpoint."""
    detail = build_fixture(tmp_path)
    config = InferenceConfig(
        checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
        mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
        manifest_path=detail["manifest_path"],  # type: ignore[arg-type]
        device="cpu",
    )
    return SASRecInferenceEngine(config)


# --------------------------------------------------------------------------- #
# 1-3. Checkpoint loading and identity validation
# --------------------------------------------------------------------------- #


def test_checkpoint_loads_into_exact_architecture(tmp_path: Path) -> None:
    """The checkpoint reconstructs the exact trained architecture."""
    detail = build_fixture(tmp_path)
    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
            mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            manifest_path=detail["manifest_path"],  # type: ignore[arg-type]
        )
    )
    assert engine.num_items == NUM_ITEMS
    assert engine.max_seq_len == MAX_SEQ_LEN
    assert engine.model_config.num_blocks == 1
    assert engine.model_config.num_heads == 2
    assert isinstance(engine.checkpoint_sha256, str) and len(engine.checkpoint_sha256) == 64


def test_checkpoint_sha256_is_verified_when_expected(tmp_path: Path) -> None:
    """An expected-digest mismatch fails startup."""
    detail = build_fixture(tmp_path)
    with pytest.raises(InferenceError, match="SHA-256 mismatch"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
                expected_checkpoint_sha256="0" * 64,
            )
        )


def test_missing_checkpoint_fails_clearly(tmp_path: Path) -> None:
    """A missing checkpoint is an explicit startup error."""
    detail = build_fixture(tmp_path)
    with pytest.raises(InferenceError, match="checkpoint not found"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=tmp_path / "absent.pt",
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            )
        )


def test_checkpoint_config_mismatch_rejected(tmp_path: Path) -> None:
    """A manifest that disagrees with the checkpoint config fails startup."""
    detail = build_fixture(tmp_path)
    manifest = json.loads(Path(detail["manifest_path"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    manifest["model_config"]["hidden_size"] = 999
    Path(detail["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")  # type: ignore[arg-type]
    with pytest.raises(InferenceError, match="disagreement"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
                manifest_path=detail["manifest_path"],  # type: ignore[arg-type]
            )
        )


def test_mapping_cardinality_mismatch_rejected(tmp_path: Path) -> None:
    """A mapping whose num_items disagrees with the model fails startup."""
    detail = build_fixture(tmp_path)
    mappings = json.loads(Path(detail["mappings_path"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    mappings["num_items"] = NUM_ITEMS + 5
    Path(detail["mappings_path"]).write_text(json.dumps(mappings), encoding="utf-8")  # type: ignore[arg-type]
    with pytest.raises(InferenceError, match="cardinality mismatch|inconsistent"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            )
        )


def test_mapping_with_pad_id_assigned_is_rejected(tmp_path: Path) -> None:
    """A mapping that hands PAD to a real item is refused."""
    detail = build_fixture(tmp_path)
    mappings = json.loads(Path(detail["mappings_path"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    mappings["item2id"][PARENT_ASINS[0]] = PAD_ID
    Path(detail["mappings_path"]).write_text(json.dumps(mappings), encoding="utf-8")  # type: ignore[arg-type]
    with pytest.raises(InferenceError, match="PAD"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            )
        )


def test_non_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    """A file that is not an AgentRec-X checkpoint is refused."""
    detail = build_fixture(tmp_path)
    foreign = tmp_path / "foreign.pt"
    torch.save({"not": "a checkpoint"}, foreign)
    with pytest.raises(InferenceError, match="checkpoint"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=foreign,
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
            )
        )


# --------------------------------------------------------------------------- #
# 4-5. Mapping round trip and PAD handling
# --------------------------------------------------------------------------- #


def test_parent_asin_round_trip(engine: SASRecInferenceEngine) -> None:
    """Every catalog item round-trips parent_asin -> id -> parent_asin."""
    for item_id in range(1, engine.num_items + 1):
        asin = engine.item_id_to_parent_asin(item_id)
        assert engine.parent_asin_to_item_id(asin) == item_id
        assert engine.has_parent_asin(asin)


def test_unknown_parent_asin_is_rejected(engine: SASRecInferenceEngine) -> None:
    """An unknown external id raises rather than being coerced or dropped."""
    with pytest.raises(UnknownItemError, match="unknown parent_asin"):
        engine.parent_asin_to_item_id("B9999999999")
    assert not engine.has_parent_asin("B9999999999")


def test_pad_cannot_be_mapped_as_a_real_item(engine: SASRecInferenceEngine) -> None:
    """PAD (0) is not a real item and cannot be requested."""
    with pytest.raises(InferenceError, match="PAD"):
        engine.item_id_to_parent_asin(PAD_ID)
    for bad in (0, engine.num_items + 1, -1):
        with pytest.raises(InferenceError):
            engine.item_id_to_parent_asin(bad)


def test_non_string_history_entries_are_rejected(engine: SASRecInferenceEngine) -> None:
    """Non-string history entries fail validation."""
    with pytest.raises(RequestValidationError, match="parent_asin strings"):
        engine.parent_asin_to_item_id(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 6-11. History encoding and truncation
# --------------------------------------------------------------------------- #


def test_empty_history_is_rejected(engine: SASRecInferenceEngine) -> None:
    """An empty history is a client error."""
    with pytest.raises(RequestValidationError, match="empty"):
        engine.recommend([], k=5)


def test_unknown_history_item_is_rejected(engine: SASRecInferenceEngine) -> None:
    """A history containing an unknown item is rejected (strict behaviour)."""
    with pytest.raises(UnknownItemError):
        engine.recommend([asin_for_item(1), "B9999999999"], k=5)


def test_left_padding_is_correct(engine: SASRecInferenceEngine) -> None:
    """A short history is left-padded with PAD, preserving chronological order."""
    history = [asin_for_item(1), asin_for_item(2), asin_for_item(3)]
    encoded, truncated = engine.encode_history(history)
    assert truncated is False
    assert len(encoded) == MAX_SEQ_LEN
    assert encoded == [PAD_ID] * (MAX_SEQ_LEN - 3) + [1, 2, 3]


def test_long_history_keeps_most_recent_items(engine: SASRecInferenceEngine) -> None:
    """A history longer than max_seq_len keeps only the newest items."""
    long_history = [asin_for_item(i) for i in range(1, NUM_ITEMS + 1)]
    encoded, truncated = engine.encode_history(long_history)
    assert truncated is True
    assert encoded == list(range(NUM_ITEMS - MAX_SEQ_LEN + 1, NUM_ITEMS + 1))


def test_duplicate_interactions_are_preserved_in_encoding(engine: SASRecInferenceEngine) -> None:
    """Duplicates are real interactions and must not be collapsed before encoding."""
    history = [asin_for_item(2), asin_for_item(2), asin_for_item(2)]
    encoded, _ = engine.encode_history(history)
    assert encoded[-3:] == [2, 2, 2]
    assert encoded.count(2) == 3


def test_caller_history_is_not_mutated(engine: SASRecInferenceEngine) -> None:
    """Encoding never modifies the caller's sequence."""
    history = [asin_for_item(4), asin_for_item(2), asin_for_item(7)]
    snapshot = list(history)
    engine.recommend(history, k=3)
    engine.encode_history(history)
    assert history == snapshot


def test_effective_length_and_truncation_flags(engine: SASRecInferenceEngine) -> None:
    """Reported history lengths distinguish supplied vs. model-window length."""
    short = engine.recommend([asin_for_item(1)], k=1)
    assert short.history_length == 1 and short.effective_history_length == 1
    assert short.history_truncated is False

    long_history = [asin_for_item(i) for i in range(1, NUM_ITEMS + 1)]
    long_result = engine.recommend(long_history, k=1)
    assert long_result.history_length == NUM_ITEMS
    assert long_result.effective_history_length == MAX_SEQ_LEN
    assert long_result.history_truncated is True


# --------------------------------------------------------------------------- #
# 12-13. Full-history seen masking
# --------------------------------------------------------------------------- #


def test_full_history_seen_masking_not_just_the_window(engine: SASRecInferenceEngine) -> None:
    """Every supplied item is masked, including those outside the model window."""
    # NUM_ITEMS (12) > MAX_SEQ_LEN (4), so items 1..8 fall outside the window but
    # must still be excluded from recommendations.
    full_history = [asin_for_item(i) for i in range(1, NUM_ITEMS + 1)]
    result = engine.recommend(full_history, k=NUM_ITEMS)
    returned = {r.parent_asin for r in result.recommendations}
    assert returned == set(), "the whole catalog is in the history -> no candidates"
    assert result.returned_k == 0

    # Mask 10 of 12 items: only the two newest (11, 12) remain candidates.  Their
    # order is score-driven, so compare the *set* - the point of this assertion is
    # that the eight items outside the model window were still masked out.
    history = [asin_for_item(i) for i in range(1, NUM_ITEMS - 1)]
    partial = engine.recommend(history, k=NUM_ITEMS)
    returned = [r.parent_asin for r in partial.recommendations]
    assert set(returned) == {asin_for_item(NUM_ITEMS - 1), asin_for_item(NUM_ITEMS)}
    assert partial.returned_k == 2
    assert not (set(returned) & set(history)), "an item outside the window leaked in"


def test_recommendations_exclude_history_and_pad(engine: SASRecInferenceEngine) -> None:
    """No recommendation is a seen item, and PAD never appears."""
    history = [asin_for_item(i) for i in range(1, 5)]
    result = engine.recommend(history, k=5)
    assert result.recommendations, "expected some unseen candidates"
    for item in result.recommendations:
        assert item.parent_asin not in history
        assert item.item_id != PAD_ID
        assert 1 <= item.item_id <= engine.num_items


def test_recommendation_asin_matches_item_id(engine: SASRecInferenceEngine) -> None:
    """Each recommendation's parent_asin is the mapping of its item_id."""
    result = engine.recommend([asin_for_item(3)], k=4)
    for item in result.recommendations:
        assert item.parent_asin == engine.item_id_to_parent_asin(item.item_id)


# --------------------------------------------------------------------------- #
# 14-18. Result contract and candidate exhaustion
# --------------------------------------------------------------------------- #


def test_rank_numbering_is_contiguous_from_one(engine: SASRecInferenceEngine) -> None:
    """Ranks are 1..returned_k."""
    result = engine.recommend([asin_for_item(1)], k=5)
    assert [r.rank for r in result.recommendations] == list(range(1, result.returned_k + 1))


def test_scores_are_descending(engine: SASRecInferenceEngine) -> None:
    """Returned scores are non-increasing."""
    result = engine.recommend([asin_for_item(2)], k=5)
    scores = [r.score for r in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert all(math.isfinite(s) for s in scores)


def test_fewer_than_k_candidates(engine: SASRecInferenceEngine) -> None:
    """Requesting more than remain returns only what is available."""
    seen = [asin_for_item(i) for i in range(1, NUM_ITEMS - 1)]
    result = engine.recommend(seen, k=10)
    assert result.requested_k == 10
    assert result.returned_k == 2
    assert result.eligible_candidates == 2


def test_zero_candidates_returns_empty(engine: SASRecInferenceEngine) -> None:
    """A fully-seen catalog yields a valid empty result, not a crash."""
    result = engine.recommend([asin_for_item(i) for i in range(1, NUM_ITEMS + 1)], k=10)
    assert result.recommendations == []
    assert result.returned_k == 0
    assert result.eligible_candidates == 0


def test_k_validation_in_engine(engine: SASRecInferenceEngine) -> None:
    """k bounds are enforced by the engine as well as the API schema."""
    for bad in (0, -1, 101, 1.5, "5", True):
        with pytest.raises(Exception):
            engine.recommend([asin_for_item(1)], k=bad)  # type: ignore[arg-type]


def test_result_is_serialisable(engine: SASRecInferenceEngine) -> None:
    """The result converts to JSON-friendly primitives."""
    payload = json.loads(json.dumps(engine.recommend([asin_for_item(1)], k=3).as_dict()))
    assert payload["returned_k"] == len(payload["recommendations"])
    assert set(payload["recommendations"][0]) == {"rank", "item_id", "parent_asin", "score"}


# --------------------------------------------------------------------------- #
# 19-23. Inference-mode correctness
# --------------------------------------------------------------------------- #


def test_model_is_always_in_eval_mode(engine: SASRecInferenceEngine) -> None:
    """The engine keeps the model in eval mode across calls."""
    assert engine.model.training is False
    engine.recommend([asin_for_item(1)], k=2)
    engine.recommend([asin_for_item(2), asin_for_item(3)], k=2)
    assert engine.model.training is False
    assert engine.is_ready() is True


def test_no_gradients_are_produced(engine: SASRecInferenceEngine) -> None:
    """Inference never builds an autograd graph or populates .grad."""
    engine.recommend([asin_for_item(1), asin_for_item(2)], k=3)
    for parameter in engine.model.parameters():
        assert parameter.grad is None
        assert parameter.requires_grad is False


def test_parameters_are_unchanged_by_inference(engine: SASRecInferenceEngine) -> None:
    """Serving never mutates weights (no optimizer exists at all)."""
    before = {name: tensor.detach().clone() for name, tensor in engine.model.named_parameters()}
    for _ in range(3):
        engine.recommend([asin_for_item(4)], k=2)
    for name, tensor in engine.model.named_parameters():
        assert torch.equal(before[name], tensor.detach()), f"{name} changed during inference"


def test_pad_embedding_row_unchanged(engine: SASRecInferenceEngine) -> None:
    """The PAD embedding row is untouched by serving."""
    row = engine.model.item_embedding.weight[PAD_ID].detach().clone()
    engine.recommend([asin_for_item(5)], k=2)
    assert torch.equal(engine.model.item_embedding.weight[PAD_ID].detach(), row)


def test_repeated_request_is_deterministic(engine: SASRecInferenceEngine) -> None:
    """Identical requests produce identical recommendations."""
    history = [asin_for_item(1), asin_for_item(4), asin_for_item(6)]
    baseline = engine.recommend(history, k=5).as_dict()["recommendations"]
    for _ in range(5):
        assert engine.recommend(history, k=5).as_dict()["recommendations"] == baseline


def test_dropout_disabled_during_serving(tmp_path: Path) -> None:
    """A model configured with dropout still serves deterministically in eval mode."""
    from tests.sasrec_inference_fixture import HIDDEN_SIZE, NUM_BLOCKS, NUM_HEADS, SEED
    from recommendation.models import build_model
    from recommendation.training.checkpoint import TrainingState, save_checkpoint

    torch.manual_seed(SEED)
    model = build_model(
        num_items=NUM_ITEMS, seed=SEED, max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, dropout=0.5,
    )
    checkpoint = tmp_path / "dropout.pt"
    save_checkpoint(
        checkpoint, model=model, optimizer=None, state=TrainingState(),
        model_config=model.config.as_dict(), trainer_config={}, max_seq_len=MAX_SEQ_LEN,
        seed=SEED, num_items=NUM_ITEMS,
    )
    detail = build_fixture(tmp_path / "mapping")
    engine = SASRecInferenceEngine(
        InferenceConfig(checkpoint_path=checkpoint, mappings_path=detail["mappings_path"])  # type: ignore[arg-type]
    )
    assert engine.model_config.dropout == 0.5
    baseline = engine.recommend([asin_for_item(1)], k=3).as_dict()["recommendations"]
    for _ in range(5):
        assert engine.recommend([asin_for_item(1)], k=3).as_dict()["recommendations"] == baseline


# --------------------------------------------------------------------------- #
# 27-28. Device behaviour
# --------------------------------------------------------------------------- #


def test_cpu_device_works(engine: SASRecInferenceEngine) -> None:
    """The default serving device is CPU and reports itself."""
    assert engine.device.type == "cpu"
    assert engine.model_metadata()["device"] == "cpu"


def test_unavailable_requested_cuda_fails(tmp_path: Path) -> None:
    """Requesting CUDA without CUDA must fail rather than silently use CPU."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; the unavailable-device path is not reachable")
    detail = build_fixture(tmp_path)
    with pytest.raises(InferenceError, match="silently"):
        SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=detail["checkpoint_path"],  # type: ignore[arg-type]
                mappings_path=detail["mappings_path"],  # type: ignore[arg-type]
                device="cuda",
            )
        )


def test_device_resolution_semantics() -> None:
    """cpu works; malformed devices are rejected; cuda is availability-gated."""
    assert resolve_device("cpu").type == "cpu"
    for bad in ("gpu", "cuda:1", "cpu:0", "", None, 5):
        with pytest.raises(InferenceError):
            resolve_device(bad)  # type: ignore[arg-type]
    if not torch.cuda.is_available():
        with pytest.raises(InferenceError, match="silently"):
            resolve_device("cuda:0")


def test_model_metadata_provenance_is_split(engine: SASRecInferenceEngine) -> None:
    """Metadata separates formal-run Git state from the serving source note."""
    meta = engine.model_metadata()
    assert meta["model_type"] == "SASRec"
    assert meta["provenance"]["formal_run_git"]["commit"] == "0" * 40
    assert "serving_note" in meta["provenance"]
    assert "num_items" in meta and "checkpoint_sha256" in meta
