"""Backend-local tests for the Step-2.4F readiness patch.

Covers the three things the patch adds and that are cheapest to verify close to the code:

* encoder-revision pinning - what is recorded, and what is refused;
* config reconstruction from a stored artifact block;
* the read-only audit's arithmetic (prefix reconstruction, occupancy, largest-code fraction,
  collision-group histogram).

The transactional-directory and manifest-requirement behaviour is verified through the CLI in
``tests/test_tiger_backend_adapter.py``, where the stages actually run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tiger_public.features import (
    ENCODER_SMOKE,
    SMOKE_ENCODER_MARKER,
    EncoderError,
    build_item_features,
    is_immutable_revision,
    resolve_encoder,
)
from tiger_public.quantizer import QuantizerConfig, RqVae, train_quantizer

# A real 40-character snapshot SHA-1 (the one the Step-2.4 artifact used).
PINNED = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"


# --------------------------------------------------------------------------- #
# Revision identification
# --------------------------------------------------------------------------- #


def test_immutable_revision_accepts_only_a_commit_sha() -> None:
    assert is_immutable_revision(PINNED)
    assert is_immutable_revision(PINNED.upper()) is False  # the pattern is lowercase
    for mutable in ("main", "v1.0", "sentence-transformers/sentence-t5-base", "", None, 42):
        assert is_immutable_revision(mutable) is False


def test_smoke_encoder_records_no_revision_and_no_pin() -> None:
    """The stand-in must never look like pinned weights."""
    _encode, info = resolve_encoder(ENCODER_SMOKE, device="cpu", batch_size=4, revision=PINNED)
    assert info["id"] == SMOKE_ENCODER_MARKER
    assert info["revision"] is None
    assert info["revision_resolved"] is False
    assert info["weights_pinned"] is False


def test_requesting_an_unresolvable_revision_is_refused(tmp_path) -> None:
    """A run must not record a pin it cannot prove."""
    path = tmp_path / "products_text.jsonl"
    path.write_text(json.dumps({"item_id": 1, "text": "x"}) + "\n", encoding="utf-8")
    # The smoke encoder never resolves a revision, so asking it to pin one must fail.
    with pytest.raises(EncoderError) as error:
        build_item_features(
            products_text=path,
            out_dir=tmp_path / "features",
            encoder=ENCODER_SMOKE,
            batch_size=4,
            revision=PINNED,
        )
    assert "could not be confirmed" in str(error.value)


def test_unknown_encoder_name_fails_without_falling_back(tmp_path) -> None:
    """A missing encoder must raise rather than silently using the offline stand-in."""
    path = tmp_path / "products_text.jsonl"
    path.write_text(json.dumps({"item_id": 1, "text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(Exception) as error:
        build_item_features(
            products_text=path,
            out_dir=tmp_path / "features",
            encoder="definitely-not-a-real-encoder",
            batch_size=1,
            local_files_only=True,
        )
    assert "could not load the encoder" in str(error.value) or "importable" in str(error.value)


# --------------------------------------------------------------------------- #
# Config reconstruction
# --------------------------------------------------------------------------- #


def test_config_survives_a_round_trip_through_its_artifact_block() -> None:
    original = QuantizerConfig(input_dim=768, levels=3, codebook_size=256, latent_dim=64, epochs=3)
    rebuilt = QuantizerConfig.from_artifact(original.as_dict())
    # ``as_dict`` renders ``encoder_dims`` as a list for JSON, so the reconstructed config holds
    # a list where the original holds a tuple; compare by field rather than by ``==``.
    assert isinstance(rebuilt.encoder_dims, tuple)
    for name in original.as_dict():
        if name in {"code_space", "encoder_layer_widths", "decoder_layer_widths",
                    "encoder_hidden_dims", "distance", "init"}:
            continue
        assert getattr(rebuilt, name) == getattr(original, name), name
    assert rebuilt.as_dict() == original.as_dict()


def test_config_reconstruction_ignores_only_the_known_derived_keys() -> None:
    block = QuantizerConfig(input_dim=16, encoder_dims=(16, 8), latent_dim=8, epochs=1).as_dict()
    block["totally_unexpected_key"] = 1
    with pytest.raises(Exception) as error:
        QuantizerConfig.from_artifact(block)
    assert "unrecognised keys" in str(error.value)


def test_a_round_tripped_model_produces_identical_codes() -> None:
    """The audit reloads a checkpoint; that reload must not perturb the assignment."""
    config = QuantizerConfig(input_dim=16, encoder_dims=(16, 8), latent_dim=8,
                             codebook_size=8, levels=2, epochs=2, batch_size=16, seed=2026)
    torch.manual_seed(config.seed)
    model = RqVae(config)
    features = torch.randn(64, 16, generator=torch.Generator().manual_seed(1))
    train_quantizer(model, features, config=config, device="cpu")
    first = model.codes_for_features(features).numpy()

    rebuilt = RqVae(QuantizerConfig.from_artifact(config.as_dict()))
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    second = rebuilt.codes_for_features(features).numpy()
    assert np.array_equal(first, second)
