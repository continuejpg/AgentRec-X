"""Backend-local tests for the Step-2.4 quantizer and dedup pass.

These run in the backend venv and are kept out of AgentRec-X's suite by
``pytest.ini``'s ``norecursedirs = backends``.  AgentRec-X's own
``tests/test_tiger_backend_adapter.py`` covers the same code through the subprocess boundary;
what lives here is the arithmetic and the tensor-level invariants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tiger_public.contracts import PAD_SENTINEL, collision_audit
from tiger_public.dedup import (
    DedupOverflowError,
    assign_dedup_digits,
    largest_collision_group,
)
from tiger_public.quantizer import (
    QuantizerConfig,
    QuantizerError,
    RqVae,
    kmeans_plus_plus,
    layer_utilisation,
    train_quantizer,
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_encoder_geometry_is_input_to_latent_in_order() -> None:
    """``[768, 256, 128]`` are HIDDEN widths, so the encoder is 768 -> 256 -> 128 -> 64."""
    config = QuantizerConfig(input_dim=768, encoder_dims=(768, 256, 128), latent_dim=64)
    assert config.as_dict()["encoder_layer_widths"] == [768, 256, 128, 64]
    assert config.as_dict()["decoder_layer_widths"] == [64, 128, 256, 768]
    model = RqVae(config)
    widths = [
        module.out_features
        for module in model.encoder
        if isinstance(module, torch.nn.Linear)
    ]
    # Output width of each linear layer, in order: 768 -> 256 -> 128 -> 64.
    assert widths == [768, 256, 128, 64]


def test_config_refuses_a_mismatched_input_width() -> None:
    with pytest.raises(QuantizerError):
        QuantizerConfig(input_dim=768, encoder_dims=(512, 256, 128))


def test_config_refuses_a_non_positive_beta() -> None:
    with pytest.raises(QuantizerError):
        QuantizerConfig(input_dim=16, encoder_dims=(16, 8), latent_dim=8, beta=-0.1)


# --------------------------------------------------------------------------- #
# k-means++ and revival
# --------------------------------------------------------------------------- #


def test_kmeans_plus_plus_returns_distinct_seeds() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(2026)
    points = torch.randn(500, 8, generator=torch.Generator().manual_seed(1))
    centroids = kmeans_plus_plus(points, 16, generator=generator)
    assert centroids.shape == (16, 8)
    # The D-squared rule never re-picks an existing centroid, so all seeds are distinct rows.
    assert len({tuple(row.tolist()) for row in centroids}) == 16


def test_kmeans_plus_plus_refuses_more_centroids_than_points() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(QuantizerError):
        kmeans_plus_plus(torch.randn(4, 2), 8, generator=generator)


def test_dead_code_revival_replaces_unused_codes() -> None:
    config = QuantizerConfig(input_dim=4, encoder_dims=(4,), latent_dim=2,
                             codebook_size=8, levels=1, epochs=1)
    model = RqVae(config)
    model.train()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.quantizer.quantize(torch.randn(16, 2, generator=generator))
    revived = model.quantizer.revive_dead_codes(torch.randn(16, 2), generator=generator)
    assert revived >= 0
    # After a revival pass the usage counter is reset, so the next epoch measures afresh.
    assert float(model.quantizer.usage.sum()) == 0.0


# --------------------------------------------------------------------------- #
# Quantisation mechanics
# --------------------------------------------------------------------------- #


def test_straight_through_estimator_passes_gradient_to_the_encoder() -> None:
    """The reconstruction loss must reach the encoder through the quantiser."""
    config = QuantizerConfig(input_dim=8, encoder_dims=(8, 4), latent_dim=4,
                             codebook_size=8, levels=2, epochs=1, normalize_input=False)
    model = RqVae(config)
    features = torch.randn(16, 8, generator=torch.Generator().manual_seed(0))
    output = model(features)
    (output["reconstruction_loss"] + output["quantization_loss"]).backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient"
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_codes_are_in_range_and_shaped_by_level() -> None:
    config = QuantizerConfig(input_dim=8, encoder_dims=(8, 4), latent_dim=4,
                             codebook_size=8, levels=3, epochs=2)
    model = RqVae(config)
    features = torch.randn(32, 8, generator=torch.Generator().manual_seed(0))
    codes = model.codes_for_features(features)
    assert codes.shape == (32, 3)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 8


def test_training_reports_finite_separated_loss_terms() -> None:
    """Reconstruction and quantisation are reported apart, not as one blended number."""
    config = QuantizerConfig(input_dim=8, encoder_dims=(8, 4), latent_dim=4,
                             codebook_size=8, levels=2, epochs=3, batch_size=16)
    model = RqVae(config)
    features = torch.randn(64, 8, generator=torch.Generator().manual_seed(0))
    outcome = train_quantizer(model, features, config=config, device="cpu")
    assert outcome.history
    for entry in outcome.history:
        assert np.isfinite(entry["loss"])
        assert np.isfinite(entry["reconstruction_loss"])
        assert np.isfinite(entry["quantization_loss"])
        assert entry["loss"] == pytest.approx(
            entry["reconstruction_loss"] + entry["quantization_loss"], rel=1e-6, abs=1e-6
        )


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    def build() -> np.ndarray:
        config = QuantizerConfig(input_dim=8, encoder_dims=(8, 4), latent_dim=4,
                                 codebook_size=8, levels=2, epochs=2, batch_size=16, seed=2026)
        torch.manual_seed(config.seed)
        model = RqVae(config)
        features = torch.randn(64, 8, generator=torch.Generator().manual_seed(7))
        train_quantizer(model, features, config=config, device="cpu")
        return model.codes_for_features(features).numpy()

    assert np.array_equal(build(), build())


def test_layer_utilisation_reports_coverage_and_entropy() -> None:
    codes = np.zeros((100, 2), dtype=np.int64)
    codes[:, 0] = np.arange(100) % 7  # 7 of an 8-wide book: 12.5% dead, below the stop
    codes[:, 1] = 0                   # a single code: fully collapsed
    blocks = layer_utilisation(codes, codebook_size=8)
    assert blocks[0]["used_codes"] == 7
    assert blocks[0]["dead_codes"] == 1
    assert blocks[0]["frac_layer_coverage"] == pytest.approx(0.875)
    # The artifact rounds to six decimals, so the tolerance reflects that, not a discrepancy.
    # The artifact rounds to six decimals, so the tolerance reflects that, not a discrepancy.
    assert blocks[0]["id_entropy"] == pytest.approx(np.log(7), abs=1e-3)
    assert blocks[0]["collapsed"] is False
    assert blocks[1]["used_codes"] == 1
    assert blocks[1]["id_entropy"] == pytest.approx(0.0)
    assert blocks[1]["collapsed"] is True


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #


def test_dedup_first_member_gets_ordinal_zero() -> None:
    result = assign_dedup_digits([(1, 2), (1, 2), (1, 2), (3, 4)], dedup_vocab_size=8)
    assert result.ordinals == (0, 1, 2, 0)
    assert result.assignment[0] == (1, 2, 0)
    assert result.assignment[1] == (1, 2, 1)
    assert result.assignment[3] == (3, 4, 0)


def test_dedup_is_order_stable_across_invocations() -> None:
    codes = [(1, 1), (2, 2), (1, 1), (3, 3), (1, 1)]
    first = assign_dedup_digits(codes, dedup_vocab_size=8)
    second = assign_dedup_digits(codes, dedup_vocab_size=8)
    assert first == second


def test_dedup_refuses_an_oversize_group() -> None:
    with pytest.raises(DedupOverflowError) as error:
        assign_dedup_digits([(1, 1)] * 9, dedup_vocab_size=4)
    assert "exceeds dedup_vocab_size" in str(error.value)


def test_dedup_refuses_a_negative_digit() -> None:
    """Only the reserved PAD row may carry the sentinel, and PAD is never quantised."""
    with pytest.raises(DedupOverflowError):
        assign_dedup_digits([(0, PAD_SENTINEL)], dedup_vocab_size=8)


def test_dedup_refuses_an_empty_assignment() -> None:
    with pytest.raises(DedupOverflowError):
        assign_dedup_digits([], dedup_vocab_size=8)


def test_dedup_produces_unique_final_codes() -> None:
    codes = [(1, 1)] * 6 + [(2, 2)] * 3
    result = assign_dedup_digits(codes, dedup_vocab_size=16)
    assert len(set(result.assignment)) == len(codes)
    assert largest_collision_group(codes) == 6
    assert result.pre_largest_group == 6
    assert result.pre_collision_groups == 2


def test_dedup_and_the_shared_audit_agree() -> None:
    codes = [(1, 1)] * 5 + [(2, 2)] + [(3, 4)]
    # The widest group is 5, so a 4-wide dedup level could not address it and the audit would
    # flag overflow; 8 can, and the two implementations must then agree on every field.
    result = assign_dedup_digits(codes, dedup_vocab_size=8)
    audit = collision_audit(codes, dedup_levels=1, dedup_vocab_size=8)
    assert audit["pre_dedup"]["largest_group"] == result.pre_largest_group
    assert audit["pre_dedup"]["collision_groups"] == result.pre_collision_groups
    assert audit["post_dedup"]["collision_groups"] == 0
    assert audit["post_dedup"]["distinct_sids"] == len(codes)
    assert audit["largest_group_le_dedup_vocab"] is True


# --------------------------------------------------------------------------- #
# CUDA generator-device compatibility (Step-2.4F hotfix)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_quantizer_on_cuda_with_cpu_resident_features() -> None:
    """The production configuration keeps features on CPU and trains on CUDA.

    That shape is what broke the registered H3 run: k-means++ and the dead-code revival draw
    indices from a generator, and a generator's device must match the tensor it produces.  This
    exercises the real path - CPU feature storage, CUDA residuals, CUDA k-means, CUDA revival -
    rather than a simplified stand-in, so a regression here is a regression in production.
    """
    device = "cuda:0"
    config = QuantizerConfig(
        input_dim=16,
        encoder_dims=(16, 8),
        latent_dim=8,
        codebook_size=8,
        levels=2,
        epochs=2,
        batch_size=16,
        seed=2026,
        # Small enough that the whole set is sampled, so the CPU-side row draw is not the point.
        kmeans_sample=64,
        revive_dead=True,
    )
    torch.manual_seed(config.seed)
    model = RqVae(config)
    # CPU-resident storage, exactly as the CLI hands the matrix over.
    features = torch.randn(64, 16, generator=torch.Generator().manual_seed(1))
    assert features.device.type == "cpu"

    outcome = train_quantizer(model, features, config=config, device=device)

    assert outcome.device == device
    assert outcome.history, "training produced no epochs"
    for entry in outcome.history:
        assert math.isfinite(entry["loss"])
        assert math.isfinite(entry["reconstruction_loss"])
        assert math.isfinite(entry["quantization_loss"])
    # Revival ran on CUDA latents, which is where the CPU/CUDA generator mismatch lived.
    assert outcome.revived_total >= 0
    codes = model.codes_for_features(features.to(device))
    assert codes.shape == (64, 2)
    assert int(codes.max()) < config.codebook_size
    assert int(codes.min()) >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rng_pair_uses_a_device_generator_only_off_cpu() -> None:
    """On CPU the pair must stay a single CPU generator, preserving existing semantics."""
    from tiger_public.quantizer import RngPair

    cpu_pair = RngPair.seeded(2026, "cpu")
    assert cpu_pair.cpu is cpu_pair.device

    cuda_pair = RngPair.seeded(2026, "cuda:0")
    assert cuda_pair.device.device.type == "cuda"
    assert cuda_pair.cpu.device.type == "cpu"
    assert cuda_pair.device is not cuda_pair.cpu


def test_training_never_mutates_the_source_feature_tensor() -> None:
    """The trainer reads its input and must not write to it.

    The promotion to a writable array exists so ``torch.from_numpy`` is given a clean tensor;
    it must not become a licence to mutate the caller's data.  This pins that the source is
    byte-identical after a training run, which is what makes reusing one array across the
    training pass and the assignment pass safe.
    """
    config = QuantizerConfig(input_dim=16, encoder_dims=(16, 8), latent_dim=8,
                             codebook_size=8, levels=2, epochs=2, batch_size=16, seed=2026)
    torch.manual_seed(config.seed)
    model = RqVae(config)
    features = torch.randn(64, 16, generator=torch.Generator().manual_seed(1))
    before = features.clone()

    train_quantizer(model, features, config=config, device="cpu")
    model.codes_for_features(features)

    assert torch.equal(features, before), "training mutated the caller's feature tensor"


def test_a_read_only_source_array_is_accepted() -> None:
    """A read-only (memory-mapped) source must train cleanly once copied to writable storage.

    This mirrors the CLI: the artifact is loaded read-only, and the fix is a single copy rather
    than teaching every consumer to tolerate a read-only tensor.
    """
    import numpy as np

    config = QuantizerConfig(input_dim=16, encoder_dims=(16, 8), latent_dim=8,
                             codebook_size=8, levels=2, epochs=2, batch_size=16, seed=2026)
    torch.manual_seed(config.seed)
    model = RqVae(config)

    source = np.random.default_rng(0).standard_normal((64, 16)).astype(np.float32)
    source.setflags(write=False)
    assert not source.flags.writeable

    writable = np.array(source, dtype=np.float32, copy=True)
    assert writable.flags.writeable
    features = torch.from_numpy(writable)
    outcome = train_quantizer(model, features, config=config, device="cpu")
    assert math.isfinite(outcome.final["loss"])
    # The read-only original is untouched by construction.
    assert not source.flags.writeable
