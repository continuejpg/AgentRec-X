"""GRID-inspired residual-quantised autoencoder (Step 2.4).

A local reimplementation of the audited design, not a vendored copy.  The absorbed elements are
named in ``PROVENANCE.md``: the encoder/decoder shape, input normalisation, squared-Euclidean
distance, the ``beta = 0.25`` commitment objective, k-means++ codebook initialisation, the
straight-through estimator, and the per-level coverage / ``id_entropy`` diagnostics.  What is
**not** absorbed is the surrounding infrastructure: no Lightning, no Hydra, no TensorFlow, no
TFRecord, no TorchRec, no BigQuery and no DDP.

```text
item features [N, 768]
    -> normalise (BatchNorm1d -> L2)
    -> encoder MLP 768 -> 256 -> 128 -> 64
    -> residual quantisation over `levels` codebooks of `codebook_size` entries
    -> decoder MLP 64 -> 128 -> 256 -> 768
    -> reconstruction (MSE against the NORMALISED input)
```

The model produces codes.  It never decides identity: turning a code into an item is
:mod:`tiger_public.dedup` plus the catalogue assignment, and the PAD row is never an input here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

__all__ = [
    "QuantizerConfig",
    "QuantizerError",
    "RqVae",
    "ResidualQuantizer",
    "kmeans_plus_plus",
    "layer_utilisation",
]

#: Activation used by the encoder and decoder, per the frozen configuration.
DEFAULT_ACTIVATION = "relu"


class QuantizerError(ValueError):
    """Raised when the quantizer is configured or driven unusably."""


@dataclass(frozen=True)
class QuantizerConfig:
    """The frozen Step-2.4 quantizer configuration.

    Every default is the value fixed before the full run; changing one is a new experiment and
    requires re-deriving every downstream artifact, because the Semantic IDs change with it.
    """

    input_dim: int = 768
    #: **Hidden** widths, input first.  The encoder appends one final layer of ``latent_dim``,
    #: so ``(768, 256, 128)`` with ``latent_dim = 64`` builds exactly
    #: ``768 -> 256 -> 128 -> 64``.  This is the audited geometry: GRID's ``MLP`` takes
    #: ``hidden_dim_list`` and appends its own ``output_dim``, and the frozen Step-2.4 values are
    #: the published ``hidden_dim_list`` plus the published latent width.
    encoder_dims: tuple[int, ...] = (768, 256, 128)
    latent_dim: int = 64
    levels: int = 3
    codebook_size: int = 256
    beta: float = 0.25
    activation: str = DEFAULT_ACTIVATION
    normalize_input: bool = True
    revive_dead: bool = True
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 2026
    #: How many points k-means++ samples to seed the codebooks from.
    kmeans_sample: int = 20_000
    kmeans_iters: int = 10

    def __post_init__(self) -> None:
        for name in ("input_dim", "latent_dim", "levels", "codebook_size", "epochs",
                     "batch_size", "kmeans_sample", "kmeans_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise QuantizerError(f"{name} must be a positive int, got {value!r}")
        if not self.encoder_dims:
            raise QuantizerError("encoder_dims must not be empty")
        if self.encoder_dims[0] != self.input_dim:
            raise QuantizerError(
                f"encoder_dims must start at input_dim {self.input_dim}, got {self.encoder_dims}"
            )
        if any(isinstance(width, bool) or not isinstance(width, int) or width < 1
               for width in self.encoder_dims):
            raise QuantizerError(f"encoder_dims must be positive ints, got {self.encoder_dims}")
        if self.beta < 0.0:
            raise QuantizerError(f"beta must be >= 0, got {self.beta}")
        if self.activation not in ("relu", "gelu"):
            raise QuantizerError(f"activation must be relu or gelu, got {self.activation!r}")

    @property
    def code_space(self) -> int:
        return self.codebook_size**self.levels

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "encoder_dims": list(self.encoder_dims),
            "encoder_hidden_dims": list(self.encoder_dims),
            "encoder_layer_widths": [*self.encoder_dims, self.latent_dim],
            "decoder_layer_widths": [self.latent_dim, *reversed(self.encoder_dims)],
            "latent_dim": self.latent_dim,
            "levels": self.levels,
            "codebook_size": self.codebook_size,
            "code_space": self.code_space,
            "beta": self.beta,
            "activation": self.activation,
            "distance": "squared_euclidean",
            "init": "k-means++",
            "normalize_input": self.normalize_input,
            "revive_dead": self.revive_dead,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "kmeans_sample": self.kmeans_sample,
            "kmeans_iters": self.kmeans_iters,
        }


def _activation(name: str) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.GELU()


def _mlp(widths: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(widths) - 1):
        layers.append(nn.Linear(int(widths[index]), int(widths[index + 1])))
        if index < len(widths) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


@torch.no_grad()
def kmeans_plus_plus(points: torch.Tensor, k: int, *, generator: torch.Generator) -> torch.Tensor:
    """Seed ``k`` centroids with the k-means++ D-squared rule.

    A uniform initialisation leaves most codes dead on a real catalogue, so the seeding step is
    part of the design rather than an optimisation.  The distance-weighted draw makes the first
    centroids spread out, which is what gives the residual levels something to refine.

    The degenerate case - every remaining point already coincides with a chosen centroid -
    falls back to a uniform draw instead of dividing by zero.
    """
    if points.dim() != 2:
        raise QuantizerError(f"points must be 2-D, got shape {tuple(points.shape)}")
    n = points.shape[0]
    if k < 1:
        raise QuantizerError(f"k must be >= 1, got {k}")
    if n == 0:
        raise QuantizerError("cannot seed centroids from an empty point set")
    if n < k:
        raise QuantizerError(
            f"cannot seed {k} centroids from {n} points; the codebook is wider than the data "
            "and would necessarily contain dead codes"
        )
    device = points.device
    centroids = torch.empty((k, points.shape[1]), dtype=points.dtype, device=device)
    chosen = int(torch.randint(n, (1,), generator=generator, device=device).item())
    centroids[0] = points[chosen]
    closest = (points - centroids[0]).pow(2).sum(dim=1)
    for index in range(1, k):
        total = float(closest.sum())
        if not math.isfinite(total) or total <= 0.0:
            fill = torch.randint(n, (k - index,), generator=generator, device=device)
            centroids[index:] = points[fill]
            break
        pick = int(torch.multinomial(closest, 1, generator=generator).item())
        centroids[index] = points[pick]
        closest = torch.minimum(closest, (points - centroids[index]).pow(2).sum(dim=1))
    return centroids


@torch.no_grad()
def refine_codebook(
    codebook: torch.Tensor,
    points: torch.Tensor,
    *,
    iterations: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Run a few Lloyd iterations and report how many `revive_dead`-style slots were empty.

    (``generator`` is accepted so the caller has one seeded source of randomness for the whole
    initialisation; Lloyd's algorithm itself is deterministic given the seed.)
    """
    del generator
    centroids = codebook.clone()
    revived = 0
    for _ in range(max(1, iterations)):
        distances = (
            points.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * points @ centroids.t()
            + centroids.pow(2).sum(dim=1).unsqueeze(0)
        )
        assignment = distances.argmin(dim=1)
        empty = 0
        for index in range(centroids.shape[0]):
            members = points[assignment == index]
            if members.shape[0] == 0:
                empty += 1
                continue
            centroids[index] = members.mean(dim=0)
        revived = empty
    return centroids, revived


class ResidualQuantizer(nn.Module):
    """``levels`` codebooks applied to the residual of the previous level."""

    def __init__(self, config: QuantizerConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.ParameterList(
            nn.Parameter(torch.randn(config.codebook_size, config.latent_dim) * 0.1)
            for _ in range(config.levels)
        )
        #: How many times each code was selected since the last reset, for dead-code revival.
        self.register_buffer("usage", torch.zeros(config.levels, config.codebook_size))

    def quantize(
        self, residuals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(quantized_sum, codes, quantization_loss, reconstruction_embeddings)``.

        ``reconstruction_embeddings`` is the straight-through tensor that carries the gradient
        to the encoder: its forward value is the quantised sum and its backward value is the
        identity of ``residuals``.
        """
        residual = residuals
        total = torch.zeros_like(residuals)
        codes: list[torch.Tensor] = []
        quantization_loss = torch.zeros((), device=residuals.device, dtype=residuals.dtype)
        for level, codebook in enumerate(self.codebooks):
            weight = codebook
            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * residual @ weight.t()
                + weight.pow(2).sum(dim=-1).unsqueeze(0)
            )
            indices = distances.argmin(dim=-1)
            chosen = weight[indices]
            codes.append(indices)
            total = total + chosen
            # GRID's BetaQuantizationLoss with beta = 0.25:
            #   codebook term      moves the codebook toward the residual
            #   commitment term    moves the encoder toward the chosen code
            codebook_term = (residual.detach() - chosen).pow(2).mean()
            commitment_term = (residual - chosen.detach()).pow(2).mean()
            quantization_loss = quantization_loss + codebook_term + self.config.beta * commitment_term
            residual = residual - chosen.detach()
            if self.training:
                self.usage[level].scatter_add_(
                    0, indices, torch.ones_like(indices, dtype=self.usage.dtype)
                )
        stacked = torch.stack(codes, dim=1)
        straight_through = residuals + (total - residuals).detach()
        return straight_through, stacked, quantization_loss, straight_through

    @torch.no_grad()
    def revive_dead_codes(self, latents: torch.Tensor, *, generator: torch.Generator) -> int:
        """Reset codes never selected this epoch to random latent points.

        GRID does not do this and only *reports* collapse; amazon-genrec does, and the
        reimplementation follows amazon-genrec because reviving is what actually fixes dead
        codes rather than merely detecting them.
        """
        revived = 0
        for level in range(self.config.levels):
            dead = (self.usage[level] < 1).nonzero().flatten()
            if dead.numel() == 0:
                continue
            pool = latents[torch.randint(latents.shape[0], (dead.numel(),), generator=generator)]
            self.codebooks[level].data[dead] = pool
            revived += int(dead.numel())
        self.usage.zero_()
        return revived


class RqVae(nn.Module):
    """The full autoencoder: normalise, encode, residual-quantise, decode."""

    def __init__(self, config: QuantizerConfig) -> None:
        super().__init__()
        if not isinstance(config, QuantizerConfig):
            raise QuantizerError(f"config must be a QuantizerConfig, got {type(config).__name__}")
        self.config = config
        widths = (config.input_dim, *config.encoder_dims, config.latent_dim)
        self.encoder = _mlp(widths, config.activation)
        self.quantizer = ResidualQuantizer(config)
        decoder_widths = (config.latent_dim, *reversed(config.encoder_dims), config.input_dim)
        self.decoder = _mlp(decoder_widths, config.activation)
        # BatchNorm + L2, applied to the INPUT; the reconstruction target is the normalised
        # input, so the objective is scale-free.
        self.input_norm = nn.BatchNorm1d(config.input_dim) if config.normalize_input else nn.Identity()
        self._apply_init()

    def _apply_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # -- forward ----------------------------------------------------------- #

    def normalise(self, features: torch.Tensor) -> torch.Tensor:
        """BatchNorm then L2-normalise; the reconstruction target."""
        if not self.config.normalize_input:
            return features
        normed = self.input_norm(features)
        return normed / normed.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        target = self.normalise(features)
        latents = self.encoder(target)
        quantized, codes, quantization_loss, reconstruction_embeddings = self.quantizer.quantize(
            latents
        )
        reconstruction = self.decoder(reconstruction_embeddings)
        return {
            "target": target,
            "latents": latents,
            "reconstruction": reconstruction,
            "codes": codes,
            "quantization_loss": quantization_loss,
            "reconstruction_loss": nn.functional.mse_loss(reconstruction, target),
        }

    @torch.no_grad()
    def codes_for_features(
        self, features: torch.Tensor, *, batch_size: int | None = None
    ) -> torch.Tensor:
        """``[N, levels]`` codes in row order.  Item features only; PAD is never an input."""
        self.eval()
        size = int(batch_size or self.config.batch_size)
        device = next(self.parameters()).device
        chunks: list[torch.Tensor] = []
        for start in range(0, features.shape[0], size):
            block = features[start : start + size].to(device)
            codes = self.quantizer.quantize(self.encoder(self.normalise(block)))[1]
            chunks.append(codes.cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, self.config.levels), dtype=torch.long)

    @torch.no_grad()
    def seed_codebooks(self, features: torch.Tensor, *, generator: torch.Generator) -> dict[str, Any]:
        """k-means++ initialise each level on the residual of the previous level.

        The sampling matches ``kmeans_sample``; the residual is recomputed per level so level
        ``l + 1`` is seeded on what level ``l`` failed to explain, which is what makes the
        hierarchy coarse-to-fine rather than three copies of one clustering.
        """
        device = next(self.parameters()).device
        sample = features
        if features.shape[0] > self.config.kmeans_sample:
            index = torch.randperm(
                features.shape[0], generator=generator, device=features.device
            )[: self.config.kmeans_sample]
            sample = features[index]
        sample = sample.to(device)
        with torch.no_grad():
            residual = self.encoder(self.normalise(sample))
            revived_total = 0
            for level in range(self.config.levels):
                centroids = kmeans_plus_plus(residual, self.config.codebook_size, generator=generator)
                centroids, empty = refine_codebook(
                    centroids, residual, iterations=self.config.kmeans_iters, generator=generator
                )
                self.quantizer.codebooks[level].data = centroids
                distances = (
                    residual.pow(2).sum(dim=1, keepdim=True)
                    - 2.0 * residual @ centroids.t()
                    + centroids.pow(2).sum(dim=1).unsqueeze(0)
                )
                residual = residual - centroids[distances.argmin(dim=1)]
                revived_total += empty
        return {
            "sampled": int(sample.shape[0]),
            "empty_after_lloyd": revived_total,
            "seed_points": int(sample.shape[0]),
        }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def layer_utilisation(codes: np.ndarray, *, codebook_size: int) -> tuple[dict[str, Any], ...]:
    """Per-level ``used_codes`` / ``dead_codes`` / ``frac_layer_coverage`` / ``id_entropy``.

    Reported, not ranked.  The only hard rule built on these numbers is the
    catastrophic-collapse stop at more than 20 % dead codes, which lives in the caller; this
    function states no threshold of its own.
    """
    table = np.asarray(codes)
    if table.ndim != 2:
        raise QuantizerError(f"codes must be [N, levels], got shape {tuple(table.shape)}")
    blocks: list[dict[str, Any]] = []
    for level in range(table.shape[1]):
        counts = np.bincount(table[:, level], minlength=codebook_size).astype(np.float64)
        used = int((counts > 0).sum())
        fractions = counts[counts > 0] / counts.sum()
        entropy = float(-(fractions * np.log(fractions)).sum()) if fractions.size else 0.0
        blocks.append(
            {
                "level": level,
                "codebook_size": int(codebook_size),
                "used_codes": used,
                "dead_codes": int(codebook_size) - used,
                "frac_layer_coverage": round(used / int(codebook_size), 6),
                "id_entropy": round(entropy, 6),
                "max_entropy": round(math.log(int(codebook_size)), 6),
                "collapsed": (int(codebook_size) - used) > 0.2 * int(codebook_size),
            }
        )
    return tuple(blocks)


@dataclass
class TrainingOutcome:
    """Everything a run report needs, so nothing has to be re-derived from the checkpoint."""

    history: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    peak_allocated_bytes: int | None = None
    revived_total: int = 0
    seed_info: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    @property
    def final(self) -> dict[str, Any]:
        return self.history[-1] if self.history else {}


def train_quantizer(
    model: RqVae,
    features: torch.Tensor,
    *,
    config: QuantizerConfig,
    device: torch.device | str = "cpu",
    determinism: bool = False,
    log: Any = None,
) -> TrainingOutcome:
    """Train the RQ-VAE on catalogue item features.

    ``features`` holds one row per real item and **no PAD row**, so the reserved sentinel can
    never be quantised.  The loss reported per epoch is the sum of the reconstruction term and
    the quantisation term; the two are also reported separately, because a single blended
    number would hide which half is failing.
    """
    outcome = TrainingOutcome(device=str(device))
    model = model.to(device)
    if determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)

    outcome.seed_info = model.seed_codebooks(features, generator=generator)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = features.shape[0]
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(rows, generator=generator)
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "quantization_loss": 0.0}
        steps = 0
        for start in range(0, rows, config.batch_size):
            index = order[start : start + config.batch_size]
            if index.numel() < 2:
                # BatchNorm needs more than one row; a trailing singleton is skipped rather
                # than forcing a different normalisation path for it.
                continue
            batch = features[index].to(device)
            output = model(batch)
            loss = output["reconstruction_loss"] + output["quantization_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["reconstruction_loss"] += float(output["reconstruction_loss"].detach())
            totals["quantization_loss"] += float(output["quantization_loss"].detach())
            steps += 1
        revived = 0
        if config.revive_dead:
            sample = features[torch.randperm(rows, generator=generator)[: min(rows, 20_000)]]
            with torch.no_grad():
                latents = model.encoder(model.normalise(sample.to(device)))
            revived = model.quantizer.revive_dead_codes(latents, generator=generator)
            outcome.revived_total += revived
        divisor = max(1, steps)
        record = {
            "epoch": epoch,
            "steps": steps,
            "loss": round(totals["loss"] / divisor, 6),
            "reconstruction_loss": round(totals["reconstruction_loss"] / divisor, 6),
            "quantization_loss": round(totals["quantization_loss"] / divisor, 6),
            "revived_codes": revived,
        }
        outcome.history.append(record)
        if log is not None:
            log(record)
    outcome.seconds = time.perf_counter() - started
    if device != "cpu" and torch.cuda.is_available():
        outcome.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    return outcome
