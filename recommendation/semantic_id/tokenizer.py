"""RQ-VAE-style semantic tokenizer: catalogue item -> discrete Semantic ID.

What it builds
--------------
Every catalogue item is represented by a vector, encoded into a residual-quantised codebook
hierarchy, and assigned the resulting tuple of codebook indices - its **Semantic ID** (SID)::

    item features (id embedding ++ categorical metadata)
        -> encoder MLP
        -> residual quantisation over L codebooks of size K
        -> (c_1, ..., c_L)          the Semantic ID

Decoding sums the chosen codebook vectors, and the decoder reconstructs the item's feature
vector, so the tokenizer is trained on reconstruction.  This is the standard RQ-VAE shape
(RQ-VAE / TIGER-style), kept deliberately small.

Two properties this module guarantees and audits
------------------------------------------------
**Deterministic and total coverage.**  The features come from an item embedding table and a
categorical metadata matrix, both indexed by item id, so the mapping is a pure function of the
catalogue and the seed - no item needs text to be tokenised, and every valid id yields a code.
:func:`recommendation.semantic_id.tokenizer.audit_tokenizer` reports the coverage.

**Collisions are measured, not hidden.**  ``K ** L`` can be far smaller than the catalogue, and
two items may share a Semantic ID.  A shared code is not a bug to be papered over: it is a real
limit on how much the SID can distinguish, and the audit reports the collision rate, the number of
distinct codes and the codebook utilisation.  Identity is then resolved by
:mod:`recommendation.semantic_id.resolver`, deterministically, from the code *and* the trusted
catalogue - never from the generator alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

__all__ = [
    "QuantizerConfig",
    "RqVaeTokenizer",
    "SemanticId",
    "TokenizerError",
    "audit_tokenizer",
    "build_item_features",
]

#: Sentinel used in a Semantic ID position that no item occupies.  Kept out of the codebook range.
EMPTY_CODE = -1


class TokenizerError(ValueError):
    """Raised when the tokenizer is configured or fed unusably."""


class QuantizerConfig:
    """Codebook hierarchy and encoder/decoder widths.

    Attributes
    ----------
    input_size:
        Width of the item feature vector the encoder consumes.
    latent_size:
        Width of the residual-quantised latent space.
    levels:
        Number of codebooks (the Semantic-ID length).
    codebook_size:
        Entries per codebook (``K``).  ``K ** levels`` is the code space.
    hidden_size:
        Width of the encoder/decoder MLP.  ``0`` disables the hidden layer.
    commitment_weight:
        Weight of the commitment loss that pulls encoder outputs toward their code.
    seed:
        Initialisation seed; recorded with the artifact.
    """

    def __init__(
        self,
        *,
        input_size: int,
        latent_size: int = 32,
        levels: int = 3,
        codebook_size: int = 256,
        hidden_size: int = 128,
        commitment_weight: float = 0.25,
        seed: int = 2026,
    ) -> None:
        for name, value in (
            ("input_size", input_size),
            ("latent_size", latent_size),
            ("levels", levels),
            ("codebook_size", codebook_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TokenizerError(f"{name} must be a positive int, got {value!r}")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 0:
            raise TokenizerError(f"hidden_size must be an int >= 0, got {hidden_size!r}")
        if commitment_weight < 0.0:
            raise TokenizerError(
                f"commitment_weight must be >= 0, got {commitment_weight!r}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TokenizerError("seed must be an int")
        self.input_size = input_size
        self.latent_size = latent_size
        self.levels = levels
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.commitment_weight = float(commitment_weight)
        self.seed = seed

    @property
    def code_space(self) -> int:
        """How many distinct Semantic IDs the hierarchy can express (``K ** L``)."""
        return self.codebook_size**self.levels

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, stored with the artifact."""
        return {
            "input_size": self.input_size,
            "latent_size": self.latent_size,
            "levels": self.levels,
            "codebook_size": self.codebook_size,
            "hidden_size": self.hidden_size,
            "commitment_weight": self.commitment_weight,
            "code_space": self.code_space,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SemanticId:
    """One item's Semantic ID: a tuple of codebook indices, one per level."""

    codes: tuple[int, ...]

    def as_tuple(self) -> tuple[int, ...]:
        return self.codes

    def as_tokens(self) -> tuple[int, ...]:
        """Return the codes as generator tokens (offsets are the generator's business)."""
        return self.codes


def _mlp(input_size: int, output_size: int, hidden_size: int) -> nn.Sequential:
    """A small MLP; ``hidden_size == 0`` means a single linear layer."""
    if hidden_size <= 0:
        return nn.Sequential(nn.Linear(input_size, output_size))
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size),
    )


class RqVaeTokenizer(nn.Module):
    """Residual-quantised autoencoder producing a Semantic ID per catalogue item.

    The module holds only learned parameters.  Identity resolution deliberately lives elsewhere
    (:mod:`recommendation.semantic_id.resolver`), so this model can never be the thing that
    decides which product a code refers to.
    """

    def __init__(self, config: QuantizerConfig) -> None:
        super().__init__()
        if not isinstance(config, QuantizerConfig):
            raise TokenizerError(
                f"config must be a QuantizerConfig, got {type(config).__name__}"
            )
        self.config = config
        self.encoder = _mlp(config.input_size, config.latent_size, config.hidden_size)
        # One codebook per level; each is an embedding-like table of code vectors.
        self.codebooks = nn.ModuleList(
            nn.Embedding(config.codebook_size, config.latent_size)
            for _ in range(config.levels)
        )
        self.decoder = _mlp(config.latent_size, config.input_size, config.hidden_size)
        self.apply(self._init_weights)
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / config.codebook_size, 1.0 / config.codebook_size)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # -- forward ----------------------------------------------------------- #

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode item features into the latent space."""
        return self.encoder(features)

    def quantize(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual-quantise ``[batch, latent]`` latents.

        Returns ``(quantized, codes, commitment_loss, codebook_loss)`` where ``codes`` is
        ``[batch, levels]``.  The **straight-through** estimator passes the gradient of the
        quantised vector to the encoder, which is what the original RQ-VAE/VQ-VAE formulation
        does; commitment and codebook terms are returned separately so the caller can weight
        them explicitly instead of having them hidden inside one number.
        """
        residual = latents
        quantized = torch.zeros_like(latents)
        codes: list[torch.Tensor] = []
        commitment = torch.zeros((), device=latents.device, dtype=latents.dtype)
        codebook_loss = torch.zeros((), device=latents.device, dtype=latents.dtype)
        for level, codebook in enumerate(self.codebooks):
            weight = codebook.weight
            # Nearest code by squared L2 distance, computed as |x|^2 - 2x.w + |w|^2.
            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * residual @ weight.t()
                + weight.pow(2).sum(dim=-1).unsqueeze(0)
            )
            indices = distances.argmin(dim=-1)
            chosen = weight[indices]
            codes.append(indices)
            quantized = quantized + chosen
            commitment = commitment + (residual.detach() - chosen).pow(2).mean()
            codebook_loss = codebook_loss + (residual - chosen.detach()).pow(2).mean()
            residual = residual - chosen.detach()
            del level
        code_tensor = torch.stack(codes, dim=1)
        # Straight-through: forward value is the quantised sum, backward gradient is identity.
        quantized = latents + (quantized - latents).detach()
        return quantized, code_tensor, commitment, codebook_loss

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode, quantise, decode and return every term the loss needs."""
        latents = self.encode(features)
        quantized, codes, commitment, codebook_loss = self.quantize(latents)
        reconstruction = self.decoder(quantized)
        return {
            "reconstruction": reconstruction,
            "codes": codes,
            "commitment_loss": commitment,
            "codebook_loss": codebook_loss,
        }

    # -- inference --------------------------------------------------------- #

    @torch.no_grad()
    def codes_for_features(self, features: torch.Tensor, *, batch_size: int = 4096) -> torch.Tensor:
        """Return ``[num_rows, levels]`` codes for a feature matrix, in row order."""
        self.eval()
        device = next(self.parameters()).device
        chunks: list[torch.Tensor] = []
        for start in range(0, features.shape[0], batch_size):
            block = features[start : start + batch_size].to(device)
            _quantized, codes, _c, _cb = self.quantize(self.encode(block))
            chunks.append(codes.cpu())
        return torch.cat(chunks, dim=0)


def build_item_features(
    *,
    num_items: int,
    item_embedding_dim: int,
    categories: Any | None,
    seed: int = 2026,
    metadata_embedding_dim: int = 16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the ``[num_items + 1, input_size]`` feature matrix the tokenizer consumes.

    The item-id component is a **learned-free, deterministically initialised** embedding: the
    tokenizer's own training updates the encoder and codebooks, not this table, so the features
    stay a fixed function of the catalogue and the seed, which is what makes the whole mapping
    reproducible from the artifact alone.

    Categorical metadata is embedded the same way - a deterministic random embedding per category
    index - so an item with no metadata still has a feature vector and therefore still gets a
    Semantic ID.  That is the coverage guarantee: **no catalogue item needs a text field to be
    tokenised.**
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise TokenizerError(f"num_items must be a positive int, got {num_items!r}")
    if item_embedding_dim < 1 or metadata_embedding_dim < 1:
        raise TokenizerError("embedding dimensions must be >= 1")

    generator = torch.Generator().manual_seed(seed)
    parts: list[torch.Tensor] = []
    rows = num_items + 1  # item id 0 is PAD and is tokenised like any other row

    # PAD's row is exactly zero, so it can never collide with an initialised item row.
    id_features = torch.zeros((rows, item_embedding_dim))
    id_features[1:] = torch.randn((rows - 1, item_embedding_dim), generator=generator)
    parts.append(id_features)

    metadata_note: dict[str, Any] = {"used": False}
    if categories is not None:
        matrix = categories if isinstance(categories, torch.Tensor) else torch.as_tensor(
            categories, dtype=torch.long
        )
        if matrix.shape[0] != rows:
            raise TokenizerError(
                f"categories must have num_items + 1 = {rows} rows, got {matrix.shape[0]}"
            )
        cardinalities = [int(matrix[:, index].max()) + 1 for index in range(matrix.shape[1])]
        field_parts: list[torch.Tensor] = []
        for index, cardinality in enumerate(cardinalities):
            table = torch.zeros((cardinality, metadata_embedding_dim))
            table[1:] = torch.randn(
                (max(cardinality - 1, 0), metadata_embedding_dim), generator=generator
            )
            field_parts.append(table[matrix[:, index]])
        parts.append(torch.cat(field_parts, dim=1))
        metadata_note = {
            "used": True,
            "fields": int(matrix.shape[1]),
            "cardinalities": cardinalities,
            "embedding_dim": metadata_embedding_dim,
        }

    features = torch.cat(parts, dim=1)
    return features, {
        "rows": rows,
        "input_size": int(features.shape[1]),
        "item_embedding_dim": item_embedding_dim,
        "metadata": metadata_note,
        "seed": seed,
        "note": (
            "item-id and metadata features are deterministically initialised from the seed and "
            "are not trained; only the encoder, codebooks and decoder are"
        ),
    }


def audit_tokenizer(
    codes: Sequence[Sequence[int]],
    *,
    num_items: int,
    config: QuantizerConfig,
    codebook_histograms: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Report coverage, collisions and codebook utilisation for an assignment.

    ``codes`` is indexed by item id, so ``codes[0]`` is PAD's (excluded from every statistic) and
    ``codes[i]`` is item ``i``'s Semantic ID for ``1 <= i <= num_items``.
    """
    if len(codes) != num_items + 1:
        raise TokenizerError(
            f"codes must have num_items + 1 = {num_items + 1} entries, got {len(codes)}"
        )
    seen: dict[tuple[int, ...], list[int]] = {}
    for item_id in range(1, num_items + 1):
        key = tuple(int(value) for value in codes[item_id])
        if len(key) != config.levels:
            raise TokenizerError(
                f"item {item_id} has {len(key)} codes, expected {config.levels}"
            )
        if any(code < 0 or code >= config.codebook_size for code in key):
            raise TokenizerError(
                f"item {item_id} has an out-of-range code {key}; codes must be in "
                f"[0, {config.codebook_size})"
            )
        seen.setdefault(key, []).append(item_id)

    collisions = {key: members for key, members in seen.items() if len(members) > 1}
    colliding_items = sum(len(members) for members in collisions.values())
    # A collision group is "lossy" for ranking: only one of its members can be presented per code.
    largest_group = max((len(members) for members in collisions.values()), default=1)

    histogram: list[dict[str, Any]] = []
    if codebook_histograms is not None:
        for level, counts in enumerate(codebook_histograms):
            used = sum(1 for count in counts if count > 0)
            histogram.append(
                {
                    "level": level,
                    "used_codes": used,
                    "codebook_size": config.codebook_size,
                    "utilisation": round(used / config.codebook_size, 6),
                    "dead_codes": config.codebook_size - used,
                }
            )

    return {
        "num_items": num_items,
        "items_with_a_semantic_id": num_items,
        "coverage": 1.0,
        "levels": config.levels,
        "codebook_size": config.codebook_size,
        "code_space": config.code_space,
        "distinct_semantic_ids": len(seen),
        "collision_groups": len(collisions),
        "items_in_a_collision_group": colliding_items,
        "collision_rate": round(colliding_items / num_items, 6) if num_items else 0.0,
        "collision_share_of_codes": (
            round(len(collisions) / len(seen), 6) if seen else 0.0
        ),
        "largest_collision_group": largest_group,
        "codebook_utilisation": histogram,
        "config": config.as_dict(),
    }
