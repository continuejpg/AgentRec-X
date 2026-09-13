"""Configurable SASRec model (Milestone 3, Part B).

A conventional SASRec: item and positional embeddings, a stack of pre-norm
Transformer blocks with causal self-attention and a point-wise feed-forward
network, residual connections, layer normalization, dropout, and a final layer
normalization.

Block order (pre-normalization, documented explicitly)
-----------------------------------------------------
::

    x = item_embedding(input_ids) + positional_embedding[0..L-1]
    x = dropout(x)
    for each block:
        a = layer_norm_1(x)
        a = causal_self_attention(a)
        x = x + dropout(a)                    # residual
        f = layer_norm_2(x)
        f = feed_forward(f)                   # Linear -> GELU -> dropout -> Linear
        x = x + dropout(f)                    # residual
    x = final_layer_norm(x)                   # final normalization

Within attention the order is ``Linear -> reshape to heads -> scaled dot product ->
softmax -> dropout -> Linear``.

Deliberately **not** included: side information, ratings, text embeddings, category
features, RAG features, user embeddings, time decay, or any positional scheme other
than the learned SASRec positional embeddings.

The model is a *model*: it never masks candidates, never ranks, and never computes
metrics.  Those stay in the Milestone 2A evaluator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from recommendation import config
from recommendation.datasets.sasrec import PAD_ID

#: Identifier stored in model metadata.
MODEL_NAME = "sasrec"


class SASRecError(ValueError):
    """Raised when SASRec is configured or called with invalid inputs."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SASRecConfig:
    """Hyper-parameters of the SASRec architecture.

    Defaults are small on purpose: this milestone instantiates and smoke-tests the
    model on CPU and never trains it.
    """

    num_items: int = 100
    max_seq_len: int = 50
    hidden_size: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.1
    feed_forward_multiplier: float = 4.0
    layer_norm_eps: float = 1e-8
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if isinstance(self.num_items, bool) or not isinstance(self.num_items, int) or self.num_items < 1:
            raise SASRecError(f"num_items must be a positive int, got {self.num_items!r}")
        if isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int) or self.max_seq_len < 1:
            raise SASRecError(f"max_seq_len must be a positive int, got {self.max_seq_len!r}")
        if isinstance(self.num_blocks, bool) or not isinstance(self.num_blocks, int) or self.num_blocks < 1:
            raise SASRecError(f"num_blocks must be >= 1, got {self.num_blocks!r}")
        if self.hidden_size < 1:
            raise SASRecError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_heads < 1:
            raise SASRecError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise SASRecError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads "
                f"({self.num_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise SASRecError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_size(self) -> int:
        """Per-head dimension."""
        return self.hidden_size // self.num_heads

    @property
    def feed_forward_size(self) -> int:
        """Hidden dimension of the point-wise feed-forward network."""
        return int(self.hidden_size * self.feed_forward_multiplier)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the configuration."""
        return {
            "model": MODEL_NAME,
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "hidden_size": self.hidden_size,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "feed_forward_multiplier": self.feed_forward_multiplier,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "normalization": "pre-norm",
            "activation": "gelu",
            "padding_idx": PAD_ID,
        }


# --------------------------------------------------------------------------- #
# Masking helpers
# --------------------------------------------------------------------------- #


def build_padding_mask(input_ids: torch.Tensor, num_items: int) -> torch.Tensor:
    """Return ``[batch, seq]`` bool mask: True where the position is a real item.

    A position is valid when the id is in ``1..num_items``.  PAD ``0`` (and any
    out-of-range id) is invalid and must never behave like a real item.
    """
    _check_input_ids(input_ids, num_items)
    return (input_ids > PAD_ID) & (input_ids <= num_items)


def build_causal_attention_mask(
    input_ids: torch.Tensor,
    num_items: int,
) -> torch.Tensor:
    """Return a boolean attention mask of shape ``[batch, 1, seq, seq]``.

    ``True`` means **blocked**.  Two things are combined:

    * causality - query ``t`` may not attend to key ``t' > t``;
    * padding - padded positions are blocked as keys, and padded *query* rows are
      fully blocked so their hidden states are zero rather than a copy of a real
      item's representation.
    """
    valid = build_padding_mask(input_ids, num_items)
    seq_len = input_ids.shape[1]
    device = input_ids.device

    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    blocked = causal.unsqueeze(0).unsqueeze(0)  # [1,1,seq,seq] -> True above diagonal
    blocked = blocked.expand(input_ids.shape[0], 1, seq_len, seq_len).clone()

    # block padded keys
    blocked |= (~valid).unsqueeze(1).unsqueeze(2)  # [b,1,1,seq]
    # block padded queries entirely
    blocked |= (~valid).unsqueeze(1).unsqueeze(3)  # [b,1,seq,1]
    return blocked


def valid_position_mask(positive_ids: torch.Tensor) -> torch.Tensor:
    """Return ``[batch, seq]`` bool mask: True where a real positive target exists.

    Padding positions carry ``positive_ids == PAD_ID`` and must not contribute to the
    training loss.
    """
    return positive_ids != PAD_ID


def _check_input_ids(input_ids: torch.Tensor, num_items: int) -> None:
    if input_ids.dim() != 2:
        raise SASRecError(f"input_ids must be 2-D [batch, seq], got shape {tuple(input_ids.shape)}")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise SASRecError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
    if input_ids.numel() == 0:
        raise SASRecError("input_ids is empty")
    if int(input_ids.min()) < PAD_ID or int(input_ids.max()) > num_items:
        raise SASRecError(
            f"input_ids must lie in [0, {num_items}]; got range "
            f"[{int(input_ids.min())}, {int(input_ids.max())}]"
        )


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with an explicit causal+padding mask.

    The computation is written out explicitly instead of delegating to
    :func:`torch.nn.functional.scaled_dot_product_attention`.  The fused kernel
    returned ``NaN`` rows for this mask layout on the pinned CPU build (verified
    against a manual reference), and writing the steps out keeps causality and
    padding semantics directly inspectable and testable::

        scores  = (q @ k^T) / sqrt(head_size)
        scores  = scores.masked_fill(blocked, -inf)   # True = blocked (causal + padding)
        weights = softmax(scores)                     # exactly 0 at blocked positions
        weights = dropout(weights)
        out     = weights @ v
    """

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        """Apply attention; ``blocked`` is a ``[batch,1,seq,seq]`` True-means-blocked mask."""
        batch, seq_len, _ = hidden.shape
        heads = self.config.num_heads
        head_size = self.config.head_size

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq_len, heads, head_size).transpose(1, 2)

        query = split(self.query(hidden))
        key = split(self.key(hidden))
        value = split(self.value(hidden))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_size)
        scores = scores.masked_fill(blocked, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # A fully blocked query row (padding) is softmax(-inf everywhere), which is
        # NaN.  Neutralise it here so no NaN reaches the residual stream; the caller
        # additionally zeroes padded query rows after the whole stack.
        blocked_rows = blocked.all(dim=-1, keepdim=True)
        weights = torch.where(blocked_rows, torch.zeros_like(weights), weights)
        weights = self.attention_dropout(weights)

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.output_dropout(self.output(attended))


class PointWiseFeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> dropout -> Linear."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.linear_in = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.linear_out = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network."""
        return self.dropout(self.linear_out(torch.nn.functional.gelu(self.linear_in(hidden))))


class SASRecBlock(nn.Module):
    """Pre-norm Transformer block: attention then feed-forward, each with a residual."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = PointWiseFeedForward(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        hidden = hidden + self.residual_dropout(self.attention(self.attention_norm(hidden), blocked))
        hidden = hidden + self.residual_dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class SASRec(nn.Module):
    """SASRec sequence encoder with a shared-embedding output head.

    Parameters
    ----------
    config:
        A :class:`SASRecConfig`.

    The item embedding table has ``num_items + 1`` rows so that item id ``i`` maps to
    row ``i`` directly, with ``padding_idx=0`` so PAD's embedding is zero and never
    receives a gradient.
    """

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        if not isinstance(config, SASRecConfig):
            raise SASRecError(f"config must be a SASRecConfig, got {type(config).__name__}")
        self.config = config

        self.item_embedding = nn.Embedding(
            config.num_items + 1, config.hidden_size, padding_idx=PAD_ID
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(SASRecBlock(config) for _ in range(config.num_blocks))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.apply(self._init_weights)
        # padding_idx rows must stay zero after initialization as well
        with torch.no_grad():
            self.item_embedding.weight[PAD_ID].zero_()

    # -- initialization ---------------------------------------------------- #

    def _init_weights(self, module: nn.Module) -> None:
        """Normal-initialise Linear/Embedding weights, zero biases and norms."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # -- properties -------------------------------------------------------- #

    @property
    def num_items(self) -> int:
        """Catalogue size."""
        return self.config.num_items

    @property
    def max_seq_len(self) -> int:
        """Maximum supported sequence length."""
        return self.config.max_seq_len

    def score_vector_length(self) -> int:
        """Length of a full-catalog score vector (PAD slot included)."""
        return self.config.num_items + 1

    def positional_ids(self, seq_len: int) -> torch.Tensor:
        """Return positional ids ``[0..seq_len-1]`` for a sequence length."""
        if seq_len < 1 or seq_len > self.config.max_seq_len:
            raise SASRecError(
                f"sequence length {seq_len} outside [1, {self.config.max_seq_len}]"
            )
        return torch.arange(seq_len, dtype=torch.long, device=self.item_embedding.weight.device)

    # -- encoding ---------------------------------------------------------- #

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        return_padding_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode ``input_ids`` ``[batch, seq]`` into hidden states ``[batch, seq, h]``.

        Padded query positions are fully masked in attention, so their hidden states
        are zero: PAD never behaves like a real item.
        """
        _check_input_ids(input_ids, self.config.num_items)
        if input_ids.shape[1] > self.config.max_seq_len:
            raise SASRecError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.config.max_seq_len}; the encoder truncates, the model does not"
            )

        blocked = build_causal_attention_mask(input_ids, self.config.num_items)
        hidden = self.item_embedding(input_ids) + self.position_embedding(
            self.positional_ids(input_ids.shape[1])
        )
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, blocked)
        hidden = self.final_norm(hidden)

        # Padded query rows were fully masked; force them to exactly zero so padding
        # cannot masquerade as a real representation (and so callers can rely on it).
        valid = build_padding_mask(input_ids, self.config.num_items)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        if return_padding_mask:
            return hidden, valid
        return hidden

    # -- training logits --------------------------------------------------- #

    def training_logits(
        self,
        input_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        *,
        return_valid_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return aligned positive/negative logits for valid positions only.

        ``positive_logit[t] = dot(hidden[t], item_embedding[positive_ids[t]])`` and
        likewise for negatives, using the same (weight-tied) item embedding table as
        the input and the full-catalog scorer.

        Only positions whose ``positive_ids`` is a real item are returned; padding
        positions are dropped rather than silently contributing a loss.  Returned
        tensors are flat, of shape ``[num_valid_positions]``, aligned with each other.
        """
        for name, tensor in (("positive_ids", positive_ids), ("negative_ids", negative_ids)):
            if tensor.shape != input_ids.shape:
                raise SASRecError(
                    f"{name} shape {tuple(tensor.shape)} must match input_ids "
                    f"shape {tuple(input_ids.shape)}"
                )
            if tensor.dtype not in (torch.int32, torch.int64):
                raise SASRecError(f"{name} must be an integer tensor, got {tensor.dtype}")
            if int(tensor.min()) < PAD_ID or int(tensor.max()) > self.config.num_items:
                raise SASRecError(
                    f"{name} must lie in [0, {self.config.num_items}], got range "
                    f"[{int(tensor.min())}, {int(tensor.max())}]"
                )

        valid = valid_position_mask(positive_ids)
        hidden = self.encode(input_ids)
        assert isinstance(hidden, torch.Tensor)

        weights = self.item_embedding.weight
        positive_logits = (hidden * weights[positive_ids]).sum(dim=-1)
        negative_logits = (hidden * weights[negative_ids]).sum(dim=-1)

        positive_valid = positive_logits[valid]
        negative_valid = negative_logits[valid]
        if return_valid_mask:
            return positive_valid, negative_valid, valid
        return positive_valid, negative_valid

    # -- full-catalog scoring ---------------------------------------------- #

    def full_catalog_scores(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw scores for every catalog item, shape ``[batch, num_items + 1]``.

        ``scores[b, item_id]`` is that item's score.  The scores are the dot product
        between the hidden state at the **last valid position** and every item
        embedding, using the same table as the input embedding.

        The model performs **no** evaluation masking: seen items, PAD and targets all
        receive an ordinary score.  The PAD column is finite (and typically ~0,
        because the PAD embedding row is zero) so the evaluator's non-finite check
        covers the whole vector.
        """
        _check_input_ids(input_ids, self.config.num_items)
        valid = build_padding_mask(input_ids, self.config.num_items)
        if not bool(valid.any(dim=1).all()):
            raise SASRecError(
                "every sequence must contain at least one non-PAD item; "
                "reject empty/all-PAD histories at the encoding boundary instead"
            )

        hidden = self.encode(input_ids)
        assert isinstance(hidden, torch.Tensor)

        # last valid position per sequence
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        last_valid = (positions * valid.long()).max(dim=1).values  # [batch]
        summary = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_valid]
        return summary @ self.item_embedding.weight.t()

    # -- metadata ---------------------------------------------------------- #

    def item_dim(self) -> int:
        """Number of rows in the item embedding table (``num_items + 1``)."""
        return self.item_embedding.num_embeddings

    def item_padding_index(self) -> int | None:
        """The item embedding's ``padding_idx`` (PAD)."""
        return self.item_embedding.padding_idx

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the model."""
        return {
            "config": self.config.as_dict(),
            "parameter_count": self.parameter_count(),
            "score_vector_length": self.score_vector_length(),
        }


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #


def set_torch_seed(seed: int) -> None:
    """Seed PyTorch's global RNG for reproducible CPU initialization."""
    torch.manual_seed(seed)


def build_model(
    num_items: int,
    *,
    seed: int = 0,
    max_seq_len: int = 50,
    hidden_size: int = 64,
    num_blocks: int = 2,
    num_heads: int = 1,
    dropout: float = 0.1,
) -> SASRec:
    """Construct a SASRec model with reproducible initialization."""
    set_torch_seed(seed)
    return SASRec(
        SASRecConfig(
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
    )


def make_score_fn(
    model: SASRec,
    max_seq_len: int,
    num_items: int,
) -> Any:
    """Return a Milestone 2A compatible scorer wrapping ``model``.

    The returned callable has the evaluator's signature
    ``score_fn(history, target_item_id) -> list[float]``: it encodes the history,
    runs one full-catalog scoring pass, and hands back a plain Python list.  The
    target argument is ignored, and no masking is applied - the evaluator owns PAD
    exclusion, seen-item masking, target retention, ranking and metrics.
    """
    from recommendation.datasets.sasrec import encode_inference_history

    def score_fn(history: tuple[int, ...], target_item_id: int) -> list[float]:
        del target_item_id  # the model scores the whole catalog; the evaluator ranks
        encoded = encode_inference_history(history, max_seq_len, num_items)
        tensor = torch.tensor([encoded], dtype=torch.long)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                scores = model.full_catalog_scores(tensor)
        finally:
            model.train(was_training)
        return [float(value) for value in scores[0].tolist()]

    return score_fn


@dataclass
class ModelSmokeResult:
    """Container for the CPU smoke-test measurements."""

    config: SASRecConfig
    batch_size: int
    seq_len: int
    timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the smoke result."""
        payload = {
            "config": self.config.as_dict(),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
        }
        payload.update(self.extra)
        return payload


def smoke_forward(
    model: SASRec,
    *,
    batch_size: int = 4,
    seq_len: int | None = None,
) -> ModelSmokeResult:
    """Run the three forwards this milestone smoke-tests, on CPU, without training.

    Returns shapes/finiteness/timings; performs **no** optimizer step and computes
    **no** metrics.
    """
    seq_len = seq_len or model.max_seq_len
    if seq_len > model.max_seq_len:
        raise SASRecError(
            f"seq_len {seq_len} exceeds max_seq_len {model.max_seq_len}"
        )

    generator = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(
        1, model.num_items + 1, (batch_size, seq_len), generator=generator, dtype=torch.long
    )
    positive_ids = torch.randint(
        1, model.num_items + 1, (batch_size, seq_len), generator=generator, dtype=torch.long
    )
    negative_ids = torch.randint(
        1, model.num_items + 1, (batch_size, seq_len), generator=generator, dtype=torch.long
    )

    was_training = model.training
    model.eval()
    timings: dict[str, float] = {}
    try:
        started = time.perf_counter()
        with torch.no_grad():
            hidden = model.encode(input_ids)
            positive_logits, negative_logits = model.training_logits(
                input_ids, positive_ids, negative_ids
            )
            scores = model.full_catalog_scores(input_ids)
        timings["combined_forward"] = time.perf_counter() - started
    finally:
        model.train(was_training)

    assert isinstance(hidden, torch.Tensor)
    return ModelSmokeResult(
        config=model.config,
        batch_size=batch_size,
        seq_len=seq_len,
        timings=timings,
        extra={
            "hidden_shape": list(hidden.shape),
            "positive_logits_shape": list(positive_logits.shape),
            "negative_logits_shape": list(negative_logits.shape),
            "scores_shape": list(scores.shape),
            "hidden_finite": bool(torch.isfinite(hidden).all()),
            "positive_logits_finite": bool(torch.isfinite(positive_logits).all()),
            "negative_logits_finite": bool(torch.isfinite(negative_logits).all()),
            "scores_finite": bool(torch.isfinite(scores).all()),
            "pad_score_finite": bool(torch.isfinite(scores[:, PAD_ID]).all()),
        },
    )
