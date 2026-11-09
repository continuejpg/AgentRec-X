"""BGE-enhanced SASRec: the existing sequence model plus frozen item content.

The only change to SASRec is item representation: a trainable item-ID embedding plus a
trainable projection of the frozen P8-B BGE-base vector.  The same representation is
used for input tokens, training positive/negative logits, and full-catalogue scoring.
"""

from __future__ import annotations

import torch
from torch import nn

from recommendation.datasets.sasrec import PAD_ID
from recommendation.models.sasrec import SASRec, SASRecConfig, SASRecError

__all__ = ["BGEEnhancedSASRec", "build_bge_sasrec"]


class BGEEnhancedSASRec(SASRec):
    """SASRec whose item table is ``ID embedding + projected frozen BGE vector``."""

    def __init__(
        self,
        config: SASRecConfig,
        bge_vectors: torch.Tensor,
    ) -> None:
        super().__init__(config)
        if not isinstance(bge_vectors, torch.Tensor):
            raise SASRecError("bge_vectors must be a torch.Tensor")
        if bge_vectors.shape != (config.num_items + 1, 768):
            raise SASRecError(
                "bge_vectors must have shape "
                f"[{config.num_items + 1}, 768], got {tuple(bge_vectors.shape)}"
            )
        if not bool(torch.isfinite(bge_vectors).all()):
            raise SASRecError("bge_vectors contain NaN/Inf")
        if not bool(torch.allclose(bge_vectors[PAD_ID], torch.zeros_like(bge_vectors[PAD_ID]))):
            raise SASRecError("BGE PAD row must be zero")

        # Frozen data, not a parameter; not persisted in checkpoints (rebuild from the
        # accepted BGE index on load).
        self.register_buffer("bge_vectors", bge_vectors.detach().float(), persistent=False)
        self.bge_projection = nn.Linear(768, config.hidden_size, bias=False)
        nn.init.normal_(
            self.bge_projection.weight, mean=0.0, std=config.initializer_range
        )
        self.bge_scale = nn.Parameter(torch.tensor(1.0))

    def item_representations(self) -> torch.Tensor:
        """ID embedding plus projected frozen BGE content; PAD row stays zero."""
        return self.item_embedding.weight + self.bge_scale * self.bge_projection(
            self.bge_vectors
        )

    def describe(self) -> dict:
        """Return a JSON-serialisable description of the enhanced model."""
        description = super().describe()
        description["content"] = {
            "source": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "frozen": True,
            "fusion": "id_embedding + scale * linear(bge)",
            "projection_size": int(self.config.hidden_size),
        }
        return description


def build_bge_sasrec(
    *,
    num_items: int,
    bge_vectors: torch.Tensor,
    seed: int = 0,
    max_seq_len: int = 50,
    hidden_size: int = 64,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
) -> BGEEnhancedSASRec:
    """Build a reproducibly initialised BGE-enhanced SASRec."""
    torch.manual_seed(seed)
    config = SASRecConfig(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
    )
    return BGEEnhancedSASRec(config, bge_vectors)
