"""Model architectures.

Milestone 3 provides the SASRec architecture only.  Models expose training logits
and raw full-catalog scores; they contain no evaluation logic, no optimizer and no
training loop.
"""

from __future__ import annotations

from .sasrec import (
    MODEL_NAME,
    ModelSmokeResult,
    SASRec,
    SASRecBlock,
    SASRecConfig,
    SASRecError,
    CausalSelfAttention,
    PointWiseFeedForward,
    build_causal_attention_mask,
    build_model,
    build_padding_mask,
    make_score_fn,
    set_torch_seed,
    smoke_forward,
    valid_position_mask,
)

__all__ = [
    "MODEL_NAME",
    "ModelSmokeResult",
    "SASRec",
    "SASRecBlock",
    "SASRecConfig",
    "SASRecError",
    "CausalSelfAttention",
    "PointWiseFeedForward",
    "build_causal_attention_mask",
    "build_model",
    "build_padding_mask",
    "make_score_fn",
    "set_torch_seed",
    "smoke_forward",
    "valid_position_mask",
]

__version__ = "0.1.0"
