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
from .bge_sasrec import BGEEnhancedSASRec, build_bge_sasrec
from .lightgcn import LightGCN, LightGCNConfig, LightGCNError

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
    "BGEEnhancedSASRec",
    "build_bge_sasrec",
    "LightGCN",
    "LightGCNConfig",
    "LightGCNError",
]

__version__ = "0.1.0"
