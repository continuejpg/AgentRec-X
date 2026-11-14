"""Small neural memory-conditioned ranker.

Architecture:

* item encoder: MLP over catalog metadata features;
* history encoder: shared item MLP + masked attention pooling over the last 64
  observed interactions (rating and recency included), plus candidate-conditioned
  attention over the same history;
* memory/candidate interaction: candidate attribute vector x historical positive /
  negative attribute frequencies;
* scoring head: MLP over candidate, global/candidate-conditioned history, memory and
  attribute interaction features;
* auxiliary per-candidate rating head.

Memory OFF is the same candidate/scoring family without the history branch.  No
user_id or raw item-id embedding is a model input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    item_dim: int
    memory_dim: int
    hidden: int = 128
    dropout: float = 0.1
    memory_on: bool = True
    use_attributes: bool = True


class NeuralMemoryRanker(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        attr_indices: list[int],
        high_attr_indices: list[int],
        low_attr_indices: list[int],
    ):
        super().__init__()
        self.config = config
        self.register_buffer("attr_indices", torch.tensor(attr_indices, dtype=torch.long), persistent=False)
        self.register_buffer("high_attr_indices", torch.tensor(high_attr_indices, dtype=torch.long), persistent=False)
        self.register_buffer("low_attr_indices", torch.tensor(low_attr_indices, dtype=torch.long), persistent=False)
        h = config.hidden
        self.item_mlp = nn.Sequential(
            nn.Linear(config.item_dim, h), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(config.item_dim, h), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.memory_mlp = nn.Sequential(
            nn.Linear(config.memory_dim, h), nn.ReLU(), nn.Dropout(config.dropout),
        )
        self.history_item_mlp = nn.Sequential(
            nn.Linear(config.item_dim + 4, h), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.history_attn = nn.Linear(h, 1)
        if config.memory_on:
            if config.use_attributes:
                # candidate, history, cand*history, candidate-conditioned history,
                # memory, attribute interaction (n_attrs*5+6)
                score_dim = h * 5 + len(attr_indices) * 5 + 6
            else:
                score_dim = h * 5
        else:
            score_dim = h
        self.score_mlp = nn.Sequential(
            nn.Linear(score_dim, h * 2), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h * 2, h), nn.ReLU(),
        )
        self.score_head = nn.Linear(h, 1)
        self.rating_head = nn.Linear(h, 1)
        self.memory_on = config.memory_on

    def encode_history(
        self,
        hist_item_features: torch.Tensor,
        hist_ratings: torch.Tensor,
        hist_days: torch.Tensor,
        hist_known: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hist_item_features.shape
        rating_norm = (hist_ratings - 3.0) / 2.0
        days_norm = torch.log1p(torch.clamp(hist_days, min=0.0)) / 5.0
        inputs = torch.cat(
            [
                hist_item_features,
                rating_norm.unsqueeze(-1),
                days_norm.unsqueeze(-1),
                hist_known.unsqueeze(-1),
                hist_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.history_item_mlp(inputs)
        scores = self.history_attn(h).squeeze(-1)
        scores = scores.masked_fill(~hist_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        return context, h, hist_mask

    def forward(
        self,
        candidate_item_features: torch.Tensor,
        history_item_features: torch.Tensor,
        history_ratings: torch.Tensor,
        history_days: torch.Tensor,
        history_known: torch.Tensor,
        history_mask: torch.Tensor,
        memory_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = candidate_item_features.shape
        cand = self.candidate_mlp(candidate_item_features)
        if not self.memory_on:
            flat = cand.reshape(b * k, -1)
            hidden = self.score_mlp(flat)
            scores = self.score_head(hidden).reshape(b, k)
            rating = self.rating_head(hidden).reshape(b, k)
            return scores, rating

        hist_context, hist_items, hist_mask = self.encode_history(
            history_item_features, history_ratings, history_days, history_known, history_mask
        )
        # candidate-conditioned attention over the history item embeddings
        attention_logits = torch.einsum("bkh,blh->bkl", cand, hist_items)
        attention_logits = attention_logits / math.sqrt(self.config.hidden)
        attention_logits = attention_logits.masked_fill(~hist_mask.unsqueeze(1), -1e9)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        candidate_history = torch.einsum("bkl,blh->bkh", attention_weights, hist_items)

        mem = self.memory_mlp(memory_features)
        hist_expanded = hist_context.unsqueeze(1).expand(-1, k, -1)
        mem_expanded = mem.unsqueeze(1).expand(-1, k, -1)
        base_features = [cand, hist_expanded, cand * hist_expanded, candidate_history, mem_expanded]
        if self.config.use_attributes:
            attr = candidate_item_features[:, :, self.attr_indices]
            high = memory_features[:, self.high_attr_indices].unsqueeze(1)
            low = memory_features[:, self.low_attr_indices].unsqueeze(1)
            count = memory_features[:, :1].clamp(min=1.0).unsqueeze(1)
            high_norm = high / count
            low_norm = low / count
            inter_high = attr * high_norm
            inter_low = attr * low_norm
            high_norm_full = high_norm.expand(-1, k, -1)
            low_norm_full = low_norm.expand(-1, k, -1)
            scalars = torch.cat(
                [
                    attr.sum(dim=-1, keepdim=True),
                    high_norm_full.sum(dim=-1, keepdim=True),
                    low_norm_full.sum(dim=-1, keepdim=True),
                    inter_high.sum(dim=-1, keepdim=True),
                    inter_low.sum(dim=-1, keepdim=True),
                    (attr * (high_norm_full + low_norm_full)).sum(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            interaction = torch.cat(
                [attr, high_norm_full, low_norm_full, inter_high, inter_low, scalars],
                dim=-1,
            )
        else:
            interaction = None
        # Remove an unused local variable warning while keeping the attribute branch
        # explicit and readable.
        if self.config.use_attributes:
            assert interaction is not None
            score_input = torch.cat(base_features + [interaction], dim=-1)
        else:
            score_input = torch.cat(base_features, dim=-1)
        flat = score_input.reshape(b * k, -1)
        hidden = self.score_mlp(flat).reshape(b, k, -1)
        scores = self.score_head(hidden).squeeze(-1)
        rating = self.rating_head(hidden).squeeze(-1)
        return scores, rating
