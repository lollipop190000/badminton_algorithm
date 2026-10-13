from __future__ import annotations

import torch
from torch import nn


class ShuttleGRU(nn.Module):
    """Small causal sequence model that emits a logit at every observed stroke."""

    def __init__(
        self,
        shot_vocab_size: int,
        hitter_vocab_size: int,
        zone_vocab_size: int,
        continuous_size: int,
        hidden_size: int = 64,
        shot_embedding_dim: int = 12,
        hitter_embedding_dim: int = 2,
        zone_embedding_dim: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.shot_embedding = nn.Embedding(shot_vocab_size, shot_embedding_dim, padding_idx=0)
        self.hitter_embedding = nn.Embedding(hitter_vocab_size, hitter_embedding_dim, padding_idx=0)
        self.zone_embedding = nn.Embedding(zone_vocab_size, zone_embedding_dim, padding_idx=0)
        feature_size = shot_embedding_dim + hitter_embedding_dim + zone_embedding_dim + continuous_size
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        shot_idx: torch.Tensor,
        hitter_idx: torch.Tensor,
        zone_idx: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.shot_embedding(shot_idx),
                self.hitter_embedding(hitter_idx),
                self.zone_embedding(zone_idx),
                continuous,
            ],
            dim=-1,
        )
        hidden, _ = self.gru(self.input_dropout(features))
        return self.output(hidden).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
