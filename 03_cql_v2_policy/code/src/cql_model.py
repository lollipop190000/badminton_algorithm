from __future__ import annotations

import torch
from torch import nn


class DiscreteCQLNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_layers: list[int]) -> None:
        super().__init__()
        dimensions = [state_dim, *hidden_layers, num_actions]
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
