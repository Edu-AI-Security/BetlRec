from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class EnvironmentPropensityNet(nn.Module):
    """Auxiliary propensity estimator for observed environment/exposure assignment.

    The manuscript says propensity is estimated but does not specify the estimator.
    This lightweight classifier is an explicit reproducibility assumption in this codebase.
    """

    def __init__(self, input_dim: int, num_environments: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_environments),
        )

    def forward(self, exposure_descriptor: torch.Tensor) -> torch.Tensor:
        return self.net(exposure_descriptor)


class CausalAdaptiveFusion(nn.Module):
    """Environment-conditioned modality gate from Eq. (3)-(4)."""

    def __init__(
        self,
        modality_names: List[str],
        hidden_dim: int = 256,
        env_dim: int = 64,
        exposure_dim: int = 64,
        gate_hidden=(256, 128, 64),
    ):
        super().__init__()
        self.modality_names = modality_names
        gate_in = hidden_dim + env_dim + exposure_dim
        layers = []
        prev = gate_in
        for width in gate_hidden:
            layers += [nn.Linear(prev, width), nn.GELU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.gate_mlp = nn.Sequential(*layers)

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        environment_embedding: torch.Tensor,
        exposure_descriptor: torch.Tensor,
    ):
        gates = []
        for name in self.modality_names:
            z = modality_embeddings[name]
            g = torch.sigmoid(
                self.gate_mlp(torch.cat([z, environment_embedding, exposure_descriptor], dim=-1))
            )
            gates.append(g)
        gate_tensor = torch.cat(gates, dim=-1)
        alpha = gate_tensor / gate_tensor.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        fused = 0.0
        for idx, name in enumerate(self.modality_names):
            fused = fused + alpha[:, idx : idx + 1] * modality_embeddings[name]
        return fused, alpha
