from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


class ModalityProjection(nn.Module):
    """Project a precomputed modality feature into BeltRec's shared latent space.

    The paper uses pretrained vision/text encoders followed by trainable projection layers.
    To keep this reproduction self-contained, raw pretrained encoders are represented by
    precomputed feature vectors; this module reproduces the trainable projection + LayerNorm.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredFieldEncoder(nn.Module):
    """Encode categorical/continuous supplier, inventory, logistics and market fields."""

    def __init__(
        self,
        cardinalities: Mapping[str, int],
        continuous_fields: Iterable[str],
        field_emb_dim: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()
        self.cardinalities = dict(cardinalities)
        self.continuous_fields = list(continuous_fields)
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, field_emb_dim) for name, size in self.cardinalities.items()}
        )
        in_dim = len(self.cardinalities) * field_emb_dim + len(self.continuous_fields)
        self.proj = nn.Sequential(
            nn.Linear(max(in_dim, 1), output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        chunks = []
        for name, emb in self.embeddings.items():
            chunks.append(emb(fields[name].long()))
        for name in self.continuous_fields:
            chunks.append(fields[name].float().view(-1, 1))
        if not chunks:
            raise ValueError("StructuredFieldEncoder requires at least one field.")
        return self.proj(torch.cat(chunks, dim=-1))


class DomainStructuredMultimodalEncoder(nn.Module):
    """Equation (1)-(2): modality projection plus taxonomy/attribute prediction heads."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        num_attributes: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.projections = nn.ModuleDict(
            {
                name: ModalityProjection(dim, hidden_dim=hidden_dim, output_dim=output_dim)
                for name, dim in modality_dims.items()
            }
        )
        self.attribute_heads = nn.ModuleDict(
            {name: nn.Linear(output_dim, num_attributes) for name in modality_dims}
        )

    def forward(self, modality_features: Dict[str, torch.Tensor]):
        z = {}
        logits = {}
        for name in self.modalities:
            z[name] = self.projections[name](modality_features[name])
            logits[name] = self.attribute_heads[name](z[name])
        return z, logits

    @staticmethod
    def attribute_consistency_weight(
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        floor: float = 0.25,
    ) -> torch.Tensor:
        """Soft implementation of the paper's attribute-consistency pair weight omega_i."""
        pa = torch.sigmoid(logits_a)
        pb = torch.sigmoid(logits_b)
        agreement = F.cosine_similarity(pa, pb, dim=-1).clamp(min=0.0, max=1.0)
        return floor + (1.0 - floor) * agreement
