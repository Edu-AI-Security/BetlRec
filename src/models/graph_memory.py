from __future__ import annotations

import torch
from torch import nn


class RelationAwareGraphLayer(nn.Module):
    """Memory-efficient relation-aware message passing.

    This is a factorized form of Eq. (7): a shared linear transform is modulated by
    relation-specific vectors, consistent with the manuscript's statement that relation
    parameters are shared across semantically similar edges. Eq. (8) is applied as a
    degree/environment tempering coefficient before aggregation.
    """

    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_scale = nn.Embedding(num_relations, hidden_dim)
        nn.init.ones_(self.relation_scale.weight)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor,
        env_beta: torch.Tensor | float = 0.5,
    ) -> torch.Tensor:
        src, dst = edge_index.long()
        transformed = self.shared_transform(h)
        msg = transformed[src] * self.relation_scale(relation_ids.long())

        deg = torch.bincount(src, minlength=h.size(0)).float().clamp_min(1.0).to(h.device)
        if not torch.is_tensor(env_beta):
            env_beta = torch.tensor(float(env_beta), device=h.device)
        rho = (1.0 + torch.log1p(deg[src])).pow(-env_beta)
        msg = msg * rho.unsqueeze(-1)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        cnt = torch.bincount(dst, minlength=h.size(0)).float().clamp_min(1.0).to(h.device)
        agg = agg / cnt.unsqueeze(-1)

        g = self.gate(torch.cat([h, agg], dim=-1))
        update = g * agg + (1.0 - g) * h
        return self.norm(h + self.dropout(update))


class IndustrialBeltGraphMemory(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_relations: int = 8, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList(
            [RelationAwareGraphLayer(hidden_dim, num_relations, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, node_states, edge_index, relation_ids, env_beta=0.5):
        h = node_states
        for layer in self.layers:
            h = layer(h, edge_index, relation_ids, env_beta=env_beta)
        return h
