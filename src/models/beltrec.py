from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from .encoders import DomainStructuredMultimodalEncoder
from .causal_fusion import CausalAdaptiveFusion, EnvironmentPropensityNet
from .graph_memory import IndustrialBeltGraphMemory


class BeltRec(nn.Module):
    """Self-contained reproduction of BeltRec's trainable architecture."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_nodes: int,
        num_environments: int,
        modality_dims: Mapping[str, int],
        num_attributes: int,
        num_relations: int,
        hidden_dim: int = 256,
        env_dim: int = 64,
        exposure_dim: int = 64,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.modality_names = list(modality_dims.keys())

        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.base_node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.environment_embedding = nn.Embedding(num_environments, env_dim)

        self.encoder = DomainStructuredMultimodalEncoder(
            modality_dims=modality_dims,
            num_attributes=num_attributes,
            hidden_dim=512,
            output_dim=hidden_dim,
        )
        self.exposure_proj = nn.Sequential(
            nn.Linear(exposure_dim, exposure_dim),
            nn.GELU(),
            nn.LayerNorm(exposure_dim),
        )
        self.causal_fusion = CausalAdaptiveFusion(
            self.modality_names,
            hidden_dim=hidden_dim,
            env_dim=env_dim,
            exposure_dim=exposure_dim,
            gate_hidden=(256, 128, 64),
        )
        self.graph_memory = IndustrialBeltGraphMemory(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_layers=2,
            dropout=0.2,
        )
        self.context_score = nn.Linear(hidden_dim * 3 + env_dim, 1)
        self.taxonomy_head = nn.Linear(hidden_dim, num_attributes)
        self.propensity_net = EnvironmentPropensityNet(exposure_dim, num_environments)

        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.base_node_embedding.weight, std=0.02)
        nn.init.normal_(self.environment_embedding.weight, std=0.02)

    def encode_items(
        self,
        modality_features: Dict[str, torch.Tensor],
        environment_ids: torch.Tensor,
        exposure_descriptor: torch.Tensor,
    ):
        modality_z, modality_attr_logits = self.encoder(modality_features)
        env_h = self.environment_embedding(environment_ids.long())
        q = self.exposure_proj(exposure_descriptor)
        fused, alpha = self.causal_fusion(modality_z, env_h, q)
        tax_logits = self.taxonomy_head(fused)
        prop_logits = self.propensity_net(q)
        return {
            "fused": fused,
            "gates": alpha,
            "modality_z": modality_z,
            "modality_attr_logits": modality_attr_logits,
            "taxonomy_logits": tax_logits,
            "propensity_logits": prop_logits,
            "environment_h": env_h,
        }

    def graph_states(self, edge_index, relation_ids, env_beta=0.5):
        return self.graph_memory(self.base_node_embedding.weight, edge_index, relation_ids, env_beta=env_beta)

    def score(
        self,
        user_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        item_encoding: Dict[str, torch.Tensor],
        graph_states: torch.Tensor,
    ):
        r_u = self.user_embedding(user_ids.long())
        z_i = item_encoding["fused"]
        h_i = graph_states[item_node_ids.long()]
        h_e = item_encoding["environment_h"]
        dot_content = (r_u * z_i).sum(dim=-1)
        dot_graph = (r_u * h_i).sum(dim=-1)
        ctx = self.context_score(torch.cat([r_u, z_i, h_i, h_e], dim=-1)).squeeze(-1)
        return dot_content + dot_graph + ctx

    def forward(
        self,
        user_ids,
        item_node_ids,
        modality_features,
        environment_ids,
        exposure_descriptor,
        edge_index,
        relation_ids,
        env_beta=0.5,
    ):
        item_encoding = self.encode_items(modality_features, environment_ids, exposure_descriptor)
        graph_h = self.graph_states(edge_index, relation_ids, env_beta=env_beta)
        score = self.score(user_ids, item_node_ids, item_encoding, graph_h)
        item_encoding["score"] = score
        item_encoding["graph_states"] = graph_h
        return item_encoding
