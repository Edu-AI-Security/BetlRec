from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    interactions: List[dict]
    modality_features: Dict[str, torch.Tensor]
    exposure: torch.Tensor
    taxonomy: torch.Tensor
    edge_index: torch.Tensor
    relation_ids: torch.Tensor
    metadata: dict


class PairwiseInteractionDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, split: str = "train"):
        self.bundle = bundle
        self.rows = [x for x in bundle.interactions if x["split"] == split]
        self.num_items = int(bundle.metadata["num_items"])
        positive_by_user = {}
        for row in bundle.interactions:
            positive_by_user.setdefault(int(row["user_id"]), set()).add(int(row["item_id"]))
        self.positive_by_user = positive_by_user
        self.rng = np.random.default_rng(2026)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        u = int(row["user_id"])
        pos = int(row["item_id"])
        neg = int(self.rng.integers(0, self.num_items))
        tries = 0
        while neg in self.positive_by_user[u] and tries < 20:
            neg = int(self.rng.integers(0, self.num_items))
            tries += 1
        return {
            "user_id": u,
            "pos_item": pos,
            "neg_item": neg,
            "env_id": int(row["env_id"]),
        }


def _load_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    return torch.load(path, map_location="cpu")


def load_bundle(root: str | Path) -> DatasetBundle:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    interactions = json.loads((root / "interactions.json").read_text())
    modalities = {}
    for name in metadata["modality_dims"]:
        modalities[name] = _load_tensor(root / f"modality_{name}.npy")
    exposure = _load_tensor(root / "exposure.npy")
    taxonomy = _load_tensor(root / "taxonomy.npy")
    graph = np.load(root / "graph.npz")
    edge_index = torch.from_numpy(graph["edge_index"]).long()
    relation_ids = torch.from_numpy(graph["relation_ids"]).long()
    return DatasetBundle(
        interactions=interactions,
        modality_features=modalities,
        exposure=exposure,
        taxonomy=taxonomy,
        edge_index=edge_index,
        relation_ids=relation_ids,
        metadata=metadata,
    )


def get_item_batch(bundle: DatasetBundle, item_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {name: feats[item_ids.long()] for name, feats in bundle.modality_features.items()}
