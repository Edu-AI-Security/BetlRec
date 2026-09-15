from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from src.data import get_item_batch, load_bundle
from src.metrics import average_metrics, ranking_metrics
from src.models import BeltRec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_beltrec")
    ap.add_argument("--checkpoint", default="checkpoints/beltrec.pt")
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bundle = load_bundle(args.data)
    meta = bundle.metadata
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = BeltRec(
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        num_nodes=meta["num_nodes"],
        num_environments=meta["num_environments"],
        modality_dims=meta["modality_dims"],
        num_attributes=meta["num_attributes"],
        num_relations=meta["num_relations"],
        hidden_dim=256,
        env_dim=64,
        exposure_dim=meta["exposure_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    positives_by_user = defaultdict(set)
    for row in bundle.interactions:
        positives_by_user[row["user_id"]].add(row["item_id"])

    edge_index = bundle.edge_index.to(device)
    relation_ids = bundle.relation_ids.to(device)
    item_offset = int(meta["item_node_offset"])
    num_items = int(meta["num_items"])

    rows = [r for r in bundle.interactions if r["split"] == "test"]
    metric_rows = []
    with torch.no_grad():
        graph_h = model.graph_states(edge_index, relation_ids, env_beta=0.5)
        for row in rows:
            u, pos, env = int(row["user_id"]), int(row["item_id"]), int(row["env_id"])
            banned = positives_by_user[u]
            negs = []
            while len(negs) < max(args.candidates - 1, 1):
                cand = int(rng.integers(0, num_items))
                if cand != pos and cand not in banned and cand not in negs:
                    negs.append(cand)
            candidates = np.array([pos] + negs, dtype=np.int64)
            ids = torch.from_numpy(candidates).to(device)
            feats = {k: v[ids.cpu()].to(device) for k, v in bundle.modality_features.items()}
            env_ids = torch.full((len(candidates),), env, dtype=torch.long, device=device)
            exp = bundle.exposure[ids.cpu()].to(device)
            enc = model.encode_items(feats, env_ids, exp)
            users = torch.full((len(candidates),), u, dtype=torch.long, device=device)
            scores = model.score(users, ids + item_offset, enc, graph_h)
            ranked = candidates[torch.argsort(scores, descending=True).cpu().numpy()].tolist()
            metric_rows.append(ranking_metrics(ranked, [pos]))

    result = average_metrics(metric_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
