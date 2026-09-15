from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import PairwiseInteractionDataset, get_item_batch, load_bundle
from src.losses import total_beltrec_loss
from src.models import BeltRec


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(features, device):
    return {k: v.to(device) for k, v in features.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_beltrec")
    ap.add_argument("--out", default="checkpoints/beltrec.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    seed_everything(args.seed)
    bundle = load_bundle(args.data)
    meta = bundle.metadata
    device = torch.device(args.device)

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

    train_ds = PairwiseInteractionDataset(bundle, "train")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    edge_index = bundle.edge_index.to(device)
    relation_ids = bundle.relation_ids.to(device)
    item_offset = int(meta["item_node_offset"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {k: 0.0 for k in ["total", "rank", "align", "env", "tax", "propensity"]}
        n_batches = 0
        for batch in loader:
            user = batch["user_id"].to(device)
            pos = batch["pos_item"].to(device)
            neg = batch["neg_item"].to(device)
            env = batch["env_id"].to(device)

            graph_h = model.graph_states(edge_index, relation_ids, env_beta=0.5)

            pos_feat = to_device(get_item_batch(bundle, pos.cpu()), device)
            neg_feat = to_device(get_item_batch(bundle, neg.cpu()), device)
            pos_exp = bundle.exposure[pos.cpu()].to(device)
            neg_exp = bundle.exposure[neg.cpu()].to(device)

            pos_out = model.encode_items(pos_feat, env, pos_exp)
            neg_out = model.encode_items(neg_feat, env, neg_exp)
            pos_out["score"] = model.score(user, pos + item_offset, pos_out, graph_h)
            neg_out["score"] = model.score(user, neg + item_offset, neg_out, graph_h)

            tax = bundle.taxonomy[pos.cpu()].to(device)
            loss, parts = total_beltrec_loss(
                model,
                pos_out,
                neg_out,
                user,
                env,
                tax,
                lambda_align=0.15,
                lambda_env=0.25,
                lambda_tax=0.10,
                lambda_l2=1e-5,
                temperature=0.2,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            for k in running:
                running[k] += float(parts[k])
            n_batches += 1

        msg = " ".join(f"{k}={running[k]/max(n_batches,1):.4f}" for k in running)
        print(f"epoch={epoch:03d} {msg}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": meta, "args": vars(args)}, out)
    print(f"Checkpoint saved to {out.resolve()}")


if __name__ == "__main__":
    main()
