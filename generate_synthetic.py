from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(x):
    x = x - np.max(x)
    y = np.exp(x)
    return y / y.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_beltrec")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--items", type=int, default=500)
    ap.add_argument("--interactions", type=int, default=4000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    num_users = args.users
    num_items = args.items
    num_suppliers = 30
    num_attributes = 12
    num_env = 8
    num_relations = 8
    exposure_dim = 64
    latent_dim = 32
    modality_dims = {"visual": 128, "text": 128, "review": 64, "supplier": 32, "context": 32}

    # Latent factors used only to create non-random synthetic behavior.
    user_lat = rng.normal(size=(num_users, latent_dim)).astype(np.float32)
    item_lat = rng.normal(size=(num_items, latent_dim)).astype(np.float32)
    env_lat = rng.normal(scale=0.4, size=(num_env, latent_dim)).astype(np.float32)
    supplier_ids = rng.integers(0, num_suppliers, size=num_items)
    supplier_quality = rng.normal(scale=0.5, size=num_suppliers).astype(np.float32)

    taxonomy = np.zeros((num_items, num_attributes), dtype=np.float32)
    for i in range(num_items):
        k = int(rng.integers(1, 4))
        taxonomy[i, rng.choice(num_attributes, size=k, replace=False)] = 1.0

    # Modalities are correlated noisy transforms of the same item semantics.
    for name, dim in modality_dims.items():
        w = rng.normal(scale=1 / np.sqrt(latent_dim), size=(latent_dim, dim)).astype(np.float32)
        noise = rng.normal(scale=0.3, size=(num_items, dim)).astype(np.float32)
        x = item_lat @ w + noise
        if name == "supplier":
            x += supplier_quality[supplier_ids, None]
        np.save(out / f"modality_{name}.npy", x.astype(np.float32))

    exposure = rng.normal(size=(num_items, exposure_dim)).astype(np.float32)
    # Promotion-like signal tied to supplier popularity / campaign confounding.
    exposure[:, 0] = supplier_quality[supplier_ids] + rng.normal(scale=0.2, size=num_items)
    np.save(out / "exposure.npy", exposure)
    np.save(out / "taxonomy.npy", taxonomy)

    interactions = []
    all_t = np.arange(args.interactions)
    for t in all_t:
        u = int(rng.integers(0, num_users))
        e = int(rng.integers(0, num_env))
        candidate = rng.choice(num_items, size=min(120, num_items), replace=False)
        stable = (user_lat[u] @ item_lat[candidate].T) / np.sqrt(latent_dim)
        env_pref = (env_lat[e] @ item_lat[candidate].T) / np.sqrt(latent_dim)
        promotion = 0.9 * exposure[candidate, 0]
        logits = stable + 0.35 * env_pref + promotion
        i = int(rng.choice(candidate, p=softmax(logits)))
        frac = t / max(args.interactions - 1, 1)
        split = "train" if frac < 0.8 else ("val" if frac < 0.9 else "test")
        interactions.append({"user_id": u, "item_id": i, "env_id": e, "timestamp": int(t), "split": split})

    (out / "interactions.json").write_text(json.dumps(interactions, indent=2))

    # Global graph node order: users | items | suppliers | attributes | markets/environments
    user_offset = 0
    item_offset = num_users
    supplier_offset = item_offset + num_items
    attr_offset = supplier_offset + num_suppliers
    env_offset = attr_offset + num_attributes
    num_nodes = env_offset + num_env

    edges = []
    rels = []
    # relation 0/1: user-item interaction and reverse
    for row in interactions[: int(0.8 * len(interactions))]:
        u = user_offset + row["user_id"]
        i = item_offset + row["item_id"]
        edges += [(u, i), (i, u)]
        rels += [0, 1]
    # relation 2/3: item-supplier and reverse
    for i, s in enumerate(supplier_ids):
        inode, snode = item_offset + i, supplier_offset + int(s)
        edges += [(inode, snode), (snode, inode)]
        rels += [2, 3]
    # relation 4/5: item-attribute and reverse
    for i in range(num_items):
        for a in np.where(taxonomy[i] > 0)[0]:
            inode, anode = item_offset + i, attr_offset + int(a)
            edges += [(inode, anode), (anode, inode)]
            rels += [4, 5]
    # relation 6/7: item-market environment and reverse (based on observed train rows)
    seen = set()
    for row in interactions[: int(0.8 * len(interactions))]:
        key = (row["item_id"], row["env_id"])
        if key in seen:
            continue
        seen.add(key)
        inode, enode = item_offset + row["item_id"], env_offset + row["env_id"]
        edges += [(inode, enode), (enode, inode)]
        rels += [6, 7]

    edge_index = np.array(edges, dtype=np.int64).T
    relation_ids = np.array(rels, dtype=np.int64)
    np.savez(out / "graph.npz", edge_index=edge_index, relation_ids=relation_ids)

    metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "num_suppliers": num_suppliers,
        "num_attributes": num_attributes,
        "num_environments": num_env,
        "num_relations": num_relations,
        "num_nodes": num_nodes,
        "item_node_offset": item_offset,
        "exposure_dim": exposure_dim,
        "modality_dims": modality_dims,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Synthetic BeltRec dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
