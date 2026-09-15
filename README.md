# BeltRec Python Reproduction

## 1. What is reproduced

The code follows the manuscript's four main components:

1. **Domain Structured Multimodal Encoding**
   - one projection network per modality;
   - 256-dimensional shared representations;
   - an attribute/taxonomy prediction head;
   - attribute-aware visual-text contrastive alignment.
2. **Industrial Belt Graph Memory**
   - heterogeneous nodes for users/items/suppliers/attributes/markets;
   - two relation-aware propagation layers;
   - degree-tempered message passing.
3. **Causal Adaptive Fusion**
   - environment-conditioned modality gates;
   - environment embedding + exposure/supplier descriptor;
   - normalized adaptive modality weights.
4. **Joint Learning Objective**
   - BPR pairwise ranking loss;
   - multimodal alignment loss;
   - environment-balanced contrastive loss;
   - taxonomy loss;
   - L2 regularization.

The default hyperparameters mirror the paper where specified: hidden size 256, two graph layers, graph dropout 0.2, gate widths 256/128/64, AdamW with learning rate 1e-3, temperature 0.2, and loss weights 0.15/0.25/0.10/1e-5.

## 2. Important reproducibility boundary

The manuscript does **not** provide the WFT-CB raw dataset. It reports 158,432 users, 12,684 SKUs, 486 suppliers/stores and 2.71M interactions, but the data itself is said to be available on reasonable request. Therefore this repository does not invent those records or claim to reproduce the paper's reported HR/NDCG numbers without the original data.

The manuscript also says that the algorithm estimates an **environment propensity**, but it does not specify the estimator. To make the training pipeline executable, this reproduction uses a small MLP environment classifier and derives clipped inverse-propensity weights from it. This choice is explicitly an implementation assumption, not a hidden claim about the authors' original code.

Similarly, the manuscript names a frozen ViT-B/16-style image encoder and a pretrained sentence encoder but does not specify exact checkpoints. This implementation therefore expects **precomputed modality features** and reproduces the trainable projection layers. You can replace these arrays with CLIP/ViT/sentence-transformer features if you have the desired checkpoints.

## 3. Project structure

```text
BeltRec_reproduction/
├── README.md
├── requirements.txt
├── generate_synthetic.py
├── prepare_real_data.py
├── train.py
├── evaluate.py
├── run_demo.py
├── configs/
│   └── default.json
└── src/
    ├── __init__.py
    ├── data.py
    ├── losses.py
    ├── metrics.py
    └── models/
        ├── __init__.py
        ├── encoders.py
        ├── graph_memory.py
        ├── causal_fusion.py
        └── beltrec.py
```

## 4. Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

Or run the steps separately:

```bash
python generate_synthetic.py --out data/synthetic_beltrec
python train.py --data data/synthetic_beltrec --out checkpoints/beltrec.pt --epochs 10
python evaluate.py --data data/synthetic_beltrec --checkpoint checkpoints/beltrec.pt
```

The demo data is intentionally synthetic. Its purpose is to verify that the complete pipeline executes end-to-end and that all modules receive gradients.

## 5. Real-data directory format

A processed dataset directory should contain:

```text
data/my_belt/
├── metadata.json
├── interactions.json
├── graph.npz
├── exposure.npy
├── taxonomy.npy
├── modality_visual.npy
├── modality_text.npy
├── modality_review.npy
├── modality_supplier.npy
└── modality_context.npy
```

### `interactions.json`

A JSON list of records:

```json
{
  "user_id": 12,
  "item_id": 87,
  "env_id": 3,
  "timestamp": 1720000000,
  "split": "train"
}
```

Use the paper's timestamp protocol: first 80% train, next 10% validation, last 10% test.

### `metadata.json`

See `generate_synthetic.py`. It records the number of users/items/nodes/environments/relations, the item-node offset, exposure dimension, attribute count and input feature dimensions.

### `graph.npz`

Contains:

- `edge_index`: shape `[2, E]`, global source/destination node ids;
- `relation_ids`: shape `[E]`, relation type for each edge.

A recommended global node order is:

```text
users | items | suppliers | attributes | markets/environments
```

## 6. Mapping from paper equations to code

- Modality projection + LayerNorm: `src/models/encoders.py`
- Attribute distribution head: `src/models/encoders.py`
- Attribute-aware contrastive alignment: `src/losses.py::weighted_info_nce`
- Causal gate and normalized modality weights: `src/models/causal_fusion.py`
- Environment-balanced contrastive objective: `src/losses.py::environment_balanced_contrastive_loss`
- Relation-aware graph update: `src/models/graph_memory.py`
- Degree tempering: `src/models/graph_memory.py`
- Final score: `src/models/beltrec.py::score`
- Pairwise ranking objective: `src/losses.py::bpr_ranking_loss`
- Taxonomy objective: `src/losses.py::taxonomy_loss`
- Full objective: `src/losses.py::total_beltrec_loss`
- HR@10 / NDCG@10 / Recall@20 / MRR@10: `src/metrics.py`

## 7. Differences from the unavailable original implementation

These are deliberate and documented:

- **Pretrained encoders:** represented by precomputed feature arrays because exact checkpoints are not specified.
- **Environment propensity:** implemented with an MLP classifier because the paper does not define the estimator.
- **Relation transforms:** implemented as a memory-efficient shared transform with relation-specific scaling, reflecting the paper's parameter-sharing statement while avoiding a full dense matrix per edge.
- **Evaluation candidates:** `evaluate.py` uses one held-out positive plus sampled negatives by default. Set a larger candidate count or adapt it to full-catalog ranking for your exact protocol.
- **Early stopping:** the minimal training script saves the final model; adding validation-based early stopping is straightforward if exact reproduction of the stated patience-15 protocol is required.

## 8. Expected paper-level settings

From the manuscript:

- AdamW learning rate: `1e-3`
- weight decay: `1e-5`
- batch size: `2048`
- temperature: `0.2`
- max sequence length: `50`
- early stopping patience: `15`
- image projection hidden/output: `512 -> 256`
- structured context embedding: `64 -> 256`
- graph: 2 layers, hidden 256, dropout 0.2
- gate hidden widths: `256, 128, 64`
- objective weights: `lambda_align=0.15`, `lambda_env=0.25`, `lambda_tax=0.10`, `lambda_l2=1e-5`

## 9. Reproducing the reported experimental numbers

To reproduce the exact tables, you still need the original WFT-CB data, the exact pretrained vision/text checkpoint choices, the exact construction of cross-environment balanced samples, and the original baseline configurations/seeds. This repository gives you a clean executable implementation of the architecture and objective rather than fabricating unavailable experimental evidence.
