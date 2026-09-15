from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def bpr_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Eq. (10): pairwise ranking objective."""
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def weighted_info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    weights: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    per_example = F.cross_entropy(logits, labels, reduction="none")
    if weights is None:
        return per_example.mean()
    weights = weights.detach().view(-1)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-8)


def environment_balanced_contrastive_loss(
    user_repr: torch.Tensor,
    item_repr: torch.Tensor,
    inverse_propensity: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Practical batch InfoNCE version of Eq. (5), optionally propensity-weighted."""
    user_repr = F.normalize(user_repr, dim=-1)
    item_repr = F.normalize(item_repr, dim=-1)
    logits = user_repr @ item_repr.t() / temperature
    labels = torch.arange(user_repr.size(0), device=user_repr.device)
    loss = F.cross_entropy(logits, labels, reduction="none")
    if inverse_propensity is not None:
        w = inverse_propensity.detach().clamp(max=20.0)
        return (loss * w).sum() / w.sum().clamp_min(1e-8)
    return loss.mean()


def taxonomy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-label form of Eq. (11), suitable for multiple fishing-tackle attributes."""
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def propensity_loss(propensity_logits: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(propensity_logits, env_ids.long())


def l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    return sum((p ** 2).sum() for p in model.parameters())


def total_beltrec_loss(
    model,
    pos_output: Dict[str, torch.Tensor],
    neg_output: Dict[str, torch.Tensor],
    user_ids: torch.Tensor,
    env_ids: torch.Tensor,
    taxonomy_labels: torch.Tensor,
    lambda_align: float = 0.15,
    lambda_env: float = 0.25,
    lambda_tax: float = 0.10,
    lambda_l2: float = 1e-5,
    lambda_propensity: float = 0.05,
    temperature: float = 0.2,
):
    rank = bpr_ranking_loss(pos_output["score"], neg_output["score"])

    modalities = list(pos_output["modality_z"].keys())
    if "visual" in modalities and "text" in modalities:
        attr_a = pos_output["modality_attr_logits"]["visual"]
        attr_b = pos_output["modality_attr_logits"]["text"]
        weight = model.encoder.attribute_consistency_weight(attr_a, attr_b)
        align = weighted_info_nce(
            pos_output["modality_z"]["visual"],
            pos_output["modality_z"]["text"],
            weights=weight,
            temperature=temperature,
        )
    elif len(modalities) >= 2:
        a, b = modalities[:2]
        align = weighted_info_nce(
            pos_output["modality_z"][a],
            pos_output["modality_z"][b],
            temperature=temperature,
        )
    else:
        align = rank.new_zeros(())

    prop_logits = pos_output["propensity_logits"]
    prop_prob = torch.softmax(prop_logits, dim=-1).gather(1, env_ids.long().view(-1, 1)).squeeze(1)
    inv_prop = (1.0 / prop_prob.clamp_min(0.05)).detach()
    user_repr = model.user_embedding(user_ids.long())
    env = environment_balanced_contrastive_loss(
        user_repr,
        pos_output["fused"],
        inverse_propensity=inv_prop,
        temperature=temperature,
    )

    tax = taxonomy_loss(pos_output["taxonomy_logits"], taxonomy_labels)
    prop = propensity_loss(prop_logits, env_ids)
    reg = l2_penalty(model)

    total = (
        rank
        + lambda_align * align
        + lambda_env * env
        + lambda_tax * tax
        + lambda_propensity * prop
        + lambda_l2 * reg
    )
    parts = {
        "total": total.detach(),
        "rank": rank.detach(),
        "align": align.detach(),
        "env": env.detach(),
        "tax": tax.detach(),
        "propensity": prop.detach(),
    }
    return total, parts
