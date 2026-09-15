from __future__ import annotations

import math
from typing import Dict, Iterable


def ranking_metrics(ranked_item_ids: Iterable[int], relevant_item_ids: Iterable[int], k10: int = 10, k20: int = 20) -> Dict[str, float]:
    ranked = list(ranked_item_ids)
    relevant = set(relevant_item_ids)
    top10 = ranked[:k10]
    top20 = ranked[:k20]

    hr10 = float(any(i in relevant for i in top10))
    recall20 = sum(i in relevant for i in top20) / max(len(relevant), 1)

    dcg = 0.0
    for rank, item in enumerate(top10, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k10)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    mrr10 = 0.0
    for rank, item in enumerate(top10, start=1):
        if item in relevant:
            mrr10 = 1.0 / rank
            break

    return {"HR@10": hr10, "NDCG@10": ndcg10, "Recall@20": recall20, "MRR@10": mrr10}


def average_metrics(rows):
    if not rows:
        return {"HR@10": 0.0, "NDCG@10": 0.0, "Recall@20": 0.0, "MRR@10": 0.0}
    keys = rows[0].keys()
    return {k: sum(row[k] for row in rows) / len(rows) for k in keys}
