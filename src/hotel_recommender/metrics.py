from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    return len(set(top_k).intersection(relevant)) / len(relevant)


def hit_rate_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return float(bool(set(list(recommended)[:k]).intersection(relevant)))


def reciprocal_rank_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    for rank, item_id in enumerate(list(recommended)[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(top_k, start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
