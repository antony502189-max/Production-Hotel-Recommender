from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hotel_recommender.data import SyntheticDataConfig, generate_synthetic_data, temporal_split
from hotel_recommender.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from hotel_recommender.pipeline import train_bundle
from hotel_recommender.service import RecommendationService


def evaluate(
    users: int, hotels: int, interactions_n: int, seed: int, k: int
) -> dict[str, object]:
    items, interactions = generate_synthetic_data(
        SyntheticDataConfig(users=users, hotels=hotels, interactions=interactions_n, seed=seed)
    )
    train, test = temporal_split(interactions, 0.82)
    bundle = train_bundle(items, train, seed=seed)
    service = RecommendationService(bundle, candidate_pool_size=min(max(100, k * 10), hotels))

    relevant: dict[int, set[int]] = defaultdict(set)
    for row in test[test["event"] == "booking"].itertuples(index=False):
        relevant[int(row.user_id)].add(int(row.hotel_id))

    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    hits: list[float] = []
    for user_id, booked in relevant.items():
        ranked = service.recommend(user_id=user_id, k=k, exclude_seen=False)
        ids = ranked["hotel_id"].astype(int).tolist()
        recalls.append(recall_at_k(ids, booked, k))
        ndcgs.append(ndcg_at_k(ids, booked, k))
        mrrs.append(reciprocal_rank_at_k(ids, booked, k))
        hits.append(hit_rate_at_k(ids, booked, k))

    if not recalls:
        raise RuntimeError("evaluation holdout has no booking targets")

    popular_ids = (
        bundle.item_stats.nlargest(k, "item_popularity")["hotel_id"].astype(int).tolist()
    )
    baseline_recalls = [recall_at_k(popular_ids, booked, k) for booked in relevant.values()]
    baseline_ndcgs = [ndcg_at_k(popular_ids, booked, k) for booked in relevant.values()]
    baseline_mrrs = [reciprocal_rank_at_k(popular_ids, booked, k) for booked in relevant.values()]
    baseline_hits = [hit_rate_at_k(popular_ids, booked, k) for booked in relevant.values()]

    model_recall = float(np.mean(recalls))
    baseline_recall = float(np.mean(baseline_recalls))
    recall_lift = (model_recall / baseline_recall - 1.0) if baseline_recall > 0 else float("inf")

    return {
        "evaluated_users": len(recalls),
        "model": {
            f"recall@{k}": model_recall,
            f"ndcg@{k}": float(np.mean(ndcgs)),
            f"mrr@{k}": float(np.mean(mrrs)),
            f"hit_rate@{k}": float(np.mean(hits)),
        },
        "popularity_baseline": {
            f"recall@{k}": baseline_recall,
            f"ndcg@{k}": float(np.mean(baseline_ndcgs)),
            f"mrr@{k}": float(np.mean(baseline_mrrs)),
            f"hit_rate@{k}": float(np.mean(baseline_hits)),
        },
        f"recall@{k}_relative_lift": float(recall_lift),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluation on deterministic synthetic data"
    )
    parser.add_argument("--users", type=int, default=1_200)
    parser.add_argument("--hotels", type=int, default=320)
    parser.add_argument("--interactions", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="reports/offline_metrics.json")
    args = parser.parse_args()

    result = evaluate(args.users, args.hotels, args.interactions, args.seed, args.k)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
