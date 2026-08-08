from __future__ import annotations

import pandas as pd

from .candidates import CandidateGenerator
from .domain import RecommenderBundle
from .features import candidate_features
from .model import predict_scores


class RecommendationService:
    def __init__(self, bundle: RecommenderBundle, candidate_pool_size: int = 100) -> None:
        self.bundle = bundle
        self.candidate_pool_size = candidate_pool_size
        self.generator = CandidateGenerator(
            bundle.items,
            bundle.history,
            bundle.item_stats,
            bundle.user_profiles,
        )

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = False) -> pd.DataFrame:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        pool_size = max(k, self.candidate_pool_size)
        candidates = self.generator.generate(
            user_id, pool_size=pool_size, exclude_seen=exclude_seen
        )
        if candidates.empty:
            return candidates.assign(score=pd.Series(dtype=float))

        features = candidate_features(
            user_id,
            candidates,
            self.bundle.history,
            self.bundle.items,
            self.bundle.item_stats,
            self.bundle.user_profiles,
        )
        scores = predict_scores(self.bundle.model, features)
        ranked = candidates.copy()
        ranked["score"] = scores
        return (
            ranked.sort_values(
                ["score", "hotel_id"],
                ascending=[False, True],
                kind="stable",
            )
            .head(k)
            .reset_index(drop=True)
        )
