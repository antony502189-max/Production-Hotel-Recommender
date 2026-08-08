from __future__ import annotations

import pandas as pd


class CandidateGenerator:
    """Hybrid retrieval: popularity plus user geography/type affinity."""

    def __init__(
        self,
        items: pd.DataFrame,
        history: pd.DataFrame,
        item_stats: pd.DataFrame,
        user_profiles: pd.DataFrame,
    ) -> None:
        self.items = items
        self.history = history
        self.item_stats = item_stats
        self.user_profiles = user_profiles
        self._item_index = items.set_index("hotel_id")

    def generate(
        self,
        user_id: int,
        pool_size: int = 100,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        pool_size = max(1, min(pool_size, len(self.items)))
        popularity = self.item_stats.nlargest(
            min(pool_size, len(self.item_stats)), "item_popularity"
        )["hotel_id"].tolist()

        city_table = self.user_profiles.attrs.get("city_affinity", pd.DataFrame())
        type_table = self.user_profiles.attrs.get("type_affinity", pd.DataFrame())
        preferred_cities = (
            city_table[city_table["user_id"] == user_id]
            .nlargest(2, "city_affinity")["city"]
            .tolist()
            if not city_table.empty
            else []
        )
        preferred_types = (
            type_table[type_table["user_id"] == user_id]
            .nlargest(2, "type_affinity")["hotel_type"]
            .tolist()
            if not type_table.empty
            else []
        )

        affinity_mask = self.items["city"].isin(preferred_cities) | self.items[
            "hotel_type"
        ].isin(preferred_types)
        affinity_ids = self.items.loc[affinity_mask, "hotel_id"].tolist()

        ordered_ids = list(dict.fromkeys(popularity + affinity_ids))
        if exclude_seen:
            seen = set(self.history.loc[self.history["user_id"] == user_id, "hotel_id"])
            ordered_ids = [hotel_id for hotel_id in ordered_ids if hotel_id not in seen]

        if len(ordered_ids) < pool_size:
            fallback = [
                hotel_id
                for hotel_id in self.items["hotel_id"].tolist()
                if hotel_id not in ordered_ids
            ]
            if exclude_seen:
                seen = set(self.history.loc[self.history["user_id"] == user_id, "hotel_id"])
                fallback = [hotel_id for hotel_id in fallback if hotel_id not in seen]
            ordered_ids.extend(fallback)

        chosen = ordered_ids[:pool_size]
        return self._item_index.loc[chosen].reset_index() if chosen else self.items.iloc[0:0].copy()
