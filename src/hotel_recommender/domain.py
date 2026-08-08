from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


EVENT_WEIGHTS: Final[dict[str, float]] = {
    "view": 1.0,
    "click": 3.0,
    "booking": 8.0,
}

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "item_popularity",
    "city_affinity",
    "type_affinity",
    "price_fit",
    "star_fit",
    "prior_views",
    "prior_clicks",
    "historical_booking_rate",
    "log_price",
    "stars_scaled",
)


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    trained_at_utc: datetime
    training_rows: int
    positive_rate: float
    feature_names: tuple[str, ...]
    training_fingerprint: str


@dataclass
class RecommenderBundle:
    model: HistGradientBoostingClassifier
    items: pd.DataFrame
    history: pd.DataFrame
    item_stats: pd.DataFrame
    user_profiles: pd.DataFrame
    metadata: ModelMetadata
