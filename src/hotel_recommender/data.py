from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataConfig:
    users: int = 1_200
    hotels: int = 320
    interactions: int = 45_000
    days: int = 240
    seed: int = 42


def validate_catalog(items: pd.DataFrame) -> None:
    required = {"hotel_id", "city", "hotel_type", "price", "stars"}
    missing = required.difference(items.columns)
    if missing:
        raise ValueError(f"catalog is missing columns: {sorted(missing)}")
    if items.empty or items["hotel_id"].duplicated().any():
        raise ValueError("hotel_id must be unique and catalog must not be empty")
    if (items["price"] <= 0).any() or not items["stars"].between(1, 5).all():
        raise ValueError("catalog contains invalid price or star values")


def validate_interactions(interactions: pd.DataFrame) -> None:
    required = {"user_id", "hotel_id", "event", "timestamp"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions are missing columns: {sorted(missing)}")
    if interactions.empty:
        raise ValueError("interactions must not be empty")
    allowed = {"view", "click", "booking"}
    invalid_events = set(interactions["event"].unique()).difference(allowed)
    if invalid_events:
        raise ValueError(f"unsupported events: {sorted(invalid_events)}")


def generate_synthetic_data(
    config: SyntheticDataConfig = SyntheticDataConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic behavior with a learnable recommendation signal.

    User-to-hotel affinity is computed as a vectorized latent utility matrix. Events then sample
    mostly from each user's high-affinity pool with a small exploration probability. This keeps the
    demo realistic enough for ranking experiments while making dataset generation fast enough for
    local development and CI.
    """

    if min(config.users, config.hotels, config.interactions, config.days) <= 0:
        raise ValueError("all synthetic data dimensions must be positive")

    rng = np.random.default_rng(config.seed)
    cities = np.array(["Warsaw", "Krakow", "Gdansk", "Prague", "Berlin", "Vienna", "Budapest"])
    hotel_types = np.array(["hotel", "apartment", "hostel", "resort", "boutique"])

    city_codes = rng.integers(0, len(cities), config.hotels)
    type_codes = rng.choice(
        len(hotel_types),
        config.hotels,
        p=[0.42, 0.25, 0.10, 0.08, 0.15],
    )
    prices = np.round(rng.lognormal(mean=4.72, sigma=0.43, size=config.hotels), 2)
    stars = rng.integers(1, 6, config.hotels)

    items = pd.DataFrame(
        {
            "hotel_id": np.arange(1, config.hotels + 1, dtype=int),
            "city": cities[city_codes],
            "hotel_type": hotel_types[type_codes],
            "price": prices,
            "stars": stars,
        }
    )
    validate_catalog(items)

    favorite_city = rng.integers(0, len(cities), config.users)
    favorite_type = rng.integers(0, len(hotel_types), config.users)
    preferred_stars = rng.integers(2, 6, config.users)
    budget = rng.lognormal(mean=4.72, sigma=0.38, size=config.users)

    city_match = favorite_city[:, None] == city_codes[None, :]
    type_match = favorite_type[:, None] == type_codes[None, :]
    price_gap = np.abs(prices[None, :] - budget[:, None]) / np.maximum(budget[:, None], 1.0)
    star_gap = np.abs(stars[None, :] - preferred_stars[:, None])

    latent_utility = (
        1.75 * city_match
        + 1.15 * type_match
        - 1.05 * price_gap
        - 0.22 * star_gap
        + 0.16 * stars[None, :]
    )

    affinity_pool_size = min(24, config.hotels)
    affinity_pool = np.argpartition(
        -latent_utility,
        kth=affinity_pool_size - 1,
        axis=1,
    )[:, :affinity_pool_size]

    user_idx = rng.integers(0, config.users, config.interactions)
    pool_slot = rng.integers(0, affinity_pool_size, config.interactions)
    preferred_hotel_idx = affinity_pool[user_idx, pool_slot]
    exploratory_hotel_idx = rng.integers(0, config.hotels, config.interactions)
    explore = rng.random(config.interactions) < 0.12
    hotel_idx = np.where(explore, exploratory_hotel_idx, preferred_hotel_idx)

    intent = latent_utility[user_idx, hotel_idx] + rng.normal(0.0, 0.50, config.interactions)
    booking_probability = 1.0 / (1.0 + np.exp(-(intent - 2.35)))
    click_probability = np.clip(0.20 + 0.12 * np.maximum(intent, 0.0), 0.20, 0.70)
    draw = rng.random(config.interactions)
    events = np.where(
        draw < 0.20 * booking_probability,
        "booking",
        np.where(draw < click_probability, "click", "view"),
    )

    seconds = rng.integers(0, config.days * 24 * 3600, config.interactions)
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    timestamps = base_time + pd.to_timedelta(seconds, unit="s")

    interactions = pd.DataFrame(
        {
            "user_id": user_idx + 1,
            "hotel_id": hotel_idx + 1,
            "event": events,
            "timestamp": timestamps,
        }
    )
    interactions = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    validate_interactions(interactions)
    return items, interactions


def temporal_split(
    interactions: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    ordered = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
