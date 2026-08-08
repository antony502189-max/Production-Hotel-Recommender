from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import CandidateGenerator
from .data import SyntheticDataConfig, generate_synthetic_data, temporal_split
from .domain import FEATURE_NAMES, ModelMetadata, RecommenderBundle
from .features import build_item_stats, build_user_profiles, candidate_features
from .model import fit_ranker


def _fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()
    return digest[:16]


def build_training_frame(
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    history_fraction: float = 0.75,
    candidates_per_user: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    history, label_window = temporal_split(interactions, history_fraction)
    positives = label_window[label_window["event"] == "booking"]
    if positives.empty:
        raise ValueError("label window contains no bookings; increase data size")

    item_stats = build_item_stats(history, items)
    profiles = build_user_profiles(history, items)
    generator = CandidateGenerator(items, history, item_stats, profiles)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    positive_by_user = positives.groupby("user_id")["hotel_id"].agg(lambda s: set(map(int, s)))
    for user_id, booked in positive_by_user.items():
        candidates = generator.generate(
            int(user_id), pool_size=candidates_per_user, exclude_seen=False
        )
        missing_positive = items[
            items["hotel_id"].isin(booked.difference(set(candidates["hotel_id"])))
        ]
        candidates = pd.concat(
            [candidates, missing_positive], ignore_index=True
        ).drop_duplicates("hotel_id")

        negatives = candidates[~candidates["hotel_id"].isin(booked)]
        max_negatives = max(10, 8 * len(booked))
        if len(negatives) > max_negatives:
            negatives = negatives.iloc[
                rng.choice(len(negatives), size=max_negatives, replace=False)
            ]
        positives_frame = candidates[candidates["hotel_id"].isin(booked)]
        sampled = pd.concat(
            [positives_frame, negatives], ignore_index=True
        ).drop_duplicates("hotel_id")

        x = candidate_features(int(user_id), sampled, history, items, item_stats, profiles)
        x = x.reset_index(drop=True)
        x["user_id"] = int(user_id)
        x["hotel_id"] = sampled["hotel_id"].to_numpy()
        x["target"] = sampled["hotel_id"].isin(booked).astype(int).to_numpy()
        frames.append(x)

    if not frames:
        raise ValueError("could not build any training examples")
    return pd.concat(frames, ignore_index=True)


def train_bundle(
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    seed: int = 42,
) -> RecommenderBundle:
    frame = build_training_frame(items, interactions, seed=seed)
    model = fit_ranker(frame, random_state=seed)
    item_stats = build_item_stats(interactions, items)
    profiles = build_user_profiles(interactions, items)
    metadata = ModelMetadata(
        model_version="1.0.0",
        trained_at_utc=datetime.now(UTC),
        training_rows=len(frame),
        positive_rate=float(frame["target"].mean()),
        feature_names=FEATURE_NAMES,
        training_fingerprint=_fingerprint(frame),
    )
    return RecommenderBundle(
        model=model,
        items=items.copy(),
        history=interactions.copy(),
        item_stats=item_stats,
        user_profiles=profiles,
        metadata=metadata,
    )


def save_bundle(bundle: RecommenderBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def cli_train() -> None:
    parser = argparse.ArgumentParser(description="Train a production hotel recommender model")
    parser.add_argument("--output", default="artifacts/recommender.joblib")
    parser.add_argument("--users", type=int, default=1_200)
    parser.add_argument("--hotels", type=int, default=320)
    parser.add_argument("--interactions", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SyntheticDataConfig(
        users=args.users,
        hotels=args.hotels,
        interactions=args.interactions,
        seed=args.seed,
    )
    items, interactions = generate_synthetic_data(config)
    train, _ = temporal_split(interactions, 0.82)
    bundle = train_bundle(items, train, seed=args.seed)
    destination = save_bundle(bundle, args.output)
    print(f"model saved to {destination}")
    print(f"training rows: {bundle.metadata.training_rows:,}")
    print(f"positive rate: {bundle.metadata.positive_rate:.3f}")


if __name__ == "__main__":
    cli_train()
