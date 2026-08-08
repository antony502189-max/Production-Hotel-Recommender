import pandas as pd
import pytest

from hotel_recommender.data import SyntheticDataConfig, generate_synthetic_data, temporal_split


def test_synthetic_generator_is_deterministic():
    config = SyntheticDataConfig(users=40, hotels=30, interactions=500, days=20, seed=7)
    items_a, events_a = generate_synthetic_data(config)
    items_b, events_b = generate_synthetic_data(config)
    pd.testing.assert_frame_equal(items_a, items_b)
    pd.testing.assert_frame_equal(events_a, events_b)


def test_temporal_split_is_strictly_ordered():
    _, events = generate_synthetic_data(
        SyntheticDataConfig(users=30, hotels=20, interactions=300, days=10, seed=3)
    )
    train, test = temporal_split(events, 0.8)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(events)


def test_temporal_split_rejects_bad_fraction():
    events = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10)})
    with pytest.raises(ValueError):
        temporal_split(events, 1.0)
