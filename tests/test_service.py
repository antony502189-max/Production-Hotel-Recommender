from hotel_recommender.data import SyntheticDataConfig, generate_synthetic_data, temporal_split
from hotel_recommender.pipeline import train_bundle
from hotel_recommender.service import RecommendationService


def _service() -> RecommendationService:
    items, interactions = generate_synthetic_data(
        SyntheticDataConfig(users=120, hotels=70, interactions=4_000, days=60, seed=11)
    )
    train, _ = temporal_split(interactions, 0.85)
    return RecommendationService(train_bundle(items, train, seed=11), candidate_pool_size=50)


def test_recommendations_are_ranked_and_unique():
    service = _service()
    result = service.recommend(user_id=1, k=8, exclude_seen=True)
    assert len(result) == 8
    assert result["hotel_id"].is_unique
    assert result["score"].is_monotonic_decreasing
    assert result["score"].between(0.0, 1.0).all()


def test_cold_start_user_receives_recommendations():
    service = _service()
    result = service.recommend(user_id=999_999, k=5)
    assert len(result) == 5
