import pytest

from hotel_recommender.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank_at_k


def test_ranking_metrics_on_known_example():
    recommended = [7, 3, 9, 2]
    relevant = {3, 2}
    assert recall_at_k(recommended, relevant, 2) == pytest.approx(0.5)
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0
    assert reciprocal_rank_at_k(recommended, relevant, 4) == pytest.approx(0.5)
    assert 0.0 < ndcg_at_k(recommended, relevant, 4) <= 1.0


def test_metrics_return_zero_without_relevant_items():
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert hit_rate_at_k([1, 2], set(), 2) == 0.0
    assert reciprocal_rank_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0
