"""Unit tests for recommendation metrics."""

from techchallenge_fase2.pipeline.metrics import (
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_top_k_metrics_for_partial_hits() -> None:
    """Metrics reward hits at the expected ranks."""
    recommended = [10, 20, 30, 40]
    relevant = {20, 40, 50}

    assert precision_at_k(recommended, relevant, top_k=4) == 0.5
    assert recall_at_k(recommended, relevant, top_k=4) == 2 / 3
    assert hit_rate_at_k(recommended, relevant, top_k=4) == 1.0
    assert round(map_at_k(recommended, relevant, top_k=4), 4) == 0.3333
    assert round(ndcg_at_k(recommended, relevant, top_k=4), 4) == 0.4982


def test_metrics_return_zero_without_relevant_items() -> None:
    """Metrics are safe when no relevant item exists."""
    recommended = [1, 2, 3]
    relevant: set[int] = set()

    assert recall_at_k(recommended, relevant, top_k=3) == 0.0
    assert map_at_k(recommended, relevant, top_k=3) == 0.0
    assert ndcg_at_k(recommended, relevant, top_k=3) == 0.0
