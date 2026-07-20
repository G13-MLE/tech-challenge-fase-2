"""Unit tests for recommendation metrics."""

from techchallenge_fase2.pipelines.splits import filter_seen_items_from_ground_truth
from techchallenge_fase2.training.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_top_k_metrics_for_partial_hits() -> None:
    """Metrics reward hits at the expected ranks."""
    relevant = {"20", "40", "50"}
    recommended = ["10", "20", "30", "40"]

    assert precision_at_k(relevant, recommended, k=4) == 0.5
    assert recall_at_k(relevant, recommended, k=4) == 2 / 3
    assert hit_rate_at_k(relevant, recommended, k=4) == 1.0
    assert round(average_precision_at_k(relevant, recommended, k=4), 4) == 0.3333
    assert round(ndcg_at_k(relevant, recommended, k=4), 4) == 0.4982


def test_metrics_return_zero_without_relevant_items() -> None:
    """Metrics are safe when no relevant item exists."""
    relevant: set[str] = set()
    recommended = ["1", "2", "3"]

    assert recall_at_k(relevant, recommended, k=3) == 0.0
    assert average_precision_at_k(relevant, recommended, k=3) == 0.0
    assert ndcg_at_k(relevant, recommended, k=3) == 0.0


def test_filter_seen_items_from_ground_truth_removes_seen() -> None:
    """Itens ja consumidos no treino sao removidos do ground truth."""
    ground_truth = {
        "u1": {"i1", "i2", "i3"},
        "u2": {"i4"},
        "u3": {"i5"},
    }
    train_interactions = [
        ("u1", "i1"),
        ("u1", "i2"),
        ("u2", "i4"),
    ]

    filtered = filter_seen_items_from_ground_truth(ground_truth, train_interactions)

    assert filtered == {"u1": {"i3"}, "u3": {"i5"}}


def test_filter_seen_items_from_ground_truth_drops_users_without_remaining() -> None:
    """Usuarios sem itens relevantes apos o filtro sao omitidos."""
    ground_truth = {"u1": {"i1"}, "u2": {"i5", "i6"}}
    train_interactions = [("u1", "i1")]

    filtered = filter_seen_items_from_ground_truth(ground_truth, train_interactions)

    assert filtered == {"u2": {"i5", "i6"}}


def test_filter_seen_items_from_ground_truth_handles_empty_train() -> None:
    """Sem treino, nenhum item e removido."""
    ground_truth = {"u1": {"i1"}}
    filtered = filter_seen_items_from_ground_truth(ground_truth, [])
    assert filtered == ground_truth
