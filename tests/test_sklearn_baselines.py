"""Testes unitarios para os baselines scikit-learn (ItemKNN e LogReg)."""

from __future__ import annotations

from techchallenge_fase2.models.base import Interaction
from techchallenge_fase2.models.sklearn_baselines import (
    ItemKNNRecommender,
    LogisticRegressionRecommender,
)

SAMPLE_INTERACTIONS: list[Interaction] = [
    ("u1", "i1"),
    ("u1", "i2"),
    ("u1", "i3"),
    ("u2", "i1"),
    ("u2", "i4"),
    ("u3", "i2"),
    ("u3", "i4"),
    ("u3", "i5"),
    ("u4", "i3"),
    ("u4", "i5"),
]


def make_item_knn(
    interactions: list[Interaction] | None = None,
) -> ItemKNNRecommender:
    """Cria e treina um ItemKNN para uso nos testes."""
    model = ItemKNNRecommender(default_limit=3)
    model.fit(interactions or SAMPLE_INTERACTIONS)
    return model


def make_logreg(
    interactions: list[Interaction] | None = None,
) -> LogisticRegressionRecommender:
    """Cria e treina um LogisticRegression para uso nos testes."""
    model = LogisticRegressionRecommender(default_limit=3)
    model.fit(interactions or SAMPLE_INTERACTIONS)
    return model


def test_item_knn_fit_builds_knn_index() -> None:
    """O treino deve construir o indice KNN."""
    model = make_item_knn()
    assert model._knn is not None
    assert model._item_vectors is not None


def test_item_knn_recommend_returns_items_for_warm_user() -> None:
    """Recomendacoes para usuario conhecido devem retornar itens validos."""
    model = make_item_knn()
    recs = model.recommend("u1", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_item_knn_recommend_excludes_seen_items() -> None:
    """Itens ja consumidos nao devem ser recomendados."""
    model = make_item_knn()
    seen = {"i1", "i2", "i3"}
    # Catalogo tem 5 itens; pedir limit=2 garante itens nao vistos
    recs = model.recommend("u1", limit=2)
    for item in recs:
        assert item not in seen


def test_item_knn_cold_start_returns_popular_items() -> None:
    """Usuario desconhecido deve receber itens populares."""
    model = make_item_knn()
    recs = model.recommend("unknown", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_item_knn_recommend_before_fit_raises() -> None:
    """Recomendar antes do treino deve lancar RuntimeError."""
    model = ItemKNNRecommender()
    try:
        model.recommend("u1", limit=3)
    except RuntimeError:
        return
    raise AssertionError("RuntimeError esperada")


def test_logreg_fit_builds_model() -> None:
    """O treino deve produzir um modelo LogisticRegression treinado."""
    model = make_logreg()
    assert model._model is not None


def test_logreg_recommend_returns_items_for_warm_user() -> None:
    """Recomendacoes para usuario conhecido devem retornar itens validos."""
    model = make_logreg()
    recs = model.recommend("u1", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_logreg_recommend_excludes_seen_items() -> None:
    """Itens ja consumidos nao devem ser recomendados."""
    model = make_logreg()
    seen = {"i1", "i2", "i3"}
    # Catalogo tem 5 itens; pedir limit=2 garante itens nao vistos
    recs = model.recommend("u1", limit=2)
    for item in recs:
        assert item not in seen


def test_logreg_cold_start_returns_popular_items() -> None:
    """Usuario desconhecido deve receber itens populares."""
    model = make_logreg()
    recs = model.recommend("unknown", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_logreg_recommend_before_fit_raises() -> None:
    """Recomendar antes do treino deve lancar RuntimeError."""
    model = LogisticRegressionRecommender()
    try:
        model.recommend("u1", limit=3)
    except RuntimeError:
        return
    raise AssertionError("RuntimeError esperada")


def test_logreg_recommend_respects_limit() -> None:
    """O limite deve ser respeitado no numero de recomendacoes."""
    model = make_logreg()
    recs = model.recommend("u1", limit=2)
    assert len(recs) == 2
