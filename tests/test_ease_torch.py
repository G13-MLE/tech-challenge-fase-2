"""Testes unitarios para o recomendador EASE^ (PyTorch)."""

from __future__ import annotations

import numpy as np
import torch

from techchallenge_fase2.models.base import Interaction
from techchallenge_fase2.models.ease_torch import EASEConfig, EASETorchRecommender

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


def make_trained_model(
    interactions: list[Interaction] | None = None,
    config: EASEConfig | None = None,
) -> EASETorchRecommender:
    """Cria e treina um modelo EASE^ para uso nos testes."""
    model = EASETorchRecommender(config or EASEConfig(lambda_reg=250.0, max_items=0))
    model.fit(interactions or SAMPLE_INTERACTIONS)
    return model


def test_fit_builds_b_matrix_with_zero_diagonal() -> None:
    """O treino deve produzir B com diagonal exatamente zero."""
    model = make_trained_model()
    assert model._b_matrix is not None
    diag = model._b_matrix.diag()
    assert torch.allclose(diag, torch.zeros_like(diag))


def test_fit_creates_item_and_user_mappings() -> None:
    """O treino deve popular mapeamentos str<->int."""
    model = make_trained_model()
    assert len(model._user_to_idx) == 4
    assert len(model._item_to_idx) == 5
    assert model._idx_to_item[0] in {"i1", "i2", "i3", "i4", "i5"}


def test_recommend_returns_items_for_warm_start_user() -> None:
    """Recomendacoes para usuário conhecido devem retornar itens validos."""
    model = make_trained_model()
    recs = model.recommend("u1", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_recommend_excludes_seen_items() -> None:
    """Itens já consumidos pelo usuário não devem ser recomendados."""
    model = make_trained_model()
    seen = {"i1", "i2", "i3"}
    # Catalogo tem 5 itens; pedir limit=2 garante espaco para itens não vistos
    recs = model.recommend("u1", limit=2)
    for item in recs:
        assert item not in seen


def test_recommend_cold_start_returns_popular_items() -> None:
    """Usuario desconhecido deve receber itens mais populares."""
    model = make_trained_model()
    recs = model.recommend("unknown_user", limit=3)
    assert len(recs) == 3
    assert all(item.startswith("i") for item in recs)


def test_recommend_before_fit_raises_runtime_error() -> None:
    """Recomendar antes do treino deve lancar RuntimeError."""
    model = EASETorchRecommender(EASEConfig())
    try:
        model.recommend("u1", limit=3)
    except RuntimeError:
        return
    raise AssertionError("RuntimeError esperada")


def test_ease_config_validates_lambda_reg_positive() -> None:
    """lambda_reg deve ser positivo."""
    try:
        EASEConfig(lambda_reg=0.0)
    except ValueError:
        return
    raise AssertionError("ValueError esperado para lambda_reg=0")


def test_ease_config_validates_max_items_non_negative() -> None:
    """max_items deve ser não negativo."""
    try:
        EASEConfig(max_items=-1)
    except ValueError:
        return
    raise AssertionError("ValueError esperado para max_items=-1")


def test_recommend_respects_limit() -> None:
    """O limite deve ser respeitado no número de recomendações."""
    model = make_trained_model()
    recs = model.recommend("u1", limit=2)
    assert len(recs) == 2


def test_popularity_blending_does_not_break_predictions() -> None:
    """Blending com popularidade não deve quebrar o pipeline de predição."""
    config = EASEConfig(lambda_reg=250.0, max_items=0, popularity_blending=0.5)
    model = make_trained_model(config=config)
    recs = model.recommend("u1", limit=3)
    assert len(recs) == 3


def test_b_matrix_is_square_with_item_count() -> None:
    """B deve ser quadrada com dimensao igual ao número de itens."""
    model = make_trained_model()
    assert model._b_matrix is not None
    assert model._b_matrix.shape == (5, 5)


def test_max_items_filter_limits_catalog_size() -> None:
    """Com max_items menor que o catalogo, B deve ter dimensao reduzida."""
    config = EASEConfig(lambda_reg=250.0, max_items=3)
    model = make_trained_model(config=config)
    assert model._b_matrix is not None
    assert model._b_matrix.shape == (3, 3)


def test_top_item_indices_subset_of_catalog() -> None:
    """Os índices filtrados devem ser subset dos índices originais."""
    model = make_trained_model(config=EASEConfig(max_items=3))
    assert len(model._top_item_indices) == 3
    assert all(0 <= idx < 5 for idx in model._top_item_indices)


def test_ease_config_validates_device() -> None:
    """device deve ser 'auto' ou 'cpu'."""
    try:
        EASEConfig(device="cuda")
    except ValueError:
        return
    raise AssertionError("ValueError esperado para device='cuda'")


def test_ease_config_device_cpu_resolves_correctly() -> None:
    """Config com device='cpu' deve resolver para torch.device('cpu')."""
    model = EASETorchRecommender(EASEConfig(device="cpu"))
    assert model._device == torch.device("cpu")


def test_ease_config_device_auto_resolves_to_valid_device() -> None:
    """Config com device='auto' deve resolver para um dispositivo valido."""
    model = EASETorchRecommender(EASEConfig(device="auto"))
    assert model._device in (
        torch.device("cpu"),
        torch.device("mps"),
        torch.device("cuda"),
    )


def test_recommend_batch_returns_recommendations_for_all_users() -> None:
    """recommend_batch deve retornar recomendações para todos os usuários."""
    model = make_trained_model()
    user_ids = ["u1", "u2", "u3", "unknown_user"]
    results = model.recommend_batch(user_ids, limit=2)
    assert set(results.keys()) == {"u1", "u2", "u3", "unknown_user"}
    for uid in user_ids:
        assert len(results[uid]) == 2
        assert all(item.startswith("i") for item in results[uid])


def test_recommend_batch_excludes_seen_items() -> None:
    """recommend_batch não deve recomendar itens já consumidos."""
    model = make_trained_model()
    results = model.recommend_batch(["u1"], limit=2)
    seen = {"i1", "i2", "i3"}
    for item in results["u1"]:
        assert item not in seen


def test_recommend_batch_respects_batch_size() -> None:
    """recommend_batch deve funcionar com batch_size menor que total."""
    config = EASEConfig(lambda_reg=250.0, max_items=0, batch_size=2)
    model = make_trained_model(config=config)
    user_ids = ["u1", "u2", "u3", "u4"]
    results = model.recommend_batch(user_ids, limit=3)
    assert all(len(results[uid]) == 3 for uid in user_ids)


def test_recommend_batch_before_fit_raises_runtime_error() -> None:
    """recommend_batch antes do treino deve lancar RuntimeError."""
    model = EASETorchRecommender(EASEConfig())
    try:
        model.recommend_batch(["u1"], limit=2)
    except RuntimeError:
        return
    raise AssertionError("RuntimeError esperada")


def test_recommend_batch_matches_single_recommend() -> None:
    """recommend_batch deve produzir os mesmos itens que recommend individual."""
    model = make_trained_model()
    single = model.recommend("u1", limit=2)
    batch_results = model.recommend_batch(["u1"], limit=2)
    assert single == batch_results["u1"]


# Interacoes com popularidades desbalanceadas para forcar top-3 com
# índices não naturalmente ordenados (i5 > i0 ~= i3 > demais).
# Com max_items=3, os índices filtrados por popularidade decrescente
# seriam [5, 0, 3] (não ordenado ascendentemente), expondo o bug do
# searchsorted em catalogo não ordenado.
UNBALANCED_INTERACTIONS: list[Interaction] = [
    ("u1", "i5"),
    ("u2", "i5"),
    ("u3", "i5"),
    ("u4", "i5"),
    ("u1", "i0"),
    ("u2", "i0"),
    ("u3", "i0"),
    ("u1", "i3"),
    ("u2", "i3"),
    ("u3", "i3"),
    ("u4", "i1"),
    ("u4", "i2"),
    ("u4", "i4"),
]


def test_top_item_indices_sorted_ascending_when_filtered() -> None:
    """Indices filtrados devem estar ordenados ascendentemente.

    compute_user_scores, exclude_seen_items e build_batch_matrix usam
    np.searchsorted, que requer um array ordenado ascendentemente.
    """
    model = EASETorchRecommender(EASEConfig(lambda_reg=250.0, max_items=3))
    model.fit(UNBALANCED_INTERACTIONS)
    top = model._top_item_indices
    assert len(top) == 3
    assert np.all(np.diff(top) >= 0), f"top_item_indices deve ser crescente: {top}"


def test_warm_user_scores_nonzero_with_filtered_catalog() -> None:
    """Usuario com interações no catalogo filtrado deve ter scores não nulos.

    Com índices não ordenados, searchsorted falha em localizar itens
    consumidos, resultando em user_row todo zero (cold-start para warm).
    """
    model = EASETorchRecommender(EASEConfig(lambda_reg=250.0, max_items=3))
    model.fit(UNBALANCED_INTERACTIONS)
    user_idx = model._user_to_idx["u1"]
    scores = model.compute_user_scores(user_idx)
    assert scores.abs().sum() > 0, "Scores do usuário warm não devem ser todos zero"


def test_exclude_seen_items_works_with_filtered_catalog() -> None:
    """exclude_seen_items deve atribuir -inf a todos os itens vistos.

    u1 consumiu i0, i3 e i5, todos presentes no top-3 filtrado; portanto
    todas as posições do vetor de scores devem ser -inf apos a exclusão.
    """
    model = EASETorchRecommender(EASEConfig(lambda_reg=250.0, max_items=3))
    model.fit(UNBALANCED_INTERACTIONS)
    user_idx = model._user_to_idx["u1"]
    scores = torch.ones(3, dtype=torch.float64)
    model.exclude_seen_items(user_idx, scores)
    assert torch.all(scores == float("-inf")), f"seen items não excluidos: {scores}"


def test_recommend_excludes_seen_items_with_filtered_catalog() -> None:
    """recommend não deve recomendar itens já consumidos com catalogo filtrado.

    Cenario com 6 itens onde o top-3 filtrado (por popularidade) e
    {i0, i3, i5}. O usuário u1 consumiu apenas i0 (no catalogo filtrado),
    deixando i3 e i5 como não consumidos; a recomendação deve vir desse
    conjunto, nunca de i0.
    """
    interactions: list[Interaction] = [
        ("u1", "i0"),
        ("u2", "i0"),
        ("u2", "i5"),
        ("u2", "i3"),
        ("u3", "i0"),
        ("u3", "i5"),
        ("u3", "i3"),
        ("u4", "i5"),
        ("u5", "i5"),
        ("u5", "i3"),
        ("u4", "i1"),
        ("u4", "i2"),
        ("u5", "i4"),
    ]
    model = EASETorchRecommender(EASEConfig(lambda_reg=250.0, max_items=3))
    model.fit(interactions)
    recs = model.recommend("u1", limit=2)
    seen = {"i0"}
    for item in recs:
        assert item not in seen, f"item já consumido recomendado: {item}"
