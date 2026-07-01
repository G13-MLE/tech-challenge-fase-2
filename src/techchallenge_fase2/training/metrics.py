"""Métricas de avaliação para sistemas de recomendação.

Fornece funções para computar métricas padrão de recomendação,
suportando avaliação por usuário e agregação multi-usuário.

Métricas implementadas:
- Precision@K: Dos K itens recomendados, quantos são relevantes.
- Recall@K: Dos itens relevantes, quantos estão nos K recomendados.
- NDCG@K: Normalized Discounted Cumulative Gain, considera a posição.
- MAP@K: Mean Average Precision, média das precision em diferentes cutoffs.
- Hit Rate@K: Pelo menos 1 item relevante nos K recomendados.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import log2


def precision_at_k(
    relevant: set[str],
    recommended: list[str],
    k: int,
) -> float:
    """Computa Precision@K para um único usuário.

    Dos K itens mais recomendados, qual fração é relevante.

    Args:
        relevant: Conjunto de itens relevantes para o usuário.
        recommended: Lista ordenada de itens recomendados.
        k: Número de itens a considerar no topo da lista.

    Returns:
        Fração dos K primeiros recomendados que são relevantes.
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    relevant: set[str],
    recommended: list[str],
    k: int,
) -> float:
    """Computa Recall@K para um único usuário.

    Dos itens relevantes, quantos estão nos K primeiros recomendados.

    Args:
        relevant: Conjunto de itens relevantes para o usuário.
        recommended: Lista ordenada de itens recomendados.
        k: Número de itens a considerar no topo da lista.

    Returns:
        Fração dos itens relevantes que aparecem nos K primeiros
        recomendados. Retorna 0.0 se não há itens relevantes.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    relevant: set[str],
    recommended: list[str],
    k: int,
) -> float:
    """Computa NDCG@K (Normalized Discounted Cumulative Gain).

    Mede a qualidade da ordenação considerando que itens relevantes
    nas posições iniciais são mais importantes.

    Args:
        relevant: Conjunto de itens relevantes para o usuário.
        recommended: Lista ordenada de itens recomendados.
        k: Número de itens a considerar no topo da lista.

    Returns:
        NDCG@K no intervalo [0.0, 1.0]. Retorna 0.0 se não há
        itens relevantes.
    """
    if k <= 0 or not relevant:
        return 0.0

    # DCG@K
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / log2(i + 2)

    # IDCG@K (ideal ordering: all relevant items first)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(
    relevant: set[str],
    recommended: list[str],
    k: int,
) -> float:
    """Computa Average Precision@K para um único usuário.

    Média das precision calculadas em cada posição onde um
    item relevante foi encontrado.

    Args:
        relevant: Conjunto de itens relevantes para o usuário.
        recommended: Lista ordenada de itens recomendados.
        k: Número de itens a considerar no topo da lista.

    Returns:
        Average Precision@K. Retorna 0.0 se não há itens relevantes.
    """
    if k <= 0 or not relevant:
        return 0.0

    precisions: list[float] = []
    num_relevant_found = 0

    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            num_relevant_found += 1
            precisions.append(num_relevant_found / (i + 1))

    if not precisions:
        return 0.0

    return sum(precisions) / min(len(relevant), k)


def hit_rate_at_k(
    relevant: set[str],
    recommended: list[str],
    k: int,
) -> float:
    """Computa Hit Rate@K para um único usuário.

    Retorna 1.0 se pelo menos 1 item relevante está nos K
    primeiros recomendados, 0.0 caso contrário.

    Args:
        relevant: Conjunto de itens relevantes para o usuário.
        recommended: Lista ordenada de itens recomendados.
        k: Número de itens a considerar no topo da lista.

    Returns:
        1.0 se há pelo menos 1 hit nos K primeiros, 0.0 caso
        contrário.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def mean_precision_at_k(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k: int,
) -> float:
    """Computa Mean Precision@K agregando por usuário.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k: Número de itens a considerar no topo.

    Returns:
        Média aritmética de Precision@K por usuário. Retorna 0.0
        se o mapeamento está vazio.
    """
    if not all_relevant:
        return 0.0
    scores = [
        precision_at_k(all_relevant[uid], all_recommended[uid], k)
        for uid in all_relevant
        if uid in all_recommended
    ]
    return sum(scores) / len(scores) if scores else 0.0


def mean_recall_at_k(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k: int,
) -> float:
    """Computa Mean Recall@K agregando por usuário.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k: Número de itens a considerar no topo.

    Returns:
        Média aritmética de Recall@K por usuário. Retorna 0.0
        se o mapeamento está vazio.
    """
    if not all_relevant:
        return 0.0
    scores = [
        recall_at_k(all_relevant[uid], all_recommended[uid], k)
        for uid in all_relevant
        if uid in all_recommended
    ]
    return sum(scores) / len(scores) if scores else 0.0


def mean_ndcg_at_k(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k: int,
) -> float:
    """Computa Mean NDCG@K agregando por usuário.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k: Número de itens a considerar no topo.

    Returns:
        Média aritmética de NDCG@K por usuário. Retorna 0.0
        se o mapeamento está vazio.
    """
    if not all_relevant:
        return 0.0
    scores = [
        ndcg_at_k(all_relevant[uid], all_recommended[uid], k)
        for uid in all_relevant
        if uid in all_recommended
    ]
    return sum(scores) / len(scores) if scores else 0.0


def mean_average_precision_at_k(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k: int,
) -> float:
    """Computa MAP@K (Mean Average Precision@K) agregando por usuário.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k: Número de itens a considerar no topo.

    Returns:
        Média aritmética de AP@K por usuário. Retorna 0.0
        se o mapeamento está vazio.
    """
    if not all_relevant:
        return 0.0
    scores = [
        average_precision_at_k(all_relevant[uid], all_recommended[uid], k)
        for uid in all_relevant
        if uid in all_recommended
    ]
    return sum(scores) / len(scores) if scores else 0.0


def mean_hit_rate_at_k(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k: int,
) -> float:
    """Computa Mean Hit Rate@K agregando por usuário.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k: Número de itens a considerar no topo.

    Returns:
        Média aritmética de Hit Rate@K por usuário. Retorna 0.0
        se o mapeamento está vazio.
    """
    if not all_relevant:
        return 0.0
    scores = [
        hit_rate_at_k(all_relevant[uid], all_recommended[uid], k)
        for uid in all_relevant
        if uid in all_recommended
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_recommender_metrics(
    all_relevant: Mapping[str, set[str]],
    all_recommended: Mapping[str, list[str]],
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Computa todas as métricas de recomendação para múltiplos K.

    Função de conveniência que calcula Precision, Recall, NDCG,
    MAP e Hit Rate para cada valor de K especificado.

    Args:
        all_relevant: Mapeamento user_id -> conjunto de itens relevantes.
        all_recommended: Mapeamento user_id -> lista de itens recomendados.
        k_values: Valores de K para os quais computar as métricas.

    Returns:
        Dicionário com chaves no formato "métrica@K" e valores float.
        Exemplo: {"precision@5": 0.4, "recall@5": 0.2, "ndcg@5": 0.35, ...}
    """
    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"precision@{k}"] = mean_precision_at_k(
            all_relevant, all_recommended, k
        )
        metrics[f"recall@{k}"] = mean_recall_at_k(all_relevant, all_recommended, k)
        metrics[f"ndcg@{k}"] = mean_ndcg_at_k(all_relevant, all_recommended, k)
        metrics[f"map@{k}"] = mean_average_precision_at_k(
            all_relevant, all_recommended, k
        )
        metrics[f"hit_rate@{k}"] = mean_hit_rate_at_k(all_relevant, all_recommended, k)

    metrics["num_users"] = float(len(all_relevant))
    return metrics
