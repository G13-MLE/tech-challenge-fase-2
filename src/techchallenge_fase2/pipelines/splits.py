"""Estrategias de divisao treino/teste para dados de interacao."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from techchallenge_fase2.models.base import Interaction


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Resultado de um split cronologico global.

    Args:
        train_interactions: Interacoes reservadas para treino.
        test_interactions: Interacoes reservadas para teste.
        ground_truth: Mapeamento user_id -> conjunto de itens relevantes.
        train_cutoff: Timestamp do corte entre treino e teste.
    """

    train_interactions: list[Interaction]
    test_interactions: list[Interaction]
    ground_truth: dict[str, set[str]]
    train_cutoff: pd.Timestamp


def chronological_holdout_split(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
) -> ChronologicalSplit:
    """Divide interacoes em treino e teste usando corte temporal global.

    Ordena interacoes por timestamp e separa as mais recentes para teste,
    reproduzindo o cenario real de previsao do futuro.

    Args:
        interactions_df: DataFrame com colunas 'user_id', 'item_id' e
            a coluna de timestamp informada.
        test_ratio: Fracao das interacoes mais recentes reservada para teste.
        timestamp_col: Nome da coluna de timestamp para ordenacao cronologica.

    Returns:
        Objeto ChronologicalSplit com treino, teste e ground truth.
    """
    ordered = interactions_df.sort_values(timestamp_col).reset_index(drop=True)
    n_total = len(ordered)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test
    train_df = ordered.iloc[:n_train]
    test_df = ordered.iloc[n_train:]
    train_interactions = _materialize_interactions(train_df)
    test_interactions = _materialize_interactions(test_df)
    ground_truth = _build_ground_truth(test_df)
    cutoff = train_df[timestamp_col].iloc[-1] if n_train > 0 else pd.Timestamp.min
    return ChronologicalSplit(
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        ground_truth=ground_truth,
        train_cutoff=cutoff,
    )


def filter_warm_start(
    split: ChronologicalSplit,
) -> ChronologicalSplit:
    """Remove cold-start users e itens do ground truth.

    Mantem apenas usuarios e itens presentes no treino para que todos os
    modelos possam gerar predicoes (EASE^ e KNN nao suportam cold-start).

    Args:
        split: Resultado de um split cronologico.

    Returns:
        Novo ChronologicalSplit com ground truth filtrado.
    """
    train_users = {uid for uid, _ in split.train_interactions}
    train_items = {iid for _, iid in split.train_interactions}
    filtered_truth: dict[str, set[str]] = {}
    filtered_test: list[Interaction] = []
    for user_id, items in split.ground_truth.items():
        if user_id not in train_users:
            continue
        warm_items = {item for item in items if item in train_items}
        if warm_items:
            filtered_truth[user_id] = warm_items
            for item_id in warm_items:
                filtered_test.append((user_id, item_id))
    return ChronologicalSplit(
        train_interactions=split.train_interactions,
        test_interactions=filtered_test,
        ground_truth=filtered_truth,
        train_cutoff=split.train_cutoff,
    )


def _materialize_interactions(df: pd.DataFrame) -> list[Interaction]:
    """Converte um DataFrame em lista de tuplas (user_id, item_id)."""
    return [(str(row.user_id), str(row.item_id)) for row in df.itertuples(index=False)]


def _build_ground_truth(test_df: pd.DataFrame) -> dict[str, set[str]]:
    """Agrupa itens relevantes por usuario a partir do DataFrame de teste."""
    truth: dict[str, set[str]] = {}
    grouped = test_df.groupby("user_id")["item_id"].apply(set)
    for user_id, items in grouped.items():
        truth[str(user_id)] = {str(item) for item in items}
    return truth
