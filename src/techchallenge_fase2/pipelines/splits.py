"""Estrategias de divisao treino/teste para dados de interacao.

Fornece duas estrategias de split cronologico:
- `chronological_holdout_split`: 2-way (treino/teste) para avaliacao final.
- `chronological_train_val_test_split`: 3-way (treino/validacao/teste)
  com razao 70/15/15 por padrao, para selecao de hiperparametros e
  early stopping, conforme o plano da issue #15.

Helpers adicionais:
- `filter_warm_start` / `filter_warm_start_three_way`: removem cold-start.
- `filter_seen_items_from_ground_truth`: remove do ground truth itens ja
  consumidos no treino (semantica unificada entre DVC e comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from techchallenge_fase2.models.base import Interaction


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Resultado de um split cronologico global (2-way).

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


@dataclass(frozen=True, slots=True)
class ChronologicalThreeWaySplit:
    """Resultado de um split cronologico global (3-way: treino/val/teste).

    Args:
        train_interactions: Interacoes reservadas para treino (70%).
        val_interactions: Interacoes reservadas para validacao (15%).
        test_interactions: Interacoes reservadas para teste (15%).
        val_ground_truth: Mapeamento user_id -> itens relevantes na validacao.
        test_ground_truth: Mapeamento user_id -> itens relevantes no teste.
        train_cutoff: Timestamp do corte entre treino e validacao.
        val_cutoff: Timestamp do corte entre validacao e teste.
        train_df: DataFrame ordenado com as interacoes de treino.
        val_df: DataFrame ordenado com as interacoes de validacao.
        test_df: DataFrame ordenado com as interacoes de teste.
    """

    train_interactions: list[Interaction]
    val_interactions: list[Interaction]
    test_interactions: list[Interaction]
    val_ground_truth: dict[str, set[str]]
    test_ground_truth: dict[str, set[str]]
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp
    train_df: pd.DataFrame | None = None
    val_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None


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
    train_interactions = materialize_interactions(train_df)
    test_interactions = materialize_interactions(test_df)
    ground_truth = build_ground_truth(test_df)
    cutoff = train_df[timestamp_col].iloc[-1] if n_train > 0 else pd.Timestamp.min
    return ChronologicalSplit(
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        ground_truth=ground_truth,
        train_cutoff=cutoff,
    )


def chronological_train_val_test_split(
    interactions_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
) -> ChronologicalThreeWaySplit:
    """Divide interacoes em treino/validacao/teste com corte temporal global.

    Ordena interacoes por timestamp e separa as fracoes mais recentes
    para validacao e teste. Por padrao usa 70/15/15, conforme o plano da
    issue #15, reproduzindo o cenario real de previsao do futuro.

    Args:
        interactions_df: DataFrame com colunas 'user_id', 'item_id' e
            a coluna de timestamp informada.
        val_ratio: Fracao das interacoes reservada para validacao.
        test_ratio: Fracao das interacoes mais recentes reservada para teste.
        timestamp_col: Nome da coluna de timestamp para ordenacao.

    Returns:
        Objeto ChronologicalThreeWaySplit com treino, validacao, teste
        e os respectivos ground truths, alem dos DataFrames ordenados
        (``train_df``, ``val_df``, ``test_df``).
    """
    ordered = interactions_df.sort_values(timestamp_col).reset_index(drop=True)
    n_total = len(ordered)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val - n_test)
    train_df = ordered.iloc[:n_train]
    val_df = ordered.iloc[n_train : n_train + n_val]
    test_df = ordered.iloc[n_train + n_val :]
    train_interactions = materialize_interactions(train_df)
    val_interactions = materialize_interactions(val_df)
    test_interactions = materialize_interactions(test_df)
    val_ground_truth = build_ground_truth(val_df)
    test_ground_truth = build_ground_truth(test_df)
    train_cutoff = train_df[timestamp_col].iloc[-1] if n_train > 0 else pd.Timestamp.min
    val_cutoff = val_df[timestamp_col].iloc[-1] if len(val_df) > 0 else pd.Timestamp.min
    return ChronologicalThreeWaySplit(
        train_interactions=train_interactions,
        val_interactions=val_interactions,
        test_interactions=test_interactions,
        val_ground_truth=val_ground_truth,
        test_ground_truth=test_ground_truth,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def filter_warm_start(split: ChronologicalSplit) -> ChronologicalSplit:
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


def filter_warm_start_three_way(
    split: ChronologicalThreeWaySplit,
) -> ChronologicalThreeWaySplit:
    """Remove cold-start users e itens do ground truth de validacao e teste.

    Mantem apenas usuarios e itens presentes no treino para que todos os
    modelos possam gerar predicoes (EASE^ e KNN nao suportam cold-start).
    Filtra tambem os DataFrames ``train_df``, ``val_df`` e ``test_df``
    (quando presentes) para conter apenas usuarios e itens warm-start.

    Args:
        split: Resultado de um split cronologico 3-way.

    Returns:
        Novo ChronologicalThreeWaySplit com ground truths e DataFrames
        filtrados.
    """
    train_users = {uid for uid, _ in split.train_interactions}
    train_items = {iid for _, iid in split.train_interactions}
    filtered_val_truth, filtered_val = filter_ground_truth(
        split.val_ground_truth, train_users, train_items
    )
    filtered_test_truth, filtered_test = filter_ground_truth(
        split.test_ground_truth, train_users, train_items
    )
    filtered_train_df = filter_frame_warm_start(
        split.train_df, train_users, train_items
    )
    filtered_val_df = filter_frame_warm_start(split.val_df, train_users, train_items)
    filtered_test_df = filter_frame_warm_start(split.test_df, train_users, train_items)
    return ChronologicalThreeWaySplit(
        train_interactions=split.train_interactions,
        val_interactions=filtered_val,
        test_interactions=filtered_test,
        val_ground_truth=filtered_val_truth,
        test_ground_truth=filtered_test_truth,
        train_cutoff=split.train_cutoff,
        val_cutoff=split.val_cutoff,
        train_df=filtered_train_df,
        val_df=filtered_val_df,
        test_df=filtered_test_df,
    )


def filter_frame_warm_start(
    frame: pd.DataFrame | None,
    train_users: set[str],
    train_items: set[str],
) -> pd.DataFrame | None:
    """Filtra um DataFrame mantendo apenas (user, item) warm-start."""
    if frame is None:
        return None
    is_warm = frame.apply(
        lambda row: str(row.user_id) in train_users and str(row.item_id) in train_items,
        axis=1,
    )
    return frame.loc[is_warm].reset_index(drop=True)


def filter_ground_truth(
    ground_truth: dict[str, set[str]],
    train_users: set[str],
    train_items: set[str],
) -> tuple[dict[str, set[str]], list[Interaction]]:
    """Filtra usuarios e itens do ground truth mantendo apenas warm-start."""
    filtered_truth: dict[str, set[str]] = {}
    filtered_interactions: list[Interaction] = []
    for user_id, items in ground_truth.items():
        if user_id not in train_users:
            continue
        warm_items = {item for item in items if item in train_items}
        if warm_items:
            filtered_truth[user_id] = warm_items
            for item_id in warm_items:
                filtered_interactions.append((user_id, item_id))
    return filtered_truth, filtered_interactions


def filter_seen_items_from_ground_truth(
    ground_truth: dict[str, set[str]],
    train_interactions: list[Interaction],
) -> dict[str, set[str]]:
    """Remove do ground truth os itens ja consumidos no treino.

    Avalia a recomendacao de itens NOVOS: um item que o usuario ja viu
    em treino nao conta como relevante no teste, mesmo que apareca no
    periodo de teste. Semantica unificada entre o pipeline DVC do NCF
    (``evaluation.py``) e o pipeline de comparacao (``run_baselines``).

    Args:
        ground_truth: Mapeamento user_id -> itens relevantes (apos
            split cronologico e, opcionalmente, warm-start).
        train_interactions: Interacoes de treino (user_id, item_id).

    Returns:
        Novo mapeamento user_id -> itens relevantes filtrado. Usuarios
        sem nenhum item relevante apos o filtro sao omitidos.
    """
    seen_by_user: dict[str, set[str]] = {}
    for user_id, item_id in train_interactions:
        seen_by_user.setdefault(user_id, set()).add(item_id)
    filtered: dict[str, set[str]] = {}
    for user_id, items in ground_truth.items():
        seen = seen_by_user.get(user_id, set())
        remaining = items - seen
        if remaining:
            filtered[user_id] = remaining
    return filtered


def materialize_interactions(df: pd.DataFrame) -> list[Interaction]:
    """Converte um DataFrame em lista de tuplas (user_id, item_id)."""
    return [(str(row.user_id), str(row.item_id)) for row in df.itertuples(index=False)]


def build_ground_truth(test_df: pd.DataFrame) -> dict[str, set[str]]:
    """Agrupa itens relevantes por usuario a partir do DataFrame de teste."""
    truth: dict[str, set[str]] = {}
    grouped = test_df.groupby("user_id")["item_id"].apply(set)
    for user_id, items in grouped.items():
        truth[str(user_id)] = {str(item) for item in items}
    return truth
