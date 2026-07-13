"""Testes unitarios para a divisao cronologica de treino/teste."""

from __future__ import annotations

import pandas as pd

from techchallenge_fase2.pipelines.splits import (
    chronological_holdout_split,
    chronological_train_val_test_split,
    filter_warm_start,
    filter_warm_start_three_way,
)


def make_interactions_df() -> pd.DataFrame:
    """Cria um DataFrame de interacoes com timestamps ordenados."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["i1", "i2", "i1", "i3", "i2", "i4", "i3", "i4"],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


def make_large_interactions_df() -> pd.DataFrame:
    """Cria um DataFrame com mais interacoes para o split 3-way."""
    return pd.DataFrame(
        {
            "user_id": [f"u{i % 5}" for i in range(20)],
            "item_id": [f"i{i % 10}" for i in range(20)],
            "timestamp": list(range(1, 21)),
        }
    )


def test_chronological_split_separates_train_and_test() -> None:
    """Split cronologico deve separar treino e teste por timestamp."""
    df = make_interactions_df()
    split = chronological_holdout_split(df, test_ratio=0.25)
    assert len(split.train_interactions) == 6
    assert len(split.test_interactions) == 2


def test_chronological_split_preserves_time_order() -> None:
    """Treino deve conter interacoes mais antigas que teste."""
    df = make_interactions_df()
    split = chronological_holdout_split(df, test_ratio=0.25)
    assert split.train_cutoff < df["timestamp"].iloc[-1]


def test_chronological_split_builds_ground_truth() -> None:
    """Ground truth deve agregar itens do periodo de teste."""
    df = make_interactions_df()
    split = chronological_holdout_split(df, test_ratio=0.25)
    assert len(split.ground_truth) > 0
    for items in split.ground_truth.values():
        assert isinstance(items, set)


def test_filter_warm_start_removes_cold_users() -> None:
    """Usuarios sem interacoes no treino devem ser removidos do ground truth."""
    df = make_interactions_df()
    split = chronological_holdout_split(df, test_ratio=0.25)
    filtered = filter_warm_start(split)
    train_users = {uid for uid, _ in filtered.train_interactions}
    for user_id in filtered.ground_truth:
        assert user_id in train_users


def test_filter_warm_start_removes_cold_items() -> None:
    """Itens ausentes do treino devem ser removidos do ground truth."""
    df = make_interactions_df()
    split = chronological_holdout_split(df, test_ratio=0.25)
    filtered = filter_warm_start(split)
    train_items = {iid for _, iid in filtered.train_interactions}
    for items in filtered.ground_truth.values():
        for item in items:
            assert item in train_items


def test_chronological_3way_split_produces_three_partitions() -> None:
    """Split 3-way deve produzir treino, validacao e teste nao vazios."""
    df = make_large_interactions_df()
    split = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    assert len(split.train_interactions) > 0
    assert len(split.val_interactions) > 0
    assert len(split.test_interactions) > 0


def test_chronological_3way_split_proportions_match() -> None:
    """As proporcoes do split 3-way devem respeitar val_ratio e test_ratio."""
    df = make_large_interactions_df()
    n_total = len(df)
    split = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    assert len(split.train_interactions) == n_total - 3 - 3
    assert len(split.val_interactions) == 3
    assert len(split.test_interactions) == 3


def test_chronological_3way_split_preserves_time_order() -> None:
    """Treino deve ser mais antigo que validacao, que e mais antiga que teste."""
    df = make_large_interactions_df()
    split = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    assert split.train_cutoff < split.val_cutoff


def test_chronological_3way_split_builds_both_ground_truths() -> None:
    """Split 3-way deve construir ground truth de validacao e teste."""
    df = make_large_interactions_df()
    split = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    assert isinstance(split.val_ground_truth, dict)
    assert isinstance(split.test_ground_truth, dict)


def test_filter_warm_start_3way_removes_cold_users() -> None:
    """filter_warm_start_three_way remove usuarios sem interacoes no treino."""
    df = make_large_interactions_df()
    split = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    filtered = filter_warm_start_three_way(split)
    train_users = {uid for uid, _ in filtered.train_interactions}
    for user_id in filtered.val_ground_truth:
        assert user_id in train_users
    for user_id in filtered.test_ground_truth:
        assert user_id in train_users
