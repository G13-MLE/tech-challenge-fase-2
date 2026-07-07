"""Testes unitarios para a divisao cronologica de treino/teste."""

from __future__ import annotations

import pandas as pd

from techchallenge_fase2.pipelines.splits import (
    chronological_holdout_split,
    filter_warm_start,
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
