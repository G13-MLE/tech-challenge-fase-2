"""Unit tests for feature engineering split guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from techchallenge_fase2.pipelines.features import split_interactions


def _frame(n_rows: int) -> pd.DataFrame:
    """Gera DataFrame sintetico com cols visitorid/itemid/timestamp/event."""
    return pd.DataFrame(
        {
            "visitorid": [f"u{i % 5}" for i in range(n_rows)],
            "itemid": [f"i{i % 5}" for i in range(n_rows)],
            "timestamp": np.arange(n_rows, dtype=int),
            "event": ["view"] * n_rows,
        }
    )


def test_split_interactions_returns_three_non_empty_partitions() -> None:
    """Split cronologico produz treino/validacao/teste nao vazios."""
    train, validation, test = split_interactions(_frame(100), 0.7, 0.15)
    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert len(train) + len(validation) + len(test) == 100


def test_split_interactions_rejects_frames_too_small() -> None:
    """Frames pequenos demais geram erro explicito."""
    with pytest.raises(ValueError):
        split_interactions(_frame(2), 0.7, 0.15)


def test_split_interactions_rejects_empty_partition() -> None:
    """Frames que resultariam em particao vazia geram erro explicito."""
    with pytest.raises(ValueError):
        split_interactions(_frame(4), 0.99, 0.005)


def test_split_interactions_preserves_warm_start_only() -> None:
    """Apos o split, teste contem apenas usuarios e itens presentes em treino."""
    frame = _frame(100)
    train, _, test = split_interactions(frame, 0.7, 0.15)
    train_users = set(train["visitorid"])
    train_items = set(train["itemid"])
    test_users = set(test["visitorid"])
    test_items = set(test["itemid"])
    assert test_users.issubset(train_users)
    assert test_items.issubset(train_items)
