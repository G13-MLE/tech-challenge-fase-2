"""Unit tests for feature engineering split guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from techchallenge_fase2.pipelines.features import split_frame


def _frame(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "visitorid": np.arange(n_rows, dtype=int),
            "itemid": np.arange(n_rows, dtype=int),
            "timestamp": np.arange(n_rows, dtype=int),
            "event": ["view"] * n_rows,
        }
    )


def test_split_frame_returns_three_non_empty_partitions() -> None:
    train, validation, test = split_frame(_frame(100), 0.7, 0.15)
    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert len(train) + len(validation) + len(test) == 100


def test_split_frame_rejects_frames_too_small() -> None:
    with pytest.raises(ValueError, match="pequeno demais"):
        split_frame(_frame(2), 0.7, 0.15)


def test_split_frame_rejects_empty_partition() -> None:
    """Tiny frames that would yield empty partitions raise a clear error."""
    with pytest.raises(ValueError):
        split_frame(_frame(4), 0.99, 0.005)
