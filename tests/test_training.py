"""Unit tests for training helpers (negative sampling saturation safety)."""

from __future__ import annotations

import numpy as np
import pytest

from techchallenge_fase2.pipelines.training import (
    build_negative_rows,
    sample_negative_item,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_sample_negative_item_never_returns_a_seen_item() -> None:
    """Sampled negatives are always outside the seen set."""
    seen = {0, 2, 4}
    rng = _rng()
    for _ in range(50):
        assert sample_negative_item(seen, num_items=5, rng=rng) not in seen


def test_sample_negative_item_raises_when_catalog_saturated() -> None:
    """Saturation raises instead of mislabeling a positive as negative."""
    seen = {0, 1, 2}
    with pytest.raises(ValueError, match="catalogo"):
        sample_negative_item(seen, num_items=3, rng=_rng())


def test_build_negative_rows_skips_user_on_saturation() -> None:
    """No negative rows are produced when the user saw every item."""
    user_items = {0, 1}
    rows = build_negative_rows(
        user=7,
        user_items=user_items,
        num_items=2,
        negative_samples=4,
        rng=_rng(),
    )
    assert rows == []


def test_build_negative_rows_returns_only_non_seen_items() -> None:
    """All sampled item índices lie outside the user's seen set."""
    user_items = {0, 3}
    rows = build_negative_rows(
        user=7,
        user_items=user_items,
        num_items=5,
        negative_samples=8,
        rng=_rng(),
    )
    assert len(rows) == 8
    assert all(row[1] not in user_items for row in rows)
    assert all(row[0] == 7 and row[2] == 0.0 for row in rows)
