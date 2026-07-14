"""Tests for the MLflow pyfunc wrapper around RecommenderModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from techchallenge_fase2.inference.mlflow_wrapper import (
    RecommenderPythonModel,
    deserialize_recommender,
    normalize_input,
    recommend_one,
    serialize_recommender,
)
from techchallenge_fase2.models.base import RecommenderModel
from techchallenge_fase2.models.sklearn_baselines import ItemKNNRecommender


class _StubRecommender(RecommenderModel):
    """Recommender returning a deterministic list per user."""

    def __init__(self) -> None:
        self._limit_default = 10

    def fit(self, interactions: Any) -> None:
        """No-op fit."""

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return predictable recommendations for testing."""
        k = limit if limit is not None else self._limit_default
        return [f"{user_id}_{i}" for i in range(k)]


def test_serialize_and_deserialize_roundtrip(tmp_path: Path) -> None:
    """Pickle round-trip preserves RecommenderModel identity."""
    model = _StubRecommender()
    path = tmp_path / "model.pkl"
    serialize_recommender(model, path)
    restored = deserialize_recommender(path)
    assert isinstance(restored, _StubRecommender)
    assert restored.recommend("u1", limit=2) == ["u1_0", "u1_1"]


def test_deserialize_rejects_non_recommender(tmp_path: Path) -> None:
    """Type guard rejects pickles that are not RecommenderModel."""
    import pickle

    path = tmp_path / "other.pkl"
    with path.open("wb") as handle:
        pickle.dump({"not": "a model"}, handle)
    with pytest.raises(TypeError):
        deserialize_recommender(path)


def test_recommend_one_uses_limit_from_row() -> None:
    """recommend_one pulls ``limit`` from the row when present."""
    row = next(pd.DataFrame({"user_id": ["u9"], "limit": [3]}).itertuples(index=False))
    recs = recommend_one(_StubRecommender(), row)
    assert recs == ["u9_0", "u9_1", "u9_2"]


def test_recommend_one_uses_default_when_limit_missing() -> None:
    """recommend_one returns the model default limit when row lacks it."""
    row = next(pd.DataFrame({"user_id": ["u9"]}).itertuples(index=False))
    recs = recommend_one(_StubRecommender(), row)
    assert len(recs) == 10


def test_normalize_input_dataframe_keeps_user_id_column() -> None:
    """normalize_input keeps dataframes that already have user_id."""
    frame = pd.DataFrame({"user_id": ["a", "b"]})
    out = normalize_input(frame)
    assert list(out.columns) == ["user_id"]


def test_normalize_input_series_becomes_dataframe() -> None:
    """normalize_input promotes a Series into a one-row DataFrame."""
    series = pd.Series({"user_id": "a"})
    out = normalize_input(series)
    assert isinstance(out, pd.DataFrame)
    assert "user_id" in out.columns


def test_predict_returns_recommendations_column() -> None:
    """predict produces a DataFrame with user_id and recommendations."""
    model = RecommenderPythonModel()
    model._recommender = _StubRecommender()
    frame = pd.DataFrame({"user_id": ["a", "b"], "limit": [2, 3]})
    out = model.predict(context=None, model_input=frame)
    assert list(out.columns) == ["user_id", "recommendations"]
    assert out.iloc[0]["recommendations"] == ["a_0", "a_1"]
    assert out.iloc[1]["recommendations"] == ["b_0", "b_1", "b_2"]


def test_predict_with_real_item_knn() -> None:
    """End-to-end predict with a real ItemKNN integrated into the wrapper."""
    interactions: list[tuple[str, str]] = [
        ("u1", "i1"),
        ("u1", "i2"),
        ("u2", "i1"),
        ("u2", "i3"),
        ("u3", "i2"),
        ("u3", "i3"),
        ("u4", "i1"),
        ("u4", "i3"),
    ]
    knn = ItemKNNRecommender(default_limit=2)
    knn.fit(interactions)
    model = RecommenderPythonModel()
    model._recommender = knn
    frame = pd.DataFrame({"user_id": ["u1"], "limit": [2]})
    out = model.predict(context=None, model_input=frame)
    assert len(out) == 1
    recs = out.iloc[0]["recommendations"]
    assert isinstance(recs, list)
    assert len(recs) <= 2


def test_predict_with_real_item_knn_excludes_seen_items() -> None:
    """Predict does not return items already seen by the warm user."""
    interactions: list[tuple[str, str]] = [
        ("u1", "i1"),
        ("u1", "i2"),
        ("u2", "i1"),
        ("u2", "i3"),
        ("u3", "i2"),
        ("u3", "i3"),
        ("u4", "i1"),
        ("u4", "i3"),
    ]
    knn = ItemKNNRecommender(default_limit=1)
    knn.fit(interactions)
    recs = knn.recommend("u1", limit=1)
    assert recs == ["i3"]
