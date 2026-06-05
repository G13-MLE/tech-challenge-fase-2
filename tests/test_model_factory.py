# ruff: noqa: E402
"""Unit tests for the recommendation model factory."""

import os
import sys
import unittest
from collections.abc import Iterable
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, os.fspath(SRC_PATH))

from techchallenge_fase2.models import (
    ModelConfig,
    ModelType,
    PopularityRecommender,
    RecentItemsRecommender,
    RecommenderModel,
    RecommenderModelFactory,
)
from techchallenge_fase2.models.base import Interaction


def create_fake_recommender(config: ModelConfig) -> RecommenderModel:
    """Create a fake recommender from model configuration."""
    return FakeRecommender(default_limit=config.recommendation_limit)


class FakeRecommender(RecommenderModel):
    """Test double used to validate custom factory registration."""

    def __init__(self, default_limit: int) -> None:
        """Initialize the fake recommender."""
        self._default_limit = default_limit
        self._items: list[str] = []

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Store item identifiers in input order."""
        self._items = [item_id for _, item_id in interactions]

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return stored items according to the requested limit."""
        _ = user_id
        recommendation_limit = self._default_limit if limit is None else limit
        return self._items[:recommendation_limit]


class RecommenderModelFactoryTest(unittest.TestCase):
    """Validate model creation through the factory pattern."""

    def setUp(self) -> None:
        """Create a default factory for each test."""
        self.factory = RecommenderModelFactory.default()

    def test_creates_popularity_model(self) -> None:
        """Factory creates a popularity recommender from enum config."""
        config = ModelConfig(ModelType.POPULARITY, recommendation_limit=2)
        model = self.factory.create(config)

        model.fit([
            ("user-1", "item-2"),
            ("user-2", "item-1"),
            ("user-3", "item-2"),
        ])

        self.assertIsInstance(model, PopularityRecommender)
        self.assertEqual(model.recommend("user-5"), ["item-2", "item-1"])

    def test_creates_recent_items_model(self) -> None:
        """Factory creates a recent-items recommender from string config."""
        config = ModelConfig("recent_items", recommendation_limit=2)
        interactions = [
            ("user-1", "item-1"),
            ("user-2", "item-2"),
            ("user-3", "item-1"),
            ("user-4", "item-3"),
        ]

        model = self.factory.create(config)
        model.fit(interactions)

        self.assertIsInstance(model, RecentItemsRecommender)
        self.assertEqual(model.recommend("user-5"), ["item-3", "item-1"])

    def test_registers_custom_model_creator(self) -> None:
        """Factory can be extended with a custom model creator."""
        self.factory.register("fake", create_fake_recommender)
        config = ModelConfig("fake", recommendation_limit=1)

        model = self.factory.create(config)
        model.fit([("user-1", "item-1"), ("user-2", "item-2")])

        self.assertIsInstance(model, RecommenderModel)
        self.assertEqual(model.recommend("user-3"), ["item-1"])

    def test_rejects_unknown_model_type(self) -> None:
        """Factory raises a clear error for unregistered model types."""
        config = ModelConfig("unknown_model")

        with self.assertRaisesRegex(ValueError, "Unknown model type"):
            self.factory.create(config)

    def test_rejects_invalid_recommendation_limit(self) -> None:
        """Model configuration requires a positive limit."""
        with self.assertRaisesRegex(ValueError, "recommendation_limit"):
            ModelConfig(recommendation_limit=0)


if __name__ == "__main__":
    unittest.main()
