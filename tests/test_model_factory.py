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
    EASETorchRecommender,
    ItemKNNRecommender,
    LogisticRegressionRecommender,
    ModelConfig,
    ModelType,
    NeuralCollaborativeFiltering,
    NeuralRecommender,
    PopularityRecommender,
    RandomRecommender,
    RecentItemsRecommender,
    RecommenderModel,
    RecommenderModelFactory,
    TorchEmbeddingRecommender,
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

        model.fit(
            [
                ("user-1", "item-2"),
                ("user-2", "item-1"),
                ("user-3", "item-2"),
            ]
        )

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

    def test_creates_random_model(self) -> None:
        """Factory creates a random recommender that samples the catalog."""
        config = ModelConfig(ModelType.RANDOM, recommendation_limit=3)
        interactions = [
            ("user-1", "item-1"),
            ("user-2", "item-2"),
            ("user-3", "item-3"),
        ]

        model = self.factory.create(config)
        model.fit(interactions)

        self.assertIsInstance(model, RandomRecommender)
        recommendations = model.recommend("user-5")
        self.assertEqual(len(recommendations), 3)
        # Todos os itens recomendados pertencem ao catálogo treinado
        self.assertTrue(set(recommendations).issubset({"item-1", "item-2", "item-3"}))

    def test_random_model_is_reproducible(self) -> None:
        """Random recommender produces same output for same seed."""
        config = ModelConfig("random", recommendation_limit=3)
        interactions = [
            ("user-1", "a"),
            ("user-2", "b"),
            ("user-3", "c"),
            ("user-4", "d"),
        ]

        model_a = self.factory.create(config)
        model_a.fit(interactions)
        recs_a = model_a.recommend("user-5")

        model_b = self.factory.create(config)
        model_b.fit(interactions)
        recs_b = model_b.recommend("user-5")

        self.assertEqual(recs_a, recs_b)

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

    def test_creates_torch_embedding_model(self) -> None:
        """Factory creates the PyTorch embedding recommender."""
        config = ModelConfig(
            ModelType.TORCH_EMBEDDING,
            recommendation_limit=2,
            num_users=3,
            num_items=4,
            embedding_dim=8,
        )

        model = self.factory.create(config)

        self.assertIsInstance(model, TorchEmbeddingRecommender)
        self.assertEqual(model.network.user_embeddings.num_embeddings, 3)
        self.assertEqual(model.network.item_embeddings.num_embeddings, 4)

    def test_rejects_invalid_recommendation_limit(self) -> None:
        """Model configuration requires a positive limit."""
        with self.assertRaisesRegex(ValueError, "recommendation_limit"):
            ModelConfig(recommendation_limit=0)

    def test_creates_neural_ncf_model(self) -> None:
        """Factory cria o recomendador NCF (GMF + MLP)."""
        config = ModelConfig(
            ModelType.NEURAL_NCF,
            recommendation_limit=2,
            num_users=3,
            num_items=4,
            embedding_dim=8,
        )

        model = self.factory.create(config)

        self.assertIsInstance(model, NeuralRecommender)
        self.assertIsInstance(model._model, NeuralCollaborativeFiltering)
        self.assertEqual(model._model.config.num_users, 3)
        self.assertEqual(model._model.config.num_items, 4)

    def test_creates_ease_torch_model(self) -> None:
        """Factory cria o recomendador EASE^ (candidato a campeão)."""
        config = ModelConfig(
            ModelType.EASE_TORCH,
            recommendation_limit=3,
            lambda_reg=250.0,
            max_items=0,
            batch_size=2,
        )

        model = self.factory.create(config)

        self.assertIsInstance(model, EASETorchRecommender)
        model.fit([("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i3")])
        recs = model.recommend("u1", limit=2)
        self.assertEqual(len(recs), 2)

    def test_creates_item_knn_model(self) -> None:
        """Factory cria o recomendador ItemKNN (scikit-learn)."""
        config = ModelConfig(ModelType.ITEM_KNN, recommendation_limit=3)

        model = self.factory.create(config)

        self.assertIsInstance(model, ItemKNNRecommender)
        model.fit([("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i3")])
        recs = model.recommend("u1", limit=2)
        self.assertEqual(len(recs), 2)

    def test_creates_logistic_regression_model(self) -> None:
        """Factory cria o recomendador LogisticRegression (scikit-learn)."""
        config = ModelConfig(ModelType.LOGISTIC_REGRESSION, recommendation_limit=3)

        model = self.factory.create(config)

        self.assertIsInstance(model, LogisticRegressionRecommender)
        model.fit([("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i3")])
        recs = model.recommend("u1", limit=2)
        self.assertEqual(len(recs), 2)

    def test_available_types_includes_new_models(self) -> None:
        """Factory default registra todos os modelos novos."""
        types = self.factory.available_types()
        self.assertIn("ease_torch", types)
        self.assertIn("item_knn", types)
        self.assertIn("logistic_regression", types)

    def test_ease_config_passes_device(self) -> None:
        """ModelConfig deve repassar device para EASEConfig."""
        config = ModelConfig(
            ModelType.EASE_TORCH,
            recommendation_limit=3,
            lambda_reg=100.0,
            max_items=0,
            batch_size=2,
            device="cpu",
        )
        ease_cfg = config.ease_config()
        self.assertEqual(ease_cfg.device, "cpu")

    def test_model_config_validates_device(self) -> None:
        """ModelConfig deve rejeitar device invalido."""
        try:
            ModelConfig(ModelType.EASE_TORCH, device="invalid")
        except ValueError:
            return
        raise AssertionError("ValueError esperado para device='invalid'")


if __name__ == "__main__":
    unittest.main()
