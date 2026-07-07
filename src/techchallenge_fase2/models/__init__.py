"""Model abstractions and factories for recommendations."""

from techchallenge_fase2.models.base import Interaction, RecommenderModel
from techchallenge_fase2.models.baselines import (
    PopularityRecommender,
    RandomRecommender,
    RecentItemsRecommender,
)
from techchallenge_fase2.models.config import ModelConfig, ModelType
from techchallenge_fase2.models.ease_torch import EASEConfig, EASETorchRecommender
from techchallenge_fase2.models.embedding import (
    EmbeddingScoringModel,
    TorchEmbeddingRecommender,
)
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.models.ncf import (
    NCFConfig,
    NCFTrainingConfig,
    NeuralCollaborativeFiltering,
    NeuralRecommender,
)
from techchallenge_fase2.models.sklearn_baselines import (
    ItemKNNConfig,
    ItemKNNRecommender,
    LogisticRegressionConfig,
    LogisticRegressionRecommender,
)

__all__ = [
    "EASEConfig",
    "EASETorchRecommender",
    "EmbeddingScoringModel",
    "Interaction",
    "ItemKNNConfig",
    "ItemKNNRecommender",
    "LogisticRegressionConfig",
    "LogisticRegressionRecommender",
    "ModelConfig",
    "ModelType",
    "NCFConfig",
    "NCFTrainingConfig",
    "NeuralCollaborativeFiltering",
    "NeuralRecommender",
    "PopularityRecommender",
    "RandomRecommender",
    "RecentItemsRecommender",
    "RecommenderModel",
    "RecommenderModelFactory",
    "TorchEmbeddingRecommender",
]
