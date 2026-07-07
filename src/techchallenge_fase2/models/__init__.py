"""Model abstractions and factories for recommendations."""

from techchallenge_fase2.models.base import Interaction, RecommenderModel
from techchallenge_fase2.models.baselines import (
    PopularityRecommender,
    RandomRecommender,
    RecentItemsRecommender,
)
from techchallenge_fase2.models.config import ModelConfig, ModelType
from techchallenge_fase2.models.embedding import (
    EmbeddingScoringModel,
    TorchEmbeddingRecommender,
)
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.models.ncf import (
    NCFConfig,
    NeuralCollaborativeFiltering,
    NeuralRecommender,
)

__all__ = [
    "EmbeddingScoringModel",
    "Interaction",
    "ModelConfig",
    "ModelType",
    "NCFConfig",
    "NeuralCollaborativeFiltering",
    "NeuralRecommender",
    "PopularityRecommender",
    "RandomRecommender",
    "RecentItemsRecommender",
    "RecommenderModel",
    "RecommenderModelFactory",
    "TorchEmbeddingRecommender",
]
