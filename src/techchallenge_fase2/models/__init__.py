"""Model abstractions and factories for recommendations."""

from techchallenge_fase2.models.base import Interaction, RecommenderModel
from techchallenge_fase2.models.baselines import (
    PopularityRecommender,
    RecentItemsRecommender,
)
from techchallenge_fase2.models.config import ModelConfig, ModelType
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.models.ncf import (
    NCFConfig,
    NeuralCollaborativeFiltering,
    NeuralRecommender,
)

__all__ = [
    "Interaction",
    "ModelConfig",
    "ModelType",
    "NCFConfig",
    "NeuralCollaborativeFiltering",
    "NeuralRecommender",
    "PopularityRecommender",
    "RecentItemsRecommender",
    "RecommenderModel",
    "RecommenderModelFactory",
]
