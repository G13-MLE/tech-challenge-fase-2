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

__all__ = [
    "EmbeddingScoringModel",
    "Interaction",
    "ModelConfig",
    "ModelType",
    "PopularityRecommender",
    "RandomRecommender",
    "RecentItemsRecommender",
    "RecommenderModel",
    "RecommenderModelFactory",
    "TorchEmbeddingRecommender",
]
