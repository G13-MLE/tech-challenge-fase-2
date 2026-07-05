"""Configuration objects for recommendation models."""

from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    """Built-in model identifiers supported by the default factory."""

    POPULARITY = "popularity"
    RECENT_ITEMS = "recent_items"
    TORCH_EMBEDDING = "torch_embedding"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration consumed by the model factory.

    Args:
        model_type: Built-in model type or custom registered model key.
        recommendation_limit: Default number of items returned by a model.
        num_users: Number of encoded users for neural models.
        num_items: Number of encoded items for neural models.
        embedding_dim: Embedding vector size for neural models.
    """

    model_type: ModelType | str = ModelType.POPULARITY
    recommendation_limit: int = 10
    num_users: int = 1
    num_items: int = 1
    embedding_dim: int = 16

    def __post_init__(self) -> None:
        """Validate model configuration values."""
        if self.recommendation_limit < 1:
            raise ValueError("recommendation_limit must be positive")
        if self.num_users < 1:
            raise ValueError("num_users must be positive")
        if self.num_items < 1:
            raise ValueError("num_items must be positive")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
