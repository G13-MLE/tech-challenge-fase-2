"""Configuration objects for recommendation models."""

from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    """Built-in model identifiers supported by the default factory."""

    POPULARITY = "popularity"
    RECENT_ITEMS = "recent_items"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration consumed by the model factory.

    Args:
        model_type: Built-in model type or custom registered model key.
        recommendation_limit: Default number of items returned by a model.
    """

    model_type: ModelType | str = ModelType.POPULARITY
    recommendation_limit: int = 10

    def __post_init__(self) -> None:
        """Validate model configuration values."""
        if self.recommendation_limit < 1:
            raise ValueError("recommendation_limit must be positive")

