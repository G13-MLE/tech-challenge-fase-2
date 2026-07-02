"""Configuration objects for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from techchallenge_fase2.models.ncf import NCFConfig


class ModelType(StrEnum):
    """Built-in model identifiers supported by the default factory."""

    POPULARITY = "popularity"
    RECENT_ITEMS = "recent_items"
    NEURAL_NCF = "neural_ncf"
    TORCH_EMBEDDING = "torch_embedding"


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
    # Dimensoes necessarias apenas para modelos neurais (NCF/TorchEmbedding).
    num_users: int = 0
    num_items: int = 0
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        """Validate model configuration values."""
        if self.recommendation_limit < 1:
            raise ValueError("recommendation_limit must be positive")
        if self.num_users < 0 or self.num_items < 0:
            raise ValueError("num_users e num_items devem ser nao negativos")
        if self.embedding_dim is not None and self.embedding_dim < 1:
            raise ValueError("embedding_dim deve ser positivo")

    def neural_config(self, embedding_dim: int = 64) -> "NCFConfig":
        """Constroi a configuracao do NCF a partir deste ModelConfig.

        Args:
            embedding_dim: Dimensao dos embeddings quando nao definida no config.

        Returns:
            NCFConfig valido para instanciar o modelo neural.
        """
        from techchallenge_fase2.models.ncf import (
            NCFConfig,
        )  # import tardio evita ciclo

        if self.num_users <= 0 or self.num_items <= 0:
            raise ValueError("num_users e num_items devem ser positivos para o NCF")
        resolved_dim = (
            self.embedding_dim if self.embedding_dim is not None else embedding_dim
        )
        return NCFConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=resolved_dim,
        )

    def embedding_dim_or(self, default: int) -> int:
        """Retorna embedding_dim ou um valor padrao quando ausente."""
        return self.embedding_dim if self.embedding_dim is not None else default
