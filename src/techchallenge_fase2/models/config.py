"""Configuration objects for recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from techchallenge_fase2.models.ease_torch import EASEConfig
    from techchallenge_fase2.models.embedding import EmbeddingTrainingConfig
    from techchallenge_fase2.models.ncf import NCFConfig, NCFTrainingConfig


class ModelType(StrEnum):
    """Built-in model identifiers supported by the default factory."""

    POPULARITY = "popularity"
    RECENT_ITEMS = "recent_items"
    TORCH_EMBEDDING = "torch_embedding"
    RANDOM = "random"
    NEURAL_NCF = "neural_ncf"
    EASE_TORCH = "ease_torch"
    ITEM_KNN = "item_knn"
    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration consumed by the model factory.

    Args:
        model_type: Built-in model type or custom registered model key.
        recommendation_limit: Default number of items returned by a model.
        num_users: Number of encoded users for neural models.
        num_items: Number of encoded items for neural models.
        embedding_dim: Embedding vector size for neural models.
        lambda_reg: Regularizacao L2 do EASE^ (faixa tipica 1e2 a 1e4).
        max_items: Numero maximo de itens considerados no catalogo do EASE^.
            Use 0 para catalogo completo.
        batch_size: Tamanho do bloco de usuarios para predicao do EASE^.
        popularity_blending: Peso de popularidade adicionado ao score do EASE^.
    """

    model_type: ModelType | str = ModelType.POPULARITY
    recommendation_limit: int = 10
    num_users: int = 1
    num_items: int = 1
    embedding_dim: int = 64
    lambda_reg: float = 250.0
    max_items: int = 20000
    batch_size: int = 1000
    popularity_blending: float = 0.0
    ncf_epochs: int = 30
    ncf_learning_rate: float = 0.001
    ncf_negatives_per_positive: int = 4

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
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg deve ser positivo")
        if self.max_items < 0:
            raise ValueError("max_items deve ser nao negativo")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        if self.popularity_blending < 0:
            raise ValueError("popularity_blending deve ser nao negativo")

    def neural_config(self) -> "NCFConfig":
        """Constroi a configuracao do NCF a partir deste ModelConfig.

        Returns:
            NCFConfig valido para instanciar o modelo neural (GMF + MLP).

        Raises:
            ValueError: Quando num_users/num_items nao sao positivos.
        """
        from techchallenge_fase2.models.ncf import (
            NCFConfig,
        )  # import tardio evita ciclo

        if self.num_users <= 0 or self.num_items <= 0:
            raise ValueError("num_users e num_items devem ser positivos para o NCF")
        return NCFConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
        )

    def ncf_training_config(self) -> "NCFTrainingConfig":
        """Constroi a configuracao de treino inline do NCF.

        Returns:
            NCFTrainingConfig para treinar o NCF dentro do pipeline
            de baselines comparativo.
        """
        from techchallenge_fase2.models.ncf import (  # import tardio evita ciclo
            NCFTrainingConfig,
        )

        return NCFTrainingConfig(
            epochs=self.ncf_epochs,
            learning_rate=self.ncf_learning_rate,
            negatives_per_positive=self.ncf_negatives_per_positive,
            batch_size=self.batch_size,
        )

    def embedding_training_config(self) -> "EmbeddingTrainingConfig":
        """Constroi a configuracao de treino inline do TorchEmbedding.

        Returns:
            EmbeddingTrainingConfig para treinar o modelo de embeddings
            com BPR loss dentro do pipeline de baselines comparativo.
        """
        from techchallenge_fase2.models.embedding import (  # import tardio evita ciclo
            EmbeddingTrainingConfig,
        )

        return EmbeddingTrainingConfig(
            epochs=self.ncf_epochs,
            learning_rate=self.ncf_learning_rate * 5,
            negatives_per_positive=self.ncf_negatives_per_positive,
            batch_size=max(self.batch_size, 256),
            random_seed=42,
        )

    def ease_config(self) -> "EASEConfig":
        """Constroi a configuracao do EASE^ a partir deste ModelConfig.

        Returns:
            EASEConfig valido para instanciar o EASE^ em PyTorch.
        """
        from techchallenge_fase2.models.ease_torch import (  # import tardio evita ciclo
            EASEConfig,
        )

        return EASEConfig(
            lambda_reg=self.lambda_reg,
            max_items=self.max_items,
            batch_size=self.batch_size,
            popularity_blending=self.popularity_blending,
        )
