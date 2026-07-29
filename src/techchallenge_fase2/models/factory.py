"""Factory for creating recommendation models from configuration."""

from collections.abc import Callable
from typing import Self

from techchallenge_fase2.models.base import RecommenderModel
from techchallenge_fase2.models.baselines import (
    PopularityRecommender,
    RandomRecommender,
    RecentItemsRecommender,
)
from techchallenge_fase2.models.config import ModelConfig, ModelType
from techchallenge_fase2.models.ease_torch import EASETorchRecommender
from techchallenge_fase2.models.embedding import TorchEmbeddingRecommender
from techchallenge_fase2.models.ncf import (
    NeuralCollaborativeFiltering,
    NeuralRecommender,
)
from techchallenge_fase2.models.sklearn_baselines import (
    ItemKNNRecommender,
    LogisticRegressionRecommender,
)

ModelKey = ModelType | str
ModelCreator = Callable[[ModelConfig], RecommenderModel]


def normalize_model_type(model_type: ModelKey) -> str:
    """Normaliza o identificador de modelo para uso no registro."""
    raw_value = model_type.value if isinstance(model_type, ModelType) else model_type
    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        raise ValueError("model_type must not be empty")
    return normalized_value


def create_popularity_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador baseado em popularidade."""
    return PopularityRecommender(default_limit=config.recommendation_limit)


def create_recent_items_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador baseado em itens recentes."""
    return RecentItemsRecommender(default_limit=config.recommendation_limit)


def create_torch_embedding_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador neural baseado em embeddings com treino BPR."""
    return TorchEmbeddingRecommender(
        num_users=config.num_users,
        num_items=config.num_items,
        embedding_dim=config.embedding_dim,
        default_limit=config.recommendation_limit,
        training_config=config.embedding_training_config(),
    )


def create_neural_ncf_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador neural (NCF: GMF + MLP) a partir da config."""
    ncf = NeuralCollaborativeFiltering(config.neural_config())
    return NeuralRecommender(ncf, config.ncf_training_config())


def create_random_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador aleatório (lower bound)."""
    return RandomRecommender(default_limit=config.recommendation_limit)


def create_ease_torch_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador EASE^ (candidato a campeão) a partir da config."""
    return EASETorchRecommender(config.ease_config())


def create_item_knn_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador baseado em vizinhos mais próximos de itens."""
    return ItemKNNRecommender(default_limit=config.recommendation_limit)


def create_logistic_regression_model(config: ModelConfig) -> RecommenderModel:
    """Cria o recomendador baseado em regressão logistica binaria."""
    return LogisticRegressionRecommender(default_limit=config.recommendation_limit)


class RecommenderModelFactory:
    """Create recommendation models without coupling clients to classes."""

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._creators: dict[str, ModelCreator] = {}

    @classmethod
    def default(cls) -> Self:
        """Build a factory with the project default model registry."""
        factory = cls()
        factory.register(ModelType.POPULARITY, create_popularity_model)
        factory.register(ModelType.RECENT_ITEMS, create_recent_items_model)
        factory.register(ModelType.TORCH_EMBEDDING, create_torch_embedding_model)
        factory.register(ModelType.RANDOM, create_random_model)
        factory.register(ModelType.NEURAL_NCF, create_neural_ncf_model)
        factory.register(ModelType.EASE_TORCH, create_ease_torch_model)
        factory.register(ModelType.ITEM_KNN, create_item_knn_model)
        factory.register(
            ModelType.LOGISTIC_REGRESSION, create_logistic_regression_model
        )
        return factory

    def register(self, model_type: ModelKey, creator: ModelCreator) -> None:
        """Register a model creator under a model key.

        Args:
            model_type: Model identifier selected by configuration.
            creator: Callable responsible for creating the model instance.
        """
        self._creators[normalize_model_type(model_type)] = creator

    def create(self, config: ModelConfig) -> RecommenderModel:
        """Create a model using the configured model type.

        Args:
            config: Model creation configuration.

        Returns:
            A recommender model that follows the project contract.
        """
        model_type = normalize_model_type(config.model_type)
        creator = self._creators.get(model_type)
        if creator is None:
            raise ValueError(self.unknown_model_message(model_type))
        return creator(config)

    def available_types(self) -> tuple[str, ...]:
        """Return all model keys currently registered in the factory."""
        return tuple(sorted(self._creators))

    def unknown_model_message(self, model_type: str) -> str:
        """Monta a mensagem para tipos de modelo não registrados."""
        available_types = ", ".join(self.available_types()) or "none"
        return f"Unknown model type '{model_type}'. Available types: {available_types}."
