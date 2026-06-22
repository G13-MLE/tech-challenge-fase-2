"""Command-line entry points for local recommendation demos."""

from techchallenge_fase2.models import (
    ModelConfig,
    ModelType,
    RecommenderModelFactory,
)
from techchallenge_fase2.models.base import Interaction

DEMO_INTERACTIONS: tuple[Interaction, ...] = (
    ("user-1", "product-1"),
    ("user-2", "product-2"),
    ("user-3", "product-1"),
    ("user-4", "product-3"),
)


def build_demo_recommendations() -> list[str]:
    """Build demo recommendations through the model factory.

    Returns:
        Product identifiers ordered by the selected recommendation model.
    """
    config = ModelConfig(model_type=ModelType.POPULARITY, recommendation_limit=3)
    model = RecommenderModelFactory.default().create(config)
    model.fit(DEMO_INTERACTIONS)
    return model.recommend(user_id="demo-user")


def main() -> None:
    """Run a small local recommendation demo."""
    recommendations = build_demo_recommendations()
    print("Recommended items:", ", ".join(recommendations))
