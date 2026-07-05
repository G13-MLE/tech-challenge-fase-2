"""Simple baseline recommenders used by the model factory."""

import random
from collections import Counter
from collections.abc import Iterable

from techchallenge_fase2.models.base import Interaction, RecommenderModel


def validate_limit(value: int) -> int:
    """Valida se o limite de recomendações é positivo."""
    if value < 1:
        raise ValueError("limit must be positive")
    return value


def resolve_limit(default_limit: int, limit: int | None) -> int:
    """Resolve o limite informado ou usa o padrão configurado."""
    return validate_limit(default_limit if limit is None else limit)


class PopularityRecommender(RecommenderModel):
    """Recommend the most frequent items in the interaction history."""

    def __init__(self, default_limit: int = 10) -> None:
        """Initialize the recommender.

        Args:
            default_limit: Default number of items returned by recommend.
        """
        self._default_limit = validate_limit(default_limit)
        self._ranked_items: list[str] = []

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Rank items by global interaction frequency."""
        counts = Counter(item_id for _, item_id in interactions)
        ranked_pairs = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        self._ranked_items = [item_id for item_id, _ in ranked_pairs]

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return the most popular known items."""
        _ = user_id
        recommendation_limit = resolve_limit(self._default_limit, limit)
        return self._ranked_items[:recommendation_limit]


class RecentItemsRecommender(RecommenderModel):
    """Recommend the most recently observed unique items."""

    def __init__(self, default_limit: int = 10) -> None:
        """Initialize the recommender.

        Args:
            default_limit: Default number of items returned by recommend.
        """
        self._default_limit = validate_limit(default_limit)
        self._ranked_items: list[str] = []

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Rank unique items by recency in the interaction history."""
        seen_items: set[str] = set()
        ranked_items: list[str] = []
        for _, item_id in reversed(list(interactions)):
            if item_id not in seen_items:
                seen_items.add(item_id)
                ranked_items.append(item_id)
        self._ranked_items = ranked_items

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return the most recently observed unique items."""
        _ = user_id
        recommendation_limit = resolve_limit(self._default_limit, limit)
        return self._ranked_items[:recommendation_limit]


class RandomRecommender(RecommenderModel):
    """Recommend items sampled uniformly at random from the catalog.

    Serve como baseline de lower bound: qualquer modelo personalizado
    deve superar o acaso para justificar sua complexidade.
    """

    def __init__(self, default_limit: int = 10, random_seed: int = 42) -> None:
        """Initialize the recommender with a fixed random seed.

        Args:
            default_limit: Default number of items returned by recommend.
            random_seed: Seed para reprodutibilidade do sampling.
        """
        self._default_limit = validate_limit(default_limit)
        self._random_seed = random_seed
        self._items: list[str] = []
        self._rng: random.Random | None = None

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Cataloga itens únicos observados nas interações."""
        seen: set[str] = set()
        for _, item_id in interactions:
            if item_id not in seen:
                seen.add(item_id)
        self._items = list(seen)
        self._rng = random.Random(self._random_seed)

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Amostra itens do catálogo de forma aleatória e reprodutível."""
        _ = user_id
        recommendation_limit = resolve_limit(self._default_limit, limit)
        if not self._items or self._rng is None:
            return []
        n = min(recommendation_limit, len(self._items))
        return self._rng.sample(self._items, n)
