"""Base contracts shared by recommendation models."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

Interaction = tuple[str, str]


class RecommenderModel(ABC):
    """Common interface for recommendation models."""

    @abstractmethod
    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Prepare the model with user-item interactions.

        Args:
            interactions: Iterable of user and item identifier pairs.
        """

    @abstractmethod
    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return recommendations for a user.

        Args:
            user_id: Identifier of the user receiving recommendations.
            limit: Optional maximum number of recommended items.

        Returns:
            Ordered item identifiers.
        """

