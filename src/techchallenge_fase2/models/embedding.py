"""PyTorch embedding recommender models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn

from techchallenge_fase2.models.base import Interaction, RecommenderModel
from techchallenge_fase2.models.baselines import resolve_limit, validate_limit


class EmbeddingScoringModel(nn.Module):
    """Score user-item pairs with trainable embeddings."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        """Initialize embedding tables.

        Args:
            num_users: Number of encoded users.
            num_items: Number of encoded items.
            embedding_dim: Embedding vector size.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for user-item pairs."""
        user_vectors = self.user_embeddings(user_ids)
        item_vectors = self.item_embeddings(item_ids)
        dot_scores = (user_vectors * item_vectors).sum(dim=1)
        user_bias = self.user_bias(user_ids).squeeze(dim=1)
        item_bias = self.item_bias(item_ids).squeeze(dim=1)
        return dot_scores + user_bias + item_bias


class TorchEmbeddingRecommender(RecommenderModel):
    """Recommendation model backed by a PyTorch embedding network."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        default_limit: int = 10,
    ) -> None:
        """Initialize the recommender.

        Args:
            num_users: Number of encoded users.
            num_items: Number of encoded items.
            embedding_dim: Embedding vector size.
            default_limit: Default number of recommendations.
        """
        self.network = EmbeddingScoringModel(num_users, num_items, embedding_dim)
        self._default_limit = validate_limit(default_limit)
        self._ranked_items: list[str] = []

    def fit(self, interactions: Iterable[Interaction]) -> None:
        """Prepare a popularity fallback for the recommender contract."""
        counts = Counter(item_id for _, item_id in interactions)
        ranked_pairs = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        self._ranked_items = [item_id for item_id, _ in ranked_pairs]

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Return fallback recommendations for a user."""
        _ = user_id
        recommendation_limit = resolve_limit(self._default_limit, limit)
        return self._ranked_items[:recommendation_limit]
