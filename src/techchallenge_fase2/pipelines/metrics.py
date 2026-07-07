"""Top-K recommendation metrics."""

from __future__ import annotations

import math


def precision_at_k(recommended: list[int], relevant: set[int], top_k: int) -> float:
    """Compute Precision@K."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    hits = count_hits(recommended[:top_k], relevant)
    return hits / top_k


def recall_at_k(recommended: list[int], relevant: set[int], top_k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    hits = count_hits(recommended[:top_k], relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], top_k: int) -> float:
    """Compute NDCG@K."""
    if not relevant:
        return 0.0
    dcg = discounted_gain(recommended[:top_k], relevant)
    ideal = ideal_discounted_gain(min(len(relevant), top_k))
    return dcg / ideal if ideal else 0.0


def map_at_k(recommended: list[int], relevant: set[int], top_k: int) -> float:
    """Compute average precision at K."""
    if not relevant:
        return 0.0
    scores = precision_hits(recommended[:top_k], relevant)
    return sum(scores) / min(len(relevant), top_k)


def hit_rate_at_k(recommended: list[int], relevant: set[int], top_k: int) -> float:
    """Compute HitRate@K."""
    return float(count_hits(recommended[:top_k], relevant) > 0)


def count_hits(recommended: list[int], relevant: set[int]) -> int:
    """Count recommendations that are relevant."""
    return sum(1 for item in recommended if item in relevant)


def discounted_gain(recommended: list[int], relevant: set[int]) -> float:
    """Compute discounted cumulative gain."""
    return sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(recommended)
        if item in relevant
    )


def ideal_discounted_gain(relevant_count: int) -> float:
    """Compute ideal discounted cumulative gain."""
    return sum(1 / math.log2(rank + 2) for rank in range(relevant_count))


def precision_hits(recommended: list[int], relevant: set[int]) -> list[float]:
    """Return precision values at every hit rank."""
    hits = 0
    scores: list[float] = []
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            scores.append(hits / rank)
    return scores
