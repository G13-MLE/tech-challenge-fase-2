"""Evaluate the trained embedding recommender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from techchallenge_fase2.models.embedding import EmbeddingScoringModel
from techchallenge_fase2.pipeline.config import PipelineParams, load_params
from techchallenge_fase2.pipeline.metrics import (
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint safely.

    Usa ``weights_only=True`` para impedir execução arbitrária de pickle ao
    desserializar checkpoints compartilhados pelo DVC remote.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(checkpoint: dict[str, Any]) -> EmbeddingScoringModel:
    """Rebuild the trained embedding model."""
    model = EmbeddingScoringModel(
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_features(path: Path) -> pd.DataFrame:
    """Load a feature frame."""
    return pd.read_parquet(path)


def build_relevance(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Build relevant items per user."""
    relevance: dict[int, set[int]] = {}
    unique_pairs = frame[["user_index", "item_index"]].drop_duplicates()
    for row in unique_pairs.itertuples(index=False):
        relevance.setdefault(int(row.user_index), set()).add(int(row.item_index))
    return relevance


def select_users(relevance: dict[int, set[int]], max_users: int) -> list[int]:
    """Select a deterministic user subset for evaluation."""
    users = sorted(relevance)
    return users if max_users <= 0 else users[:max_users]


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Build train items that should not be recommended back."""
    seen: dict[int, set[int]] = {}
    unique_pairs = frame[["user_index", "item_index"]].drop_duplicates()
    for row in unique_pairs.itertuples(index=False):
        seen.setdefault(int(row.user_index), set()).add(int(row.item_index))
    return seen


def build_candidate_items(frame: pd.DataFrame) -> set[int]:
    """Build the item catalog learned during training."""
    return set(frame["item_index"].astype("int64").unique().tolist())


def filter_evaluable_relevance(
    relevance: dict[int, set[int]],
    seen: dict[int, set[int]],
    candidate_items: set[int],
) -> dict[int, set[int]]:
    """Keep only warm-start users and recommendable relevant items."""
    filtered: dict[int, set[int]] = {}
    for user, relevant_items in relevance.items():
        if user not in seen:
            continue
        available_items = (relevant_items & candidate_items) - seen[user]
        if available_items:
            filtered[user] = available_items
    return filtered


def recommend_for_user(
    model: EmbeddingScoringModel,
    user: int,
    candidate_items: set[int],
    excluded: set[int],
    top_k: int,
) -> list[int]:
    """Recommend top items for one encoded user."""
    candidates = sorted(candidate_items - excluded)
    if not candidates:
        return []
    item_ids = torch.as_tensor(candidates, dtype=torch.long)
    user_ids = torch.full((len(candidates),), user, dtype=torch.long)
    with torch.no_grad():
        scores = model(user_ids, item_ids).numpy()
    top_indexes = np.argsort(scores)[::-1][:top_k]
    return [int(candidates[index]) for index in top_indexes]


def evaluate_users(
    model: EmbeddingScoringModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: PipelineParams,
) -> dict[str, float]:
    """Evaluate recommendations for selected users."""
    seen = build_seen_items(train)
    candidate_items = build_candidate_items(train)
    relevance = filter_evaluable_relevance(
        build_relevance(test),
        seen,
        candidate_items,
    )
    users = select_users(relevance, params.evaluation.max_users)
    scores = [
        score_user(model, user, relevance, seen, candidate_items, params)
        for user in users
    ]
    return summarize_scores(scores, params.evaluation.top_k, len(users))


def score_user(
    model: EmbeddingScoringModel,
    user: int,
    relevance: dict[int, set[int]],
    seen: dict[int, set[int]],
    candidate_items: set[int],
    params: PipelineParams,
) -> dict[str, float]:
    """Score recommendations for one user."""
    top_k = params.evaluation.top_k
    recommended = recommend_for_user(
        model,
        user,
        candidate_items=candidate_items,
        excluded=seen.get(user, set()),
        top_k=top_k,
    )
    return compute_scores(recommended, relevance[user], top_k)


def compute_scores(
    recommended: list[int],
    relevant: set[int],
    top_k: int,
) -> dict[str, float]:
    """Compute all ranking metrics for one user."""
    return {
        "hit_rate": hit_rate_at_k(recommended, relevant, top_k),
        "map": map_at_k(recommended, relevant, top_k),
        "ndcg": ndcg_at_k(recommended, relevant, top_k),
        "precision": precision_at_k(recommended, relevant, top_k),
        "recall": recall_at_k(recommended, relevant, top_k),
    }


def summarize_scores(
    scores: list[dict[str, float]],
    top_k: int,
    users: int,
) -> dict[str, float]:
    """Aggregate per-user metrics."""
    if not scores:
        return empty_metrics(top_k)
    keys = scores[0].keys()
    metrics = {f"{key}_at_{top_k}": average_score(scores, key) for key in keys}
    metrics["evaluated_users"] = float(users)
    return metrics


def average_score(scores: list[dict[str, float]], key: str) -> float:
    """Average one metric key across users."""
    return float(np.mean([score[key] for score in scores]))


def empty_metrics(top_k: int) -> dict[str, float]:
    """Return zero metrics when no evaluable user exists."""
    names = ["hit_rate", "map", "ndcg", "precision", "recall"]
    metrics = {f"{name}_at_{top_k}": 0.0 for name in names}
    metrics["evaluated_users"] = 0.0
    return metrics


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Save metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def run(params: PipelineParams) -> None:
    """Run the evaluation stage."""
    checkpoint = load_checkpoint(params.paths.model_checkpoint)
    model = load_model(checkpoint)
    train = load_features(params.paths.train_features)
    test = load_features(params.paths.test_features)
    metrics = evaluate_users(model, train, test, params)
    save_metrics(metrics, params.paths.metrics)


def main() -> None:
    """CLI entry point for the evaluation stage."""
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
