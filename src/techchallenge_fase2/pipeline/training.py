"""Train the PyTorch embedding recommender."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from techchallenge_fase2.models import ModelConfig, ModelType, RecommenderModelFactory
from techchallenge_fase2.models.embedding import TorchEmbeddingRecommender
from techchallenge_fase2.pipeline.config import PipelineParams, load_params


@dataclass(frozen=True, slots=True)
class TrainingTensors:
    """Tensor bundle used by the training loop."""

    users: torch.Tensor
    items: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features(path: Path) -> pd.DataFrame:
    """Load training features."""
    return pd.read_parquet(path)


def load_entity_counts(path: Path) -> tuple[int, int]:
    """Count all encoded users and items from the mappings file."""
    mappings = json.loads(path.read_text(encoding="utf-8"))
    return len(mappings["user_ids"]), len(mappings["item_ids"])


def create_recommender(
    params: PipelineParams,
    users: int,
    items: int,
) -> TorchEmbeddingRecommender:
    """Create the neural recommender through the project factory."""
    config = ModelConfig(
        model_type=ModelType.TORCH_EMBEDDING,
        recommendation_limit=params.evaluation.top_k,
        num_users=users,
        num_items=items,
        embedding_dim=params.training.embedding_dim,
    )
    model = RecommenderModelFactory.default().create(config)
    if not isinstance(model, TorchEmbeddingRecommender):
        raise TypeError("Factory did not return TorchEmbeddingRecommender")
    return model


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Build known positive items per user."""
    seen: dict[int, set[int]] = {}
    for row in frame[["user_index", "item_index"]].itertuples(index=False):
        seen.setdefault(int(row.user_index), set()).add(int(row.item_index))
    return seen


def sample_negative_item(
    user_items: set[int],
    num_items: int,
    rng: np.random.Generator,
) -> int:
    """Sample one item that the user has not interacted with."""
    if len(user_items) >= num_items:
        return int(rng.integers(0, num_items))
    candidate = int(rng.integers(0, num_items))
    while candidate in user_items:
        candidate = int(rng.integers(0, num_items))
    return candidate


def build_training_tensors(
    frame: pd.DataFrame,
    params: PipelineParams,
    num_items: int,
) -> TrainingTensors:
    """Build positive and negative examples for implicit feedback."""
    rng = np.random.default_rng(params.training.random_seed)
    seen_items = build_seen_items(frame)
    positive = frame[["user_index", "item_index", "event_weight"]].drop_duplicates()
    rows = build_rows(
        positive,
        seen_items,
        num_items,
        params.training.negative_samples,
        rng,
    )
    array = np.asarray(rows, dtype=np.float32)
    return TrainingTensors(
        users=torch.as_tensor(array[:, 0], dtype=torch.long),
        items=torch.as_tensor(array[:, 1], dtype=torch.long),
        labels=torch.as_tensor(array[:, 2], dtype=torch.float32),
        weights=torch.as_tensor(array[:, 3], dtype=torch.float32),
    )


def build_rows(
    positive: pd.DataFrame,
    seen_items: dict[int, set[int]],
    num_items: int,
    negative_samples: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, float, float]]:
    """Create labeled rows with sampled negatives."""
    rows: list[tuple[int, int, float, float]] = []
    for user_index, item_index, weight in positive.itertuples(index=False):
        user = int(user_index)
        rows.append((user, int(item_index), 1.0, float(weight)))
        negative_rows = build_negative_rows(
            user,
            seen_items[user],
            num_items,
            negative_samples,
            rng,
        )
        rows.extend(negative_rows)
    return rows


def build_negative_rows(
    user: int,
    user_items: set[int],
    num_items: int,
    negative_samples: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, float, float]]:
    """Create sampled negative rows for one user."""
    return [
        (user, sample_negative_item(user_items, num_items, rng), 0.0, 1.0)
        for _ in range(negative_samples)
    ]


def make_loader(
    tensors: TrainingTensors,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Build a PyTorch DataLoader."""
    dataset = TensorDataset(
        tensors.users,
        tensors.items,
        tensors.labels,
        tensors.weights,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Train the model for one epoch."""
    model.train()
    losses: list[float] = []
    for users, items, labels, weights in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(users, items), labels)
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        optimizer.step()
        losses.append(float(weighted_loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(
    recommender: TorchEmbeddingRecommender,
    params: PipelineParams,
    users: int,
    items: int,
    final_loss: float,
) -> None:
    """Persist model weights and training metadata."""
    params.paths.model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_checkpoint(recommender, params, users, items, final_loss),
        params.paths.model_checkpoint,
    )


def build_checkpoint(
    recommender: TorchEmbeddingRecommender,
    params: PipelineParams,
    users: int,
    items: int,
    final_loss: float,
) -> dict[str, object]:
    """Build a serializable checkpoint."""
    return {
        "embedding_dim": params.training.embedding_dim,
        "final_loss": final_loss,
        "model_state": recommender.network.state_dict(),
        "num_items": items,
        "num_users": users,
    }


def run(params: PipelineParams) -> None:
    """Run the training stage."""
    set_seed(params.training.random_seed)
    frame = load_features(params.paths.train_features)
    users, items = load_entity_counts(params.paths.mappings)
    recommender = create_recommender(params, users, items)
    tensors = build_training_tensors(frame, params, items)
    loader = make_loader(tensors, params.training.batch_size)
    optimizer = torch.optim.Adam(
        recommender.network.parameters(),
        lr=params.training.learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    final_loss = 0.0
    for _ in range(params.training.epochs):
        final_loss = train_epoch(recommender.network, loader, optimizer, loss_fn)
    save_checkpoint(recommender, params, users, items, final_loss)


def main() -> None:
    """CLI entry point for the training stage."""
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
