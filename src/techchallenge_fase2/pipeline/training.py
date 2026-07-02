"""Train the PyTorch neural recommender (NCF) for the DVC pipeline.

Orquestra a leitura das features de treino/validacao, a amostragem negativa,
a instanciacao do NCF via Factory e o loop de treinamento delegando ao
``Trainer`` do modulo ``techchallenge_fase2.training``, que aplica early
stopping, valida por AUC a cada epoca e persiste checkpoints best/last.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from techchallenge_fase2.data import InteractionData
from techchallenge_fase2.models.config import ModelConfig
from techchallenge_fase2.models.ncf import NeuralCollaborativeFiltering
from techchallenge_fase2.pipeline.config import PipelineParams, load_params
from techchallenge_fase2.training import Trainer, TrainingConfig

LabelRow = tuple[int, int, float]
PositiveRow = tuple[int, int]


@dataclass(frozen=True, slots=True)
class LabeledTensors:
    """Tensores rotulados para alimentar o loop de treino do NCF."""

    users: torch.Tensor
    items: torch.Tensor
    labels: torch.Tensor


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


def create_ncf_model(
    params: PipelineParams, users: int, items: int
) -> NeuralCollaborativeFiltering:
    """Create the NCF model from pipeline parameters."""
    config = ModelConfig(
        num_users=users,
        num_items=items,
        embedding_dim=params.training.embedding_dim,
        recommendation_limit=params.evaluation.top_k,
    )
    return NeuralCollaborativeFiltering(config.neural_config())


def build_training_config(params: PipelineParams) -> TrainingConfig:
    """Build the TrainerConfig from the pipeline parameters."""
    return TrainingConfig(
        learning_rate=params.training.learning_rate,
        batch_size=params.training.batch_size,
        epochs=params.training.epochs,
        patience=params.training.patience,
        min_delta=params.training.min_delta,
        device="cpu",
    )


def build_seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Build known positive items per user."""
    seen: dict[int, set[int]] = {}
    for row in frame[["user_index", "item_index"]].itertuples(index=False):
        seen.setdefault(int(row.user_index), set()).add(int(row.item_index))
    return seen


def sample_negative_item(
    user_items: set[int], num_items: int, rng: np.random.Generator
) -> int:
    """Sample one item that the user has not interacted with.

    Usa a diferenca de conjuntos em vez de rejeicao amostral, evitando laco
    infinito em catalogos parcialmente saturados e nunca retorna um item
    positivo disfarcado de negativo.

    Raises:
        ValueError: Quando o usuario ja interagiu com todos os itens do
            catalogo (nao existe negativo valido).
    """
    missing = list(set(range(num_items)) - user_items)
    if not missing:
        raise ValueError(
            "Nao ha item negativo disponivel: usuario consumiu todo o catalogo",
        )
    return int(rng.choice(missing))


def build_negative_rows(
    user: int,
    user_items: set[int],
    num_items: int,
    negative_samples: int,
    rng: np.random.Generator,
) -> list[LabelRow]:
    """Create sampled negative rows for one user.

    Ignora o usuario quando ele ja interagiu com todos os itens do catalogo,
    evitando rotular um positivo como negativo e corromper o treino.
    """
    if len(user_items) >= num_items:
        return []
    return [
        (user, sample_negative_item(user_items, num_items, rng), 0.0)
        for _ in range(negative_samples)
    ]


def build_positive_rows(positives: pd.DataFrame) -> list[LabelRow]:
    """Build positive rows from the feature frame."""
    return [
        (int(user), int(item), 1.0)
        for user, item in positives[["user_index", "item_index"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]


def build_label_rows(
    frame: pd.DataFrame, params: PipelineParams, num_items: int
) -> list[LabelRow]:
    """Create labeled rows with sampled negatives for implicit feedback."""
    rng = np.random.default_rng(params.training.random_seed)
    seen_items = build_seen_items(frame)
    rows = build_positive_rows(frame)
    for user, seen in seen_items.items():
        rows.extend(
            build_negative_rows(
                user, seen, num_items, params.training.negative_samples, rng
            )
        )
    return rows


def rows_to_tensors(rows: list[LabelRow]) -> LabeledTensors:
    """Convert sampled rows to label-only tensors for the Trainer."""
    array = np.asarray(rows, dtype=np.int64)
    return LabeledTensors(
        users=torch.as_tensor(array[:, 0], dtype=torch.long),
        items=torch.as_tensor(array[:, 1], dtype=torch.long),
        labels=torch.as_tensor(array[:, 2], dtype=torch.float32),
    )


def build_labeled_tensors(
    frame: pd.DataFrame, params: PipelineParams, num_items: int
) -> LabeledTensors:
    """Build positive and negative examples for implicit feedback."""
    rows = build_label_rows(frame, params, num_items)
    if not rows:
        raise ValueError(
            "Nao foi possivel gerar amostras de treino; verifique o split",
        )
    return rows_to_tensors(rows)


def to_interaction_data(
    train: LabeledTensors,
    validation: LabeledTensors,
    users: int,
    items: int,
) -> InteractionData:
    """Monta o InteractionData consumido pelo Trainer."""
    return InteractionData(
        user_ids=train.users,
        item_ids=train.items,
        labels=train.labels,
        val_user_ids=validation.users,
        val_item_ids=validation.items,
        val_labels=validation.labels,
        num_users=users,
        num_items=items,
    )


def save_pipeline_checkpoint(
    model: NeuralCollaborativeFiltering,
    params: PipelineParams,
    history: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    """Persist a weights-only compatible checkpoint for the evaluate stage."""
    train_losses, val_metrics = history
    final_loss = float(train_losses[-1]) if train_losses else 0.0
    best_metric = float(max(val_metrics)) if val_metrics else 0.0
    params.paths.model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "ncf",
            "model_state": model.state_dict(),
            "num_users": model.config.num_users,
            "num_items": model.config.num_items,
            "embedding_dim": model.config.embedding_dim,
            "mlp_hidden_sizes": list(model.config.mlp_hidden_sizes),
            "dropout": float(model.config.dropout),
            "final_loss": final_loss,
            "best_metric": best_metric,
        },
        params.paths.model_checkpoint,
    )


def run(params: PipelineParams) -> None:
    """Run the training stage."""
    set_seed(params.training.random_seed)
    train_frame = load_features(params.paths.train_features)
    val_frame = load_features(params.paths.validation_features)
    users, items = load_entity_counts(params.paths.mappings)
    model = create_ncf_model(params, users, items)
    train_tensors = build_labeled_tensors(train_frame, params, items)
    val_tensors = build_labeled_tensors(val_frame, params, items)
    data = to_interaction_data(train_tensors, val_tensors, users, items)
    trainer = Trainer(build_training_config(params), params.paths.checkpoint_dir)
    history = trainer.train(model, data)
    save_pipeline_checkpoint(model, params, (history.train_losses, history.val_metrics))


def main() -> None:
    """CLI entry point for the training stage."""
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
