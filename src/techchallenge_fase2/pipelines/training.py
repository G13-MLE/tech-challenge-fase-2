"""Train the PyTorch neural recommender (NCF) for the DVC pipeline.

Orquestra a leitura das features de treino/validacao, a amostragem negativa,
a instanciacao do NCF via Factory e o loop de treinamento delegando ao
``Trainer`` do modulo ``techchallenge_fase2.training``, que aplica early
stopping, valida por AUC a cada epoca e persiste checkpoints best/last.
Tambem registra hiperparametros, metricas, artefatos e o modelo no MLflow,
reaproveitando os utilitarios de ``techchallenge_fase2.training.mlflow_tracking``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch

from techchallenge_fase2.data import InteractionData
from techchallenge_fase2.models.config import ModelConfig
from techchallenge_fase2.models.ncf import NeuralCollaborativeFiltering
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from techchallenge_fase2.pipelines.config import PipelineParams, load_params
from techchallenge_fase2.training import (
    Trainer,
    TrainingConfig,
    TrainingHistory,
)
from techchallenge_fase2.training.mlflow_tracking import (
    MLflowConfig,
    log_artifacts,
    log_hyperparameters,
    log_input_data_summary,
    log_metrics,
    log_recommender_model,
    log_system_info,
    setup_mlflow,
)
from techchallenge_fase2.training.model_card import build_model_card

logger = logging.getLogger(__name__)

LabelRow = tuple[int, int, float]
PositiveRow = tuple[int, int]

# Nome do experimento MLflow para o treino do NCF orquestrado pelo DVC.
DEFAULT_EXPERIMENT_NAME = "tech-challenge-fase2-ncf"


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

    Usa amostragem por rejeicao (custo amortizado O(1) por chamada em
    catalogos esparsos), amostrando um item uniformemente em [0, num_items)
    e descartando candidatos que ja foram vistos. Para o caso edge de
    catalogos pequenos/saturados faz fallback para a diferenca de conjuntos,
    garantindo que nunca retorna um item positivo disfarcado de negativo.

    Raises:
        ValueError: Quando o usuario ja interagiu com todos os itens do
            catalogo (nao existe negativo valido).
    """
    max_attempts = 50
    for _ in range(max_attempts):
        candidate = int(rng.integers(0, num_items))
        if candidate not in user_items:
            return candidate
    return sample_negative_from_missing(user_items, num_items, rng)


def sample_negative_from_missing(
    user_items: set[int], num_items: int, rng: np.random.Generator
) -> int:
    """Fallback deterministico: amostra um item negativo da diferenca.

    Usado quando a rejeicao falha (catalogo pequeno/saturado). Levanta
    ``ValueError`` quando nenhum item negativo existe.
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


def build_training_hyperparameters(
    params: PipelineParams, users: int, items: int, dataset_version: str
) -> dict[str, Any]:
    """Constroi o dict de hiperparametros para logar no MLflow."""
    train_df = load_features(params.paths.train_features)
    val_df = load_features(params.paths.validation_features)
    return {
        "model_type": "neural_ncf",
        "architecture": "NeuralCollaborativeFiltering",
        "num_users": users,
        "num_items": items,
        "embedding_dim": params.training.embedding_dim,
        "batch_size": params.training.batch_size,
        "epochs": params.training.epochs,
        "learning_rate": params.training.learning_rate,
        "negative_samples": params.training.negative_samples,
        "patience": params.training.patience,
        "min_delta": params.training.min_delta,
        "random_seed": params.training.random_seed,
        "top_k": params.evaluation.top_k,
        "num_train_interactions": int(len(train_df)),
        "num_validation_interactions": int(len(val_df)),
        "dataset_version": dataset_version,
    }


def build_input_summary(
    params: PipelineParams, users: int, items: int, dataset_version: str
) -> dict[str, Any]:
    """Constroi resumo dos dados de entrada para artefato MLflow."""
    train_df = load_features(params.paths.train_features)
    val_df = load_features(params.paths.validation_features)
    num_train = int(len(train_df))
    num_val = int(len(val_df))
    sparsity = 1.0 - (num_train / (users * items)) if users and items else 1.0
    return {
        "dataset": "RetailRocket E-Commerce",
        "dataset_version": dataset_version,
        "split_strategy": "temporal_holdout",
        "num_users": users,
        "num_items": items,
        "num_train_interactions": num_train,
        "num_validation_interactions": num_val,
        "sparsity": float(sparsity),
        "features_train_path": str(params.paths.train_features),
        "features_validation_path": str(params.paths.validation_features),
    }


def summarize_history(history: TrainingHistory) -> dict[str, float]:
    """Extrai metricas finais agregadas do historico de treino."""
    train_losses = history.train_losses
    val_metrics = history.val_metrics
    return {
        "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
        "best_val_auc": float(max(val_metrics)) if val_metrics else 0.0,
        "last_val_auc": float(val_metrics[-1]) if val_metrics else 0.0,
        "num_epochs_run": float(len(train_losses)),
        "stopped_epoch": float(history.stopped_epoch),
    }


def save_history_artifact(history: TrainingHistory, artifact_path: Path) -> Path:
    """Salva o historico de treino/validacao como JSON para artefato MLflow."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_losses": list(history.train_losses),
        "val_auc": list(history.val_metrics),
        "stopped_epoch": history.stopped_epoch,
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact_path


def log_training_run(
    model: NeuralCollaborativeFiltering,
    params: PipelineParams,
    history: TrainingHistory,
    users: int,
    items: int,
    dataset_version: str,
) -> None:
    """Registra hiperparametros, metricas, modelo e artefatos no MLflow.

    Reutiliza os utilitarios genericos de ``mlflow_tracking`` para manter
    consistencia com a pipeline de baselines.
    """
    log_hyperparameters(
        build_training_hyperparameters(params, users, items, dataset_version)
    )
    log_system_info(random_seed=params.training.random_seed)
    log_input_data_summary(build_input_summary(params, users, items, dataset_version))
    log_metrics(summarize_history(history))

    # Tags para distinguir o run do NCF orquestrado pelo DVC.
    mlflow.set_tag("model_type", "neural_ncf")
    mlflow.set_tag("model_name", "ncf")
    mlflow.set_tag("stage", "train")
    mlflow.set_tag("orchestrator", "dvc")
    mlflow.set_tag("random_seed", str(params.training.random_seed))
    mlflow.set_tag("dataset_version", dataset_version)

    # Log do modelo PyTorch como artefato do MLflow.
    log_recommender_model(model, "neural_ncf", artifact_path="model")

    # Salva e loga o historico de treino como artefato JSON.
    history_path = save_history_artifact(
        history, Path("models") / "training_history.json"
    )
    log_artifacts([history_path])

    # Loga o checkpoint final do pipeline como artefato.
    if params.paths.model_checkpoint.exists():
        log_artifacts([params.paths.model_checkpoint])

    # Model Card do NCF (reaproveita o builder generico de model_card.py).
    card = build_model_card(
        "neural_ncf",
        random_seed=params.training.random_seed,
        dataset_version=dataset_version,
        num_users=users,
        num_items=items,
        final_train_loss=float(
            history.train_losses[-1] if history.train_losses else 0.0
        ),
        best_val_auc=float(max(history.val_metrics) if history.val_metrics else 0.0),
    )
    mlflow.log_dict(card, "model_card.json")


def run(params: PipelineParams) -> None:
    """Run the training stage with MLflow tracking."""
    logger.info("=" * 70)
    logger.info("[STAGE 3/4] TRAIN - treino do NCF (PyTorch) com MLflow tracking")
    load_dotenv_silent()
    set_seed(params.training.random_seed)

    # Configura MLflow (tracking URI + experimento) e abre um run dedicado.
    ncf_exp_name = get_experiment_name(
        cli_arg=None,
        env_var_name="MLFLOW_NCF_EXPERIMENT_NAME",
        default_name=DEFAULT_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=ncf_exp_name)
    setup_mlflow(mlflow_config)
    dataset_version = safe_get_dataset_version()

    train_frame = load_features(params.paths.train_features)
    val_frame = load_features(params.paths.validation_features)
    users, items = load_entity_counts(params.paths.mappings)
    model = create_ncf_model(params, users, items)
    logger.info(
        "  catalogo: usuarios=%d itens=%d embedding_dim=%d",
        users,
        items,
        params.training.embedding_dim,
    )
    logger.info(
        "  treino: %d interacoes | validacao: %d interacoes",
        len(train_frame),
        len(val_frame),
    )
    train_tensors = build_labeled_tensors(train_frame, params, items)
    val_tensors = build_labeled_tensors(val_frame, params, items)
    data = to_interaction_data(train_tensors, val_tensors, users, items)
    logger.info(
        "  exemplos rotulados: treino=%d (positivos+%d neg/usuario) validacao=%d",
        len(train_tensors.users),
        params.training.negative_samples,
        len(val_tensors.users),
    )
    trainer = Trainer(build_training_config(params), params.paths.checkpoint_dir)

    with mlflow.start_run(run_name="ncf_train"):
        history = trainer.train(model, data)
        save_pipeline_checkpoint(
            model, params, (history.train_losses, history.val_metrics)
        )
        log_training_run(model, params, history, users, items, dataset_version)

    logger.info(
        "Treino concluido: best_val_auc=%.4f epochs_run=%d",
        max(history.val_metrics) if history.val_metrics else 0.0,
        len(history.train_losses),
    )
    logger.info("[STAGE 3/4] TRAIN concluido")
    logger.info("=" * 70)


def main() -> None:
    """CLI entry point for the training stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
