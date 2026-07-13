"""Evaluate the trained embedding recommender.

Avalia o NCF treinado pelo estagio ``train`` e persiste as metricas Top-K
em ``metrics/recommendation_metrics.json``. Tambem registra hiperparametros,
metricas, artefatos e o Model Card no MLflow, reaproveitando os utilitarios
de ``techchallenge_fase2.training.mlflow_tracking``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch

from techchallenge_fase2.models.ncf import (
    NCFConfig,
    NeuralCollaborativeFiltering,
)
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
)
from techchallenge_fase2.pipelines.config import PipelineParams, load_params
from techchallenge_fase2.pipelines.metrics import (
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from techchallenge_fase2.training.mlflow_tracking import (
    MLflowConfig,
    log_artifacts,
    log_hyperparameters,
    log_metrics,
    log_system_info,
    setup_mlflow,
)
from techchallenge_fase2.training.model_card import build_model_card

logger = logging.getLogger(__name__)

# Nome do experimento MLflow para a avaliacao do NCF orquestrada pelo DVC.
DEFAULT_EXPERIMENT_NAME = "tech-challenge-fase2-ncf"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint safely.

    Usa ``weights_only=True`` para impedir execucao arbitrária de pickle ao
    desserializar checkpoints compartilhados pelo DVC remote.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(checkpoint: dict[str, Any]) -> NeuralCollaborativeFiltering:
    """Rebuild the trained NCF model from the pipeline checkpoint."""
    config = NCFConfig(
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        mlp_hidden_sizes=tuple(int(size) for size in checkpoint["mlp_hidden_sizes"]),
        dropout=float(checkpoint["dropout"]),
    )
    model = NeuralCollaborativeFiltering(config)
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
    model: NeuralCollaborativeFiltering,
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
    model: NeuralCollaborativeFiltering,
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
    model: NeuralCollaborativeFiltering,
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


def build_evaluation_hyperparameters(
    params: PipelineParams,
    checkpoint: dict[str, Any],
    num_evaluated_users: int,
    dataset_version: str,
) -> dict[str, Any]:
    """Constroi o dict de hiperparametros de avaliacao para o MLflow."""
    return {
        "model_type": "neural_ncf",
        "architecture": "NeuralCollaborativeFiltering",
        "embedding_dim": int(checkpoint["embedding_dim"]),
        "top_k": params.evaluation.top_k,
        "max_users": params.evaluation.max_users,
        "num_users": int(checkpoint["num_users"]),
        "num_items": int(checkpoint["num_items"]),
        "num_evaluated_users": num_evaluated_users,
        "dataset_version": dataset_version,
        "best_train_metric": float(checkpoint.get("best_metric", 0.0)),
        "final_train_loss": float(checkpoint.get("final_loss", 0.0)),
    }


def log_evaluation_run(
    params: PipelineParams,
    checkpoint: dict[str, Any],
    metrics: dict[str, float],
    dataset_version: str,
) -> None:
    """Registra hiperparametros, metricas e artefatos de avaliacao no MLflow."""
    num_evaluated_users = int(metrics.get("evaluated_users", 0.0))
    log_hyperparameters(
        build_evaluation_hyperparameters(
            params, checkpoint, num_evaluated_users, dataset_version
        )
    )
    log_system_info(random_seed=params.training.random_seed)
    log_metrics(metrics)

    # Tags distinguindo o run de avaliacao do NCF orquestrado pelo DVC.
    mlflow.set_tag("model_type", "neural_ncf")
    mlflow.set_tag("model_name", "ncf")
    mlflow.set_tag("stage", "evaluate")
    mlflow.set_tag("orchestrator", "dvc")
    mlflow.set_tag("dataset_version", dataset_version)

    # Loga o arquivo de metricas como artefato do MLflow.
    if params.paths.metrics.exists():
        log_artifacts([params.paths.metrics])

    # Model Card do NCF com as metricas de avaliacao preenchidas.
    card = build_model_card(
        "neural_ncf",
        random_seed=params.training.random_seed,
        dataset_version=dataset_version,
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        num_evaluated_users=num_evaluated_users,
        **card_metrics(metrics, params.evaluation.top_k),
    )
    mlflow.log_dict(card, "model_card.json")


def card_metrics(metrics: dict[str, float], top_k: int) -> dict[str, float]:
    """Mapeia metricas agregadas para as chaves esperadas pelo Model Card."""
    return {
        f"hit_rate@{top_k}": metrics.get(f"hit_rate_at_{top_k}", 0.0),
        f"map@{top_k}": metrics.get(f"map_at_{top_k}", 0.0),
        f"ndcg@{top_k}": metrics.get(f"ndcg_at_{top_k}", 0.0),
        f"precision@{top_k}": metrics.get(f"precision_at_{top_k}", 0.0),
        f"recall@{top_k}": metrics.get(f"recall_at_{top_k}", 0.0),
    }


def run(params: PipelineParams) -> None:
    """Run the evaluation stage with MLflow tracking."""
    load_dotenv_silent()

    # Configura MLflow (tracking URI + experimento) e abre um run dedicado.
    ncf_exp_name = get_experiment_name(
        cli_arg=None,
        env_var_name="MLFLOW_NCF_EXPERIMENT_NAME",
        default_name=DEFAULT_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=ncf_exp_name)
    setup_mlflow(mlflow_config)
    dataset_version = safe_get_dataset_version()

    checkpoint = load_checkpoint(params.paths.model_checkpoint)
    model = load_model(checkpoint)
    train = load_features(params.paths.train_features)
    test = load_features(params.paths.test_features)
    metrics = evaluate_users(model, train, test, params)
    save_metrics(metrics, params.paths.metrics)

    with mlflow.start_run(run_name="ncf_evaluate"):
        log_evaluation_run(params, checkpoint, metrics, dataset_version)

    logger.info(
        "Avaliacao concluida: evaluated_users=%d top_k=%d",
        int(metrics.get("evaluated_users", 0.0)),
        params.evaluation.top_k,
    )


def main() -> None:
    """CLI entry point for the evaluation stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
