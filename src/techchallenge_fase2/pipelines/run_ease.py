"""Pipeline dedicado ao EASE^ com tracking MLflow no experimento proprio.

Orquestra o fluxo completo do candidato a campeao:
1. Carrega configuracao e ambiente
2. Carrega dados de interacoes
3. Divide em treino/teste (split cronologico global)
4. Treina e avalia o EASE^ (Embarrassingly Shallow Autoencoder)
5. Registra hiperparametros, metricas e artefatos no MLflow
6. Salva relatorio markdown individual do EASE^

Como usar:
    $ uv run python -m techchallenge_fase2.pipelines.run_ease
    $ uv run python -m techchallenge_fase2.pipelines.run_ease --lambda-reg 500
    $ uv run python -m techchallenge_fase2.pipelines.run_ease --max-items 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from techchallenge_fase2.models.base import Interaction
from techchallenge_fase2.models.config import ModelConfig, ModelType
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
    set_global_seed,
)
from techchallenge_fase2.pipelines.model_result import ModelResult
from techchallenge_fase2.pipelines.report import (
    generate_markdown_report,
    save_markdown_report,
)
from techchallenge_fase2.pipelines.run_baselines import (
    K_VALUES,
    build_input_data_summary,
    load_interactions,
    temporal_holdout_split,
)
from techchallenge_fase2.pipelines.splits import (
    chronological_holdout_split,
    filter_warm_start,
)
from techchallenge_fase2.training.metrics import compute_recommender_metrics
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
from techchallenge_fase2.training.plots import (
    save_item_popularity_distribution,
    save_metrics_bar_chart,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "tech-challenge-ease"
CHAMPION_K = 10


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando para o pipeline do EASE^.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Treina e avalia o EASE^ com tracking MLflow dedicado"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Diretorio com dados brutos (default: data/raw)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fracao de interacoes para teste (default: 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (default: 42)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=250.0,
        help="Regularizacao L2 do EASE^ (default: 250.0)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20000,
        help="Numero maximo de itens no catalogo (default: 20000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tamanho do lote para predicao (default: 1000)",
    )
    parser.add_argument(
        "--popularity-blending",
        type=float,
        default=0.0,
        help="Peso de popularidade no score (default: 0.0)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nome do experimento MLflow (override)",
    )
    return parser.parse_args()


def run_ease_pipeline(  # noqa: PLR0913
    data_dir: str | Path = "data/raw",
    test_ratio: float = 0.15,
    random_seed: int = 42,
    lambda_reg: float = 250.0,
    max_items: int = 20000,
    batch_size: int = 1000,
    popularity_blending: float = 0.0,
    experiment_name: str | None = None,
    k_values: tuple[int, ...] = K_VALUES,
) -> ModelResult:
    """Executa pipeline dedicada do EASE^ com MLflow tracking.

    Args:
        data_dir: Diretorio com dados brutos.
        test_ratio: Fracao de interacoes para teste.
        random_seed: Seed para reprodutibilidade.
        lambda_reg: Regularizacao L2 do EASE^.
        max_items: Numero maximo de itens no catalogo.
        batch_size: Tamanho do lote para predicao.
        popularity_blending: Peso de popularidade no score.
        experiment_name: Nome do experimento MLflow (override).
        k_values: Valores de K para as metricas.

    Returns:
        ModelResult com as metricas do EASE^.
    """
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_global_seed(random_seed)

    exp_name = get_experiment_name(
        cli_arg=experiment_name,
        env_var_name="MLFLOW_EASE_EXPERIMENT_NAME",
        default_name=DEFAULT_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=exp_name)
    setup_mlflow(mlflow_config)

    interactions_df = load_interactions(data_dir)
    train_interactions, ground_truth, _ = _split_chronological(
        interactions_df, test_ratio
    )
    dataset_version = safe_get_dataset_version()

    config = ModelConfig(
        model_type=ModelType.EASE_TORCH,
        recommendation_limit=max(k_values),
        lambda_reg=lambda_reg,
        max_items=max_items,
        batch_size=batch_size,
        popularity_blending=popularity_blending,
    )
    factory = RecommenderModelFactory.default()
    model = factory.create(config)

    logger.info("Treinando EASE^ (lambda=%s, max_items=%s)...", lambda_reg, max_items)
    train_time = _time_fit(model, train_interactions)
    recommended, infer_time = _time_recommend(model, ground_truth, max(k_values))
    metrics = compute_recommender_metrics(ground_truth, recommended, k_values)

    result = ModelResult(
        model_name=ModelType.EASE_TORCH.value,
        model_role="champion_candidate",
        metrics=metrics,
        train_time_sec=float(train_time),
        infer_time_sec=float(infer_time),
        num_users_evaluated=int(metrics.get("num_users", 0)),
    )

    logger.info(
        "EASE^: precision@10=%.4f recall@10=%.4f ndcg@10=%.4f "
        "map@10=%.4f hit_rate@10=%.4f treino=%.2fs infer=%.2fs",
        result.metric_at_k("precision", 10),
        result.metric_at_k("recall", 10),
        result.metric_at_k("ndcg", 10),
        result.metric_at_k("map", 10),
        result.metric_at_k("hit_rate", 10),
        train_time,
        infer_time,
    )

    _log_to_mlflow(
        model=model,
        result=result,
        config=config,
        interactions_df=interactions_df,
        train_interactions=train_interactions,
        ground_truth=ground_truth,
        test_ratio=test_ratio,
        random_seed=random_seed,
        dataset_version=dataset_version,
        k_values=k_values,
    )

    _save_ease_report(
        result=result,
        interactions_df=interactions_df,
        train_interactions=train_interactions,
        ground_truth=ground_truth,
        test_ratio=test_ratio,
        random_seed=random_seed,
        k_values=k_values,
    )

    return result


def _split_chronological(
    interactions_df: pd.DataFrame, test_ratio: float
) -> tuple[list[Interaction], dict[str, set[str]], dict[str, list[str]]]:
    """Divide interacoes usando split cronologico global com warm-start filter.

    Args:
        interactions_df: DataFrame com colunas user_id, item_id, timestamp.
        test_ratio: Fracao para teste.

    Returns:
        Tupla (train_interactions, ground_truth, all_items_by_user).
    """
    if "timestamp" not in interactions_df.columns:
        logger.warning(
            "Coluna timestamp ausente; usando fallback temporal_holdout_split"
        )
        return temporal_holdout_split(interactions_df, test_ratio=test_ratio)
    split = chronological_holdout_split(interactions_df, test_ratio)
    split = filter_warm_start(split)
    all_items_by_user: dict[str, list[str]] = {}
    for uid, iid in split.train_interactions:
        all_items_by_user.setdefault(uid, []).append(iid)
    return (
        split.train_interactions,
        split.ground_truth,
        all_items_by_user,
    )


def _time_fit(model: Any, interactions: list[Interaction]) -> float:
    """Mede o tempo de treino em segundos."""
    t0 = time.perf_counter()
    model.fit(interactions)
    return time.perf_counter() - t0


def _time_recommend(
    model: Any, ground_truth: dict[str, set[str]], limit: int
) -> tuple[dict[str, list[str]], float]:
    """Mede o tempo de inferencia e retorna recomendacoes."""
    t0 = time.perf_counter()
    recommended: dict[str, list[str]] = {}
    for user_id in ground_truth:
        recommended[user_id] = model.recommend(user_id, limit=limit)
    return recommended, time.perf_counter() - t0


def _log_to_mlflow(  # noqa: PLR0913
    model: Any,
    result: ModelResult,
    config: ModelConfig,
    interactions_df: pd.DataFrame,
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    test_ratio: float,
    random_seed: int,
    dataset_version: str,
    k_values: tuple[int, ...],
) -> None:
    """Registra hiperparametros, metricas e artefatos do EASE^ no MLflow.

    Args:
        model: Instancia do EASETorchRecommender treinado.
        result: ModelResult com metricas e tempos.
        config: ModelConfig usada no treino.
        interactions_df: DataFrame completo de interacoes.
        train_interactions: Interacoes de treino.
        ground_truth: Itens relevantes por usuario.
        test_ratio: Fracao de teste.
        random_seed: Seed usado.
        dataset_version: Versao do dataset via DVC.
        k_values: Valores de K das metricas.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="ease_torch_dedicated"):
        log_hyperparameters(
            {
                "model_type": ModelType.EASE_TORCH.value,
                "lambda_reg": config.lambda_reg,
                "max_items": config.max_items,
                "batch_size": config.batch_size,
                "popularity_blending": config.popularity_blending,
                "recommendation_limit": config.recommendation_limit,
                "random_seed": random_seed,
                "test_ratio": test_ratio,
                "dataset_version": dataset_version,
                "num_train_interactions": len(train_interactions),
                "num_evaluated_users": len(ground_truth),
                "k_values": str(k_values),
                "architecture": type(model).__name__,
                "model_role": "champion_candidate",
                "train_time_sec": result.train_time_sec,
                "infer_time_sec": result.infer_time_sec,
            }
        )

        log_system_info(random_seed)
        log_metrics(_sanitize_metric_names(result.metrics))
        log_input_data_summary(
            build_input_data_summary(
                interactions_df=interactions_df,
                train_interactions=train_interactions,
                ground_truth=ground_truth,
                test_ratio=test_ratio,
                dataset_version=dataset_version,
                random_seed=random_seed,
            )
        )
        log_recommender_model(model, ModelType.EASE_TORCH.value)

        mlflow.set_tag("issue", "15")
        mlflow.set_tag("model_role", "champion_candidate")
        mlflow.set_tag("baseline_family", "ease_torch")
        mlflow.set_tag("pipeline", "run_ease")
        mlflow.set_tag("random_seed", str(random_seed))

        metrics_chart_path = reports_dir / "metrics_ease_torch.png"
        save_metrics_bar_chart(
            result.metrics,
            metrics_chart_path,
            title="Metrics - EASE^ (dedicated)",
        )
        log_artifacts([metrics_chart_path])

        popularity_chart_path = reports_dir / "popularity_ease_torch.png"
        from collections import Counter

        item_counts = Counter(item_id for _, item_id in train_interactions)
        save_item_popularity_distribution(
            dict(item_counts.most_common(30)),
            popularity_chart_path,
            top_n=30,
            title="Item Popularity - EASE^",
        )
        log_artifacts([popularity_chart_path])

        card = build_model_card(
            ModelType.EASE_TORCH.value,
            random_seed=random_seed,
            dataset_version=dataset_version,
            num_evaluated_users=len(ground_truth),
            **result.metrics,
        )
        mlflow.log_dict(card, "model_card.json")

    logger.info("Run MLflow registrada para EASE^ no experimento dedicado")


def _save_ease_report(  # noqa: PLR0913
    result: ModelResult,
    interactions_df: pd.DataFrame,
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    test_ratio: float,
    random_seed: int,
    k_values: tuple[int, ...],
) -> None:
    """Gera e salva relatorio markdown individual do EASE^.

    Args:
        result: ModelResult com metricas do EASE^.
        interactions_df: DataFrame completo de interacoes.
        train_interactions: Interacoes de treino.
        ground_truth: Itens relevantes por usuario.
        test_ratio: Fracao de teste.
        random_seed: Seed usado.
        k_values: Valores de K das metricas.
    """
    report_content = generate_markdown_report(
        results=[result],
        k_values=k_values,
        champion_k=CHAMPION_K,
        num_users=int(interactions_df["user_id"].nunique()),
        num_items=int(interactions_df["item_id"].nunique()),
        num_interactions=len(interactions_df),
        num_evaluated_users=len(ground_truth),
        dataset_name="RetailRocket E-Commerce",
        split_strategy="chronological_holdout",
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    report_path = save_markdown_report(
        report_content,
        Path("reports") / "ease_dedicated_report.md",
    )
    logger.info("Relatorio markdown do EASE^ salvo em: %s", report_path)


def _sanitize_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    """Sanitiza nomes de metricas para compatibilidade com MLflow.

    MLflow nao aceita '@' em nomes de metricas; substitui por '_at_'.
    """
    return {key.replace("@", "_at_"): value for key, value in metrics.items()}


def main() -> int:
    """Ponto de entrada do pipeline dedicado do EASE^.

    Returns:
        Codigo de saida (0 para sucesso).
    """
    args = parse_args()
    try:
        result = run_ease_pipeline(
            data_dir=args.data_dir,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            lambda_reg=args.lambda_reg,
            max_items=args.max_items,
            batch_size=args.batch_size,
            popularity_blending=args.popularity_blending,
            experiment_name=args.experiment_name,
        )
        print(f"\nEASE^ H-Mean@10: {result.harmonic_mean_at_k(CHAMPION_K):.4f}")
        print(f"Precision@10: {result.metric_at_k('precision', 10):.4f}")
        print(f"Recall@10: {result.metric_at_k('recall', 10):.4f}")
        print(f"NDCG@10: {result.metric_at_k('ndcg', 10):.4f}")
        print(f"MAP@10: {result.metric_at_k('map', 10):.4f}")
        print(f"HitRate@10: {result.metric_at_k('hit_rate', 10):.4f}")
        print(f"Treino: {result.train_time_sec:.2f}s")
        print(f"Inferencia: {result.infer_time_sec:.2f}s")
        return 0
    except FileNotFoundError as e:
        logger.error("Arquivo nao encontrado: %s", e)
        return 1
    except Exception:
        logger.exception("Erro na execucao do pipeline do EASE^")
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    raise SystemExit(main())
