"""Pipeline de comparacao entre modelos de recomendacao e baselines.

Orquestra o fluxo completo de comparacao:
1. Carrega configuracao e ambiente
2. Carrega dados de interacoes
3. Divide em treino/teste (split cronologico global)
4. Treina e avalia todos os modelos (baselines + candidato a campeao)
5. Compara usando no minimo 4 metricas (precision, recall, NDCG, MAP)
6. Declara o campeao e gera relatorio comparativo
7. Registra tudo no MLflow

Uso:
    $ uv run python -m techchallenge_fase2.pipelines.run_compare_models
    $ uv run python -m techchallenge_fase2.pipelines.run_compare_models --skip-ease
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from techchallenge_fase2.models import ModelConfig
from techchallenge_fase2.models.base import Interaction
from techchallenge_fase2.models.config import ModelType
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
    set_global_seed,
)
from techchallenge_fase2.pipelines.model_result import (
    ModelResult,
    declare_champion,
)
from techchallenge_fase2.pipelines.report import (
    generate_markdown_report,
    save_markdown_report,
)
from techchallenge_fase2.pipelines.run_baselines import (
    EASE_HYPERPARAMS,
    K_VALUES,
    MODEL_ROLES,
    build_input_data_summary,
    load_interactions,
    temporal_holdout_split,
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
    save_model_comparison_chart,
)

logger = logging.getLogger(__name__)

# Metricas canonicas para a comparacao (no minimo 4)
COMPARISON_METRICS = ("precision", "recall", "ndcg", "map")
DEFAULT_EXPERIMENT_NAME = "tech-challenge-comparison"
CHAMPION_K = 10


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Comparacao de modelos de recomendacao vs baselines"
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
        help="Fracao de interacoes para teste (default: 0.15, com 70/15/15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (default: 42)",
    )
    parser.add_argument(
        "--skip-ease",
        action="store_true",
        help="Pular treinamento do EASE^ (apenas baselines simples)",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nome do experimento MLflow (override)",
    )
    return parser.parse_args()


def build_model_config(model_name: str, k_values: tuple[int, ...]) -> ModelConfig:
    """Constroi a configuracao apropriada para cada modelo.

    Args:
        model_name: Nome do modelo (valor do ModelType).
        k_values: Valores de K para as metricas.

    Returns:
        ModelConfig configurado para o modelo.
    """
    if model_name == ModelType.EASE_TORCH.value:
        return ModelConfig(
            model_type=model_name,
            recommendation_limit=max(k_values),
            lambda_reg=EASE_HYPERPARAMS["lambda_reg"],
            max_items=EASE_HYPERPARAMS["max_items"],
            batch_size=EASE_HYPERPARAMS["batch_size"],
            popularity_blending=EASE_HYPERPARAMS["popularity_blending"],
        )
    return ModelConfig(
        model_type=model_name,
        recommendation_limit=max(k_values),
    )


def time_fit(model: Any, interactions: list[Interaction]) -> float:
    """Mede o tempo de treino do modelo em segundos."""
    t0 = time.perf_counter()
    model.fit(interactions)
    return time.perf_counter() - t0


def time_recommend(
    model: Any, ground_truth: dict[str, set[str]], limit: int
) -> tuple[dict[str, list[str]], float]:
    """Mede o tempo total de inferencia e retorna recomendacoes."""
    t0 = time.perf_counter()
    recommended: dict[str, list[str]] = {}
    for user_id in ground_truth:
        recommended[user_id] = model.recommend(user_id, limit=limit)
    return recommended, time.perf_counter() - t0


def sanitize_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    """Sanitiza nomes de metricas para compatibilidade com MLflow.

    MLflow nao aceita '@' em nomes de metricas; substitui por '_at_'.
    """
    return {key.replace("@", "_at_"): value for key, value in metrics.items()}


def evaluate_all_models(  # noqa: PLR0913
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    k_values: tuple[int, ...] = K_VALUES,
    skip_ease: bool = False,
    random_seed: int = 42,
) -> tuple[list[ModelResult], dict[str, Any]]:
    """Treina e avalia todos os modelos para comparacao.

    Args:
        train_interactions: Interacoes de treino.
        ground_truth: Itens relevantes por usuario para avaliacao.
        k_values: Valores de K para computar metricas.
        skip_ease: Se True, pula o EASE^ na avaliacao.
        random_seed: Seed para reprodutibilidade.

    Returns:
        Tupla com:
        - Lista de ModelResult com resultados estruturados por modelo.
        - Dicionario mapeando nome do modelo para instancia treinada.
    """
    _ = random_seed
    factory = RecommenderModelFactory.default()
    model_names = list(factory.available_types())

    if skip_ease and ModelType.EASE_TORCH.value in model_names:
        model_names.remove(ModelType.EASE_TORCH.value)

    model_results: list[ModelResult] = []
    trained_models: dict[str, Any] = {}

    for model_name in model_names:
        config = build_model_config(model_name, k_values)
        model = factory.create(config)
        try:
            train_time = time_fit(model, train_interactions)
            recommended, infer_time = time_recommend(model, ground_truth, max(k_values))
            metrics = compute_recommender_metrics(ground_truth, recommended, k_values)
        except Exception as exc:
            logger.warning("Modelo %s falhou (%s); pulando.", model_name, exc)
            continue
        trained_models[model_name] = model

        result = ModelResult(
            model_name=model_name,
            model_role=MODEL_ROLES.get(model_name, "baseline"),
            metrics=metrics,
            train_time_sec=float(train_time),
            infer_time_sec=float(infer_time),
            num_users_evaluated=int(metrics.get("num_users", 0)),
        )
        model_results.append(result)
        logger.info(
            "Modelo %s: precision@10=%.4f recall@10=%.4f ndcg@10=%.4f "
            "map@10=%.4f hit_rate@10=%.4f treino=%.2fs infer=%.2fs",
            model_name,
            result.metric_at_k("precision", 10),
            result.metric_at_k("recall", 10),
            result.metric_at_k("ndcg", 10),
            result.metric_at_k("map", 10),
            result.metric_at_k("hit_rate", 10),
            train_time,
            infer_time,
        )

    return model_results, trained_models


def build_comparison_summary(
    results: list[ModelResult],
    k: int = CHAMPION_K,
    comparison_metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> str:
    """Constroi resumo textual da comparacao com no minimo 4 metricas.

    Args:
        results: Lista de ModelResult ordenados por performance.
        k: Valor de K para comparacao.
        comparison_metrics: Tupla com nomes das metricas canonicas.

    Returns:
        String com resumo comparativo.
    """
    if not results:
        return "Nenhum modelo avaliado."

    champion, runner_up = declare_champion(results, k)
    lines = [
        f"Comparacao de modelos com {len(comparison_metrics)}+ metricas "
        f"(K={k}): {', '.join(m.upper() for m in comparison_metrics)}",
        "",
    ]

    for result in results:
        metric_strs = [
            f"{m}@{k}={result.metric_at_k(m, k):.4f}" for m in comparison_metrics
        ]
        lines.append(
            f"  {result.model_name} ({result.model_role}): "
            f"{', '.join(metric_strs)} "
            f"H-Mean@{k}={result.harmonic_mean_at_k(k):.4f} "
            f"treino={result.train_time_sec:.2f}s "
            f"infer={result.infer_time_sec:.2f}s"
        )

    if champion is not None:
        lines.append("")
        h_mean = champion.harmonic_mean_at_k(k)
        lines.append(f"Campeao: {champion.model_name} (H-Mean@{k}={h_mean:.4f})")
        if runner_up is not None:
            gain = champion.harmonic_mean_at_k(k) - runner_up.harmonic_mean_at_k(k)
            lines.append(f"Vantagem sobre {runner_up.model_name}: +{gain:.4f}")

    return "\n".join(lines)


def run_compare_pipeline(  # noqa: PLR0913
    data_dir: str | Path = "data/raw",
    test_ratio: float = 0.15,
    random_seed: int = 42,
    skip_ease: bool = False,
    experiment_name: str | None = None,
    k_values: tuple[int, ...] = K_VALUES,
) -> pd.DataFrame:
    """Executa pipeline completa de comparacao de modelos com MLflow.

    Args:
        data_dir: Diretorio com dados brutos.
        test_ratio: Fracao de interacoes para teste.
        random_seed: Seed para reprodutibilidade.
        skip_ease: Se True, pula o EASE^.
        experiment_name: Nome do experimento MLflow.
        k_values: Valores de K para as metricas.

    Returns:
        DataFrame comparativo com metricas por modelo.
    """
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_global_seed(random_seed)

    # Configura MLflow
    exp_name = get_experiment_name(
        cli_arg=experiment_name,
        env_var_name="MLFLOW_COMPARISON_EXPERIMENT_NAME",
        default_name=DEFAULT_EXPERIMENT_NAME,
    )
    mlflow_config = MLflowConfig(experiment_name=exp_name)
    setup_mlflow(mlflow_config)

    # Carrega dados
    interactions_df = load_interactions(data_dir)

    # Divide treino/teste
    train_interactions, ground_truth, _ = temporal_holdout_split(
        interactions_df, test_ratio=test_ratio, random_seed=random_seed
    )

    # Obtem versao do dataset
    dataset_version = safe_get_dataset_version()

    # Conta popularidade dos itens para grafico
    item_counts = Counter(item_id for _, item_id in train_interactions)

    # Treina e avalia todos os modelos
    model_results, trained_models = evaluate_all_models(
        train_interactions, ground_truth, k_values, skip_ease, random_seed
    )

    # Constroi resumo dos dados de entrada
    input_data_summary = build_input_data_summary(
        interactions_df=interactions_df,
        train_interactions=train_interactions,
        ground_truth=ground_truth,
        test_ratio=test_ratio,
        dataset_version=dataset_version,
        random_seed=random_seed,
    )

    # Registra cada modelo como uma run separada no MLflow
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_data: list[dict[str, Any]] = []

    for result in model_results:
        model = trained_models[result.model_name]
        metrics = result.metrics

        with mlflow.start_run(run_name=f"compare_{result.model_name}"):
            # Log de hiperparametros
            log_hyperparameters(
                {
                    "model_type": result.model_name,
                    "recommendation_limit": max(k_values),
                    "random_seed": random_seed,
                    "test_ratio": test_ratio,
                    "dataset_version": dataset_version,
                    "num_train_interactions": len(train_interactions),
                    "num_evaluated_users": len(ground_truth),
                    "k_values": str(k_values),
                    "architecture": type(model).__name__,
                    "model_role": result.model_role,
                    "train_time_sec": result.train_time_sec,
                    "infer_time_sec": result.infer_time_sec,
                }
            )

            # Log de informacoes de sistema
            log_system_info(random_seed)

            # Log de metricas (sanitiza nomes para MLflow: '@' -> '_at_')
            log_metrics(sanitize_metric_names(metrics))

            # Log do resumo dos dados de entrada como artefato
            log_input_data_summary(input_data_summary)

            # Log do modelo como artefato
            log_recommender_model(model, result.model_name)

            # Tags
            mlflow.set_tag("issue", "15")
            mlflow.set_tag("pipeline", "run_compare_models")
            mlflow.set_tag("model_role", result.model_role)
            mlflow.set_tag("baseline_family", result.model_name)
            mlflow.set_tag("random_seed", str(random_seed))
            mlflow.set_tag("comparison_metrics", ",".join(COMPARISON_METRICS))

            # Salva e log graficos de metricas
            metrics_chart_path = (
                reports_dir / f"comparison_metrics_{result.model_name}.png"
            )
            save_metrics_bar_chart(
                metrics,
                metrics_chart_path,
                title=f"Comparison Metrics - {result.model_name}",
            )
            log_artifacts([metrics_chart_path])

            # Salva e log grafico de popularidade
            popularity_chart_path = (
                reports_dir / f"comparison_popularity_{result.model_name}.png"
            )
            save_item_popularity_distribution(
                dict(item_counts.most_common(30)),
                popularity_chart_path,
                top_n=30,
                title=f"Item Popularity - {result.model_name}",
            )
            log_artifacts([popularity_chart_path])

            # Model Card
            card = build_model_card(
                result.model_name,
                random_seed=random_seed,
                dataset_version=dataset_version,
                num_evaluated_users=len(ground_truth),
                **metrics,
            )
            mlflow.log_dict(card, "model_card.json")

        comparison_data.append(result.to_comparison_dict())
        logger.info("Run MLflow registrada para %s", result.model_name)

    # Constroi dict de metricas por modelo para grafico comparativo
    all_metrics_dict = {r.model_name: r.metrics for r in model_results}

    # Salva e log grafico comparativo
    if len(model_results) > 1:
        comparison_chart_path = reports_dir / "model_comparison.png"
        metric_keys = [
            f"{m}@{k}" for k in k_values for m in COMPARISON_METRICS + ("hit_rate",)
        ]
        save_model_comparison_chart(
            all_metrics_dict,
            metric_keys=metric_keys,
            output_path=comparison_chart_path,
            title="Model Comparison - Baselines vs Champion",
        )

    # Salva CSV comparativo
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(
        "harmonic_mean_at_10", ascending=False
    ).reset_index(drop=True)
    comparison_path = Path("models") / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Comparativo salvo em: %s", comparison_path)

    # Declara o campeao
    champion, runner_up = declare_champion(model_results, k=CHAMPION_K)

    # Imprime resumo comparativo
    summary = build_comparison_summary(model_results, k=CHAMPION_K)
    print(summary)

    if champion is not None:
        logger.info("Campeao declarado: %s", champion.model_name)
        if runner_up is not None:
            logger.info("Segundo colocado: %s", runner_up.model_name)

    # Gera relatorio markdown automatico
    report_content = generate_markdown_report(
        results=model_results,
        k_values=k_values,
        champion_k=CHAMPION_K,
        num_users=int(interactions_df["user_id"].nunique()),
        num_items=int(interactions_df["item_id"].nunique()),
        num_interactions=len(interactions_df),
        num_evaluated_users=len(ground_truth),
        dataset_name="RetailRocket E-Commerce",
        split_strategy="chronological_3way",
        test_ratio=test_ratio,
        val_ratio=test_ratio,
        random_seed=random_seed,
    )
    report_path = save_markdown_report(
        report_content,
        Path("reports") / "model_comparison_report.md",
    )
    logger.info("Relatorio comparativo salvo em: %s", report_path)

    # Log resumo
    logger.info(
        "Pipeline de comparacao concluida com %d modelos avaliados",
        len(model_results),
    )
    for result in model_results:
        logger.info(
            "  %s: precision@10=%.4f recall@10=%.4f ndcg@10=%.4f "
            "map@10=%.4f harmonic@10=%.4f",
            result.model_name,
            result.metric_at_k("precision", 10),
            result.metric_at_k("recall", 10),
            result.metric_at_k("ndcg", 10),
            result.metric_at_k("map", 10),
            result.harmonic_mean_at_k(CHAMPION_K),
        )

    return comparison_df


def main() -> int:
    """Ponto de entrada do script de comparacao.

    Returns:
        Codigo de saida (0 para sucesso).
    """
    args = parse_args()
    try:
        comparison_df = run_compare_pipeline(
            data_dir=args.data_dir,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            skip_ease=args.skip_ease,
            experiment_name=args.experiment_name,
        )
        print("\nResultados comparativos:")
        print(comparison_df.to_string(index=False))
        return 0
    except FileNotFoundError as e:
        logger.error("Arquivo nao encontrado: %s", e)
        return 1
    except Exception:
        logger.exception("Erro na execucao da pipeline de comparacao")
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    raise SystemExit(main())
