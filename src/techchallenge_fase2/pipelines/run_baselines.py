"""Pipeline de avaliação dos modelos baseline com tracking MLflow.

Orquestra o fluxo completo:
1. Carrega configuração e ambiente
2. Carrega dados de interações
3. Divide em treino/teste (temporal holdout)
4. Treina e avalia modelos baseline (popularity, recent_items)
5. Registra hiperparâmetros, métricas e artefatos no MLflow
6. Salva resultados comparativos
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from techchallenge_fase2.models import ModelConfig
from techchallenge_fase2.models.base import Interaction
from techchallenge_fase2.models.factory import RecommenderModelFactory
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
    safe_get_dataset_version,
    set_global_seed,
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

DEFAULT_EXPERIMENT_NAME = "tech-challenge-baselines"
K_VALUES = (5, 10, 20)


def build_input_data_summary(
    interactions_df: pd.DataFrame,
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    test_ratio: float,
    dataset_version: str,
    random_seed: int,
) -> dict[str, Any]:
    """Constrói resumo estatístico dos dados de entrada para artefato MLflow.

    Args:
        interactions_df: DataFrame completo de interações carregadas.
        train_interactions: Interações reservadas para treino.
        ground_truth: Itens relevantes por usuário no teste.
        test_ratio: Fração reservada para teste.
        dataset_version: Versão do dataset via DVC.
        random_seed: Seed usado na divisão treino/teste.

    Returns:
        Dicionário JSON-serializável com estatísticas dos dados.
    """
    num_users = int(interactions_df["user_id"].nunique())
    num_items = int(interactions_df["item_id"].nunique())
    num_interactions = int(len(interactions_df))
    if num_users and num_items:
        sparsity = 1.0 - (num_interactions / (num_users * num_items))
    else:
        sparsity = 1.0

    interactions_per_user = interactions_df.groupby("user_id").size()
    interactions_per_item = interactions_df.groupby("item_id").size()

    return {
        "dataset": "RetailRocket E-Commerce",
        "dataset_version": dataset_version,
        "split_strategy": "temporal_holdout",
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "num_total_interactions": num_interactions,
        "num_train_interactions": len(train_interactions),
        "num_test_interactions": int(
            sum(len(items) for items in ground_truth.values())
        ),
        "num_users": num_users,
        "num_items": num_items,
        "num_evaluated_users": len(ground_truth),
        "sparsity": float(sparsity),
        "interactions_per_user": {
            "min": int(interactions_per_user.min()),
            "max": int(interactions_per_user.max()),
            "mean": float(interactions_per_user.mean()),
            "median": float(interactions_per_user.median()),
        },
        "interactions_per_item": {
            "min": int(interactions_per_item.min()),
            "max": int(interactions_per_item.max()),
            "mean": float(interactions_per_item.mean()),
            "median": float(interactions_per_item.median()),
        },
    }


def load_interactions(data_dir: str | Path) -> pd.DataFrame:
    """Carrega interações do dataset RetailRocket.

    Procura por arquivos CSV com colunas de interação user-item
    no diretório data/raw/.

    Args:
        data_dir: Caminho para o diretório de dados brutos.

    Returns:
        DataFrame com colunas mínimas de interação.

    Raises:
        FileNotFoundError: Se nenhum arquivo de interações for encontrado.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        msg = f"Nenhum arquivo CSV encontrado em {data_path}"
        raise FileNotFoundError(msg)

    # Tenta carregar o primeiro CSV encontrado
    df = pd.read_csv(csv_files[0])
    logger.info(
        "Carregado %s com %d linhas e %d colunas",
        csv_files[0],
        len(df),
        len(df.columns),
    )

    # Identifica colunas de interação
    # RetailRocket: visitorid, itemid, event
    possible_user_cols = ["visitorid", "user_id", "userId", "user"]
    possible_item_cols = ["itemid", "item_id", "itemId", "item"]

    user_col = None
    item_col = None

    for col in possible_user_cols:
        if col in df.columns:
            user_col = col
            break

    for col in possible_item_cols:
        if col in df.columns:
            item_col = col
            break

    if user_col is None or item_col is None:
        # Fallback: usa as duas primeiras colunas como user e item
        if len(df.columns) >= 2:
            user_col = df.columns[0]
            item_col = df.columns[1]
            logger.info(
                "Colunas não identificadas, usando %s como user e %s como item",
                user_col,
                item_col,
            )
        else:
            msg = (
                f"Não foi possível identificar colunas de "
                f"interação em {df.columns.tolist()}"
            )
            raise ValueError(msg)

    result = df[[user_col, item_col]].copy()
    result.columns = ["user_id", "item_id"]

    # Converte para string para compatibilidade com baselines
    result["user_id"] = result["user_id"].astype(str)
    result["item_id"] = result["item_id"].astype(str)

    # Remove duplicatas (mesmo user-item pode ter múltiplas interações)
    result = result.drop_duplicates()

    logger.info(
        "Interações carregadas: %d usuários, %d itens",
        result["user_id"].nunique(),
        result["item_id"].nunique(),
    )

    return result


def temporal_holdout_split(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[Interaction], dict[str, set[str]], dict[str, list[str]]]:
    """Divide interações em treino e teste usando holdout temporal.

    Para cada usuário, separa uma fração das interações como teste
    e usa o restante como treino. Usuários com apenas 1 interação
    são mantidos integralmente no treino.

    Args:
        interactions_df: DataFrame com colunas 'user_id' e 'item_id'.
        test_ratio: Fração de interações a reservar para teste.
        random_seed: Seed para reprodutibilidade.

    Returns:
        Tupla com:
        - train_interactions: lista de tuplas (user_id, item_id) de treino
        - ground_truth: mapeamento user_id -> conjunto de itens relevantes
        - all_items_by_user: mapeamento user_id -> lista de itens do usuário
    """
    rng = np.random.RandomState(random_seed)

    train_interactions: list[Interaction] = []
    ground_truth: dict[str, set[str]] = {}
    all_items_by_user: dict[str, list[str]] = {}

    for user_id, group in interactions_df.groupby("user_id"):
        items = group["item_id"].tolist()
        all_items_by_user[user_id] = items

        if len(items) <= 1:
            # Usuários com 1 interação: tudo no treino
            for item_id in items:
                train_interactions.append((str(user_id), item_id))
            continue

        # Embaralha e divide
        items_shuffled = items.copy()
        rng.shuffle(items_shuffled)
        n_test = max(1, int(len(items_shuffled) * test_ratio))
        test_items = items_shuffled[:n_test]
        train_items = items_shuffled[n_test:]

        for item_id in train_items:
            train_interactions.append((str(user_id), item_id))

        ground_truth[str(user_id)] = set(test_items)

    return train_interactions, ground_truth, all_items_by_user


def evaluate_baselines(  # noqa: PLR0913
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    k_values: tuple[int, ...] = K_VALUES,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Treina e avalia todos os modelos baseline.

    Args:
        train_interactions: Interações de treino.
        ground_truth: Itens relevantes por usuário para avaliação.
        k_values: Valores de K para computar métricas.
        random_seed: Seed para reprodutibilidade.

    Returns:
        Dicionário mapeando nome do modelo para métricas.
    """
    factory = RecommenderModelFactory.default()

    # Gera recomendações para cada modelo e cada K
    model_names = list(factory.available_types())
    all_results: dict[str, dict[str, float]] = {}

    for model_name in model_names:
        config = ModelConfig(
            model_type=model_name,
            recommendation_limit=max(k_values),
        )
        model = factory.create(config)
        model.fit(train_interactions)

        # Gera recomendações para cada usuário no ground truth
        all_recommended: dict[str, list[str]] = {}
        for user_id in ground_truth:
            recommendations = model.recommend(user_id, limit=max(k_values))
            all_recommended[user_id] = recommendations

        # Computa métricas
        metrics = compute_recommender_metrics(ground_truth, all_recommended, k_values)

        all_results[model_name] = metrics

        logger.info(
            "Modelo %s: precision@5=%.4f "
            "recall@5=%.4f ndcg@5=%.4f "
            "map@5=%.4f hit_rate@5=%.4f",
            model_name,
            metrics.get("precision@5", 0.0),
            metrics.get("recall@5", 0.0),
            metrics.get("ndcg@5", 0.0),
            metrics.get("map@5", 0.0),
            metrics.get("hit_rate@5", 0.0),
        )

    return all_results


def run_baseline_pipeline(  # noqa: PLR0913
    data_dir: str | Path = "data/raw",
    test_ratio: float = 0.2,
    random_seed: int = 42,
    experiment_name: str | None = None,
    k_values: tuple[int, ...] = K_VALUES,
) -> pd.DataFrame:
    """Executa pipeline completa de avaliação dos baselines com MLflow.

    Args:
        data_dir: Diretório com dados brutos.
        test_ratio: Fração de interações para teste.
        random_seed: Seed para reprodutibilidade.
        experiment_name: Nome do experimento MLflow.
        k_values: Valores de K para as métricas.

    Returns:
        DataFrame comparativo com métricas por modelo.
    """
    # Carrega ambiente e configura logging
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Define seed global
    set_global_seed(random_seed)

    # Configura MLflow
    exp_name = get_experiment_name(
        cli_arg=experiment_name,
        env_var_name="MLFLOW_EXPERIMENT_NAME",
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

    # Obtém versão do dataset
    dataset_version = safe_get_dataset_version()

    # Conta popularidade dos itens para gráfico
    item_counts = Counter(item_id for _, item_id in train_interactions)

    # Treina e avalia baselines
    all_results = evaluate_baselines(
        train_interactions, ground_truth, k_values, random_seed
    )

    # Constrói resumo dos dados de entrada para log como artefato
    input_data_summary = build_input_data_summary(
        interactions_df=interactions_df,
        train_interactions=train_interactions,
        ground_truth=ground_truth,
        test_ratio=test_ratio,
        dataset_version=dataset_version,
        random_seed=random_seed,
    )

    # Registra cada modelo como uma run separada no MLflow
    factory = RecommenderModelFactory.default()
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_data: list[dict[str, Any]] = []

    for model_name, metrics in all_results.items():
        config = ModelConfig(
            model_type=model_name,
            recommendation_limit=max(k_values),
        )
        model = factory.create(config)
        model.fit(train_interactions)

        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            # Log de hiperparâmetros
            log_hyperparameters(
                {
                    "model_type": model_name,
                    "recommendation_limit": max(k_values),
                    "random_seed": random_seed,
                    "test_ratio": test_ratio,
                    "dataset_version": dataset_version,
                    "num_train_interactions": len(train_interactions),
                    "num_evaluated_users": len(ground_truth),
                    "k_values": str(k_values),
                    "architecture": type(model).__name__,
                }
            )

            # Log de informações de sistema
            log_system_info(random_seed)

            # Log de métricas
            log_metrics(metrics)

            # Log do resumo dos dados de entrada como artefato
            log_input_data_summary(input_data_summary)

            # Log do modelo como artefato
            log_recommender_model(model, model_name)

            # Tags
            mlflow.set_tag("issue", "13")
            mlflow.set_tag("baseline_family", model_name)
            mlflow.set_tag("model_baseline", model_name)
            mlflow.set_tag("random_seed", str(random_seed))

            # Salva e log gráficos de métricas
            metrics_chart_path = reports_dir / f"metrics_{model_name}.png"
            save_metrics_bar_chart(
                metrics,
                metrics_chart_path,
                title=f"Metrics - {model_name}",
            )
            log_artifacts([metrics_chart_path])

            # Salva e log gráfico de popularidade
            popularity_chart_path = reports_dir / f"popularity_{model_name}.png"
            save_item_popularity_distribution(
                dict(item_counts.most_common(30)),
                popularity_chart_path,
                top_n=30,
                title=f"Item Popularity - {model_name}",
            )
            log_artifacts([popularity_chart_path])

            # Model Card
            card = build_model_card(
                model_name,
                random_seed=random_seed,
                dataset_version=dataset_version,
                num_evaluated_users=len(ground_truth),
                **metrics,
            )
            mlflow.log_dict(card, "model_card.json")

        comparison_data.append({"model": model_name, **metrics})
        logger.info("Run MLflow registrada para %s", model_name)

    # Salva e log gráfico comparativo
    if len(all_results) > 1:
        comparison_chart_path = reports_dir / "model_comparison.png"
        metric_keys = [
            f"{m}@{k}"
            for k in k_values
            for m in ("precision", "recall", "ndcg", "map", "hit_rate")
        ]
        save_model_comparison_chart(
            all_results,
            metric_keys=metric_keys,
            output_path=comparison_chart_path,
            title="Baseline Model Comparison",
        )

    # Salva CSV comparativo
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = Path("models") / "baseline_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Comparativo salvo em: %s", comparison_path)

    # Log resumo
    logger.info(
        "Pipeline concluída com %d modelos avaliados",
        len(all_results),
    )
    for model_name, metrics in all_results.items():
        logger.info(
            "  %s: precision@5=%.4f recall@5=%.4f ndcg@5=%.4f",
            model_name,
            metrics.get("precision@5", 0.0),
            metrics.get("recall@5", 0.0),
            metrics.get("ndcg@5", 0.0),
        )

    return comparison_df


def main() -> int:
    """Ponto de entrada do script.

    Executa a pipeline de avaliação dos baselines com MLflow.

    Returns:
        Código de saída (0 para sucesso).
    """
    try:
        comparison_df = run_baseline_pipeline()
        print("\nResultados comparativos:")
        print(comparison_df.to_string(index=False))
        return 0
    except FileNotFoundError as e:
        logger.error("Arquivo não encontrado: %s", e)
        return 1
    except Exception:
        logger.exception("Erro na execução da pipeline de baselines")
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    raise SystemExit(main())
