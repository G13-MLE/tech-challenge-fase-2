"""Pipeline de avaliacao dos modelos baseline com tracking MLflow.

Orquestra o fluxo completo:
1. Carrega configuracao e ambiente
2. Carrega dados de interacoes
3. Divide em treino/validacao/teste (split cronologico global 70/15/15)
4. Treina e avalia todos os modelos (popularity, recent_items, random,
   torch_embedding, neural_ncf, ease_torch, item_knn, logistic_regression)
5. Registra hiperparametros, metricas e artefatos no MLflow
6. Salva resultados comparativos com criterio de campeao
7. Gera relatorio markdown automatico
"""

from __future__ import annotations

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
from techchallenge_fase2.pipelines.splits import (
    chronological_holdout_split,
    chronological_train_val_test_split,
    filter_warm_start,
    filter_warm_start_three_way,
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
CHAMPION_K = 10
CHAMPION_MIN_RELATIVE_GAIN = 0.01

# Tag de papel de cada modelo na comparacao
MODEL_ROLES: dict[str, str] = {
    ModelType.POPULARITY.value: "baseline",
    ModelType.RECENT_ITEMS.value: "baseline",
    ModelType.RANDOM.value: "baseline",
    ModelType.TORCH_EMBEDDING.value: "baseline",
    ModelType.NEURAL_NCF.value: "baseline_neural",
    ModelType.ITEM_KNN.value: "baseline",
    ModelType.LOGISTIC_REGRESSION.value: "baseline",
    ModelType.EASE_TORCH.value: "champion_candidate",
}

# Hiperparametros especificos do EASE^ para o candidato a campeao
EASE_HYPERPARAMS: dict[str, Any] = {
    "lambda_reg": 250.0,
    "max_items": 20000,
    "batch_size": 1000,
    "popularity_blending": 0.0,
}


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
        "split_strategy": "chronological_3way",
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

    # Prioriza arquivos de eventos (interacoes user-item) sobre outros CSVs
    event_files = [f for f in csv_files if "event" in f.name.lower()]
    chosen_file = event_files[0] if event_files else csv_files[0]
    df = pd.read_csv(chosen_file)
    logger.info(
        "Carregado %s com %d linhas e %d colunas",
        chosen_file,
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

    # Preserva timestamp para split cronologico se disponivel
    possible_time_cols = ["timestamp", "event_time", "time", "datetime", "ts"]
    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    keep_cols = [user_col, item_col]
    rename_map = {user_col: "user_id", item_col: "item_id"}
    if time_col is not None:
        keep_cols.append(time_col)
        rename_map[time_col] = "timestamp"

    result = df[keep_cols].copy()
    result = result.rename(columns=rename_map)

    # Converte para string para compatibilidade com baselines
    result["user_id"] = result["user_id"].astype(str)
    result["item_id"] = result["item_id"].astype(str)

    # Remove duplicatas (mesmo user-item pode ter múltiplas interações)
    # mantendo a primeira ocorrencia cronologica se timestamp presente
    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").drop_duplicates(
            subset=["user_id", "item_id"], keep="first"
        )
    else:
        result = result.drop_duplicates()

    logger.info(
        "Interações carregadas: %d usuários, %d itens",
        result["user_id"].nunique(),
        result["item_id"].nunique(),
    )

    return result


def temporal_holdout_split(
    interactions_df: pd.DataFrame,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    timestamp_col: str = "timestamp",
) -> tuple[list[Interaction], dict[str, set[str]], dict[str, list[str]]]:
    """Divide interacoes em treino/validacao/teste com corte cronologico global.

    Usa split cronologico 3-way (70/15/15 por padrao) conforme o plano da
    issue #15. O conjunto de validacao fica disponivel para selecao de
    hiperparametros; a avaliacao final usa o conjunto de teste.

    Mantem compatibilidade com a assinatura anterior.

    Args:
        interactions_df: DataFrame com colunas 'user_id', 'item_id' e
            timestamp_col.
        test_ratio: Fracao das interacoes mais recentes para teste.
        random_seed: Seed mantido por compatibilidade (nao usado no split
            cronologico deterministico).
        timestamp_col: Nome da coluna de timestamp para ordenacao.

    Returns:
        Tupla com:
        - train_interactions: lista de tuplas (user_id, item_id) de treino
        - ground_truth: mapeamento user_id -> conjunto de itens relevantes
        - all_items_by_user: mapeamento user_id -> lista de itens do usuario
    """
    _ = random_seed
    if timestamp_col not in interactions_df.columns:
        split = chronological_holdout_split(interactions_df, test_ratio, timestamp_col)
        split = filter_warm_start(split)
        all_items_by_user = group_items_by_user(interactions_df)
        return (
            split.train_interactions,
            split.ground_truth,
            all_items_by_user,
        )
    val_ratio = test_ratio
    split = chronological_train_val_test_split(
        interactions_df, val_ratio=val_ratio, test_ratio=test_ratio
    )
    split = filter_warm_start_three_way(split)
    all_items_by_user = group_items_by_user(interactions_df)
    return (
        split.train_interactions,
        split.test_ground_truth,
        all_items_by_user,
    )


def group_items_by_user(df: pd.DataFrame) -> dict[str, list[str]]:
    """Agrupa itens por usuario mantendo a ordem de ocorrencia."""
    grouped: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(str(row.user_id), []).append(str(row.item_id))
    return grouped


def evaluate_baselines(  # noqa: PLR0913
    train_interactions: list[Interaction],
    ground_truth: dict[str, set[str]],
    k_values: tuple[int, ...] = K_VALUES,
    random_seed: int = 42,
) -> tuple[list[ModelResult], dict[str, Any]]:
    """Treina e avalia todos os modelos baseline e o candidato a campeao.

    Args:
        train_interactions: Interacoes de treino.
        ground_truth: Itens relevantes por usuario para avaliacao.
        k_values: Valores de K para computar metricas.
        random_seed: Seed para reprodutibilidade.

    Returns:
        Tupla com:
        - Lista de ModelResult com resultados estruturados por modelo.
        - Dicionario mapeando nome do modelo para instancia treinada.
    """
    _ = random_seed
    factory = RecommenderModelFactory.default()
    model_names = list(factory.available_types())
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


def build_model_config(model_name: str, k_values: tuple[int, ...]) -> ModelConfig:
    """Constroi a configuracao apropriada para cada modelo."""
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


def encode_role(role: str) -> int:
    """Codifica o papel do modelo como inteiro para serializacao no MLflow."""
    mapping = {"baseline": 0, "baseline_neural": 1, "champion_candidate": 2}
    return mapping.get(role, 0)


def sanitize_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    """Sanitiza nomes de metricas para compatibilidade com MLflow.

    MLflow nao aceita '@' em nomes de metricas; substitui por '_at_'.
    """
    return {key.replace("@", "_at_"): value for key, value in metrics.items()}


def run_baseline_pipeline(  # noqa: PLR0913
    data_dir: str | Path = "data/raw",
    test_ratio: float = 0.15,
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
        env_var_name="MLFLOW_BASELINE_EXPERIMENT_NAME",
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
    model_results, trained_models = evaluate_baselines(
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
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_data: list[dict[str, Any]] = []

    for result in model_results:
        model = trained_models[result.model_name]
        metrics = result.metrics

        with mlflow.start_run(run_name=f"baseline_{result.model_name}"):
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
            mlflow.set_tag("baseline_family", result.model_name)
            mlflow.set_tag("model_baseline", result.model_name)
            mlflow.set_tag("model_role", result.model_role)
            mlflow.set_tag("random_seed", str(random_seed))

            # Salva e log graficos de metricas
            metrics_chart_path = reports_dir / f"metrics_{result.model_name}.png"
            save_metrics_bar_chart(
                metrics,
                metrics_chart_path,
                title=f"Metrics - {result.model_name}",
            )
            log_artifacts([metrics_chart_path])

            # Salva e log grafico de popularidade
            popularity_chart_path = reports_dir / f"popularity_{result.model_name}.png"
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
            f"{m}@{k}"
            for k in k_values
            for m in ("precision", "recall", "ndcg", "map", "hit_rate")
        ]
        save_model_comparison_chart(
            all_metrics_dict,
            metric_keys=metric_keys,
            output_path=comparison_chart_path,
            title="Baseline Model Comparison",
        )

    # Salva CSV comparativo
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(
        "harmonic_mean_at_10", ascending=False
    ).reset_index(drop=True)
    comparison_path = Path("models") / "baseline_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Comparativo salvo em: %s", comparison_path)

    # Declara o campeao com base no criterio de media harmonica em K=10
    champion, runner_up = declare_champion(model_results, k=CHAMPION_K)
    log_champion_summary(champion, runner_up)

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
        Path("reports") / "baseline_comparison_report.md",
    )
    logger.info("Relatorio markdown salvo em: %s", report_path)

    # Log resumo
    logger.info(
        "Pipeline concluida com %d modelos avaliados",
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


def log_champion_summary(
    champion: ModelResult | None, runner_up: ModelResult | None
) -> None:
    """Registra no log o resumo do campeao declarado."""
    if champion is None:
        logger.warning("Nenhum modelo avaliado; campeao nao declarado.")
        return
    logger.info("Campeao declarado: %s", champion.model_name)
    if runner_up is not None:
        logger.info("Segundo colocado: %s", runner_up.model_name)
    if champion.model_name == ModelType.EASE_TORCH.value:
        logger.info("EASE^ confirmado como campeao por simplicidade e performance.")
    else:
        logger.info(
            "EASE^ nao venceu; modelo %s sera o principal do projeto.",
            champion.model_name,
        )


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
