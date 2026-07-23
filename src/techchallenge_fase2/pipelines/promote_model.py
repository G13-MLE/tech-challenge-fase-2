"""Valida o modelo em Staging e promove para Production.

Implementa a etapa de validacao exigida pela issue #16 antes da
promocao para Production:

1. Carrega a versao em Staging do Model Registry.
2. Reavalia as recomendacoes num conjunto de teste derivado do
   ``data/features/test.parquet`` splitado cronologicamente.
3. Compara a media harmonica @10 com a referencia registrada em
   ``models/model_comparison.csv``.
4. Se dentro da tolerancia configuravel, promove para Production
   (arquivando versoes anteriores).

Uso:
    $ uv run python -m techchallenge_fase2.pipelines.promote_model
    $ uv run python -m techchallenge_fase2.pipelines.promote_model \\
        --model-name TechChallengeFase2Recommender --tolerance 0.05
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
)
from techchallenge_fase2.pipelines.register_model import (
    DEFAULT_MODEL_NAME,
    PRODUCTION_STAGE,
    STAGING_STAGE,
    transition_to_stage,
)
from techchallenge_fase2.pipelines.run_baselines import (
    K_VALUES,
    load_interactions,
    temporal_holdout_split,
)
from techchallenge_fase2.training.metrics import compute_recommender_metrics
from techchallenge_fase2.training.mlflow_tracking import MLflowConfig, setup_mlflow

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.05
DEFAULT_TEST_PATH = "data/features/test.parquet"
DEFAULT_REFERENCE_CSV = "models/model_comparison.csv"
REGISTRY_EXPERIMENT_ENV = "MLFLOW_REGISTRY_EXPERIMENT_NAME"
DEFAULT_REGISTRY_EXPERIMENT = "tech-challenge-registry"


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
        help=f"Nome do modelo no Registry (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=float(
            os.getenv("MLFLOW_REGISTRY_STAGING_TOLERANCE", DEFAULT_TOLERANCE)
        ),
        help="Tolerancia relativa aceitavel vs referencia (default: 0.05)",
    )
    parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help="Parquet de teste para reavaliar o modelo em Staging",
    )
    parser.add_argument(
        "--reference-csv",
        default=DEFAULT_REFERENCE_CSV,
        help="CSV com a metrica de referencia do campeao",
    )
    parser.add_argument(
        "--registry-experiment",
        default=None,
        help="Experimento MLflow de registro (override)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Avalia sem promover para Production",
    )
    return parser.parse_args()


def load_staging_model(model_name: str) -> tuple[Any, str]:
    """Carrega a versao atual em Staging do modelo.

    Args:
        model_name: Nome do modelo no Registry.

    Returns:
        Tupla (modelo_pyfunc, version) onde ``modelo_pyfunc`` segue
        o contrato ``predict`` do ``RecommenderPythonModel``.

    Raises:
        RuntimeError: Se nao houver versao em Staging.
    """
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[STAGING_STAGE])
    if not versions:
        msg = (
            f"Nenhuma versao de '{model_name}' em {STAGING_STAGE}. "
            "Rode 'make register' antes de 'make promote'."
        )
        raise RuntimeError(msg)
    version = versions[0]
    model_uri = f"models:/{model_name}/{STAGING_STAGE}"
    logger.info(
        "Carregando %s versao %s (%s)",
        model_name,
        version.version,
        STAGING_STAGE,
    )
    return mlflow.pyfunc.load_model(model_uri), version.version


def build_test_ground_truth(test_path: str) -> tuple[Any, dict[str, set[str]]]:
    """Constroi interacoes de treino e ground truth de teste.

    Reusa o ``data/raw`` + ``temporal_holdout_split`` para gerar um
    conjunto de avaliacao coerente com o pipeline de comparacao.

    Args:
        test_path: Caminho do parquet de teste (usado so para validar
            a existencia; o split reexecuta a partir do raw).

    Returns:
        Tupla (train_interactions, ground_truth).

    Raises:
        FileNotFoundError: Se ``test_path`` ou ``data/raw`` ausente.
    """
    if not Path(test_path).exists():
        msg = f"Arquivo de teste ausente: {test_path}. Rode 'dvc pull'."
        raise FileNotFoundError(msg)
    interactions_df = load_interactions("data/raw")
    train_interactions, ground_truth, _ = temporal_holdout_split(interactions_df)
    return train_interactions, ground_truth


def predict_batch(
    model: Any,
    ground_truth: dict[str, set[str]],
    limit: int,
) -> dict[str, list[str]]:
    """Gera recomendacoes em late chamando o pyfunc carregado.

    Args:
        model: Modelo pyfunc carregado via ``mlflow.pyfunc.load_model``.
        ground_truth: Mapeamento user_id -> itens relevantes.
        limit: Numero de recomendacoes por usuario.

    Returns:
        Dicionario user_id -> lista de itens recomendados.
    """
    frame = pd.DataFrame(
        {"user_id": list(ground_truth), "limit": [limit] * len(ground_truth)}
    )
    result = model.predict(frame)
    return {
        str(row.user_id): list(row.recommendations)
        for row in result.itertuples(index=False)
    }


def evaluate_staging(model: Any, ground_truth: dict[str, set[str]]) -> float:
    """Avalia o modelo em Staging e devolve a media harmonica @10.

    Args:
        model: Modelo pyfunc carregado.
        ground_truth: Itens relevantes por usuario.

    Returns:
        Valor da metrica ``harmonic_mean_at_10``.
    """
    recommended = predict_batch(model, ground_truth, limit=max(K_VALUES))
    metrics = compute_recommender_metrics(ground_truth, recommended, K_VALUES)
    return harmonic_mean_at_k(metrics, k=10)


def harmonic_mean_at_k(metrics: dict[str, float], k: int = 10) -> float:
    """Computa a media harmonica das 4 metricas canonicas em K.

    Args:
        metrics: Dicionario com chaves no formato ``metrica@K``
            (ex: ``precision@10``).
        k: Valor de K para o calculo.

    Returns:
        Media harmonica de precision/recall/ndcg/map@K. Retorna 0.0
        se alguma metrica for ausente ou zero.
    """
    keys = (f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"map@{k}")
    values = [metrics.get(key) for key in keys]
    if any(v is None for v in values):
        return 0.0
    numeric = [float(v) for v in values if v is not None]
    if any(v <= 0 for v in numeric):
        return 0.0
    return len(numeric) / sum(1.0 / v for v in numeric)


def load_reference_harmonic_mean(csv_path: str) -> float | None:
    """Le a media harmonica @10 do campeao no CSV comparativo.

    Args:
        csv_path: Caminho do CSV ``model_comparison.csv``.

    Returns:
        Valor da media harmonica do primeiro modelo (campeao) ou None.
    """
    path = Path(csv_path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame.sort_values("harmonic_mean_at_10", ascending=False)
    if frame.empty:
        return None
    return float(frame.iloc[0]["harmonic_mean_at_10"])


def validate_and_promote(
    model_name: str,
    tolerance: float,
    test_path: str,
    reference_csv: str,
    dry_run: bool,
) -> str:
    """Valida a versao em Staging e promove para Production se aprovada.

    Args:
        model_name: Nome do modelo no Registry.
        tolerance: Tolerancia relativa aceitavel.
        test_path: Caminho do parquet de teste.
        reference_csv: CSV comparativo com a metrica de referencia.
        dry_run: Se True, nao promove para Production.

    Returns:
        Versao avaliada.
    """
    model, version = load_staging_model(model_name)
    _, ground_truth = build_test_ground_truth(test_path)
    observed = evaluate_staging(model, ground_truth)
    reference = load_reference_harmonic_mean(reference_csv)

    logger.info(
        "Staging %s versao %s: harmonic_mean@10=%.4f (referencia=%s)",
        model_name,
        version,
        observed,
        f"{reference:.4f}" if reference is not None else "indisponivel",
    )

    if reference is not None:
        if reference <= 0:
            msg = "Referencia de harmonic_mean@10 e zero; abortando promocao."
            raise RuntimeError(msg)
        relative_diff = abs(observed - reference) / reference
        if relative_diff > tolerance:
            msg = (
                f"Validacao falhou: |{observed:.4f} - {reference:.4f}|/"
                f"{reference:.4f} = {relative_diff:.4%} > tolerancia {tolerance:.4%}."
            )
            raise RuntimeError(msg)
        logger.info("Validacao aprovada (tolerancia %.2f%%).", tolerance * 100)

    if dry_run:
        logger.info("Dry-run: sem promocao para Production.")
        return version

    client = mlflow.tracking.MlflowClient()
    transition_to_stage(
        client, model_name, version, PRODUCTION_STAGE, archive_existing=True
    )
    logger.info(
        "Modelo %s versao %s promovido para %s.",
        model_name,
        version,
        PRODUCTION_STAGE,
    )
    return version


def run(
    model_name: str,
    tolerance: float,
    test_path: str,
    reference_csv: str,
    dry_run: bool,
    registry_experiment_name: str | None,
) -> str:
    """Executa validacao em Staging e promocao para Production.

    Args:
        model_name: Nome do modelo no Registry.
        tolerance: Tolerancia relativa aceitavel.
        test_path: Caminho do parquet de teste.
        reference_csv: CSV comparativo com a metrica de referencia.
        dry_run: Se True, avalia sem promover.
        registry_experiment_name: Nome do experimento MLflow de registro.

    Returns:
        Versao do modelo avaliada/promovida.
    """
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    registry_exp = get_experiment_name(
        cli_arg=registry_experiment_name,
        env_var_name=REGISTRY_EXPERIMENT_ENV,
        default_name=DEFAULT_REGISTRY_EXPERIMENT,
    )
    setup_mlflow(MLflowConfig(experiment_name=registry_exp))
    return validate_and_promote(
        model_name, tolerance, test_path, reference_csv, dry_run
    )


def main() -> int:
    """Ponto de entrada do script de promocao.

    Returns:
        Codigo de saida (0 sucesso, 1 erro).
    """
    args = parse_args()
    try:
        version = run(
            model_name=args.model_name,
            tolerance=args.tolerance,
            test_path=args.test_path,
            reference_csv=args.reference_csv,
            dry_run=args.dry_run,
            registry_experiment_name=args.registry_experiment,
        )
        print(f"Versao avaliada: {version}")
        return 0
    except Exception:
        logger.exception("Erro na promocao para Production")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
