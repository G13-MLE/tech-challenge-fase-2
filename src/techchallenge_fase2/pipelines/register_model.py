"""Pipeline de registro do modelo campeao no MLflow Model Registry.

Orquestra o fluxo exigido pela issue #16:
1. Descobre o campeao do experimento ``tech-challenge-comparison``
   (maior ``harmonic_mean_at_10`` entre os runs ``compare_*``).
2. Carrega o modelo treinado a partir do artefato pickle do run.
3. Empacota como ``RecommenderPythonModel`` (pyfunc) num novo run
   ``register_champion`` do experimento ``tech-challenge-registry``.
4. Registra o modelo no Model Registry com nome configuravel.
5. Promove a versao recem-registrada para ``Staging``.

Uso:
    $ uv run python -m techchallenge_fase2.pipelines.register_model
    $ uv run python -m techchallenge_fase2.pipelines.register_model \\
        --model-name TechChallengeFase2Recommender
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from techchallenge_fase2.inference.mlflow_wrapper import (
    ARTIFACT_FILENAME,
    ARTIFACT_KEY,
    RecommenderPythonModel,
    serialize_recommender,
)
from techchallenge_fase2.models.base import RecommenderModel
from techchallenge_fase2.pipelines.common import (
    get_experiment_name,
    load_dotenv_silent,
)
from techchallenge_fase2.training.mlflow_tracking import (
    MLflowConfig,
    log_system_info,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "tech-challenge-registry"
DEFAULT_MODEL_NAME = "TechChallengeFase2Recommender"
STAGING_STAGE = "Staging"
PRODUCTION_STAGE = "Production"
COMPARISON_EXPERIMENT_ENV = "MLFLOW_COMPARISON_EXPERIMENT_NAME"
DEFAULT_COMPARISON_EXPERIMENT = "tech-challenge-comparison"
CHAMPION_METRIC_KEY = "harmonic_mean_at_10"
CANONICAL_METRIC_KEYS = ("precision_at_10", "recall_at_10", "ndcg_at_10", "map_at_10")


def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando.

    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
        help=f"Nome do modelo no Model Registry (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--comparison-experiment",
        default=None,
        help="Experimento MLflow de comparacao (override)",
    )
    parser.add_argument(
        "--registry-experiment",
        default=None,
        help="Experimento MLflow de registro (override)",
    )
    return parser.parse_args()


def discover_champion_run(
    client: mlflow.tracking.MlflowClient,
    comparison_experiment_name: str,
) -> tuple[str, str, float]:
    """Encontra o run campeao no experimento de comparacao.

    Busca runs ``compare_*`` ordenados por ``harmonic_mean_at_10``
    decrescente e devolve o primeiro com artefato de modelo.

    Args:
        client: Cliente MLflow.
        comparison_experiment_name: Nome do experimento de comparacao.

    Returns:
        Tupla (run_id, model_name, harmonic_mean_at_10) do campeao.

    Raises:
        RuntimeError: Se o experimento nao existir ou nenhum run elegivel.
    """
    experiment = client.get_experiment_by_name(comparison_experiment_name)
    if experiment is None:
        msg = (
            f"Experimento de comparacao '{comparison_experiment_name}' "
            "nao encontrado. Rode 'make compare-models' antes do registro."
        )
        raise RuntimeError(msg)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=200,
    )

    scored = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if not run_name.startswith("compare_"):
            continue
        harmonic = compute_harmonic_mean_from_metrics(run.data.metrics)
        if harmonic is None:
            continue
        scored.append((harmonic, run.info.run_id, run_name.removeprefix("compare_")))

    scored.sort(key=lambda item: item[0], reverse=True)
    for harmonic, run_id, model_name in scored:
        if not has_model_artifact(client, run_id):
            logger.warning(
                "Run %s (%s) sem artefato de modelo; pulando.",
                run_id,
                model_name,
            )
            continue
        logger.info(
            "Campeao: %s (run %s, %s=%.4f)",
            model_name,
            run_id,
            CHAMPION_METRIC_KEY,
            harmonic,
        )
        return run_id, model_name, harmonic

    msg = (
        f"Nenhum run elegivel no experimento '{comparison_experiment_name}'. "
        "Verifique se 'make compare-models' registrou artefatos de modelo."
    )
    raise RuntimeError(msg)


def has_model_artifact(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> bool:
    """Verifica se o run possui artefato de modelo relevante."""
    artifacts = client.list_artifacts(run_id)
    names = {item.path for item in artifacts}
    if "model" in names:
        return True
    return any(path.endswith(".pkl") for path in names)


def compute_harmonic_mean_from_metrics(metrics: dict[str, float]) -> float | None:
    """Calcula a media harmonica @10 das metricas canonicas logadas.

    Args:
        metrics: Dicionario de metricas do run (precision@10, etc).

    Returns:
        Media harmonica de precision/recall/ndcg/map @10, ou None se
        alguma metrica canonica estiver ausente.
    """
    values = [metrics.get(key) for key in CANONICAL_METRIC_KEYS]
    if any(v is None for v in values):
        return None
    numeric = [float(v) for v in values if v is not None]
    if any(v <= 0 for v in numeric):
        return 0.0
    return len(numeric) / sum(1.0 / v for v in numeric)


def load_champion_model(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> RecommenderModel:
    """Carrega o modelo treinado a partir do artefato do run.

    Tenta primeiro carregar como pyfunc (``runs:/{run_id}/model``);
    se houver apenas pickle cru (baselines sklearn), baixa e
    desserializa diretamente.
    """
    try:
        return load_pyfunc_model(run_id)
    except mlflow.exceptions.MlflowException:
        logger.info(
            "pyfunc indisponivel para run %s; tentando pickle cru.",
            run_id,
        )
        pkl_path = find_pickle_artifact(client, run_id)
        return load_pickle_artifact(run_id, pkl_path)


def find_pickle_artifact(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> str:
    """Localiza um artefato pickle dentro do diretorio ``model``.

    Args:
        client: Cliente MLflow.
        run_id: Run ID do campeao.

    Returns:
        Caminho relativo do artefato pickle (ex: ``model/tmpXXX.pkl``).

    Raises:
        RuntimeError: Se nenhum pickle for encontrado.
    """
    model_artifacts = client.list_artifacts(run_id, path="model")
    for artifact in model_artifacts:
        if artifact.path.endswith(".pkl"):
            return artifact.path
    msg = (
        f"Nenhum artefato .pkl encontrado em 'model/' do run {run_id}. "
        "Verifique como o campeao foi logado."
    )
    raise RuntimeError(msg)


def load_pyfunc_model(run_id: str) -> RecommenderModel:
    """Carrega o modelo salvo como pyfunc/PyTorch do run do campeao."""
    logger.info("Carregando pyfunc do run %s/", run_id)
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    return _RecommenderFromPyFunc(loaded)


def load_pickle_artifact(run_id: str, artifact_path: str) -> RecommenderModel:
    """Desserializa um artefato pickle diretamente do run."""
    logger.info("Carregando pickle %s do run %s", artifact_path, run_id)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    with Path(local_path).open("rb") as handle:
        obj = pickle.load(handle)  # noqa: S301
    if not isinstance(obj, RecommenderModel):
        msg = (
            f"Artefato {artifact_path} do run {run_id} nao e "
            f"RecommenderModel: {type(obj).__name__}."
        )
        raise TypeError(msg)
    return obj


class _RecommenderFromPyFunc:
    """Adapta um pyfunc MLflow para o contrato ``RecommenderModel``.

    Necessario quando o run original salvou via ``mlflow.pyfunc`` ou
    ``mlflow.pytorch``; encapsula ``predict`` como ``recommend``.
    """

    def __init__(self, pyfunc_model: Any) -> None:
        self._pyfunc = pyfunc_model

    def recommend(self, user_id: str, limit: int | None = None) -> list[str]:
        """Delega para o pyfunc, traduzindo o contrato."""
        import pandas as pd

        frame = pd.DataFrame({"user_id": [user_id], "limit": [limit]})
        result = self._pyfunc.predict(frame)
        recommendations = result.iloc[0].to_dict().get("recommendations")
        if recommendations is None:
            return list(result.iloc[0].tolist())
        return list(recommendations)


def package_and_log(
    model: RecommenderModel,
    champion_run_id: str,
    champion_model_name: str,
    harmonic_mean: float,
) -> str:
    """Empacota o modelo como pyfunc e loga num novo run de registro.

    Args:
        model: Instancia treinada do recomendador campeao.
        champion_run_id: Run ID do campeao no experimento de comparacao.
        champion_model_name: Nome canonico do modelo campeao.
        harmonic_mean: Metrica H-Mean@10 do campeao.

    Returns:
        Run ID do novo run de registro.
    """
    with mlflow.start_run(run_name="register_champion") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = serialize_recommender(model, Path(tmpdir) / ARTIFACT_FILENAME)
            artifacts = {ARTIFACT_KEY: str(pickle_path)}
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RecommenderPythonModel(),
                artifacts=artifacts,
            )

        mlflow.log_metric(CHAMPION_METRIC_KEY, harmonic_mean)
        mlflow.set_tag("source_run_id", champion_run_id)
        mlflow.set_tag("champion_model_name", champion_model_name)
        mlflow.set_tag("stage", "register")
        mlflow.set_tag("pipeline", "register_model")
        mlflow.set_tag("issue", "16")
        log_system_info(random_seed=int(os.getenv("RANDOM_SEED", "42")))
        logger.info("Modelo empacotado no run %s", run.info.run_id)
        return run.info.run_id


def register_model_version(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    model_name: str,
) -> mlflow.entities.model_registry.ModelVersion:
    """Registra a versao do modelo no Model Registry.

    Args:
        client: Cliente MLflow.
        run_id: Run ID do run de registro com o artefato ``model``.
        model_name: Nome do modelo no Registry.

    Returns:
        Objeto ``ModelVersion`` criado.
    """
    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(
        "Modelo registrado: %s versao %s (stage %s)",
        model_name,
        version.version,
        version.current_stage,
    )
    return version


def transition_to_stage(
    client: mlflow.tracking.MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> mlflow.entities.model_registry.ModelVersion:
    """Promove a versao para o stage informado.

    Args:
        client: Cliente MLflow.
        model_name: Nome do modelo no Registry.
        version: Versao a promover.
        stage: Stage de destino (``Staging`` ou ``Production``).
        archive_existing: Arquiva versoes anteriores no mesmo stage.

    Returns:
        Objeto ``ModelVersion`` atualizado.
    """
    updated = client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Modelo %s versao %s promovido para %s",
        model_name,
        version,
        stage,
    )
    return updated


def run(
    model_name: str,
    comparison_experiment_name: str,
    registry_experiment_name: str,
) -> str:
    """Executa o fluxo completo de registro em Staging.

    Args:
        model_name: Nome do modelo no Registry.
        comparison_experiment_name: Nome do experimento de comparacao.
        registry_experiment_name: Nome do experimento de registro.

    Returns:
        Versao do modelo registrado.
    """
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    comparison_exp = get_experiment_name(
        cli_arg=comparison_experiment_name,
        env_var_name=COMPARISON_EXPERIMENT_ENV,
        default_name=DEFAULT_COMPARISON_EXPERIMENT,
    )
    registry_exp = get_experiment_name(
        cli_arg=registry_experiment_name,
        env_var_name="MLFLOW_REGISTRY_EXPERIMENT_NAME",
        default_name=DEFAULT_EXPERIMENT_NAME,
    )
    setup_mlflow(MLflowConfig(experiment_name=registry_exp))

    client = mlflow.tracking.MlflowClient()
    champion_run_id, champion_model_name, harmonic_mean = discover_champion_run(
        client, comparison_exp
    )
    model = load_champion_model(client, champion_run_id)
    register_run_id = package_and_log(
        model, champion_run_id, champion_model_name, harmonic_mean
    )
    version = register_model_version(client, register_run_id, model_name)
    transition_to_stage(client, model_name, str(version.version), STAGING_STAGE)
    logger.info(
        "Registro concluido: %s versao %s em %s",
        model_name,
        version.version,
        STAGING_STAGE,
    )
    return str(version.version)


def main() -> int:
    """Ponto de entrada do script de registro.

    Returns:
        Codigo de saida (0 sucesso, 1 erro).
    """
    args = parse_args()
    try:
        version = run(
            model_name=args.model_name,
            comparison_experiment_name=args.comparison_experiment,
            registry_experiment_name=args.registry_experiment,
        )
        print(f"Modelo registrado em Staging. Versao: {version}")
        return 0
    except Exception:
        logger.exception("Erro no registro do modelo")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
