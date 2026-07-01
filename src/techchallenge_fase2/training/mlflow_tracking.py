"""Utilitários genéricos para tracking com MLflow.

Fornece configuração centralizada, logging de hiperparâmetros,
métricas, artefatos e informações de sistema para cada run
do experimento de recomendação.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    """Configuração de tracking do MLflow.

    Usa default_factory para lazy evaluation, garantindo que
    as variáveis de ambiente sejam lidas após carregar o .env.
    """

    tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "tech-challenge-fase2"
        )
    )


def setup_mlflow(config: MLflowConfig) -> None:
    """Configura tracking URI e experimento no MLflow.

    Args:
        config: Configuração do MLflow com URI e nome do experimento.
    """
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    tracking_uri = config.tracking_uri
    if "localhost" in tracking_uri:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault(
            "AWS_ACCESS_KEY_ID",
            os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        )
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY",
            os.getenv("MINIO_SECRET_KEY", "minioadmin_secret_key_2024"),
        )


def log_hyperparameters(params: dict[str, Any]) -> None:
    """Registra hiperparâmetros na run ativa do MLflow.

    Converte todos os valores para tipos suportados pelo MLflow
    (str, int, float, bool).

    Args:
        params: Dicionário de hiperparâmetros. Valores que não
            são str, int, float ou bool são convertidos para str.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        if isinstance(value, str | int | float | bool):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    mlflow.log_params(sanitized)


def log_metrics(metrics: dict[str, float]) -> None:
    """Registra métricas na run ativa do MLflow.

    Args:
        metrics: Dicionário de métricas com valores float.
    """
    mlflow.log_metrics(metrics)


def log_system_info(random_seed: int = 42) -> None:
    """Registra informações de sistema e versões de bibliotecas.

    Loga como tags no MLflow para facilitar filtragem e
    reprodutibilidade.

    Args:
        random_seed: Seed utilizado para reprodutibilidade.
    """
    import numpy
    import pandas

    info: dict[str, str] = {
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "random_seed": str(random_seed),
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        info["torch_version"] = "not_installed"

    try:
        info["mlflow_version"] = mlflow.__version__
    except AttributeError:
        info["mlflow_version"] = "unknown"

    mlflow.set_tags(info)


def log_artifacts(artifact_paths: list[str | Path]) -> None:
    """Registra artefatos na run ativa do MLflow.

    Args:
        artifact_paths: Lista de caminhos de arquivos ou diretórios
            para registrar como artefatos.
    """
    for path in artifact_paths:
        path = Path(path)
        if not path.exists():
            continue
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        else:
            mlflow.log_artifact(str(path))


def log_recommender_model(
    model: Any,
    model_name: str,
    artifact_path: str = "model",
) -> None:
    """Registra modelo de recomendação como artefato do MLflow.

    Para modelos baseline (que não são PyTorch), usa pickle.
    Para modelos PyTorch, usa mlflow.pytorch.log_model.

    Args:
        model: Instância do modelo treinado.
        model_name: Nome identificador do modelo.
        artifact_path: Caminho do artefato no MLflow.
    """
    model_type = type(model).__name__
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("model_name", model_name)

    try:
        import torch  # noqa: PLC0415

        if isinstance(model, torch.nn.Module):
            mlflow.pytorch.log_model(model, artifact_path)
            return
    except ImportError:
        pass

    import pickle
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name

    try:
        mlflow.log_artifact(tmp_path, artifact_path)
    finally:
        os.unlink(tmp_path)


def log_input_data_summary(
    summary: dict[str, Any],
    artifact_file: str = "input_data_summary.json",
) -> None:
    """Registra resumo dos dados de entrada como artefato JSON do MLflow.

    Loga estatísticas do dataset (num_users, num_items, num_interactions,
    sparsity, dataset_version, amostra de interações) para rastreabilidade
    e reprodutibilidade do experimento.

    Args:
        summary: Dicionário com estatísticas dos dados de entrada.
            Deve ser JSON-serializável.
        artifact_file: Nome do arquivo JSON no artifact store.
    """
    mlflow.log_dict(summary, artifact_file)
