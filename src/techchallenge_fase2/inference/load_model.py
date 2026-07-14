"""Carregador de modelo de Production do MLflow Model Registry.

Fornece utilitario para carregar a versao em Production do modelo
registrado e CLI para gerar recomendacoes de um usuario.

Uso:
    $ uv run python -m techchallenge_fase2.inference.load_model \\
        recommend --user-id 123 --limit 10
    $ uv run python -m techchallenge_fase2.inference.load_model \\
        recommend --user-id 123 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import mlflow
import pandas as pd

from techchallenge_fase2.pipelines.common import load_dotenv_silent
from techchallenge_fase2.pipelines.register_model import (
    DEFAULT_MODEL_NAME,
    PRODUCTION_STAGE,
)

logger = logging.getLogger(__name__)


def load_production_model(
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str = PRODUCTION_STAGE,
) -> Any:
    """Carrega a versao atual de Production do Model Registry.

    Args:
        model_name: Nome do modelo no Registry.
        stage: Stage de onde carregar (default Production).

    Returns:
        Modelo pyfunc carregado.

    Raises:
        RuntimeError: Se nao houver versao no stage informado.
    """
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        msg = (
            f"Nenhuma versao de '{model_name}' em {stage}. "
            "Rode 'make register' e 'make promote' antes da inferencia."
        )
        raise RuntimeError(msg)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(
        "Carregando %s versao %s (%s)",
        model_name,
        versions[0].version,
        stage,
    )
    return mlflow.pyfunc.load_model(model_uri)


def recommend(
    model: Any,
    user_id: str,
    limit: int | None = None,
) -> list[str]:
    """Gera recomendacoes para um usuario usando o pyfunc carregado.

    Args:
        model: Modelo pyfunc carregado.
        user_id: Identificador do usuario como string.
        limit: Numero maximo de recomendacoes.

    Returns:
        Lista de identificadores de itens recomendados.
    """
    frame = pd.DataFrame({"user_id": [user_id]})
    if limit is not None:
        frame["limit"] = [limit]
    result = model.predict(frame)
    if "recommendations" in result.columns:
        return list(result.iloc[0]["recommendations"])
    return list(result.iloc[0].tolist())


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
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend", help="Recomenda itens para um usuario")
    rec.add_argument("--user-id", required=True, help="Identificador do usuario")
    rec.add_argument("--limit", type=int, default=None, help="Top-K recomendacoes")
    rec.add_argument("--json", action="store_true", help="Saida em JSON")
    rec.add_argument(
        "--stage",
        default=PRODUCTION_STAGE,
        help="Stage de onde carregar (default: Production)",
    )
    list_versions = sub.add_parser("list-versions", help="Lista versoes registradas")
    list_versions.add_argument(
        "--stage",
        default=None,
        help="Filtrar por stage (default: todos)",
    )
    return parser.parse_args()


def list_versions(model_name: str, stage: str | None) -> int:
    """Lista versoes registradas do modelo no stdout.

    Args:
        model_name: Nome do modelo no Registry.
        stage: Stage filtro (None para todos).

    Returns:
        0 sempre.
    """
    client = mlflow.tracking.MlflowClient()
    stages = [stage] if stage else None
    versions = client.get_latest_versions(model_name, stages=stages)
    if not versions:
        print(f"Nenhuma versao de '{model_name}' encontrada.")
        return 0
    print(f"Versoes de '{model_name}':")
    for version in versions:
        print(
            f"  versao {version.version} | stage {version.current_stage} | "
            f"run_id {version.run_id} | status {version.status}"
        )
    return 0


def main() -> int:
    """Ponto de entrada da CLI de inferencia.

    Returns:
        Codigo de saida (0 sucesso, 1 erro).
    """
    args = parse_args()
    load_dotenv_silent()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "list-versions":
        return list_versions(args.model_name, args.stage)

    if args.command == "recommend":
        model = load_production_model(args.model_name, args.stage)
        recommendations = recommend(model, args.user_id, args.limit)
        if args.json:
            print(
                json.dumps(
                    {"user_id": args.user_id, "recommendations": recommendations},
                    ensure_ascii=False,
                )
            )
        else:
            print(f"Recomendacoes para {args.user_id}: {recommendations}")
        return 0

    logger.error("Comando desconhecido: %s", args.command)
    return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    raise SystemExit(main())
