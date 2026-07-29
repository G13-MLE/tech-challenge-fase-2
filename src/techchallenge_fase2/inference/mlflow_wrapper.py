"""Wrapper pyfunc do MLflow para recomendadores customizados.

Empacota qualquer ``RecommenderModel`` do projeto como um modelo
``mlflow.pyfunc.PythonModel`` para registro no Model Registry. A
serialização usa pickle padrão (dependências sklearn/scipy/torch já
estão no ambiente), preservando mappings e estado interno.

Contrato do ``predict``:
- Entrada: DataFrame com coluna ``user_id`` (str) e opcional ``limit`` (int).
- Saida: DataFrame com colunas ``user_id`` e ``recommendations``
  (lista de str).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from techchallenge_fase2.models.base import RecommenderModel

ARTIFACT_KEY = "recommender_pickle"
ARTIFACT_FILENAME = "recommender.pkl"
DEFAULT_LIMIT = 10


class RecommenderPythonModel(mlflow.pyfunc.PythonModel):
    """Empacota um ``RecommenderModel`` como pyfunc MLflow.

    O modelo serializado e desserializado em ``load_context`` e exposto
    via ``predict`` seguindo o contrato de qualquer recomendador do
    projeto.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Carrega o recomendador a partir do artefato pickle.

        Args:
            context: Contexto MLflow com o dicionário de artefatos.
        """
        pickle_path = Path(context.artifacts[ARTIFACT_KEY])
        self._recommender: RecommenderModel = deserialize_recommender(pickle_path)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Gera recomendações em lote a partir de um DataFrame.

        Args:
            context: Contexto MLflow (não usado diretamente).
            model_input: DataFrame com coluna ``user_id`` e opcional
                ``limit``. Se ``limit`` estiver ausente, usa o default
                do modelo.
            params: Parametros opcionais (não usados).

        Returns:
            DataFrame com colunas ``user_id`` (str) e
            ``recommendations`` (list[str]) por usuário.
        """
        del context, params
        frame = normalize_input(model_input)
        rows = [
            (row.user_id, recommend_one(self._recommender, row))
            for row in frame.itertuples(index=False)
        ]
        return pd.DataFrame(rows, columns=["user_id", "recommendations"])


def recommend_one(
    recommender: RecommenderModel,
    row: Any,
) -> list[str]:
    """Recomenda para uma linha do DataFrame de entrada."""
    limit = (
        int(row.limit)
        if hasattr(row, "limit") and not pd.isna(getattr(row, "limit", float("nan")))
        else None
    )
    return list(recommender.recommend(str(row.user_id), limit=limit))


def normalize_input(model_input: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normaliza Series ou DataFrame em DataFrame com colunas esperadas."""
    if isinstance(model_input, pd.Series):
        return model_input.to_frame(name="user_id").T
    if "user_id" not in model_input.columns and len(model_input.columns) >= 1:
        model_input = model_input.rename(columns={model_input.columns[0]: "user_id"})
    return model_input


def serialize_recommender(recommender: RecommenderModel, path: Path) -> Path:
    """Serializa o recomendador para pickle no caminho informado.

    Args:
        recommender: Instancia treinada de ``RecommenderModel``.
        path: Caminho do arquivo pickle a ser criado.

    Returns:
        O caminho do arquivo criado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(recommender, handle)
    return path


def deserialize_recommender(path: Path) -> RecommenderModel:
    """Desserializa um recomendador a partir do pickle.

    Args:
        path: Caminho do arquivo pickle.

    Returns:
        Instancia de ``RecommenderModel`` restaurada.
    """
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)  # noqa: S301
    if not isinstance(obj, RecommenderModel):
        msg = (
            f"Artefato pickle não e um RecommenderModel: {type(obj).__name__}. "
            "Verifique o empacotamento do modelo."
        )
        raise TypeError(msg)
    return obj
