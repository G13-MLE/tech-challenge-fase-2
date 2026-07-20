"""Testes da integracao MLflow do estagio de avaliacao (pipeline.evaluation).

Os testes mockam o modulo ``mlflow`` para evitar dependencia de servidor
real e validar que os utilitarios de tracking sao chamados corretamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from techchallenge_fase2.pipelines import evaluation as evaluation_module
from techchallenge_fase2.pipelines.config import (
    EvaluationParams,
    FeatureParams,
    PathParams,
    PipelineParams,
    PreprocessParams,
    TrainingParams,
)


def build_params(tmp_path: Path) -> PipelineParams:
    """Constroi PipelineParams valido apontando para tmp_path."""
    return PipelineParams(
        paths=PathParams(
            raw_events=tmp_path / "raw_events.csv",
            processed_interactions=tmp_path / "interactions.parquet",
            train_features=tmp_path / "train.parquet",
            validation_features=tmp_path / "val.parquet",
            test_features=tmp_path / "test.parquet",
            mappings=tmp_path / "mappings.json",
            dataset_stats=tmp_path / "dataset_stats.json",
            model_checkpoint=tmp_path / "model.pt",
            checkpoint_dir=tmp_path / "checkpoints",
            metrics=tmp_path / "metrics.json",
        ),
        preprocess=PreprocessParams(sample_size=100, random_seed=42),
        features=FeatureParams(
            train_ratio=0.7,
            validation_ratio=0.15,
            session_gap_minutes=30,
            event_weights={"view": 1.0, "addtocart": 3.0, "transaction": 5.0},
        ),
        training=TrainingParams(
            batch_size=64,
            epochs=2,
            embedding_dim=8,
            learning_rate=0.01,
            negative_samples=1,
            random_seed=42,
            patience=2,
            min_delta=1e-4,
        ),
        evaluation=EvaluationParams(top_k=5, max_users=10),
    )


def write_features(path: Path, num_users: int = 4, num_items: int = 4) -> None:
    """Escreve um parquet minimo de features com colunas visitorid/itemid/index."""
    rows: list[dict[str, object]] = []
    for user in range(num_users):
        for item in range(num_items):
            rows.append(
                {
                    "visitorid": f"user_{user}",
                    "itemid": f"item_{item}",
                    "user_index": user,
                    "item_index": item,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def build_checkpoint(num_users: int = 4, num_items: int = 4) -> dict[str, Any]:
    """Constroi um checkpoint minimo compativel com load_model."""
    from techchallenge_fase2.models.ncf import (
        NCFConfig,
        NeuralCollaborativeFiltering,
    )

    model = NeuralCollaborativeFiltering(
        NCFConfig(num_users=num_users, num_items=num_items, embedding_dim=8)
    )
    return {
        "model_kind": "ncf",
        "model_state": model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": 8,
        "mlp_hidden_sizes": [128, 64, 32],
        "dropout": 0.1,
        "final_loss": 0.5,
        "best_metric": 0.85,
    }


@pytest.fixture
def setup_eval_env(tmp_path: Path) -> tuple[PipelineParams, dict[str, Any]]:
    """Prepara features, checkpoint e params para os testes de avaliacao."""
    params = build_params(tmp_path)
    write_features(params.paths.train_features)
    write_features(params.paths.test_features)
    checkpoint = build_checkpoint()
    return params, checkpoint


class TestBuildEvaluationHyperparameters:
    """Valida construcao do dict de hiperparametros de avaliacao."""

    @staticmethod
    def test_inclui_hiperparametros_chave(
        setup_eval_env: tuple[PipelineParams, dict[str, Any]],
    ) -> None:
        """Hiperparametros de avaliacao e checkpoint estao presentes."""
        params, checkpoint = setup_eval_env
        hyperparams = evaluation_module.build_evaluation_hyperparameters(
            params, checkpoint, num_evaluated_users=10, dataset_version="abc12345"
        )
        assert hyperparams["model_type"] == "neural_ncf"
        assert hyperparams["top_k"] == params.evaluation.top_k
        assert hyperparams["max_users"] == params.evaluation.max_users
        assert hyperparams["num_evaluated_users"] == 10
        assert hyperparams["embedding_dim"] == 8
        assert hyperparams["num_users"] == checkpoint["num_users"]
        assert hyperparams["num_items"] == checkpoint["num_items"]
        assert hyperparams["dataset_version"] == "abc12345"
        assert hyperparams["best_train_metric"] == pytest.approx(0.85)
        assert hyperparams["final_train_loss"] == pytest.approx(0.5)


class TestCardMetrics:
    """Valida o mapeamento das metricas agregadas para o Model Card."""

    @staticmethod
    def test_mapeia_chaves_at_para_arroba() -> None:
        """Metricas no formato final_at_K viram metric@K."""
        metrics = {
            "hit_rate_at_5": 0.6,
            "map_at_5": 0.4,
            "ndcg_at_5": 0.5,
            "precision_at_5": 0.3,
            "recall_at_5": 0.2,
            "evaluated_users": 10.0,
        }
        result = evaluation_module.card_metrics(metrics, top_k=5)
        assert result["hit_rate@5"] == pytest.approx(0.6)
        assert result["map@5"] == pytest.approx(0.4)
        assert result["ndcg@5"] == pytest.approx(0.5)
        assert result["precision@5"] == pytest.approx(0.3)
        assert result["recall@5"] == pytest.approx(0.2)


class TestLogEvaluationRun:
    """Valida que log_evaluation_run chama os utilitarios de MLflow."""

    @staticmethod
    @patch("techchallenge_fase2.pipelines.evaluation.mlflow")
    @patch("techchallenge_fase2.pipelines.evaluation.log_artifacts")
    @patch("techchallenge_fase2.pipelines.evaluation.log_metrics")
    @patch("techchallenge_fase2.pipelines.evaluation.log_system_info")
    @patch("techchallenge_fase2.pipelines.evaluation.log_hyperparameters")
    def test_chama_utilitarios_de_tracking(
        mock_log_hyper: MagicMock,
        mock_log_sys: MagicMock,
        mock_log_metrics: MagicMock,
        mock_log_artifacts: MagicMock,
        mock_mlflow: MagicMock,
        setup_eval_env: tuple[PipelineParams, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Todos os utilitarios de tracking sao chamados durante o run."""
        params, checkpoint = setup_eval_env
        metrics = {
            "hit_rate_at_5": 0.6,
            "map_at_5": 0.4,
            "ndcg_at_5": 0.5,
            "precision_at_5": 0.3,
            "recall_at_5": 0.2,
            "evaluated_users": 10.0,
        }
        # Salva o arquivo de metricas para que log_artifacts o encontre.
        evaluation_module.save_metrics(metrics, params.paths.metrics)

        evaluation_module.log_evaluation_run(
            params, checkpoint, metrics, dataset_version="abc12345"
        )

        mock_log_hyper.assert_called_once()
        mock_log_sys.assert_called_once()
        mock_log_metrics.assert_called_once_with(metrics)
        mock_log_artifacts.assert_called_once_with([params.paths.metrics])
        # Tags definidas
        assert mock_mlflow.set_tag.call_count >= 5
        # model_card.json logado via mlflow.log_dict
        mock_mlflow.log_dict.assert_called_once()
        args, _ = mock_mlflow.log_dict.call_args
        assert args[1] == "model_card.json"

    @staticmethod
    @patch("techchallenge_fase2.pipelines.evaluation.mlflow")
    @patch("techchallenge_fase2.pipelines.evaluation.log_artifacts")
    @patch("techchallenge_fase2.pipelines.evaluation.log_metrics")
    @patch("techchallenge_fase2.pipelines.evaluation.log_system_info")
    @patch("techchallenge_fase2.pipelines.evaluation.log_hyperparameters")
    def test_define_tags_do_run_avaliacao(
        mock_log_hyper: MagicMock,
        mock_log_sys: MagicMock,
        mock_log_metrics: MagicMock,
        mock_log_artifacts: MagicMock,
        mock_mlflow: MagicMock,
        setup_eval_env: tuple[PipelineParams, dict[str, Any]],
    ) -> None:
        """Tags distinguem o run de avaliacao do NCF orquestrado pelo DVC."""
        params, checkpoint = setup_eval_env
        metrics = {"evaluated_users": 5.0}

        evaluation_module.log_evaluation_run(
            params, checkpoint, metrics, dataset_version="abc12345"
        )

        tags = {
            call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list
        }
        assert tags["model_type"] == "neural_ncf"
        assert tags["model_name"] == "ncf"
        assert tags["stage"] == "evaluate"
        assert tags["orchestrator"] == "dvc"
        assert tags["dataset_version"] == "abc12345"


class TestRunIntegratesMlflow:
    """Valida que run() abre um run do MLflow e chama os utilitarios."""

    @staticmethod
    @patch("techchallenge_fase2.pipelines.evaluation.mlflow")
    @patch("techchallenge_fase2.pipelines.evaluation.log_evaluation_run")
    @patch("techchallenge_fase2.pipelines.evaluation.setup_mlflow")
    def test_run_abre_run_mlflow_e_loga(
        mock_setup: MagicMock,
        mock_log_run: MagicMock,
        mock_mlflow: MagicMock,
        setup_eval_env: tuple[PipelineParams, dict[str, Any]],
    ) -> None:
        """run() configura MLflow, abre start_run e chama log_evaluation_run."""
        params, checkpoint = setup_eval_env
        # Salva o checkpoint para load_checkpoint funcionar.
        import torch

        torch.save(checkpoint, params.paths.model_checkpoint)

        # Mock do start_run como context manager.
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_cm
        mock_mlflow.start_run.return_value = mock_cm

        evaluation_module.run(params)

        mock_setup.assert_called_once()
        mock_mlflow.start_run.assert_called_once_with(run_name="ncf_evaluate")
        mock_log_run.assert_called_once()
