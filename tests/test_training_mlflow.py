"""Testes da integracao MLflow do estagio de treino (pipeline.training).

Os testes mockam o modulo ``mlflow`` para evitar dependencia de servidor
real e validar que os utilitarios de tracking sao chamados corretamente.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from techchallenge_fase2.pipelines import training as training_module
from techchallenge_fase2.pipelines.config import (
    EvaluationParams,
    FeatureParams,
    PathParams,
    PipelineParams,
    PreprocessParams,
    TrainingParams,
)
from techchallenge_fase2.training.trainer import TrainingHistory


def _build_params(tmp_path: Path) -> PipelineParams:
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


def _write_features(path: Path, num_users: int = 4, num_items: int = 4) -> None:
    """Escreve um parquet minimo de features com colunas user_index/item_index."""
    rows: list[dict[str, int]] = []
    for user in range(num_users):
        for item in range(num_items):
            rows.append({"user_index": user, "item_index": item})
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_mappings(path: Path, users: int = 4, items: int = 4) -> None:
    """Escreve o arquivo mappings.json minimo consumido pelo treino."""
    path.write_text(
        json.dumps(
            {
                "user_ids": list(range(users)),
                "item_ids": list(range(items)),
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def setup_training_env(tmp_path: Path) -> tuple[PipelineParams, Path]:
    """Prepara features, mappings e params para os testes de treino."""
    params = _build_params(tmp_path)
    _write_features(params.paths.train_features)
    _write_features(params.paths.validation_features)
    _write_mappings(params.paths.mappings)
    params.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    params.paths.model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return params, tmp_path


class TestBuildTrainingHyperparameters:
    """Valida construcao do dict de hiperparametros para o MLflow."""

    @staticmethod
    def test_inclui_hiperparametros_chave(
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """Hiperparametros essenciais do NCF estao presentes no dict."""
        params, _ = setup_training_env
        hyperparams = training_module.build_training_hyperparameters(
            params, users=4, items=4, dataset_version="abc12345"
        )
        assert hyperparams["model_type"] == "neural_ncf"
        assert hyperparams["embedding_dim"] == params.training.embedding_dim
        assert hyperparams["batch_size"] == params.training.batch_size
        assert hyperparams["epochs"] == params.training.epochs
        assert hyperparams["learning_rate"] == params.training.learning_rate
        assert hyperparams["random_seed"] == params.training.random_seed
        assert hyperparams["num_users"] == 4
        assert hyperparams["num_items"] == 4
        assert hyperparams["dataset_version"] == "abc12345"

    @staticmethod
    def test_conta_interacoes_de_treino_e_validacao(
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """Numero de interacoes reflete o que foi escrito nos parquets."""
        params, _ = setup_training_env
        hyperparams = training_module.build_training_hyperparameters(
            params, users=4, items=4, dataset_version="unknown"
        )
        # 4 usuarios x 4 itens = 16 interacoes em cada frame
        assert hyperparams["num_train_interactions"] == 16
        assert hyperparams["num_validation_interactions"] == 16


class TestBuildInputSummary:
    """Valida o resumo dos dados de entrada logado como artefato."""

    @staticmethod
    def test_calcula_sparsity_e_contagens(
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """Sparsity e contagens sao derivadas de users, items e interacoes."""
        params, _ = setup_training_env
        summary = training_module.build_input_summary(
            params, users=4, items=4, dataset_version="abc12345"
        )
        assert summary["num_users"] == 4
        assert summary["num_items"] == 4
        assert summary["num_train_interactions"] == 16
        assert summary["num_validation_interactions"] == 16
        # 16 interacoes / (4 * 4) = 1.0 -> sparsity = 0.0
        assert summary["sparsity"] == pytest.approx(0.0)
        assert summary["dataset_version"] == "abc12345"


class TestSummarizeHistory:
    """Valida a agregacao do historico de treino em metricas finais."""

    @staticmethod
    def test_extrai_metricas_finais_do_historico() -> None:
        """Metricas finais refletem o ultimo valor e o melhor AUC."""
        history = TrainingHistory(
            train_losses=(0.5, 0.3, 0.2),
            val_metrics=(0.7, 0.8, 0.75),
            stopped_epoch=-1,
        )
        metrics = training_module.summarize_history(history)
        assert metrics["final_train_loss"] == pytest.approx(0.2)
        assert metrics["best_val_auc"] == pytest.approx(0.8)
        assert metrics["last_val_auc"] == pytest.approx(0.75)
        assert metrics["num_epochs_run"] == 3.0
        assert metrics["stopped_epoch"] == -1.0

    @staticmethod
    def test_historico_vazio_retorna_zeros() -> None:
        """Historico vazio nao quebra a agregacao."""
        history = TrainingHistory(train_losses=(), val_metrics=(), stopped_epoch=-1)
        metrics = training_module.summarize_history(history)
        assert metrics["final_train_loss"] == 0.0
        assert metrics["best_val_auc"] == 0.0
        assert metrics["num_epochs_run"] == 0.0


class TestSaveHistoryArtifact:
    """Valida o artefato JSON do historico de treino."""

    @staticmethod
    def test_salva_json_com_perdas_e_auc(tmp_path: Path) -> None:
        """Artefato JSON contem train_losses, val_auc e stopped_epoch."""
        history = TrainingHistory(
            train_losses=(0.5, 0.3),
            val_metrics=(0.7, 0.8),
            stopped_epoch=-1,
        )
        artifact_path = tmp_path / "models" / "training_history.json"
        result = training_module.save_history_artifact(history, artifact_path)
        assert result == artifact_path
        assert artifact_path.exists()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["train_losses"] == [0.5, 0.3]
        assert payload["val_auc"] == [0.7, 0.8]
        assert payload["stopped_epoch"] == -1


class TestLogTrainingRun:
    """Valida que log_training_run chama os utilitarios de MLflow corretamente."""

    @staticmethod
    @patch("techchallenge_fase2.pipelines.training.mlflow")
    @patch("techchallenge_fase2.pipelines.training.log_recommender_model")
    @patch("techchallenge_fase2.pipelines.training.log_artifacts")
    @patch("techchallenge_fase2.pipelines.training.log_metrics")
    @patch("techchallenge_fase2.pipelines.training.log_input_data_summary")
    @patch("techchallenge_fase2.pipelines.training.log_system_info")
    @patch("techchallenge_fase2.pipelines.training.log_hyperparameters")
    def test_chama_utilitarios_de_tracking(
        mock_log_hyper: MagicMock,
        mock_log_sys: MagicMock,
        mock_log_summary: MagicMock,
        mock_log_metrics: MagicMock,
        mock_log_artifacts: MagicMock,
        mock_log_model: MagicMock,
        mock_mlflow: MagicMock,
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """Todos os utilitarios de tracking sao chamados durante o run."""
        params, tmp_path = setup_training_env
        from techchallenge_fase2.models.ncf import (
            NCFConfig,
            NeuralCollaborativeFiltering,
        )

        model = NeuralCollaborativeFiltering(
            NCFConfig(num_users=4, num_items=4, embedding_dim=8)
        )
        history = TrainingHistory(
            train_losses=(0.5,), val_metrics=(0.7,), stopped_epoch=-1
        )

        # Salva um checkpoint para que log_artifacts o encontre.
        training_module.save_pipeline_checkpoint(
            model, params, (history.train_losses, history.val_metrics)
        )

        training_module.log_training_run(
            model, params, history, users=4, items=4, dataset_version="abc12345"
        )

        mock_log_hyper.assert_called_once()
        mock_log_sys.assert_called_once()
        mock_log_summary.assert_called_once()
        mock_log_metrics.assert_called_once()
        mock_log_model.assert_called_once()
        # log_artifacts chamado para history_path e checkpoint
        assert mock_log_artifacts.call_count >= 1
        # mlflow.set_tag chamado para varias tags
        assert mock_mlflow.set_tag.call_count >= 4
        # mlflow.log_dict chamado para o model_card.json
        mock_mlflow.log_dict.assert_called_once()
        args, _ = mock_mlflow.log_dict.call_args
        assert args[1] == "model_card.json"

    @staticmethod
    @patch("techchallenge_fase2.pipelines.training.mlflow")
    @patch("techchallenge_fase2.pipelines.training.log_recommender_model")
    @patch("techchallenge_fase2.pipelines.training.log_artifacts")
    @patch("techchallenge_fase2.pipelines.training.log_metrics")
    @patch("techchallenge_fase2.pipelines.training.log_input_data_summary")
    @patch("techchallenge_fase2.pipelines.training.log_system_info")
    @patch("techchallenge_fase2.pipelines.training.log_hyperparameters")
    def test_define_tags_do_run_ncf(
        mock_log_hyper: MagicMock,
        mock_log_sys: MagicMock,
        mock_log_summary: MagicMock,
        mock_log_metrics: MagicMock,
        mock_log_artifacts: MagicMock,
        mock_log_model: MagicMock,
        mock_mlflow: MagicMock,
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """Tags distinguem o run do NCF orquestrado pelo DVC."""
        params, _ = setup_training_env
        from techchallenge_fase2.models.ncf import (
            NCFConfig,
            NeuralCollaborativeFiltering,
        )

        model = NeuralCollaborativeFiltering(
            NCFConfig(num_users=4, num_items=4, embedding_dim=8)
        )
        history = TrainingHistory(
            train_losses=(0.5,), val_metrics=(0.7,), stopped_epoch=-1
        )

        training_module.log_training_run(
            model, params, history, users=4, items=4, dataset_version="abc12345"
        )

        tags = {
            call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list
        }
        assert tags["model_type"] == "neural_ncf"
        assert tags["model_name"] == "ncf"
        assert tags["stage"] == "train"
        assert tags["orchestrator"] == "dvc"
        assert tags["dataset_version"] == "abc12345"


class TestRunIntegratesMlflow:
    """Valida que run() abre um run do MLflow e chama os utilitarios."""

    @staticmethod
    @patch("techchallenge_fase2.pipelines.training.mlflow")
    @patch("techchallenge_fase2.pipelines.training.log_training_run")
    @patch("techchallenge_fase2.pipelines.training.setup_mlflow")
    @patch("techchallenge_fase2.pipelines.training.Trainer")
    def test_run_abre_run_mlflow_e_loga(
        mock_trainer_cls: MagicMock,
        mock_setup: MagicMock,
        mock_log_run: MagicMock,
        mock_mlflow: MagicMock,
        setup_training_env: tuple[PipelineParams, Path],
    ) -> None:
        """run() configura MLflow, abre start_run e chama log_training_run."""
        params, _ = setup_training_env

        # Trainer.train() retorna um historico minimo.
        history = TrainingHistory(
            train_losses=(0.5,), val_metrics=(0.7,), stopped_epoch=-1
        )
        mock_trainer_cls.return_value.train.return_value = history

        # Mock do start_run como context manager.
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_cm
        mock_mlflow.start_run.return_value = mock_cm

        training_module.run(params)

        mock_setup.assert_called_once()
        mock_mlflow.start_run.assert_called_once_with(run_name="ncf_train")
        mock_log_run.assert_called_once()
