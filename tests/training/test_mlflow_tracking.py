"""Tests for MLflow tracking module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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


class TestMLflowConfig:
    """Tests for MLflowConfig dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Should have sensible defaults."""
        config = MLflowConfig()
        assert config.tracking_uri is not None
        assert config.experiment_name is not None

    @staticmethod
    def test_custom_values() -> None:
        """Should accept custom values."""
        config = MLflowConfig(
            tracking_uri="http://mlflow.example.com:5000",
            experiment_name="custom-experiment",
        )
        assert config.tracking_uri == "http://mlflow.example.com:5000"
        assert config.experiment_name == "custom-experiment"


class TestSetupMLflow:
    """Tests for setup_mlflow function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_sets_tracking_uri_and_experiment(
        mock_mlflow: MagicMock,
    ) -> None:
        """Should configure tracking URI and experiment."""
        config = MLflowConfig(
            tracking_uri="http://custom:5000",
            experiment_name="test-exp",
        )
        setup_mlflow(config)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://custom:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_sets_minio_env_vars_for_localhost(
        mock_mlflow: MagicMock,
    ) -> None:
        """Should set MinIO env vars when tracking URI is localhost."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test",
        )
        with patch.dict(os.environ, {}, clear=True):
            setup_mlflow(config)
            assert os.environ.get("MLFLOW_S3_ENDPOINT_URL") == "http://localhost:9000"


class TestLogHyperparameters:
    """Tests for log_hyperparameters function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_params(mock_mlflow: MagicMock) -> None:
        """Should log all parameters via MLflow."""
        params = {
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 64,
            "model_type": "popularity",
        }
        log_hyperparameters(params)
        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["learning_rate"] == 0.001
        assert logged["epochs"] == 10
        assert logged["model_type"] == "popularity"

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_converts_non_primitives_to_string(
        mock_mlflow: MagicMock,
    ) -> None:
        """Should convert non-primitive types to string."""
        params = {"hidden_dims": (128, 64, 32)}
        log_hyperparameters(params)
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["hidden_dims"] == "(128, 64, 32)"
        assert isinstance(logged["hidden_dims"], str)


class TestLogMetrics:
    """Tests for log_metrics function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_metrics(mock_mlflow: MagicMock) -> None:
        """Should log all metrics via MLflow."""
        metrics = {
            "precision@5": 0.4,
            "recall@5": 0.2,
            "ndcg@5": 0.35,
        }
        log_metrics(metrics)
        mock_mlflow.log_metrics.assert_called_once_with(metrics)


class TestLogSystemInfo:
    """Tests for log_system_info function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_seed_and_versions(mock_mlflow: MagicMock) -> None:
        """Should log system info as tags."""
        log_system_info(random_seed=42)
        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["random_seed"] == "42"
        assert "python_version" in tags
        assert "numpy_version" in tags
        assert "pandas_version" in tags


class TestLogArtifacts:
    """Tests for log_artifacts function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_file_artifact(mock_mlflow: MagicMock, tmp_path) -> None:
        """Should log existing file as artifact."""
        test_file = tmp_path / "test.png"
        test_file.write_text("fake image data")

        log_artifacts([str(test_file)])
        mock_mlflow.log_artifact.assert_called_once_with(str(test_file))

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_skips_nonexistent_files(mock_mlflow: MagicMock) -> None:
        """Should skip files that do not exist."""
        log_artifacts(["/nonexistent/file.png"])
        mock_mlflow.log_artifact.assert_not_called()


class TestLogRecommenderModel:
    """Tests for log_recommender_model function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_baseline_model(mock_mlflow: MagicMock) -> None:
        """Should log baseline model using pickle."""
        from techchallenge_fase2.models.baselines import PopularityRecommender

        model = PopularityRecommender(default_limit=10)
        log_recommender_model(model, "popularity")
        mock_mlflow.set_tag.assert_any_call("model_type", "PopularityRecommender")
        mock_mlflow.set_tag.assert_any_call("model_name", "popularity")
        # Should have logged an artifact (pickle file)
        assert mock_mlflow.log_artifact.called


class TestLogInputDataSummary:
    """Tests for log_input_data_summary function."""

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_logs_summary_as_json_artifact(mock_mlflow: MagicMock) -> None:
        """Should log input data summary via mlflow.log_dict."""
        summary = {
            "dataset": "RetailRocket E-Commerce",
            "num_users": 100,
            "num_items": 200,
            "num_interactions": 500,
            "sparsity": 0.975,
        }
        log_input_data_summary(summary)
        mock_mlflow.log_dict.assert_called_once()
        logged_payload, artifact_file = mock_mlflow.log_dict.call_args[0]
        assert logged_payload == summary
        assert artifact_file == "input_data_summary.json"

    @staticmethod
    @patch("techchallenge_fase2.training.mlflow_tracking.mlflow")
    def test_accepts_custom_artifact_file_name(mock_mlflow: MagicMock) -> None:
        """Should accept a custom artifact file name."""
        log_input_data_summary({"num_users": 1}, artifact_file="stats.json")
        mock_mlflow.log_dict.assert_called_once_with({"num_users": 1}, "stats.json")
