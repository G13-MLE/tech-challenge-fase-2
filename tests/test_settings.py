from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from configs.settings import Environment, Settings


class _DefaultsOnlySettings(Settings):
    """Settings sem leitura de .env, para testar valores padrão."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )


@pytest.fixture
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    variables = [
        "ENV",
        "RANDOM_SEED",
        "DATA_DIR",
        "MODELS_DIR",
        "CONFIGS_DIR",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "DVC_REMOTE_URL",
        "LEARNING_RATE",
        "BATCH_SIZE",
        "EPOCHS",
        "EMBEDDING_DIM",
    ]
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)


def test_settings_defaults(clear_env: None) -> None:
    settings = _DefaultsOnlySettings()

    assert settings.env == Environment.DEVELOPMENT
    assert settings.random_seed == 42
    assert settings.data_dir == Path("data")
    assert settings.models_dir == Path("models")
    assert settings.configs_dir == Path("configs")
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.mlflow_experiment_name == "tech-challenge-fase2"
    assert settings.mlflow_ncf_experiment_name == "tech-challenge-ncf"
    assert settings.mlflow_ease_experiment_name == "tech-challenge-ease"
    assert settings.mlflow_baseline_experiment_name == "tech-challenge-baselines"
    assert settings.mlflow_comparison_experiment_name == "tech-challenge-comparison"
    assert settings.dvc_remote_url == ""
    assert settings.learning_rate == 0.001
    assert settings.batch_size == 64
    assert settings.epochs == 10
    assert settings.embedding_dim == 64


def test_env_example_declares_mlflow_comparison_experiment_name() -> None:
    """O .env.example deve documentar MLFLOW_COMPARISON_EXPERIMENT_NAME.

    As tres variaveis irmao (NCF, EASE, BASELINE) ja sao declaradas; a
    de comparacao estava faltando, impedindo configuracao via .env.
    """
    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    content = env_example.read_text(encoding="utf-8")
    assert "MLFLOW_COMPARISON_EXPERIMENT_NAME" in content, (
        ".env.example deve declarar MLFLOW_COMPARISON_EXPERIMENT_NAME"
    )


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("BATCH_SIZE", "128")
    monkeypatch.setenv("LEARNING_RATE", "0.01")

    settings = Settings()

    assert settings.env == Environment.PRODUCTION
    assert settings.batch_size == 128
    assert settings.learning_rate == 0.01


def test_invalid_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "ftp://localhost:5000")

    with pytest.raises(ValidationError):
        Settings()


def test_invalid_dvc_remote_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DVC_REMOTE_URL", "ftp://bucket/path")

    with pytest.raises(ValidationError):
        Settings()


def test_invalid_mlflow_experiment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "   ")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LEARNING_RATE", "0"),
        ("BATCH_SIZE", "0"),
        ("EPOCHS", "0"),
        ("EMBEDDING_DIM", "0"),
        ("RANDOM_SEED", "-1"),
    ],
)
def test_invalid_numeric_constraints(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()
