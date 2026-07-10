from enum import Enum
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Tipos de ambiente suportados pela aplicação."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: Environment = Environment.DEVELOPMENT
    random_seed: int = Field(default=42, ge=0)

    data_dir: Path = Path("data/")
    models_dir: Path = Path("models/")
    configs_dir: Path = Path("configs/")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "tech-challenge-fase2"
    mlflow_ncf_experiment_name: str = "tech-challenge-ncf"
    mlflow_ease_experiment_name: str = "tech-challenge-ease"
    mlflow_baseline_experiment_name: str = "tech-challenge-baselines"
    mlflow_comparison_experiment_name: str = "tech-challenge-comparison"

    dvc_remote_url: str = ""

    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=10, gt=0)
    embedding_dim: int = Field(default=64, gt=0)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    recommendation_limit: int = Field(default=20, gt=0)

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_mlflow_tracking_uri(cls, value: str) -> str:
        """Valida esquema aceito para URI do MLflow."""
        parsed_uri = urlparse(value)
        allowed_schemes = {"http", "https", "file"}
        if parsed_uri.scheme not in allowed_schemes:
            msg = "MLFLOW_TRACKING_URI deve usar http://, https:// ou file://"
            raise ValueError(msg)
        return value

    @field_validator("mlflow_experiment_name")
    @classmethod
    def validate_mlflow_experiment_name(cls, value: str) -> str:
        """Garante que o nome do experimento não esteja vazio."""
        if not value.strip():
            msg = "MLFLOW_EXPERIMENT_NAME não pode ser vazio"
            raise ValueError(msg)
        return value

    @field_validator("dvc_remote_url")
    @classmethod
    def validate_dvc_remote_url(cls, value: str) -> str:
        """Valida URI do remote do DVC quando configurada."""
        stripped_value = value.strip()
        if not stripped_value:
            return ""

        parsed_uri = urlparse(stripped_value)
        allowed_schemes = {"s3", "gs", "ssh"}
        if parsed_uri.scheme not in allowed_schemes:
            msg = "DVC_REMOTE_URL deve usar s3://, gs:// ou ssh://"
            raise ValueError(msg)
        return stripped_value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna uma instancia unica de Settings por processo."""
    return Settings()
