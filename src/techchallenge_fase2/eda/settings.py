"""Settings for RetailRocket EDA."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _RetailRocketEnvironment(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    retailrocket_data_dir: Path | None = Field(
        default=None,
        validation_alias="RETAILROCKET_DATA_DIR",
    )
    eda_output_dir: Path | None = Field(default=None, validation_alias="EDA_OUTPUT_DIR")
    required_interactions: int | None = Field(
        default=None,
        validation_alias="REQUIRED_INTERACTIONS",
    )


@dataclass(frozen=True)
class RetailRocketEdaSettings:
    """Configuration values used by the RetailRocket EDA pipeline."""

    dataset_dir: Path
    output_dir: Path
    required_interactions: int

    @classmethod
    def from_project(cls, project_root: Path) -> RetailRocketEdaSettings:
        """Load settings from TOML config and environment variables.

        Args:
            project_root: Repository root used to resolve relative paths.

        Returns:
            Settings for the EDA execution.
        """
        config = _read_config(project_root / "configs" / "retailrocket_eda.toml")
        env = _environment(project_root)
        dataset_dir = _dataset_dir(config, project_root, env.retailrocket_data_dir)
        output_dir = _output_dir(config, project_root, env.eda_output_dir)
        interactions = _required_interactions(config, env.required_interactions)
        return cls(dataset_dir, output_dir, interactions)

    def with_overrides(
        self,
        dataset_dir: Path | None,
        output_dir: Path | None,
        required_interactions: int | None,
    ) -> RetailRocketEdaSettings:
        """Return settings with command-line overrides applied."""
        dataset_value = self.dataset_dir if dataset_dir is None else dataset_dir
        output_value = self.output_dir if output_dir is None else output_dir
        required_value = (
            self.required_interactions if required_interactions is None else required_interactions
        )
        return replace(
            self,
            dataset_dir=dataset_value,
            output_dir=output_value,
            required_interactions=required_value,
        )


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def _environment(project_root: Path) -> _RetailRocketEnvironment:
    return _RetailRocketEnvironment(_env_file=project_root / ".env")


def _dataset_dir(
    config: dict[str, Any],
    project_root: Path,
    env_value: Path | None,
) -> Path:
    value = env_value or config["dataset"]["default_dir"]
    return _resolve_path(value, project_root)


def _output_dir(
    config: dict[str, Any],
    project_root: Path,
    env_value: Path | None,
) -> Path:
    value = env_value or config["outputs"]["eda_dir"]
    return _resolve_path(value, project_root)


def _required_interactions(config: dict[str, Any], env_value: int | None) -> int:
    if env_value is not None:
        return env_value
    return int(config["dataset"]["required_interactions"])


def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path
