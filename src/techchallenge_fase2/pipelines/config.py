"""Configuration helpers for the DVC recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PathParams:
    """Filesystem paths used by the pipeline stages."""

    raw_events: Path
    processed_interactions: Path
    train_features: Path
    validation_features: Path
    test_features: Path
    mappings: Path
    dataset_stats: Path
    model_checkpoint: Path
    checkpoint_dir: Path
    metrics: Path


@dataclass(frozen=True, slots=True)
class PreprocessParams:
    """Preprocessing behavior.

    A filtragem do catalogo e dos usuarios e uma decisao de design documentada
    da literatura de sistemas de recomendacao para tornar o problema tratavel
    em datasets esparsos (RetailRocket tem ~2 interacoes/usuario em catalogo de
    235k itens): mantemos apenas os top `max_items` itens mais populares e
    usuarios com pelo menos `min_interactions_per_user` interacoes.
    """

    sample_size: int
    random_seed: int
    max_items: int
    min_interactions_per_user: int


@dataclass(frozen=True, slots=True)
class FeatureParams:
    """Feature engineering behavior."""

    train_ratio: float
    validation_ratio: float
    session_gap_minutes: int
    event_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Model training behavior."""

    batch_size: int
    epochs: int
    embedding_dim: int
    learning_rate: float
    negative_samples: int
    random_seed: int
    patience: int
    min_delta: float


@dataclass(frozen=True, slots=True)
class EvaluationParams:
    """Recommendation evaluation behavior."""

    top_k: int
    max_users: int


@dataclass(frozen=True, slots=True)
class PipelineParams:
    """Complete typed configuration for all pipeline stages."""

    paths: PathParams
    preprocess: PreprocessParams
    features: FeatureParams
    training: TrainingParams
    evaluation: EvaluationParams


def load_params(params_path: str | Path) -> PipelineParams:
    """Load pipeline parameters from YAML.

    Args:
        params_path: Path to the YAML parameter file.

    Returns:
        Typed pipeline parameters.
    """
    raw_params = read_yaml(Path(params_path))
    return PipelineParams(
        paths=build_paths(raw_params["paths"]),
        preprocess=build_preprocess(raw_params["preprocess"]),
        features=build_features(raw_params["features"]),
        training=build_training(raw_params["training"]),
        evaluation=build_evaluation(raw_params["evaluation"]),
    )


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_paths(raw_paths: dict[str, str]) -> PathParams:
    """Build path parameters from raw YAML values."""
    return PathParams(**{key: Path(value) for key, value in raw_paths.items()})


def build_preprocess(raw_params: dict[str, int]) -> PreprocessParams:
    """Build preprocessing parameters."""
    return PreprocessParams(
        sample_size=int(raw_params["sample_size"]),
        random_seed=int(raw_params["random_seed"]),
        max_items=int(raw_params.get("max_items", 0)),
        min_interactions_per_user=int(raw_params.get("min_interactions_per_user", 0)),
    )


def build_features(raw_params: dict[str, Any]) -> FeatureParams:
    """Build feature engineering parameters."""
    return FeatureParams(
        train_ratio=float(raw_params["train_ratio"]),
        validation_ratio=float(raw_params["validation_ratio"]),
        session_gap_minutes=int(raw_params["session_gap_minutes"]),
        event_weights=to_float_dict(raw_params["event_weights"]),
    )


def build_training(raw_params: dict[str, int | float]) -> TrainingParams:
    """Build training parameters."""
    return TrainingParams(
        batch_size=int(raw_params["batch_size"]),
        epochs=int(raw_params["epochs"]),
        embedding_dim=int(raw_params["embedding_dim"]),
        learning_rate=float(raw_params["learning_rate"]),
        negative_samples=int(raw_params["negative_samples"]),
        random_seed=int(raw_params["random_seed"]),
        patience=int(raw_params["patience"]),
        min_delta=float(raw_params["min_delta"]),
    )


def build_evaluation(raw_params: dict[str, int]) -> EvaluationParams:
    """Build evaluation parameters."""
    return EvaluationParams(
        top_k=int(raw_params["top_k"]),
        max_users=int(raw_params["max_users"]),
    )


def to_float_dict(raw_values: dict[str, int | float]) -> dict[str, float]:
    """Convert a numeric dictionary to float values."""
    return {key: float(value) for key, value in raw_values.items()}
