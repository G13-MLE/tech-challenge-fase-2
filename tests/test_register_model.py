"""Tests for the Model Registry register pipeline (Discovery + Staging).

Cobra a logica de descoberta do campeao e empacotamento sem bater em
MLflow server real: mockamos o ``MlflowClient`` onde necessario.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

from techchallenge_fase2.inference.mlflow_wrapper import deserialize_recommender
from techchallenge_fase2.models.sklearn_baselines import ItemKNNRecommender
from techchallenge_fase2.pipelines.register_model import (
    CANONICAL_METRIC_KEYS,
    compute_harmonic_mean_from_metrics,
    discover_champion_run,
    find_pickle_artifact,
    has_model_artifact,
)


class _FakeArtifact:
    """Mimica um artefato MLflow."""

    def __init__(self, path: str, is_dir: bool = False, file_size: int = 0) -> None:
        self.path = path
        self.is_dir = is_dir
        self.file_size = file_size


class _FakeRunInfo:
    """Mimica ``RunInfo`` simplificado."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.start_time = None


class _FakeRunData:
    """Mimica ``RunData`` simplificado com metrics e tags."""

    def __init__(self, metrics: dict[str, float], tags: dict[str, str]) -> None:
        self.metrics = metrics
        self.tags = tags


class _FakeRun:
    """Mimica ``Run`` simplificado."""

    def __init__(
        self,
        run_id: str,
        metrics: dict[str, float],
        tags: dict[str, str],
    ) -> None:
        self.info = _FakeRunInfo(run_id)
        self.data = _FakeRunData(metrics, tags)


class _FakeClient:
    """Cliente MLflow mockado para testes do register_model."""

    def __init__(
        self,
        experiment_id: str,
        runs: list[_FakeRun],
        artifacts_by_run: dict[str, list[_FakeArtifact]] | None = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._runs = runs
        self._artifacts = artifacts_by_run or {}

    def get_experiment_by_name(self, name: str) -> Any:
        class _Exp:
            def __init__(self, exp_id: str) -> None:
                self.experiment_id = exp_id

        if name == "missing-experiment":
            return None
        return _Exp(self._experiment_id)

    def search_runs(
        self,
        experiment_ids: list[str],
        max_results: int = 100,
        order_by: list[str] | None = None,
    ) -> list[_FakeRun]:
        return list(self._runs[:max_results])

    def list_artifacts(
        self, run_id: str, path: str | None = None
    ) -> list[_FakeArtifact]:
        artifacts = self._artifacts.get(run_id, [])
        if path is None:
            return artifacts
        prefix = path.rstrip("/") + "/"
        return [a for a in artifacts if a.path.startswith(prefix)]


def test_compute_harmonic_mean_returns_zero_for_zero_values() -> None:
    """compute_harmonic_mean returns 0.0 when any metric is zero."""
    metrics = {key: 0.1 for key in CANONICAL_METRIC_KEYS}
    metrics["recall_at_10"] = 0.0
    assert compute_harmonic_mean_from_metrics(metrics) == 0.0


def test_compute_harmonic_mean_returns_none_for_missing_key() -> None:
    """compute_harmonic_mean returns None when a canonical key is missing."""
    metrics = {
        "precision_at_10": 0.1,
        "recall_at_10": 0.1,
        "map_at_10": 0.1,
    }
    assert compute_harmonic_mean_from_metrics(metrics) is None


def test_compute_harmonic_mean_matches_known_value() -> None:
    """Harmonic mean matches a pre-computed reference value."""
    metrics = {
        "precision_at_10": 0.1391304347826087,
        "recall_at_10": 0.5489130434782609,
        "ndcg_at_10": 0.373511788301609,
        "map_at_10": 0.2550494593972855,
    }
    harmonic = compute_harmonic_mean_from_metrics(metrics)
    assert round(harmonic, 4) == 0.2563


def test_has_model_artifact_detects_directory(tmp_path: Path) -> None:
    """has_model_artifact returns True when 'model' directory is present."""
    client = _FakeClient(
        experiment_id="x",
        runs=[],
        artifacts_by_run={
            "r1": [_FakeArtifact("model", is_dir=True)],
        },
    )
    assert has_model_artifact(client, "r1") is True


def test_has_model_artifact_detects_pkl() -> None:
    """has_model_artifact returns True for any pickle artifact."""
    client = _FakeClient(
        experiment_id="x",
        runs=[],
        artifacts_by_run={
            "r1": [_FakeArtifact("foo.pkl"), _FakeArtifact("report.png")],
        },
    )
    assert has_model_artifact(client, "r1") is True


def test_has_model_artifact_returns_false_when_none() -> None:
    """has_model_artifact returns False when no relevant artifacts."""
    client = _FakeClient(
        experiment_id="x",
        runs=[],
        artifacts_by_run={"r1": [_FakeArtifact("report.png")]},
    )
    assert has_model_artifact(client, "r1") is False


def test_find_pickle_artifact_locates_pickle_inside_model_dir() -> None:
    """find_pickle_artifact locates pickle within the model directory."""
    client = _FakeClient(
        experiment_id="x",
        runs=[],
        artifacts_by_run={
            "r1": [
                _FakeArtifact("model"),
                _FakeArtifact("model/tmpXXX.pkl"),
                _FakeArtifact("model/MLmodel"),
            ],
        },
    )
    found = find_pickle_artifact(client, "r1")
    assert found == "model/tmpXXX.pkl"


def test_find_pickle_artifact_raises_when_none() -> None:
    """find_pickle_artifact raises when no pickle is found."""
    client = _FakeClient(
        experiment_id="x",
        runs=[],
        artifacts_by_run={"r1": [_FakeArtifact("model")]},
    )
    with pytest.raises(RuntimeError):
        find_pickle_artifact(client, "r1")


def test_discover_champion_run_ranks_by_harmonic_mean() -> None:
    """discover_champion_run returns the run with highest harmonic mean."""
    runs = [
        _FakeRun(
            run_id="r_a",
            metrics={
                "precision_at_10": 0.08,
                "recall_at_10": 0.315,
                "ndcg_at_10": 0.168,
                "map_at_10": 0.079,
            },
            tags={"mlflow.runName": "compare_popularity"},
        ),
        _FakeRun(
            run_id="r_b",
            metrics={
                "precision_at_10": 0.139,
                "recall_at_10": 0.549,
                "ndcg_at_10": 0.374,
                "map_at_10": 0.255,
            },
            tags={"mlflow.runName": "compare_item_knn"},
        ),
        _FakeRun(
            run_id="r_c",
            metrics={
                "precision_at_10": 0.141,
                "recall_at_10": 0.578,
                "ndcg_at_10": 0.355,
                "map_at_10": 0.225,
            },
            tags={"mlflow.runName": "compare_ease_torch"},
        ),
    ]
    client = _FakeClient(
        experiment_id="exp-1",
        runs=runs,
        artifacts_by_run={
            "r_a": [_FakeArtifact("model")],
            "r_b": [_FakeArtifact("model")],
            "r_c": [_FakeArtifact("model")],
        },
    )
    run_id, model_name, harmonic = discover_champion_run(
        client, "tech-challenge-comparison"
    )
    assert run_id == "r_b"
    assert model_name == "item_knn"
    assert harmonic == pytest.approx(0.2563, abs=1e-3)


def test_discover_champion_run_skips_runs_without_artifact() -> None:
    """Runs without model artifacts are skipped."""
    runs = [
        _FakeRun(
            run_id="r_top",
            metrics={
                "precision_at_10": 0.15,
                "recall_at_10": 0.55,
                "ndcg_at_10": 0.4,
                "map_at_10": 0.3,
            },
            tags={"mlflow.runName": "compare_ease_torch"},
        ),
        _FakeRun(
            run_id="r_low",
            metrics={
                "precision_at_10": 0.08,
                "recall_at_10": 0.32,
                "ndcg_at_10": 0.17,
                "map_at_10": 0.08,
            },
            tags={"mlflow.runName": "compare_popularity"},
        ),
    ]
    client = _FakeClient(
        experiment_id="exp-1",
        runs=runs,
        artifacts_by_run={
            "r_low": [_FakeArtifact("model")],
        },
    )
    run_id, model_name, _ = discover_champion_run(client, "tech-challenge-comparison")
    assert run_id == "r_low"
    assert model_name == "popularity"


def test_discover_champion_run_raises_for_missing_experiment() -> None:
    """Missing experiment name raises a clear RuntimeError."""
    client = _FakeClient(experiment_id="x", runs=[])
    with pytest.raises(RuntimeError, match="nao encontrado"):
        discover_champion_run(client, "missing-experiment")


def test_discover_champion_run_raises_when_no_elegible_run() -> None:
    """No elegible runs raises a clear RuntimeError."""
    runs = [
        _FakeRun(
            run_id="r1",
            metrics={"precision_at_10": 0.08},
            tags={"mlflow.runName": "compare_x"},
        ),
    ]
    client = _FakeClient(experiment_id="exp-1", runs=runs)
    with pytest.raises(RuntimeError, match="Nenhum run elegivel"):
        discover_champion_run(client, "tech-challenge-comparison")


def test_serialize_roundtrip_preserves_itemknn_state(tmp_path: Path) -> None:
    """Serializing/desserializing ItemKNN keeps trained state."""
    interactions: list[tuple[str, str]] = [
        ("u1", "i1"),
        ("u1", "i2"),
        ("u1", "i3"),
        ("u2", "i1"),
        ("u2", "i4"),
        ("u3", "i2"),
        ("u3", "i4"),
        ("u3", "i5"),
        ("u4", "i3"),
        ("u4", "i5"),
    ]
    knn = ItemKNNRecommender(default_limit=2)
    knn.fit(interactions)
    pkl = tmp_path / "knn.pkl"
    with pkl.open("wb") as handle:
        pickle.dump(knn, handle)
    restored = deserialize_recommender(pkl)
    assert isinstance(restored, ItemKNNRecommender)
    recommendations = restored.recommend("u1", limit=2)
    assert len(recommendations) == 2
    assert "i1" not in recommendations
    assert "i2" not in recommendations
    assert "i3" not in recommendations
