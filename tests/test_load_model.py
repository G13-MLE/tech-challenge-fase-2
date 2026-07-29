"""Tests for the inference CLI loader (load_model)."""

from __future__ import annotations

import pandas as pd
import pytest

from techchallenge_fase2.inference.load_model import (
    list_versions,
    recommend,
)


class _FakeVersions:
    """Mock de ModelVersion para list-versions."""

    def __init__(self, version: str, stage: str, run_id: str, status: str) -> None:
        self.version = version
        self.current_stage = stage
        self.run_id = run_id
        self.status = status


class _FakePyFunc:
    """pyfunc fake para testar ``recommend``."""

    def __init__(self, recs: dict[str, list[str]]) -> None:
        self._recs = recs

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = [self._recs.get(uid, []) for uid in frame["user_id"]]
        return pd.DataFrame({"user_id": list(frame["user_id"]), "recommendations": out})


def test_recommend_returns_predictions_from_pyfunc() -> None:
    """recommend maps pyfunc output to a flat list[str]."""
    model = _FakePyFunc({"u1": ["i1", "i2", "i3"]})
    recommendations = recommend(model, "u1", limit=3)
    assert recommendations == ["i1", "i2", "i3"]


def test_recommend_falls_back_to_first_column_when_no_recommendations() -> None:
    """recommend tolerates a flat list response from predict."""
    model = _FakePyFunc({"u1": []})
    out = pd.DataFrame({"user_id": ["u1"], "recommendations": [["i1", "i2"]]})
    real_predict = model.predict

    def patched(df: pd.DataFrame) -> pd.DataFrame:
        return out

    model.predict = patched
    recommendations = recommend(model, "u1", limit=2)
    assert recommendations == ["i1", "i2"]
    model.predict = real_predict


def test_list_versions_prints_versions(capsys: pytest.CaptureFixture[str]) -> None:
    """list_versions lists versions via the mocked client."""

    class _FakeClient:
        def get_latest_versions(self, name, stages=None):  # noqa: ANN001
            return [
                _FakeVersions("1", "Production", "r1", "READY"),
                _FakeVersions("2", "Staging", "r2", "READY"),
            ]

    import techchallenge_fase2.inference.load_model as lm

    original_client = getattr(lm, "mlflow")
    try:

        class _MlflowStub:
            tracking = type(
                "_T",
                (),
                {"MlflowClient": lambda *a, **k: _FakeClient()},
            )()

        lm.mlflow = _MlflowStub()  # type: ignore[attr-defined]
        rc = list_versions("M", stage=None)
        captured = capsys.readouterr()
        assert rc == 0
        assert "Production" in captured.out
        assert "Staging" in captured.out
    finally:
        lm.mlflow = original_client  # type: ignore[attr-defined]


def test_list_versions_handles_no_versions(capsys: pytest.CaptureFixture[str]) -> None:
    """list_versions returns 0 and prints a hint when no versions exist."""

    class _FakeClient:
        def get_latest_versions(self, name, stages=None):  # noqa: ANN001
            return []

    import techchallenge_fase2.inference.load_model as lm

    original_client = getattr(lm, "mlflow")
    try:

        class _MlflowStub:
            tracking = type(
                "_T",
                (),
                {"MlflowClient": lambda *a, **k: _FakeClient()},
            )()

        lm.mlflow = _MlflowStub()  # type: ignore[attr-defined]
        rc = list_versions("M", stage=None)
        captured = capsys.readouterr()
        assert rc == 0
        assert "Nenhuma versão" in captured.out
    finally:
        lm.mlflow = original_client  # type: ignore[attr-defined]
