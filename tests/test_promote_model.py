"""Tests for the promote pipeline (validation + Production transition).

Mock do ``MlflowClient`` para evitar dependencia de servidor real.
"""

from __future__ import annotations

import pandas as pd
import pytest

from techchallenge_fase2.pipelines.promote_model import (
    evaluate_staging,
    harmonic_mean_at_k,
    load_reference_harmonic_mean,
    predict_batch,
    validate_and_promote,
)


class _FakeRecommenderPyFunc:
    """pyfunc fake: devolve recomendacoes pre-definidas."""

    def __init__(self, recs: dict[str, list[str]]) -> None:
        self._recs = recs

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        recs_out: list[list[str]] = []
        for user_id in frame["user_id"]:
            recs_out.append(self._recs.get(user_id, []))
        return pd.DataFrame(
            {"user_id": list(frame["user_id"]), "recommendations": recs_out}
        )


def test_harmonic_mean_at_k_with_known_metrics() -> None:
    """harmonic_mean_at_k matches a pre-computed reference value."""
    metrics = {
        "precision@10": 0.1391,
        "recall@10": 0.5489,
        "ndcg@10": 0.3735,
        "map@10": 0.2550,
    }
    harmonic = harmonic_mean_at_k(metrics, k=10)
    assert harmonic == pytest.approx(0.2563, abs=1e-3)


def test_harmonic_mean_at_k_returns_zero_for_missing() -> None:
    """harmonic_mean_at_k returns 0.0 when a metric is missing."""
    metrics = {"precision@10": 0.1, "recall@10": 0.1, "map@10": 0.1}
    assert harmonic_mean_at_k(metrics, k=10) == 0.0


def test_harmonic_mean_at_k_returns_zero_with_zero_metric() -> None:
    """harmonic_mean_at_k returns 0.0 when any value is zero."""
    metrics = {
        "precision@10": 0.1,
        "recall@10": 0.0,
        "ndcg@10": 0.1,
        "map@10": 0.1,
    }
    assert harmonic_mean_at_k(metrics, k=10) == 0.0


def test_predict_batch_produces_recommendations_per_user() -> None:
    """predict_batch maps user_id -> list[str] via pyfunc."""
    pyfunc = _FakeRecommenderPyFunc(
        {
            "u1": ["i3", "i4", "i1"],
            "u2": ["i5"],
        }
    )
    ground_truth = {"u1": {"i1"}, "u2": {"i5"}, "u3": {"i9"}}
    recs = predict_batch(pyfunc, ground_truth, limit=3)
    assert set(recs.keys()) == {"u1", "u2", "u3"}
    assert recs["u1"] == ["i3", "i4", "i1"]
    assert recs["u2"] == ["i5"]
    assert recs["u3"] == []


def test_evaluate_staging_computes_real_harmonic_mean() -> None:
    """evaluate_staging returns the harmonic mean of the predictions."""
    pyfunc = _FakeRecommenderPyFunc(
        {
            "u1": ["i1", "i2", "i3", "i5", "i6", "i7", "i8", "i9", "iA", "iB"],
            "u2": ["i4", "i5", "i6", "i7", "i8", "i9", "iA", "iB", "iC", "iD"],
        }
    )
    ground_truth = {"u1": {"i1", "i2", "i3"}, "u2": {"i4", "i5"}}
    observed = evaluate_staging(pyfunc, ground_truth)
    assert observed > 0.0
    assert observed <= 1.0


def test_load_reference_harmonic_mean_reads_first_row(tmp_path) -> None:
    """load_reference_harmonic_mean returns the best model's value."""
    csv_path = tmp_path / "comparison.csv"
    pd.DataFrame(
        [
            {"model": "item_knn", "harmonic_mean_at_10": 0.2563},
            {"model": "random", "harmonic_mean_at_10": 0.05},
        ]
    ).to_csv(csv_path, index=False)
    assert load_reference_harmonic_mean(str(csv_path)) == pytest.approx(0.2563)


def test_load_reference_harmonic_mean_returns_none_missing_file() -> None:
    """Returns None when the CSV file is missing."""
    assert load_reference_harmonic_mean("/nonexistent/file.csv") is None


def test_validate_and_promote_raises_when_reference_zero(monkeypatch, tmp_path) -> None:
    """Promotion aborted when the reference value is zero."""
    csv_path = tmp_path / "comp.csv"
    pd.DataFrame([{"model": "broken", "harmonic_mean_at_10": 0.0}]).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.load_staging_model",
        lambda name: (_FakeRecommenderPyFunc({}), "1"),
    )
    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.build_test_ground_truth",
        lambda path: (None, {"u1": {"i1"}}),
    )
    with pytest.raises(RuntimeError, match="zero"):
        validate_and_promote(
            model_name="M",
            tolerance=0.05,
            test_path="ignored",
            reference_csv=str(csv_path),
            dry_run=True,
        )


def test_validate_and_promote_raises_when_outside_tolerance(
    monkeypatch, tmp_path
) -> None:
    """Promotion aborted when observed is outside relative tolerance."""
    csv_path = tmp_path / "comp.csv"
    pd.DataFrame([{"model": "champ", "harmonic_mean_at_10": 0.5}]).to_csv(
        csv_path, index=False
    )

    pyfunc = _FakeRecommenderPyFunc({"u1": ["i5", "i6"]})
    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.load_staging_model",
        lambda name: (pyfunc, "1"),
    )
    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.build_test_ground_truth",
        lambda path: (None, {"u1": {"i1"}}),
    )
    with pytest.raises(RuntimeError, match="Validacao falhou"):
        validate_and_promote(
            model_name="M",
            tolerance=0.05,
            test_path="ignored",
            reference_csv=str(csv_path),
            dry_run=True,
        )


def test_validate_and_promote_dry_run_does_not_promote(monkeypatch, tmp_path) -> None:
    """In dry-run, promotion does not transition stage even when valid."""
    csv_path = tmp_path / "comp.csv"
    pd.DataFrame([{"model": "champ", "harmonic_mean_at_10": 0.012}]).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.evaluate_staging",
        lambda model, gt: 0.012,
    )
    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.load_staging_model",
        lambda name: (_FakeRecommenderPyFunc({}), "1"),
    )
    monkeypatch.setattr(
        "techchallenge_fase2.pipelines.promote_model.build_test_ground_truth",
        lambda path: (None, {}),
    )
    version = validate_and_promote(
        model_name="M",
        tolerance=0.05,
        test_path="ignored",
        reference_csv=str(csv_path),
        dry_run=True,
    )
    assert version == "1"
