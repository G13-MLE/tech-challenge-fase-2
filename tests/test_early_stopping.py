"""Testes do EarlyStopping (modulo training)."""

from __future__ import annotations

import pytest

from techchallenge_fase2.training.early_stopping import EarlyStopping


def test_step_registra_melhora_em_modo_max() -> None:
    """Scores crescentes sao considerados melhorias em modo max."""
    early = EarlyStopping(patience=2, mode="max")
    assert early.step(0.5) is True
    assert early.best_score == 0.5
    assert early.step(0.6) is True
    assert early.step(0.55) is False
    assert not early.should_stop


def test_step_interrompe_apos_patience_sem_melhora() -> None:
    """Apos patience epocas sem melhora, should_stop passa a True."""
    early = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    early.step(0.5)
    early.step(0.505)
    early.step(0.505)
    assert early.should_stop
    assert early.best_score == 0.5


def test_modo_min_inverte_logica_de_melhora() -> None:
    """Em modo min, scores decrescentes sao melhorias significativas."""
    early = EarlyStopping(patience=1, min_delta=0.0, mode="min")
    assert early.step(1.0) is True
    assert early.step(0.9) is True
    assert early.step(0.95) is False


def test_patience_invalida_levanta_erro() -> None:
    """patience deve ser positivo."""
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)


def test_mode_invalido_levanta_erro() -> None:
    """mode deve ser 'min' ou 'max'."""
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(mode="media")
