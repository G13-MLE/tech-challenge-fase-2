"""Testes de checkpoints do módulo training."""

from __future__ import annotations

from pathlib import Path

import torch

from techchallenge_fase2.models.ncf import NCFConfig, NeuralCollaborativeFiltering
from techchallenge_fase2.training.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)


def _model() -> NeuralCollaborativeFiltering:
    return NeuralCollaborativeFiltering(
        NCFConfig(num_users=4, num_items=5, embedding_dim=4)
    )


def test_save_e_load_checkpoint_restaura_estado(tmp_path: Path) -> None:
    """Round-trip de save/load restaura pesos e otimizador."""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, optimizer, epoch=3, metric=0.7, path=path)

    novo = _model()
    novo_optimizer = torch.optim.Adam(novo.parameters(), lr=0.001)
    epoch, metric = load_checkpoint(novo, novo_optimizer, path)

    assert epoch == 3
    assert metric == 0.7
    for a, b in zip(
        model.state_dict().values(), novo.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_save_best_e_load_model_from_checkpoint(tmp_path: Path) -> None:
    """Melhor checkpoint reconstrói NCF pronto para avaliação."""
    model = _model()
    path = tmp_path / "best.pt"

    save_best_checkpoint(model, path, metric=0.9)
    loaded = load_model_from_checkpoint(path)

    assert isinstance(loaded, NeuralCollaborativeFiltering)
    assert loaded.training is False
    for a, b in zip(
        model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
