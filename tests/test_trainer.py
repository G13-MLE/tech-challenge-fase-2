"""Testes do Trainer (loop de treino com validacao e early stopping)."""

from __future__ import annotations

from pathlib import Path

import torch

from techchallenge_fase2.data import InteractionData
from techchallenge_fase2.models.ncf import NCFConfig, NeuralCollaborativeFiltering
from techchallenge_fase2.training.trainer import (
    Trainer,
    TrainingConfig,
    build_dataloader,
    compute_auc,
)


def _toy_interaction() -> InteractionData:
    users = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
    labels = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    return InteractionData(
        user_ids=users,
        item_ids=items,
        labels=labels,
        val_user_ids=users,
        val_item_ids=items,
        val_labels=labels,
        num_users=3,
        num_items=3,
    )


def test_trainer_executa_epocas_e_salva_checkpoints(tmp_path: Path) -> None:
    """Trainer treina NCF, historico coerente e melhores/last checkpoints salvos."""
    config = TrainingConfig(
        learning_rate=0.01, batch_size=2, epochs=3, patience=5, device="cpu"
    )
    trainer = Trainer(config, tmp_path)
    model = NeuralCollaborativeFiltering(
        NCFConfig(num_users=3, num_items=3, embedding_dim=4)
    )

    history = trainer.train(model, _toy_interaction())

    assert len(history.train_losses) == 3
    assert len(history.val_metrics) == 3
    assert history.stopped_epoch == -1
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()


def test_trainer_dispara_early_stopping(tmp_path: Path) -> None:
    """Sem melhora, patience atingido encerra o treino antes de todas as epocas."""
    config = TrainingConfig(
        learning_rate=0.01, batch_size=2, epochs=10, patience=1, device="cpu"
    )
    trainer = Trainer(config, tmp_path)
    model = NeuralCollaborativeFiltering(
        NCFConfig(num_users=3, num_items=3, embedding_dim=4)
    )

    history = trainer.train(model, _toy_interaction())

    assert history.stopped_epoch >= 1
    assert len(history.train_losses) < config.epochs


def test_compute_auc_retorna_neutro_quando_label_unica() -> None:
    """AUC=0.5 quando a validacao tem apenas uma classe."""
    probs = torch.tensor([0.1, 0.9])
    labels = torch.tensor([1.0, 1.0], dtype=torch.float32)
    assert compute_auc(probs, labels) == 0.5


def test_build_dataloader_embaralha_preservando_batches() -> None:
    """DataLoader criado alterna por mini-batches do tamanho configurado."""
    users = torch.zeros(8, dtype=torch.long)
    items = torch.zeros(8, dtype=torch.long)
    labels = torch.ones(8, dtype=torch.float32)
    loader = build_dataloader(users, items, labels, batch_size=4)
    assert len(loader) == 2
    batches = list(loader)
    assert all(len(batch) == 3 for batch in batches)
