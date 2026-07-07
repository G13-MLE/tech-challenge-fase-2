"""Loop de treinamento do NCF com validacao, early stopping e checkpoints.

Orquestra epocas de treino por mini-batches (BCEWithLogitsLoss + Adam),
avalia o conjunto de validacao a cada epoca com AUC-ROC e aciona o early
stopping e o salvamento automatico de checkpoints do melhor modelo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from techchallenge_fase2.data import InteractionData
from techchallenge_fase2.models.ncf import NeuralCollaborativeFiltering
from techchallenge_fase2.training.checkpoint import (
    save_best_checkpoint,
    save_checkpoint,
)
from techchallenge_fase2.training.early_stopping import EarlyStopping


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hiperparametros do loop de treinamento.

    Args:
        learning_rate: Taxa de aprendizado do otimizador Adam.
        batch_size: Tamanho do mini-batch.
        epochs: Numero maximo de epocas.
        patience: Epocas sem melhora antes de parar (early stopping).
        min_delta: Melhoria minima considerada significativa.
        device: Dispositivo alvo ("cpu" ou "cuda").
    """

    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    patience: int = 5
    min_delta: float = 1e-4
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Metricas coletadas a cada epoca de treino e validacao.

    Args:
        train_losses: Perda media por epoca no treino.
        val_metrics: AUC de validacao por epoca.
        stopped_epoch: Epoca em que o treino parou (-1 se nao parou cedo).
    """

    train_losses: tuple[float, ...]
    val_metrics: tuple[float, ...]
    stopped_epoch: int


Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_dataloader(
    users: torch.Tensor, items: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    """Cria um DataLoader por mini-batches a partir dos tensores."""
    dataset = TensorDataset(users, items, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_auc(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Calcula o AUC-ROC das probabilidades contra as labels binarias."""
    if labels.unique().numel() <= 1:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels.numpy(), probabilities.numpy()))


def move_batch(batch, device):
    """Move os tensores do batch para o dispositivo alvo."""
    return tuple(tensor.to(device) for tensor in batch)


def run_epoch(model, loader, optimizer, criterion, device):
    """Executa uma epoca de treino e retorna a perda media."""
    model.train()
    total_loss = 0.0
    for users, items, labels in loader:
        users, items, labels = move_batch((users, items, labels), device)
        optimizer.zero_grad()
        loss = criterion(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * users.size(0)
    return total_loss / len(loader.dataset)


def evaluate_auc(model, loader, device) -> float:
    """Calcula o AUC do modelo no conjunto de validacao."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for users, items, labels in loader:
            users, items, labels = move_batch((users, items, labels), device)
            all_probs.append(torch.sigmoid(model(users, items)))
            all_labels.append(labels)
    probabilities = torch.cat(all_probs).cpu()
    labels_tensor = torch.cat(all_labels).cpu()
    return compute_auc(probabilities, labels_tensor)


class Trainer:
    """Orquestra o loop de treino do NCF com early stopping e checkpoints."""

    def __init__(self, config: TrainingConfig, checkpoint_dir: Path) -> None:
        """Inicializa o treinador com hiperparametros e diretorio de saida."""
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(config.device)
        self.best_path = checkpoint_dir / "best.pt"
        self.last_path = checkpoint_dir / "last.pt"

    def train(
        self, model: NeuralCollaborativeFiltering, data: InteractionData
    ) -> TrainingHistory:
        """Executa o treino completo e retorna o historico de metricas."""
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        train_loader = build_dataloader(
            data.user_ids, data.item_ids, data.labels, self.config.batch_size
        )
        val_loader = build_dataloader(
            data.val_user_ids,
            data.val_item_ids,
            data.val_labels,
            self.config.batch_size,
        )
        early = EarlyStopping(
            patience=self.config.patience, min_delta=self.config.min_delta, mode="max"
        )
        return self._run_loop(
            model, optimizer, criterion, train_loader, val_loader, early
        )

    def _run_loop(self, model, optimizer, criterion, train_loader, val_loader, early):
        """Itera pelas epocas aplicando treino, validacao e checkpoints."""
        train_losses: list[float] = []
        val_metrics: list[float] = []
        stopped_epoch = -1
        for epoch in range(1, self.config.epochs + 1):
            loss = run_epoch(model, train_loader, optimizer, criterion, self.device)
            val_auc = evaluate_auc(model, val_loader, self.device)
            train_losses.append(loss)
            val_metrics.append(val_auc)
            improved = early.step(val_auc)
            if improved:
                save_best_checkpoint(model, self.best_path, val_auc)
            save_checkpoint(model, optimizer, epoch, val_auc, self.last_path)
            if early.should_stop:
                stopped_epoch = epoch
                break
        return TrainingHistory(
            train_losses=tuple(train_losses),
            val_metrics=tuple(val_metrics),
            stopped_epoch=stopped_epoch,
        )
