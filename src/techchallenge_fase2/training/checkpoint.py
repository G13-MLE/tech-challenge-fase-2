"""Salvamento e carregamento de checkpoints do modelo neural.

Os checkpoints preservam pesos, estado do otimizador, época atual e
melhor métrica de validação, permitindo retomar o treino ou restaurar
o melhor modelo apos early stopping.
"""

from __future__ import annotations

from pathlib import Path

import torch

from techchallenge_fase2.models.ncf import NeuralCollaborativeFiltering


def save_checkpoint(
    model: NeuralCollaborativeFiltering,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    path: Path,
) -> None:
    """Salva um checkpoint completo do estado de treino.

    Args:
        model: Modelo NCF cujos pesos serao persistidos.
        optimizer: Otimizador cujo estado sera persistido.
        epoch: Epoca atual do treinamento.
        metric: Valor da métrica de validação associada.
        path: Caminho do arquivo .pt onde o checkpoint sera escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": model.config,
            "epoch": epoch,
            "metric": metric,
        },
        path,
    )


def load_checkpoint(
    model: NeuralCollaborativeFiltering,
    optimizer: torch.optim.Optimizer | None,
    path: Path,
) -> tuple[int, float]:
    """Restaura pesos e otimizador a partir de um checkpoint.

    Args:
        model: Modelo NCF que recebera os pesos carregados.
        optimizer: Otimizador a restaurar (None para não restaurar).
        path: Caminho do arquivo .pt de checkpoint.

    Returns:
        Tupla (epoch, metric) registradas no checkpoint.
    """
    payload = torch.load(path, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]), float(payload["metric"])


def save_best_checkpoint(
    model: NeuralCollaborativeFiltering, path: Path, metric: float
) -> None:
    """Salva apenas os pesos do modelo (usado para o melhor estado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": model.config, "metric": metric},
        path,
    )


def load_model_from_checkpoint(path: Path) -> NeuralCollaborativeFiltering:
    """Reconstrui um NCF com os pesos do melhor checkpoint.

    Args:
        path: Caminho do arquivo .pt de checkpoint.

    Returns:
        Modelo NCF pronto para recomendação ou avaliação.
    """
    payload = torch.load(path, weights_only=False)
    model = NeuralCollaborativeFiltering(payload["config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
