"""Carregamento e preparacao de interacoes do dataset RetailRocket.

Responsavel por ler events.csv, mapear identificadores para indices
inteiros, gerar amostras negativas e dividir treino/validacao por usuario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

# Eventos interpretados como feedback positivo implicito.
POSITIVE_EVENTS: tuple[str, ...] = ("view", "addtocart", "transaction")

# Colunas esperadas no arquivo events.csv do RetailRocket.
EVENT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
)


@dataclass(frozen=True, slots=True)
class InteractionData:
    """Dados preprocessados prontos para alimentar o modelo neural.

    Args:
        user_ids: Tensores de indices de usuarios (treino).
        item_ids: Tensores de indices de itens (treino).
        labels: Tensores de labels (1 positiva, 0 negativa).
        val_user_ids: Tensores de indices de usuarios (validacao).
        val_item_ids: Tensores de indices de itens (validacao).
        val_labels: Tensores de labels da validacao.
        num_users: Total de usuarios unicos.
        num_items: Total de itens unicos.
    """

    user_ids: torch.Tensor
    item_ids: torch.Tensor
    labels: torch.Tensor
    val_user_ids: torch.Tensor
    val_item_ids: torch.Tensor
    val_labels: torch.Tensor
    num_users: int
    num_items: int


def load_events(events_path: Path) -> pd.DataFrame:
    """Le events.csv mantendo apenas interacoes positivas.

    Args:
        events_path: Caminho para o arquivo events.csv.

    Returns:
        DataFrame com colunas visitorid, itemid, timestamp ordenado por tempo.
    """
    df = pd.read_csv(events_path, usecols=list(EVENT_COLUMNS))
    df = df[df["event"].isin(POSITIVE_EVENTS)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["visitorid", "itemid", "timestamp"]]


def build_id_mappings(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Cria mapeamentos estaveis de visitorid/itemid para indices densos.

    Args:
        df: DataFrame com colunas visitorid e itemid.

    Returns:
        Tupla (user2idx, item2idx).
    """
    unique_users = df["visitorid"].drop_duplicates().sort_values().tolist()
    unique_items = df["itemid"].drop_duplicates().sort_values().tolist()
    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    return user2idx, item2idx


def encode_interactions(
    df: pd.DataFrame, user2idx: dict[int, int], item2idx: dict[int, int]
) -> pd.DataFrame:
    """Adiciona colunas user_idx e item_idx a partir dos mapeamentos.

    Args:
        df: DataFrame com colunas visitorid e itemid.
        user2idx: Mapeamento visitorid -> indice.
        item2idx: Mapeamento itemid -> indice.

    Returns:
        DataFrame com colunas user_idx, item_idx, timestamp.
    """
    encoded = df.copy()
    encoded["user_idx"] = encoded["visitorid"].map(user2idx)
    encoded["item_idx"] = encoded["itemid"].map(item2idx)
    return encoded[["user_idx", "item_idx", "timestamp"]]


def split_train_validation(
    encoded: pd.DataFrame, validation_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide temporal por usuario: ultimas interacoes ficam para validacao.

    Args:
        encoded: DataFrame ordenado por timestamp com colunas user_idx, item_idx.
        validation_ratio: Proporcao das interacoes de cada usuario para validacao.

    Returns:
        Tupla (train_df, val_df).
    """
    val_parts = []
    for _, group in encoded.groupby("user_idx", sort=False):
        n_val = max(1, int(len(group) * validation_ratio))
        val_parts.append(group.tail(n_val))
    val_df = pd.concat(val_parts, ignore_index=True)
    train_df = encoded.drop(val_df.index).reset_index(drop=True)
    return train_df, val_df.reset_index(drop=True)


def sample_negatives(
    positives: pd.DataFrame, num_items: int, num_negatives: int, seed: int = 42
) -> pd.DataFrame:
    """Gera interacoes negativas (itens nao consumidos pelo usuario).

    Args:
        positives: DataFrame com colunas user_idx, item_idx.
        num_items: Total de itens disponiveis.
        num_negatives: Numero de negativas por positiva.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame balanceado com label 1 (positivo) e 0 (negativo).
    """
    positives = positives[["user_idx", "item_idx"]].copy()
    positives["label"] = 1
    negatives = _sample_negative_rows(positives, num_items, num_negatives, seed)
    combined = pd.concat([positives, negatives], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def _sample_negative_rows(
    positives: pd.DataFrame, num_items: int, num_negatives: int, seed: int
) -> pd.DataFrame:
    """Amostra itens negativos por usuario respeitando o historico positivo."""
    rng = torch.Generator().manual_seed(seed)
    seen = {
        (int(user), int(item))
        for user, item in zip(positives["user_idx"], positives["item_idx"], strict=True)
    }
    rows: list[dict[str, int]] = []
    for user, pos_count in positives["user_idx"].value_counts().items():
        target = int(pos_count) * num_negatives
        sampled = 0
        while sampled < target:
            item = int(torch.randint(num_items, (1,), generator=rng).item())
            if (user, item) in seen:
                continue
            seen.add((user, item))
            rows.append({"user_idx": int(user), "item_idx": item, "label": 0})
            sampled += 1
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])


def to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converte um DataFrame com label em tensores user, item, label.

    Args:
        df: DataFrame com colunas user_idx, item_idx, label.

    Returns:
        Tupla (user_tensor, item_tensor, label_tensor).
    """
    users = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
    items = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    return users, items, labels


def prepare_interaction_data(
    events_path: Path,
    num_negatives: int = 4,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> InteractionData:
    """Pipeline completo: carrega, mapeia, divide, amostra e converte.

    Args:
        events_path: Caminho para events.csv.
        num_negatives: Negativas por positiva no treino.
        validation_ratio: Proporcao de validacao por usuario.
        seed: Semente para reprodutibilidade.

    Returns:
        InteractionData pronto para treino.
    """
    df = load_events(events_path)
    user2idx, item2idx = build_id_mappings(df)
    encoded = encode_interactions(df, user2idx, item2idx)
    train_df, val_df = split_train_validation(encoded, validation_ratio)
    train_balanced = sample_negatives(train_df, len(item2idx), num_negatives, seed)
    val_balanced = sample_negatives(val_df, len(item2idx), num_negatives, seed)
    users, items, labels = to_tensors(train_balanced)
    v_users, v_items, v_labels = to_tensors(val_balanced)
    return InteractionData(
        user_ids=users,
        item_ids=items,
        labels=labels,
        val_user_ids=v_users,
        val_item_ids=v_items,
        val_labels=v_labels,
        num_users=len(user2idx),
        num_items=len(item2idx),
    )
