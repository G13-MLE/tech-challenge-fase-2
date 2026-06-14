"""Criacao do dataset para o MLP: split cronologico + sessao, amostragem positiva/negativa.

Split cronologico 70/15/15 para definir treino/validacao/teste.
Dentro de cada particao, cada sessao e dividida em:
- historico h = [s1, ..., sm-1] (input)
- target t = sm (ultimo evento como label)

Amostras positivas: itens com os quais o usuario interagiu em t.
Amostras negativas: itens que o usuario nao viu, amostrados aleatoriamente.
"""

import random

import polars as pl

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NEG_SAMPLE_RATIO = 4


def chronological_split(
    events: pl.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Divide o DataFrame em treino, validacao e teste por timestamp.

    Args:
        events: DataFrame ordenado por timestamp.
        train_ratio: Proporcao para treino.
        val_ratio: Proporcao para validacao. Teste = 1 - train - val.

    Returns:
        Tupla (train, val, test).
    """
    total = events.height
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    events_sorted = events.sort("timestamp")
    train = events_sorted.slice(0, train_end)
    val = events_sorted.slice(train_end, val_end - train_end)
    test = events_sorted.slice(val_end)

    return train, val, test


def split_sessions_for_mlp(
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide cada sessao em historico (h) e target (t).

    Para cada sessao com >= 2 eventos:
    - historico: eventos s1 ate sm-1
    - target: ultimo evento sm

    Sessoes com 1 evento entram so no historico (cold start).

    Args:
        events: DataFrame com coluna session_id.

    Returns:
        Tupla (history, target) DataFrames.
    """
    session_groups = events.group_by("session_id").agg(
        pl.len().alias("n_events"),
        pl.col("event").last().alias("target_event"),
        pl.col("itemid").last().alias("target_itemid"),
        pl.col("datetime").last().alias("target_datetime"),
    )

    target = session_groups.filter(pl.col("n_events") >= 2).select(
        "session_id",
        "target_event",
        "target_itemid",
        "target_datetime",
    )

    history = events.join(
        target.select("session_id"),
        on="session_id",
        how="anti",
    )

    last_events = (
        events.sort(["session_id", "timestamp"])
        .group_by("session_id")
        .agg(pl.all().last())
        .join(target.select("session_id"), on="session_id", how="semi")
    )

    history = pl.concat([history, last_events]).sort(["session_id", "timestamp"])

    return history, target


def build_mlp_pairs(
    history: pl.DataFrame,
    target: pl.DataFrame,
    all_item_ids: set[int],
    neg_ratio: int = NEG_SAMPLE_RATIO,
    seed: int = SEED,
) -> pl.DataFrame:
    """Cria pares positivo/negativo para o MLP.

    Para cada sessao com target:
    - Amostra positiva: (session_id, target_itemid, label=1)
    - Amostras negativas: (session_id, random_item, label=0) x neg_ratio

    Itens negativos sao amostrados do conjunto de itens que o usuario
    NAO interagiu em toda a sessao.

    Args:
        history: DataFrame com historico de eventos por sessao.
        target: DataFrame com target por sessao.
        all_item_ids: Conjunto de todos os itemids disponiveis.
        neg_ratio: Numero de amostras negativas por positiva.
        seed: Seed para reprodutibilidade.

    Returns:
        DataFrame com colunas session_id, itemid, label.
    """
    rng = random.Random(seed)

    items_interacted = history.group_by("session_id").agg(
        pl.col("itemid").unique().alias("interacted_items"),
    )

    target_with_items = target.join(items_interacted, on="session_id")

    pairs = []

    for row in target_with_items.iter_rows(named=True):
        sid = row["session_id"]
        pos_item = row["target_itemid"]
        interacted = set(row["interacted_items"])

        candidates = all_item_ids - interacted
        candidates = list(candidates)

        if len(candidates) == 0:
            continue

        pairs.append({"session_id": sid, "itemid": pos_item, "label": 1})

        n_neg = min(neg_ratio, len(candidates))
        neg_items = rng.sample(candidates, n_neg)
        for neg_item in neg_items:
            pairs.append({"session_id": sid, "itemid": neg_item, "label": 0})

    return pl.DataFrame(pairs).cast({"itemid": pl.Int64, "label": pl.Int8})


def build_mlp_dataset(
    events: pl.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    neg_ratio: int = NEG_SAMPLE_RATIO,
    seed: int = SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Pipeline completo: split cronologico + split por sessao + amostragem.

    Args:
        events: DataFrame com session_id e todas as features.
        train_ratio: Proporcao de treino.
        val_ratio: Proporcao de validacao.
        neg_ratio: Ratio de amostras negativas.
        seed: Seed para reprodutibilidade.

    Returns:
        Tupla (train_pairs, val_pairs, test_pairs).
    """
    train_events, val_events, test_events = chronological_split(
        events,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    all_item_ids = set(events["itemid"].unique().to_list())

    datasets = []
    for split_events in [train_events, val_events, test_events]:
        history, target = split_sessions_for_mlp(split_events)
        if target.height > 0:
            pairs = build_mlp_pairs(
                history, target, all_item_ids, neg_ratio=neg_ratio, seed=seed
            )
            datasets.append(pairs)
        else:
            datasets.append(
                pl.DataFrame(
                    schema={"session_id": pl.Utf8, "itemid": pl.Int64, "label": pl.Int8}
                )
            )

    return datasets[0], datasets[1], datasets[2]
