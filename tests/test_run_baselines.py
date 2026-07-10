"""Testes unitarios para a divisao temporal via pipelines.run_baselines.

Cobrem o comportamento de `temporal_holdout_split` quando a coluna de
timestamp esta ausente do DataFrame de interacoes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from techchallenge_fase2.pipelines.run_baselines import temporal_holdout_split


def make_df_without_timestamp() -> pd.DataFrame:
    """Cria um DataFrame de interacoes sem coluna de timestamp."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "item_id": ["i1", "i2", "i1", "i3", "i2", "i4", "i3", "i4"],
        }
    )


def test_temporal_holdout_split_without_timestamp_raises_value_error() -> None:
    """Ausencia de timestamp deve gerar ValueError claro, nao KeyError.

    O fallback anterior chamava chronological_holdout_split com a coluna
    inexistente, provocando KeyError opaco. O contrato deve ser um erro
    explicito explicando a dependencia de timestamp.
    """
    df = make_df_without_timestamp()
    with pytest.raises(ValueError, match="timestamp"):
        temporal_holdout_split(df, test_ratio=0.15, random_seed=42)


def test_temporal_holdout_split_with_timestamp_works() -> None:
    """Com timestamp presente, o split cronologico deve funcionar normalmente."""
    df = make_df_without_timestamp().assign(timestamp=list(range(1, 9)))
    train_interactions, ground_truth, _ = temporal_holdout_split(
        df, test_ratio=0.25, random_seed=42
    )
    assert len(train_interactions) > 0
    assert isinstance(ground_truth, dict)
