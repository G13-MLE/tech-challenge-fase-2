"""Definicao de sessoes de usuario no dataset Retail Rocket.

Uma sessao e uma sequencia contínua de atividade de um mesmo visitor.
Nova sessao inicia quando:
1. Gap entre eventos consecutivos >= 30 minutos, OU
2. O evento anterior e 'transaction' E o evento atual NAO e 'transaction'
   (transacoes consecutivas ficam na mesma sessao; so volta a navegar
   inicia nova sessao).
"""

import polars as pl


SESSION_GAP_MINUTES = 30


def create_sessions(
    events: pl.DataFrame,
    gap_minutes: float = SESSION_GAP_MINUTES,
) -> pl.DataFrame:
    """Adiciona coluna session_id ao DataFrame de events.

    Args:
        events: DataFrame com colunas visitorid, timestamp, event, datetime.
            Deve conter a coluna 'event' com valores 'view', 'addtocart',
            'transaction'.
        gap_minutes: Gap em minutos para considerar nova sessao. Default 30.

    Returns:
        DataFrame original com coluna 'session_id' adicionada no formato
        '{visitorid}_{session_number}'.
    """
    gap_ms = gap_minutes * 60_000

    result = (
        events.sort(["visitorid", "timestamp"])
        .with_columns(
            pl.col("timestamp")
            .diff()
            .cast(pl.Float64)
            .over("visitorid")
            .alias("delta_ms"),
            pl.col("event").shift(1).over("visitorid").alias("prev_event"),
        )
        .with_columns(
            (
                pl.col("delta_ms").is_null()
                | (pl.col("delta_ms") >= gap_ms)
                | (
                    (pl.col("prev_event") == "transaction")
                    & (pl.col("event") != "transaction")
                )
            ).alias("is_new_session"),
        )
        .with_columns(
            session_per_visitor=pl.col("is_new_session")
            .cast(pl.Int32)
            .cum_sum()
            .over("visitorid"),
        )
        .with_columns(
            session_id=pl.col("visitorid").cast(pl.Utf8)
            + "_"
            + pl.col("session_per_visitor").cast(pl.Utf8),
        )
        .drop(["delta_ms", "prev_event", "is_new_session", "session_per_visitor"])
    )

    return result


def session_stats(events_with_session: pl.DataFrame) -> pl.DataFrame:
    """Calcula estatisticas por sessao.

    Args:
        events_with_session: DataFrame com coluna session_id.

    Returns:
        DataFrame com colunas session_id, n_events, duration, event_types.
    """
    return events_with_session.group_by("session_id").agg(
        pl.len().alias("n_events"),
        (pl.col("datetime").max() - pl.col("datetime").min()).alias("duration"),
        pl.col("event").unique().alias("event_types"),
    )


def compare_session_heuristics(
    events: pl.DataFrame, gap_minutes: float = SESSION_GAP_MINUTES
) -> pl.DataFrame:
    """Compara o numero de sessoes geradas por diferentes heuristicas.

    Retorna uma tabela com:
    - Gap-only: gap >= threshold
    - Gap + purchase ends session: gap >= threshold OU (prev == transaction E cur != transaction)
    """
    gap_ms = gap_minutes * 60_000

    gap_only = (
        events.sort(["visitorid", "timestamp"])
        .with_columns(
            (
                (pl.col("timestamp").diff().cast(pl.Float64) / 60_000 >= gap_minutes)
                .over("visitorid")
                .fill_null(False)
                .cast(pl.Int32)
                .cum_sum()
                .over("visitorid")
            ).alias("session_per_visitor"),
        )
        .with_columns(
            session_id_gap_only=pl.col("visitorid").cast(pl.Utf8)
            + "_"
            + pl.col("session_per_visitor").cast(pl.Utf8),
        )
    )

    with_purchase = create_sessions(events, gap_minutes=gap_minutes)

    return pl.DataFrame(
        {
            "heuristic": ["Gap 30min only", "Gap 30min + purchase ends session"],
            "total_sessions": [
                gap_only["session_id_gap_only"].n_unique(),
                with_purchase["session_id"].n_unique(),
            ],
            "diff_from_gap_only": [
                0,
                with_purchase["session_id"].n_unique()
                - gap_only["session_id_gap_only"].n_unique(),
            ],
        }
    )
