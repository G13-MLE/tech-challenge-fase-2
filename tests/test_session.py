"""Testes para o modulo de sessao."""

import polars as pl
import pytest

from data_pipeline.session import create_sessions, compare_session_heuristics


@pytest.fixture
def sample_events():
    """DataFrame de exemplo com eventos de 2 visitors."""
    return pl.DataFrame(
        {
            "visitorid": [1, 1, 1, 1, 1, 2, 2, 2],
            "timestamp": [
                1000000,
                1001000,
                1002000,
                5000000,
                5001000,
                1000000,
                5000000,
                5001000,
            ],
            "event": [
                "view",
                "addtocart",
                "transaction",
                "view",
                "view",
                "view",
                "transaction",
                "transaction",
            ],
            "itemid": [10, 20, 20, 30, 30, 40, 40, 50],
            "transactionid": [None, None, "T1", None, None, None, "T2", "T3"],
        }
    ).with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
    )


class TestCreateSessions:
    def test_basic_session_creation(self, sample_events):
        result = create_sessions(sample_events, gap_minutes=30)
        assert "session_id" in result.columns

    def test_gap_creates_new_session(self, sample_events):
        """Gap de ~66min entre timestamp 1002000 e 5000000 cria nova sessao."""
        result = create_sessions(sample_events, gap_minutes=30)
        visitor1 = result.filter(pl.col("visitorid") == 1)
        sessions_v1 = visitor1["session_id"].unique().to_list()
        assert len(sessions_v1) >= 2

    def test_transaction_then_view_creates_new_session(self):
        """Transaction seguido de view cria nova sessao."""
        events = pl.DataFrame(
            {
                "visitorid": [1, 1, 1],
                "timestamp": [1000000, 1001000, 1002000],
                "event": ["view", "transaction", "view"],
                "itemid": [10, 20, 30],
                "transactionid": [None, "T1", None],
            }
        ).with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        )
        result = create_sessions(events, gap_minutes=30)
        sessions = result["session_id"].unique().to_list()
        assert len(sessions) >= 2, "Transaction seguido de view deve criar nova sessao"

    def test_consecutive_transactions_same_session(self):
        """Transacoes consecutivas ficam na mesma sessao."""
        events = pl.DataFrame(
            {
                "visitorid": [1, 1, 1],
                "timestamp": [1000000, 1001000, 1002000],
                "event": ["view", "transaction", "transaction"],
                "itemid": [10, 20, 30],
                "transactionid": [None, "T1", "T2"],
            }
        ).with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        )
        result = create_sessions(events, gap_minutes=30)
        sessions = result["session_id"].unique().to_list()
        assert len(sessions) == 1, "Transacoes consecutivas devem ficar na mesma sessao"

    def test_no_gap_same_session(self):
        """Eventos proximos sem gap ficam na mesma sessao."""
        events = pl.DataFrame(
            {
                "visitorid": [1, 1, 1],
                "timestamp": [1000000, 1000500, 1001000],
                "event": ["view", "addtocart", "view"],
                "itemid": [10, 20, 30],
                "transactionid": [None, None, None],
            }
        ).with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        )
        result = create_sessions(events, gap_minutes=30)
        sessions = result["session_id"].unique().to_list()
        assert len(sessions) == 1, "Eventos sem gap ficam na mesma sessao"

    def test_different_visitors_different_sessions(self):
        """Visitors diferentes nunca compartilham sessao."""
        events = pl.DataFrame(
            {
                "visitorid": [1, 2],
                "timestamp": [1000000, 1000000],
                "event": ["view", "view"],
                "itemid": [10, 20],
                "transactionid": [None, None],
            }
        ).with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        )
        result = create_sessions(events, gap_minutes=30)
        sessions = result["session_id"].unique().to_list()
        assert len(sessions) == 2, "Visitors diferentes devem ter sessoes diferentes"


class TestCompareSessionHeuristics:
    def test_comparison_returns_rows(self, sample_events):
        result = compare_session_heuristics(sample_events, gap_minutes=30)
        assert result.height == 2
        assert "heuristic" in result.columns
        assert "total_sessions" in result.columns
