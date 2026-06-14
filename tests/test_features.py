"""Testes para o modulo de features."""

import polars as pl
import pytest

from data_pipeline.features import build_super_table, _add_historical_features


@pytest.fixture
def sample_events():
    return pl.DataFrame(
        {
            "visitorid": [1, 1, 2, 2],
            "timestamp": [1000000, 2000000, 1500000, 2500000],
            "event": ["view", "addtocart", "view", "transaction"],
            "itemid": [10, 20, 10, 30],
            "transactionid": [None, None, None, "T1"],
            "datetime": pl.Series(
                "datetime",
                [
                    "2015-05-01 00:00:00",
                    "2015-05-01 00:00:01",
                    "2015-05-01 00:00:01",
                    "2015-05-01 00:00:02",
                ],
            ).cast(pl.Datetime("ms")),
            "session_id": ["1_0", "1_0", "2_0", "2_0"],
        }
    )


class TestAddHistoricalFeatures:
    def test_user_events_before_increments(self, sample_events):
        result = _add_historical_features(sample_events)
        user_events = result.filter(pl.col("visitorid") == 1).sort("timestamp")
        counts = user_events["user_events_before"].to_list()
        assert counts == [1, 2], "cum_count deve incrementar por visitor"

    def test_item_events_before_increments(self, sample_events):
        result = _add_historical_features(sample_events)
        item_events = result.filter(pl.col("itemid") == 10).sort("timestamp")
        counts = item_events["item_events_before"].to_list()
        assert counts == [1, 2], "cum_count deve incrementar por item"


class TestBuildSuperTable:
    def test_targets_created(self, sample_events):
        result = build_super_table(
            sample_events, pl.DataFrame(), category_tree_path=None
        )
        assert "target_transaction" in result.columns
        assert "target_addtocart_or_transaction" in result.columns

    def test_temporal_features_created(self, sample_events):
        result = build_super_table(
            sample_events, pl.DataFrame(), category_tree_path=None
        )
        assert "event_month" in result.columns
        assert "event_hour" in result.columns
