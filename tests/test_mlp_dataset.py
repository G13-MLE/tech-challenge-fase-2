"""Testes para o modulo mlp_dataset."""

import polars as pl
import pytest

from data_pipeline.mlp_dataset import (
    build_mlp_dataset,
    build_mlp_pairs,
    chronological_split,
    split_sessions_for_mlp,
)


@pytest.fixture
def sample_events():
    return pl.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 2, 3, 3],
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800],
            "event": [
                "view",
                "addtocart",
                "transaction",
                "view",
                "view",
                "addtocart",
                "view",
                "view",
            ],
            "itemid": [10, 20, 20, 30, 40, 40, 50, 60],
            "transactionid": [None, None, "T1", None, None, None, None, None],
            "datetime": pl.Series("datetime", range(8)).cast(pl.Datetime("ms")),
            "session_id": ["1_0", "1_0", "1_0", "2_0", "2_0", "2_0", "3_0", "3_0"],
        }
    )


class TestChronologicalSplit:
    def test_split_ratios(self, sample_events):
        train, val, test = chronological_split(
            sample_events, train_ratio=0.5, val_ratio=0.25
        )
        total = sample_events.height
        assert train.height + val.height + test.height == total

    def test_no_temporal_leakage(self, sample_events):
        train, val, test = chronological_split(sample_events)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()


class TestSplitSessionsForMlp:
    def test_history_and_target_separation(self, sample_events):
        history, target = split_sessions_for_mlp(sample_events)
        assert "session_id" in target.columns
        assert "target_itemid" in target.columns

    def test_single_event_sessions_in_history(self):
        events = pl.DataFrame(
            {
                "visitorid": [1],
                "timestamp": [100],
                "event": ["view"],
                "itemid": [10],
                "transactionid": [None],
                "datetime": pl.Series("datetime", [0]).cast(pl.Datetime("ms")),
                "session_id": ["1_0"],
            }
        )
        history, target = split_sessions_for_mlp(events)
        assert target.height == 0


class TestBuildMlpPairs:
    def test_positive_and_negative_samples(self, sample_events):
        all_items = set(sample_events["itemid"].unique().to_list())
        history, target = split_sessions_for_mlp(sample_events)
        pairs = build_mlp_pairs(history, target, all_items, neg_ratio=2)
        labels = pairs["label"].unique().to_list()
        assert 0 in labels
        assert 1 in labels

    def test_negative_ratio(self, sample_events):
        all_items = set(sample_events["itemid"].unique().to_list())
        history, target = split_sessions_for_mlp(sample_events)
        pairs = build_mlp_pairs(history, target, all_items, neg_ratio=3)
        positives = pairs.filter(pl.col("label") == 1).height
        negatives = pairs.filter(pl.col("label") == 0).height
        assert negatives == positives * 3
