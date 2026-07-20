"""Create model-ready features from preprocessed interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from techchallenge_fase2.pipelines.config import PipelineParams, load_params
from techchallenge_fase2.pipelines.splits import (
    chronological_train_val_test_split,
    filter_warm_start_three_way,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def load_interactions(path: Path) -> pd.DataFrame:
    """Load preprocessed interactions."""
    return pd.read_parquet(path)


def add_session_features(frame: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    """Create session identifiers from visitor timelines."""
    sorted_frame = frame.sort_values(["visitorid", "timestamp"]).copy()
    previous_timestamp = sorted_frame.groupby("visitorid")["timestamp"].shift()
    previous_event = sorted_frame.groupby("visitorid")["event"].shift()
    gap = (sorted_frame["timestamp"] - previous_timestamp) / 60_000
    is_new = previous_timestamp.isna() | (gap >= gap_minutes)
    after_purchase = previous_event == "transaction"
    is_new = is_new | (after_purchase & (sorted_frame["event"] != "transaction"))
    sorted_frame["session_index"] = (
        is_new.astype("int64")
        .groupby(
            sorted_frame["visitorid"],
        )
        .cumsum()
    )
    sorted_frame["session_id"] = make_session_id(sorted_frame)
    return sorted_frame.sort_values(["timestamp", "visitorid", "itemid"])


def make_session_id(frame: pd.DataFrame) -> pd.Series:
    """Build stable session identifiers."""
    visitor_ids = frame["visitorid"].astype("string")
    session_indexes = frame["session_index"].astype("string")
    return visitor_ids + "_" + session_indexes


def add_encoded_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add numeric user and item identifiers."""
    encoded = frame.copy()
    encoded["user_index"], user_ids = encode_column(encoded["visitorid"])
    encoded["item_index"], item_ids = encode_column(encoded["itemid"])
    return encoded, {"user_ids": user_ids, "item_ids": item_ids}


def encode_column(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Encode a string column as stable integer identifiers."""
    values = sorted(series.astype("string").unique().tolist())
    mapping = {value: index for index, value in enumerate(values)}
    return series.astype("string").map(mapping).astype("int64"), values


def split_interactions(
    interactions_df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically using the canonical splits module.

    Encoda user_index/item_index APENAS no universo warm-start (treino),
    garantindo que treino/validacao/teste compartilhem o mesmo mapeamento
    e que o teste contenha somente usuarios e itens vistos em treino.
    """
    test_ratio = max(0.0, 1.0 - train_ratio - validation_ratio)
    split_df = interactions_df.rename(
        columns={"visitorid": "user_id", "itemid": "item_id"}
    )
    split = chronological_train_val_test_split(
        split_df,
        val_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    split = filter_warm_start_three_way(split)
    train_df = rename_back(split.train_df)
    val_df = rename_back(split.val_df)
    test_df = rename_back(split.test_df)
    if (
        train_df is None
        or val_df is None
        or test_df is None
        or train_df.empty
        or val_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "Split produziu particao vazia; aumente preprocess.sample_size",
        )
    return train_df, val_df, test_df


def rename_back(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Restaura nomes visitorid/itemid apos o split."""
    if frame is None:
        return None
    return frame.rename(columns={"user_id": "visitorid", "item_id": "itemid"})


def save_outputs(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    mappings: dict[str, Any],
    params: PipelineParams,
) -> None:
    """Save feature datasets, mappings and dataset statistics."""
    train, validation, test = splits
    save_parquet(train, params.paths.train_features)
    save_parquet(validation, params.paths.validation_features)
    save_parquet(test, params.paths.test_features)
    save_json(mappings, params.paths.mappings)
    stats = build_stats(train, validation, test, mappings)
    save_json(stats, params.paths.dataset_stats)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Save a feature frame as parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_stats(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, list[str]],
) -> dict[str, int]:
    """Build compact dataset statistics."""
    return {
        "items": len(mappings["item_ids"]),
        "test_rows": len(test),
        "train_rows": len(train),
        "users": len(mappings["user_ids"]),
        "validation_rows": len(validation),
    }


def run(params: PipelineParams) -> None:
    """Run the feature engineering stage."""
    interactions = load_interactions(params.paths.processed_interactions)
    sessions = add_session_features(interactions, params.features.session_gap_minutes)
    train_df, val_df, test_df = split_interactions(
        sessions,
        params.features.train_ratio,
        params.features.validation_ratio,
    )
    # Encode sobre o universo completo (treino + validacao + teste warm-start)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    encoded, mappings = add_encoded_ids(combined)
    encoded_train = encoded.iloc[: len(train_df)]
    encoded_val = encoded.iloc[len(train_df) : len(train_df) + len(val_df)]
    encoded_test = encoded.iloc[len(train_df) + len(val_df) :]
    save_outputs((encoded_train, encoded_val, encoded_test), mappings, params)


def main() -> None:
    """CLI entry point for the feature engineering stage."""
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
