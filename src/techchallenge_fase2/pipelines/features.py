"""Create model-ready features from preprocessed interactions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from techchallenge_fase2.pipelines.config import PipelineParams, load_params

logger = logging.getLogger(__name__)


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


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically into train, validation and test.

    Raises:
        ValueError: Quando o frame é pequeno demais para gerar splits
            não vazios, o que resultaria em parquets vazios e mapeamentos
            sem entidades nas próximas etapas do pipeline.
    """
    total_rows = len(frame)
    if total_rows < 3:
        raise ValueError(
            f"Frame pequeno demais para split: {total_rows} linhas; "
            "aumente preprocess.sample_size",
        )
    train_end = max(1, int(total_rows * train_ratio))
    validation_end = int(total_rows * (train_ratio + validation_ratio))
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, total_rows - 1)
    train, validation, test = (
        frame.iloc[:train_end],
        frame.iloc[train_end:validation_end],
        frame.iloc[validation_end:],
    )
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Split produziu partição vazia; aumente preprocess.sample_size",
        )
    return train, validation, test


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
    logger.info("=" * 70)
    logger.info("[STAGE 2/4] FEATURES - engenharia de features e split temporal")
    interactions = load_interactions(params.paths.processed_interactions)
    logger.info("  interacoes carregadas: %d linhas", len(interactions))
    sessions = add_session_features(interactions, params.features.session_gap_minutes)
    encoded, mappings = add_encoded_ids(sessions)
    logger.info(
        "  usuarios codificados: %d | itens codificados: %d",
        len(mappings["user_ids"]),
        len(mappings["item_ids"]),
    )
    splits = split_frame(
        encoded,
        params.features.train_ratio,
        params.features.validation_ratio,
    )
    save_outputs(splits, mappings, params)
    train, validation, test = splits
    logger.info(
        "  split temporal: train=%d (70%%) | validation=%d (15%%) | test=%d (15%%)",
        len(train),
        len(validation),
        len(test),
    )
    logger.info("[STAGE 2/4] FEATURES concluido")
    logger.info("=" * 70)


def main() -> None:
    """CLI entry point for the feature engineering stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
