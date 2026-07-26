"""Preprocess RetailRocket events for recommendation modeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from techchallenge_fase2.pipelines.config import PipelineParams, load_params

EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid"]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    """Load the RetailRocket event table.

    Args:
        path: Raw `events.csv` path.

    Returns:
        DataFrame with the required event columns.
    """
    return pd.read_csv(path, usecols=EVENT_COLUMNS)


def validate_columns(frame: pd.DataFrame) -> None:
    """Validate that all required columns are available."""
    missing = set(EVENT_COLUMNS).difference(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {joined}")


def apply_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Apply a deterministic sample when configured."""
    if sample_size <= 0 or sample_size >= len(frame):
        return frame
    return frame.sample(n=sample_size, random_state=seed)


def normalize_events(
    frame: pd.DataFrame,
    event_weights: dict[str, float],
) -> pd.DataFrame:
    """Normalize event types, timestamps and implicit feedback weights."""
    filtered = frame.dropna(subset=EVENT_COLUMNS).copy()
    filtered = filtered[filtered["event"].isin(event_weights)].copy()
    filtered["visitorid"] = filtered["visitorid"].astype("string")
    filtered["itemid"] = filtered["itemid"].astype("string")
    filtered["event"] = filtered["event"].astype("string")
    filtered["event_weight"] = filtered["event"].map(event_weights).astype("float32")
    filtered["event_time"] = pd.to_datetime(filtered["timestamp"], unit="ms", utc=True)
    return filtered.sort_values(["timestamp", "visitorid", "itemid"])


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def run(params: PipelineParams) -> None:
    """Run the preprocessing stage."""
    logger.info("=" * 70)
    logger.info("[STAGE 1/4] PREPROCESS - normalizando eventos do RetailRocket")
    logger.info("  raw_events: %s", params.paths.raw_events)
    events = load_events(params.paths.raw_events)
    validate_columns(events)
    logger.info("  eventos bruto carregados: %d linhas", len(events))
    logger.info(
        "  distribuicao de eventos: %s",
        events["event"].value_counts().to_dict(),
    )
    sampled = apply_sample(
        events,
        params.preprocess.sample_size,
        params.preprocess.random_seed,
    )
    if len(sampled) != len(events):
        logger.info(
            "  amostra aplicada: %d -> %d (sample_size=%d)",
            len(events),
            len(sampled),
            params.preprocess.sample_size,
        )
    else:
        logger.info(
            "  sem amostragem (sample_size=%d): usando dataset completo",
            params.preprocess.sample_size,
        )
    interactions = normalize_events(sampled, params.features.event_weights)
    save_frame(interactions, params.paths.processed_interactions)
    logger.info(
        "  interacoes normalizadas salvas: %d linhas -> %s",
        len(interactions),
        params.paths.processed_interactions,
    )
    logger.info(
        "  usuarios unicos=%d itens unicos=%d",
        interactions["visitorid"].nunique(),
        interactions["itemid"].nunique(),
    )
    logger.info("[STAGE 1/4] PREPROCESS concluido")
    logger.info("=" * 70)


def main() -> None:
    """CLI entry point for the preprocessing stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
