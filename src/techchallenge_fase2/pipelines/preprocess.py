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


def filter_catalog(
    frame: pd.DataFrame, max_items: int, min_interactions_per_user: int
) -> pd.DataFrame:
    """Filtra o catalogo para os top-`max_items` itens mais populares e mantem
    apenas usuarios com pelo menos `min_interactions_per_user` interacoes.

    Decisao de design documentada da literatura de recsys para datasets
    esparsos: reduz o catalogo de 235k para ~10k itens (tornando tratavel o
    problema e viaveis os modelos ItemKNN/LogReg/EASE^ em 16GB) e foca nos
    usuarios com historico suficiente para personalizacao. Valores <=0
    desativam o filtro respectivo.

    A filtragem itera ate convergencia para garantir que apos remover itens
    pouco populares os usuarios que so interagiram com eles tambem sejam
    removidos, evitando usuarios sem nenhum item no catalogo final.
    """
    filtered = frame
    while True:
        before = len(filtered)
        if max_items > 0:
            top_items = filtered["itemid"].value_counts().nlargest(max_items).index
            filtered = filtered[filtered["itemid"].isin(top_items)]
        if min_interactions_per_user > 0:
            user_counts = filtered["visitorid"].value_counts()
            active_users = user_counts[user_counts >= min_interactions_per_user].index
            filtered = filtered[filtered["visitorid"].isin(active_users)]
        if len(filtered) == before:
            break
    return filtered


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
    filtered = filter_catalog(
        interactions,
        params.preprocess.max_items,
        params.preprocess.min_interactions_per_user,
    )
    if len(filtered) != len(interactions):
        logger.info(
            "  filtro de catalogo aplicado: %d -> %d interacoes "
            "(max_items=%d, min_interactions_per_user=%d)",
            len(interactions),
            len(filtered),
            params.preprocess.max_items,
            params.preprocess.min_interactions_per_user,
        )
        interactions = filtered
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
