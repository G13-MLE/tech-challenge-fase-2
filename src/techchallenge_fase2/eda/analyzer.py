"""RetailRocket dataset analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from techchallenge_fase2.eda.profilers import (
    CategoryTreeProfiler,
    EventsProfiler,
    ItemPropertiesProfiler,
)


@dataclass(frozen=True)
class RetailRocketPaths:
    """Paths for the expected RetailRocket CSV files."""

    events: Path
    item_properties_part1: Path
    item_properties_part2: Path
    category_tree: Path

    @classmethod
    def from_dir(cls, dataset_dir: Path) -> RetailRocketPaths:
        """Build expected file paths from a dataset directory."""
        return cls(
            events=dataset_dir / "events.csv",
            item_properties_part1=dataset_dir / "item_properties_part1.csv",
            item_properties_part2=dataset_dir / "item_properties_part2.csv",
            category_tree=dataset_dir / "category_tree.csv",
        )

    def all_files(self) -> tuple[Path, ...]:
        """Return all expected dataset files."""
        return (
            self.events,
            self.item_properties_part1,
            self.item_properties_part2,
            self.category_tree,
        )


class RetailRocketAnalyzer:
    """Coordinate EDA profilers for the RetailRocket dataset."""

    def __init__(self, dataset_dir: Path, required_interactions: int) -> None:
        self._dataset_dir = dataset_dir
        self._required_interactions = required_interactions

    def run(self) -> dict[str, Any]:
        """Run EDA and return serializable metrics."""
        paths = RetailRocketPaths.from_dir(self._dataset_dir)
        _validate_files(paths)
        events = EventsProfiler().profile(paths.events)
        properties = ItemPropertiesProfiler().profile(_property_paths(paths))
        categories = CategoryTreeProfiler().profile(paths.category_tree)
        return self._summary(events, properties, categories)

    def _summary(
        self,
        events: dict[str, Any],
        properties: dict[str, Any],
        categories: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset": _dataset_summary(self._dataset_dir),
            "minimum_interactions": _minimum_check(events, self._required_interactions),
            "events": events,
            "item_properties": properties,
            "category_tree": categories,
            "recommendations": _recommendations(events),
        }


def _validate_files(paths: RetailRocketPaths) -> None:
    missing_files = [path for path in paths.all_files() if not path.exists()]
    if missing_files:
        names = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing RetailRocket files: {names}")


def _property_paths(paths: RetailRocketPaths) -> tuple[Path, Path]:
    return (paths.item_properties_part1, paths.item_properties_part2)


def _dataset_summary(dataset_dir: Path) -> dict[str, str]:
    return {
        "name": "RetailRocket Ecommerce Dataset",
        "source_url": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
        "local_path": str(dataset_dir),
    }


def _minimum_check(events: dict[str, Any], required_interactions: int) -> dict[str, Any]:
    unique_pairs = events["unique_user_item_pairs"]
    return {
        "required_user_item_interactions": required_interactions,
        "observed_events": events["row_count"],
        "observed_unique_user_item_pairs": unique_pairs,
        "passes": unique_pairs >= required_interactions,
    }


def _recommendations(events: dict[str, Any]) -> list[str]:
    cutoffs = events["temporal_split_cutoffs"]
    split_message = (
        f"Usar corte de treino em {cutoffs['train_until']} e "
        f"validacao em {cutoffs['validation_until']}."
    )
    return [
        "Usar split cronologico para evitar vazamento entre treino e avaliacao.",
        split_message,
        "Mapear eventos para pesos implicitos: view=1, addtocart=3, transaction=5.",
        "Usar categoryid como primeira feature de item.",
        "Tratar demais propriedades como metadados esparsos apos filtro de cardinalidade.",
        "Versionar CSVs brutos com DVC, sem commit direto no Git.",
    ]
