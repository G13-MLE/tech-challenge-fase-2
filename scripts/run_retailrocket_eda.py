"""Run RetailRocket EDA and write versionable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from techchallenge_fase2.eda.analyzer import RetailRocketAnalyzer  # noqa: E402
from techchallenge_fase2.eda.report import build_retailrocket_report  # noqa: E402
from techchallenge_fase2.eda.settings import RetailRocketEdaSettings  # noqa: E402


def main() -> None:
    """Run the RetailRocket EDA command-line interface."""
    args = _parse_args()
    settings = _settings_from_args(args)
    metrics = RetailRocketAnalyzer(
        settings.dataset_dir,
        settings.required_interactions,
    ).run()
    _write_artifacts(metrics, settings.output_dir)
    _print_success(settings.output_dir, metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RetailRocket EDA.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--required-interactions", type=int, default=None)
    return parser.parse_args()


def _settings_from_args(args: argparse.Namespace) -> RetailRocketEdaSettings:
    settings = RetailRocketEdaSettings.from_project(PROJECT_ROOT)
    return settings.with_overrides(
        args.dataset_dir,
        args.output_dir,
        args.required_interactions,
    )


def _write_artifacts(metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "retailrocket_metrics.json", metrics)
    _write_report(output_dir / "retailrocket_eda.md", metrics)


def _write_json(output_path: Path, metrics: dict[str, object]) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=True, indent=2)


def _write_report(output_path: Path, metrics: dict[str, object]) -> None:
    output_path.write_text(build_retailrocket_report(metrics), encoding="utf-8")


def _print_success(output_dir: Path, metrics: dict[str, object]) -> None:
    check = metrics["minimum_interactions"]
    print(f"EDA written to {output_dir}")
    print(f"Minimum interaction requirement passed: {check['passes']}")


if __name__ == "__main__":
    main()
