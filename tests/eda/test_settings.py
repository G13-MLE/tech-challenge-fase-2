"""Tests for RetailRocket EDA settings."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from techchallenge_fase2.eda.settings import RetailRocketEdaSettings  # noqa: E402


class RetailRocketEdaSettingsTest(unittest.TestCase):
    """Validate configuration loading and overrides."""

    def test_from_project_loads_env_file(self) -> None:
        """Load EDA settings from a project-local .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_config(project_root)
            _write_env(project_root)
            with patch.dict(os.environ, {}, clear=True):
                settings = RetailRocketEdaSettings.from_project(project_root)

        self.assertEqual(settings.dataset_dir, project_root / "raw_data")
        self.assertEqual(settings.output_dir, project_root / "custom_reports")
        self.assertEqual(settings.required_interactions, 42)

    def test_with_overrides_accepts_zero_interactions(self) -> None:
        """Apply explicit falsy override values."""
        settings = RetailRocketEdaSettings(Path("raw"), Path("reports"), 10)
        updated = settings.with_overrides(None, None, 0)
        self.assertEqual(updated.required_interactions, 0)


def _write_config(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir()
    (config_dir / "retailrocket_eda.toml").write_text(_config_text(), encoding="utf-8")


def _write_env(project_root: Path) -> None:
    (project_root / ".env").write_text(_env_text(), encoding="utf-8")


def _config_text() -> str:
    return "\n".join(
        [
            "[dataset]",
            'default_dir = "D:/Dataset/archive"',
            "required_interactions = 10000",
            "",
            "[outputs]",
            'eda_dir = "reports/eda"',
            "",
        ],
    )


def _env_text() -> str:
    return "\n".join(
        [
            "RETAILROCKET_DATA_DIR=raw_data",
            "EDA_OUTPUT_DIR=custom_reports",
            "REQUIRED_INTERACTIONS=42",
            "",
        ],
    )


if __name__ == "__main__":
    unittest.main()
