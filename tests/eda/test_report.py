"""Tests for RetailRocket EDA report rendering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from techchallenge_fase2.eda.report import build_retailrocket_report  # noqa: E402


class RetailRocketReportTest(unittest.TestCase):
    """Validate report rendering contracts."""

    def test_build_report_contains_executive_summary(self) -> None:
        """Render the expected top-level sections."""
        report = build_retailrocket_report(_metrics())
        self.assertIn("## Resumo executivo", report)
        self.assertIn("## Qualidade e schema", report)
        self.assertIn("## Features candidatas", report)
        self.assertIn("## Decisoes recomendadas", report)


def _metrics() -> dict[str, object]:
    return {
        "dataset": {"local_path": "D:/Dataset/archive"},
        "minimum_interactions": {"passes": True},
        "events": _events(),
        "item_properties": _properties(),
        "category_tree": _categories(),
        "recommendations": ["Use chronological split."],
    }


def _events() -> dict[str, object]:
    return {
        "row_count": 3,
        "unique_user_item_pairs": 2,
        "sparsity": 0.5,
        "file_name": "events.csv",
        "file_size_bytes": 100,
        "columns": ["timestamp", "visitorid", "event", "itemid", "transactionid"],
        "missing_values": {
            "timestamp": 0,
            "visitorid": 0,
            "event": 0,
            "itemid": 0,
            "transactionid": 2,
        },
        "schema": {"matches_expected": True},
        "expected_column_types": {
            "timestamp": "unix_ms",
            "visitorid": "integer_id",
            "event": "category",
            "itemid": "integer_id",
            "transactionid": "nullable_integer_id",
        },
        "event_counts": {"view": 2, "transaction": 1},
        "unique_visitors": 2,
        "unique_items": 1,
        "date_range": {"start": "2015-01-01", "end": "2015-01-02"},
        "duplicate_rows": 1,
        "monthly_counts": {"2015-01": 3},
    }


def _properties() -> dict[str, object]:
    file_data = {
        "file_name": "item_properties_part1.csv",
        "file_size_bytes": 100,
        "row_count": 4,
        "columns": ["timestamp", "itemid", "property", "value"],
        "missing_values": {"timestamp": 0, "itemid": 0, "property": 0, "value": 0},
        "schema": {"matches_expected": True},
        "expected_column_types": {
            "timestamp": "unix_ms",
            "itemid": "integer_id",
            "property": "category",
            "value": "string",
        },
        "duplicate_rows": None,
    }
    return {
        "files": [file_data],
        "row_count": 4,
        "unique_items": 2,
        "unique_properties": 2,
        "categoryid_rows": 1,
        "items_with_categoryid": 1,
        "date_range": {"start": "2015-01-01", "end": "2015-01-02"},
        "duplicate_note": "not tracked",
        "top_properties": [{"id": "categoryid", "count": 1}],
    }


def _categories() -> dict[str, object]:
    return {
        "file_name": "category_tree.csv",
        "file_size_bytes": 100,
        "row_count": 1,
        "columns": ["categoryid", "parentid"],
        "missing_values": {"categoryid": 0, "parentid": 1},
        "schema": {"matches_expected": True},
        "expected_column_types": {
            "categoryid": "integer_id",
            "parentid": "nullable_integer_id",
        },
        "unique_categories": 1,
        "unique_parent_categories": 0,
        "root_categories": 1,
        "duplicate_rows": 0,
    }


if __name__ == "__main__":
    unittest.main()
