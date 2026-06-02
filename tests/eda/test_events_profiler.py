"""Tests for RetailRocket EDA profilers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from techchallenge_fase2.eda.profilers import EventsProfiler  # noqa: E402


class EventsProfilerTest(unittest.TestCase):
    """Validate event profiling on a small fixture."""

    def test_profile_events_counts_unique_pairs(self) -> None:
        """Count events and unique user-item pairs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "events.csv"
            csv_path.write_text(_events_csv(), encoding="utf-8")
            profile = EventsProfiler().profile(csv_path)
        self.assertEqual(profile["row_count"], 3)
        self.assertEqual(profile["event_counts"]["view"], 2)
        self.assertEqual(profile["unique_user_item_pairs"], 2)
        self.assertEqual(profile["duplicate_rows"], 1)
        self.assertTrue(profile["schema"]["matches_expected"])
        self.assertEqual(profile["expected_column_types"]["timestamp"], "unix_ms")


def _events_csv() -> str:
    return "\n".join(
        [
            "timestamp,visitorid,event,itemid,transactionid",
            "1433221332117,1,view,10,",
            "1433221332117,1,view,10,",
            "1433224214164,2,transaction,10,100",
            "",
        ],
    )


if __name__ == "__main__":
    unittest.main()
