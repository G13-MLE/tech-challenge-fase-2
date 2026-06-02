"""CSV profilers for the RetailRocket dataset."""

from __future__ import annotations

import csv
import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_COLUMNS = {
    "events.csv": ("timestamp", "visitorid", "event", "itemid", "transactionid"),
    "item_properties_part1.csv": ("timestamp", "itemid", "property", "value"),
    "item_properties_part2.csv": ("timestamp", "itemid", "property", "value"),
    "category_tree.csv": ("categoryid", "parentid"),
}

EXPECTED_COLUMN_TYPES = {
    "events.csv": {
        "timestamp": "unix_ms",
        "visitorid": "integer_id",
        "event": "category",
        "itemid": "integer_id",
        "transactionid": "nullable_integer_id",
    },
    "item_properties_part1.csv": {
        "timestamp": "unix_ms",
        "itemid": "integer_id",
        "property": "category",
        "value": "string",
    },
    "item_properties_part2.csv": {
        "timestamp": "unix_ms",
        "itemid": "integer_id",
        "property": "category",
        "value": "string",
    },
    "category_tree.csv": {
        "categoryid": "integer_id",
        "parentid": "nullable_integer_id",
    },
}


class CsvProfiler(ABC):
    """Template method for CSV file profiling."""

    def __init__(self, track_duplicate_rows: bool) -> None:
        self._track_duplicate_rows = track_duplicate_rows

    def profile(self, csv_path: Path) -> dict[str, Any]:
        """Profile a CSV file.

        Args:
            csv_path: CSV path to inspect.

        Returns:
            Dictionary with file-level and domain metrics.
        """
        state = self._new_state(csv_path)
        with csv_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            self._set_columns(state, reader.fieldnames)
            for row in reader:
                self._update_common(row, state)
                self._update(row, state)
        return self._build_summary(state)

    @abstractmethod
    def _update(self, row: dict[str, str], state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _new_state(self, csv_path: Path) -> dict[str, Any]:
        return {
            "file_name": csv_path.name,
            "file_size_bytes": csv_path.stat().st_size,
            "row_count": 0,
            "columns": [],
            "missing_values": {},
            "duplicate_rows": 0,
            "seen_rows": set(),
        }

    def _set_columns(self, state: dict[str, Any], columns: list[str] | None) -> None:
        state["columns"] = columns or []
        state["missing_values"] = dict.fromkeys(state["columns"], 0)

    def _update_common(self, row: dict[str, str], state: dict[str, Any]) -> None:
        state["row_count"] += 1
        self._update_missing_values(row, state)
        self._update_duplicate_rows(row, state)

    def _update_missing_values(self, row: dict[str, str], state: dict[str, Any]) -> None:
        for column, value in row.items():
            if value == "":
                state["missing_values"][column] += 1

    def _update_duplicate_rows(self, row: dict[str, str], state: dict[str, Any]) -> None:
        if not self._track_duplicate_rows:
            state["duplicate_rows"] = None
            return
        signature = _row_signature(row, state["columns"])
        if signature in state["seen_rows"]:
            state["duplicate_rows"] += 1
        state["seen_rows"].add(signature)

    def _common_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        file_name = state["file_name"]
        return {
            "file_name": file_name,
            "file_size_bytes": state["file_size_bytes"],
            "row_count": state["row_count"],
            "columns": state["columns"],
            "missing_values": state["missing_values"],
            "duplicate_rows": state["duplicate_rows"],
            "expected_columns": EXPECTED_COLUMNS[file_name],
            "expected_column_types": EXPECTED_COLUMN_TYPES[file_name],
            "schema": _schema_summary(state["columns"], EXPECTED_COLUMNS[file_name]),
        }


class EventsProfiler(CsvProfiler):
    """Profiler for user-item events."""

    def __init__(self) -> None:
        super().__init__(track_duplicate_rows=True)

    def _new_state(self, csv_path: Path) -> dict[str, Any]:
        state = super()._new_state(csv_path)
        state.update(_events_state())
        return state

    def _update(self, row: dict[str, str], state: dict[str, Any]) -> None:
        timestamp = int(row["timestamp"])
        visitor_id = row["visitorid"]
        item_id = row["itemid"]
        event = row["event"]
        state["timestamps"].append(timestamp)
        state["event_counts"][event] += 1
        state["visitor_ids"].add(visitor_id)
        state["item_ids"].add(item_id)
        state["visitor_item_pairs"].add((visitor_id, item_id))
        state["visitor_counts"][visitor_id] += 1
        state["item_counts"][item_id] += 1
        state["monthly_counts"][_month_from_ms(timestamp)] += 1

    def _build_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        common = self._common_summary(state)
        timestamps = sorted(state["timestamps"])
        return common | _events_summary(state, timestamps)


class CategoryTreeProfiler(CsvProfiler):
    """Profiler for category hierarchy rows."""

    def __init__(self) -> None:
        super().__init__(track_duplicate_rows=True)

    def _new_state(self, csv_path: Path) -> dict[str, Any]:
        state = super()._new_state(csv_path)
        state.update({"category_ids": set(), "parent_ids": set(), "root_categories": set()})
        return state

    def _update(self, row: dict[str, str], state: dict[str, Any]) -> None:
        state["category_ids"].add(row["categoryid"])
        if row["parentid"]:
            state["parent_ids"].add(row["parentid"])
            return
        state["root_categories"].add(row["categoryid"])

    def _build_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        common = self._common_summary(state)
        common["unique_categories"] = len(state["category_ids"])
        common["unique_parent_categories"] = len(state["parent_ids"])
        common["root_categories"] = len(state["root_categories"])
        return common


class ItemPropertiesProfiler:
    """Profiler for the two item property CSV files."""

    def profile(self, csv_paths: tuple[Path, Path]) -> dict[str, Any]:
        """Profile both item property files as one logical table."""
        state = _properties_state()
        file_summaries = [self._profile_file(csv_path, state) for csv_path in csv_paths]
        return _properties_summary(state, file_summaries)

    def _profile_file(self, csv_path: Path, state: dict[str, Any]) -> dict[str, Any]:
        file_state = _properties_file_state(csv_path)
        with csv_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            file_state["columns"] = reader.fieldnames or []
            file_state["missing_values"] = dict.fromkeys(file_state["columns"], 0)
            _set_property_schema(file_state, csv_path.name)
            for row in reader:
                _update_property_file(row, file_state)
                _update_property_global(row, state)
        return file_state


def _events_state() -> dict[str, Any]:
    return {
        "event_counts": Counter(),
        "visitor_ids": set(),
        "item_ids": set(),
        "visitor_item_pairs": set(),
        "visitor_counts": Counter(),
        "item_counts": Counter(),
        "monthly_counts": Counter(),
        "timestamps": [],
    }


def _events_summary(state: dict[str, Any], timestamps: list[int]) -> dict[str, Any]:
    unique_pairs = len(state["visitor_item_pairs"])
    unique_visitors = len(state["visitor_ids"])
    unique_items = len(state["item_ids"])
    return {
        "event_counts": dict(state["event_counts"]),
        "unique_visitors": unique_visitors,
        "unique_items": unique_items,
        "unique_user_item_pairs": unique_pairs,
        "sparsity": _sparsity(unique_pairs, unique_visitors, unique_items),
        "date_range": _date_range(timestamps),
        "temporal_split_cutoffs": _split_cutoffs(timestamps),
        "monthly_counts": dict(sorted(state["monthly_counts"].items())),
        "top_visitors": _top_counter(state["visitor_counts"]),
        "top_items": _top_counter(state["item_counts"]),
    }


def _properties_state() -> dict[str, Any]:
    return {
        "row_count": 0,
        "item_ids": set(),
        "property_counts": Counter(),
        "category_item_ids": set(),
        "min_timestamp": None,
        "max_timestamp": None,
        "missing_values": Counter(),
    }


def _properties_file_state(csv_path: Path) -> dict[str, Any]:
    return {
        "file_name": csv_path.name,
        "file_size_bytes": csv_path.stat().st_size,
        "row_count": 0,
        "columns": [],
        "missing_values": {},
        "duplicate_rows": None,
        "expected_columns": EXPECTED_COLUMNS[csv_path.name],
        "expected_column_types": EXPECTED_COLUMN_TYPES[csv_path.name],
        "schema": {},
    }


def _set_property_schema(state: dict[str, Any], file_name: str) -> None:
    state["schema"] = _schema_summary(state["columns"], EXPECTED_COLUMNS[file_name])


def _update_property_file(row: dict[str, str], state: dict[str, Any]) -> None:
    state["row_count"] += 1
    for column, value in row.items():
        if value == "":
            state["missing_values"][column] += 1


def _update_property_global(row: dict[str, str], state: dict[str, Any]) -> None:
    timestamp = int(row["timestamp"])
    state["row_count"] += 1
    state["item_ids"].add(row["itemid"])
    state["property_counts"][row["property"]] += 1
    _update_property_dates(timestamp, state)
    if row["property"] == "categoryid":
        state["category_item_ids"].add(row["itemid"])


def _update_property_dates(timestamp: int, state: dict[str, Any]) -> None:
    min_timestamp = state["min_timestamp"]
    max_timestamp = state["max_timestamp"]
    state["min_timestamp"] = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
    state["max_timestamp"] = timestamp if max_timestamp is None else max(max_timestamp, timestamp)


def _properties_summary(
    state: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "files": file_summaries,
        "row_count": state["row_count"],
        "unique_items": len(state["item_ids"]),
        "unique_properties": len(state["property_counts"]),
        "categoryid_rows": state["property_counts"]["categoryid"],
        "items_with_categoryid": len(state["category_item_ids"]),
        "date_range": _property_date_range(state),
        "top_properties": _top_counter(state["property_counts"]),
        "duplicate_rows": None,
        "duplicate_note": "Duplicidades exatas nao sao rastreadas nos arquivos de propriedades.",
    }


def _property_date_range(state: dict[str, Any]) -> dict[str, str]:
    return {
        "start": _date_from_ms(state["min_timestamp"]),
        "end": _date_from_ms(state["max_timestamp"]),
    }


def _date_range(timestamps: list[int]) -> dict[str, str]:
    return {"start": _date_from_ms(timestamps[0]), "end": _date_from_ms(timestamps[-1])}


def _split_cutoffs(timestamps: list[int]) -> dict[str, str]:
    return {
        "train_until": _date_from_ms(_percentile(timestamps, 0.70)),
        "validation_until": _date_from_ms(_percentile(timestamps, 0.85)),
    }


def _percentile(values: list[int], percentile: float) -> int:
    index = int((len(values) - 1) * percentile)
    return values[index]


def _sparsity(unique_pairs: int, unique_visitors: int, unique_items: int) -> float:
    possible_pairs = unique_visitors * unique_items
    return 1 - (unique_pairs / possible_pairs)


def _top_counter(counter: Counter[str], limit: int = 10) -> list[dict[str, int | str]]:
    return [{"id": key, "count": value} for key, value in counter.most_common(limit)]


def _month_from_ms(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC).strftime("%Y-%m")


def _date_from_ms(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC).date().isoformat()


def _row_signature(row: dict[str, str], columns: list[str]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for column in columns:
        digest.update(row[column].encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
    return digest.digest()


def _schema_summary(columns: list[str], expected_columns: tuple[str, ...]) -> dict[str, Any]:
    actual_columns = tuple(columns)
    return {
        "matches_expected": actual_columns == expected_columns,
        "missing_columns": [column for column in expected_columns if column not in actual_columns],
        "unexpected_columns": [
            column for column in actual_columns if column not in expected_columns
        ],
    }
