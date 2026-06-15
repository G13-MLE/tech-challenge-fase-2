.PHONY: sync test lint

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
