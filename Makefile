.PHONY: sync test lint verify

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

verify:
	uv run python scripts/validate_env.py
