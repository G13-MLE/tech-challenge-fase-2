.PHONY: sync test lint verify docker-build docker-build-gpu

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

verify:
	uv run python scripts/validate_env.py
docker-build:
	docker build -f docker/Dockerfile --target cpu -t techchallenge-fase2 .

docker-build-gpu:
	docker build -f docker/Dockerfile --target gpu -t techchallenge-fase2:gpu .
