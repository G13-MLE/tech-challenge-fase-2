.PHONY: sync test lint verify docker-build docker-build-gpu mlflow-up mlflow-down

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

mlflow-up:
	docker compose -f docker/docker-compose.yml --env-file .env up -d --build

mlflow-down:
	docker compose -f docker/docker-compose.yml --env-file .env down
