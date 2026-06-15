.PHONY: help setup test lint verify format docker-build docker-build-gpu mlflow-up mlflow-down

# Help padrão
help:
	@echo "Tech Challenge Fase 2 - Comandos Disponíveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup         - Configurar ambiente (uv sync + pre-commit + .env)"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make test          - Rodar testes (pytest)"
	@echo "  make lint          - Verificar código com ruff"
	@echo "  make format        - Formatar código com ruff"
	@echo ""
	@echo "Docker (aplicação):"
	@echo "  make docker-build     - Build imagem CPU (default)"
	@echo "  make docker-build-gpu - Build imagem GPU (com CUDA)"
	@echo ""
	@echo "Docker (MLflow):"
	@echo "  make mlflow-up    - Iniciar stack MLflow em background (requer .env)"
	@echo "  make mlflow-down  - Parar containers MLflow"
	@echo ""

setup:
	@echo "Configurando ambiente..."
	@if [ ! -f .env ]; then \
		echo "Criando .env a partir de .env.example..."; \
		cp .env.example .env; \
	fi
	uv sync
	uv run pre-commit install
	@echo "Setup concluído!"

test:
	@echo "Executando testes..."
	uv run pytest

lint:
	@echo "Verificando código com ruff..."
	uv run ruff check .

verify:
	uv run python scripts/validate_env.py

format:
	@echo "Formatando código com ruff..."
	uv run ruff format .

docker-build:
	docker build -f docker/Dockerfile --target cpu -t techchallenge-fase2 .

docker-build-gpu:
	docker build -f docker/Dockerfile --target gpu -t techchallenge-fase2:gpu .

mlflow-up:
	@echo "Docker: Iniciando MLflow em background..."
	docker compose -f docker/docker-compose.yml --env-file .env up -d --build
	@echo "[OK] MLflow iniciado! Acesse http://localhost:$$(grep -E '^MLFLOW_PORT=' .env | cut -d '=' -f2) para usar."

mlflow-down:
	@echo "[STOP] Parando containers MLflow..."
	docker compose -f docker/docker-compose.yml --env-file .env down
	@echo "[OK] Containers parados!"
