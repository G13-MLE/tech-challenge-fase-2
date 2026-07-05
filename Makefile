.PHONY: help sync setup test lint verify format data dvc-push dvc-pull dvc-status docker-build docker-build-gpu mlflow-up mlflow-down

PYTHON := uv run python
UV_CACHE_DIR ?= .uv-cache
PRE_COMMIT_HOME ?= .pre-commit-cache

export UV_CACHE_DIR
export PRE_COMMIT_HOME

# Help padrão
help:
	@echo "Tech Challenge Fase 2 - Comandos Disponíveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup         - Configurar ambiente (uv sync + pre-commit + .env + DVC remote)"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make test          - Rodar testes (pytest)"
	@echo "  make lint          - Verificar código com ruff"
	@echo "  make format        - Formatar código com ruff"
	@echo "  make data          - Baixar dataset RetailRocket via Kaggle"
	@echo "  make dvc-push      - Enviar cache DVC ao remote"
	@echo "  make dvc-pull      - Restaurar dados pelo DVC"
	@echo "  make dvc-status    - Verificar status DVC"
	@echo ""
	@echo "Dados (DVC + OneDrive):"
	@echo "  make data          - Baixar dataset RetailRocket do Kaggle"
	@echo "  make dvc-push      - Enviar cache DVC ao remote OneDrive"
	@echo "  make dvc-pull      - Baixar cache DVC do remote OneDrive"
	@echo "  make dvc-status    - Verificar estado do versionamento DVC"
	@echo ""
	@echo "Docker (aplicação):"
	@echo "  make docker-build     - Build imagem CPU (default)"
	@echo "  make docker-build-gpu - Build imagem GPU (com CUDA)"
	@echo ""
	@echo "Docker (MLflow):"
	@echo "  make mlflow-up    - Iniciar stack MLflow em background (requer .env)"
	@echo "  make mlflow-down  - Parar containers MLflow"
	@echo ""

sync:
	uv sync

# Setup inicial do ambiente: .env, deps, pre-commit e remote DVC OneDrive.
setup:
	@echo "Configurando ambiente..."
	@$(PYTHON) -c "from pathlib import Path; src = Path('.env.example'); dst = Path('.env'); created = not dst.exists(); dst.write_bytes(src.read_bytes()) if created else None; print('Criando .env a partir de .env.example...' if created else '.env já existe; mantendo arquivo local.')"
	uv sync
	@$(PYTHON) -c "import subprocess, sys; hooks_path = subprocess.run(['git', 'config', '--get', 'core.hooksPath'], capture_output=True, text=True).stdout.strip(); print('core.hooksPath configurado; instalando apenas ambientes do pre-commit.' if hooks_path else 'Instalando hook do pre-commit...'); command = [sys.executable, '-m', 'pre_commit', 'install-hooks'] if hooks_path else [sys.executable, '-m', 'pre_commit', 'install', '--install-hooks']; raise SystemExit(subprocess.call(command))"
	@$(PYTHON) scripts/setup_environment.py
	@echo "Setup concluído!"

# Download do dataset RetailRocket via Kaggle (requer KAGGLE_USERNAME/KAGGLE_KEY no .env).
data:
	@echo "Baixando dataset RetailRocket para data/raw/ ..."
	uv run python scripts/download_dataset.py
	@echo "[OK] dataset disponível em data/raw/."

# Versionamento DVC: envia o cache ao remote OneDrive.
dvc-push:
	uv run dvc push

# Versionamento DVC: baixa o cache do remote OneDrive e restaura os dados.
dvc-pull:
	uv run dvc pull

# Status do versionamento DVC.
dvc-status:
	uv run dvc status

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
	@$(PYTHON) -c "from pathlib import Path; values = dict(line.split('=', 1) for line in Path('.env').read_text().splitlines() if line.startswith('MLFLOW_PORT=')); port = values.get('MLFLOW_PORT', '5000').split('#', 1)[0].strip() or '5000'; print(f'[OK] MLflow iniciado! Acesse http://localhost:{port} para usar.')"

mlflow-down:
	@echo "[STOP] Parando containers MLflow..."
	docker compose -f docker/docker-compose.yml --env-file .env down
	@echo "[OK] Containers parados!"
