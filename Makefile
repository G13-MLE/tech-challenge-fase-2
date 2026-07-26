# =============================================================================
# Tech Challenge Fase 2 - Makefile
# =============================================================================

PYTHON := uv run python
UV_CACHE_DIR ?= .uv-cache
PRE_COMMIT_HOME ?= .pre-commit-cache

export UV_CACHE_DIR
export PRE_COMMIT_HOME

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: \
	help \
	sync setup verify \
	test lint format \
	data train pipeline pipeline-force pipeline-live train-live \
	dvc-push dvc-pull dvc-status \
	docker-build docker-build-gpu \
	mlflow-up mlflow-down docker-train \
	baselines ease compare-models \
	register promote promote-dry-run inference

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Tech Challenge Fase 2 - Comandos Disponíveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            - Configurar ambiente (.env, deps, pre-commit, DVC remote)"
	@echo "  make sync             - Sincronizar dependências (uv sync)"
	@echo "  make verify           - Validar ambiente (Python, deps, .env, DVC, Docker, GPU)"
	@echo ""
	@echo "Qualidade:"
	@echo "  make test             - Rodar testes (pytest)"
	@echo "  make lint             - Verificar código com ruff"
	@echo "  make format           - Formatar código com ruff"
	@echo ""
	@echo "Pipeline DVC:"
	@echo "  make data             - Baixar dataset RetailRocket via Kaggle"
	@echo "  make train            - Reexecutar stage de treino (dvc repro train -v)"
	@echo "  make train-live       - Rodar apenas o treino direto (sem DVC) com tqdm"
	@echo "  make pipeline         - Reexecutar pipeline completo (dvc repro -v)"
	@echo "  make pipeline-force   - Reexecutar pipeline completo forcado (dvc repro --force)"
	@echo "  make pipeline-live    - Rodar preprocess->features->train->evaluate direto (tqdm/log live)"
	@echo "  make dvc-push         - Enviar cache DVC ao remote"
	@echo "  make dvc-pull         - Restaurar dados do remote DVC"
	@echo "  make dvc-status       - Verificar status do versionamento DVC"
	@echo ""
	@echo "Avaliacao comparativa:"
	@echo "  make baselines        - Rodar pipeline de baselines (8 modelos) no MLflow"
	@echo "  make ease             - Rodar pipeline dedicado do EASE^ no MLflow"
	@echo "  make compare-models   - Comparar modelos vs baselines (min. 4 metricas)"
	@echo "  make register         - Registrar campeao no MLflow Model Registry (Staging)"
	@echo "  make promote          - Validar Staging e promover para Production"
	@echo "  make inference        - Carregar modelo de Production e recomendar"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build imagem CPU (default)"
	@echo "  make docker-build-gpu - Build imagem GPU (com CUDA)"
	@echo "  make mlflow-up        - Iniciar stack MLflow em background (requer .env)"
	@echo "  make mlflow-down      - Parar containers MLflow"
	@echo "  make docker-train     - Rodar pipeline DVC dentro do container CPU (perfil train)"
	@echo ""

# ---------------------------------------------------------------------------
# Setup e ambiente
# ---------------------------------------------------------------------------
sync:
	uv sync

setup: sync
	@echo "Configurando ambiente..."
	@$(PYTHON) -c "import subprocess, sys; hooks_path = subprocess.run(['git', 'config', '--get', 'core.hooksPath'], capture_output=True, text=True).stdout.strip(); print('core.hooksPath configurado; instalando apenas ambientes do pre-commit.' if hooks_path else 'Instalando hook do pre-commit...'); command = [sys.executable, '-m', 'pre_commit', 'install-hooks'] if hooks_path else [sys.executable, '-m', 'pre_commit', 'install', '--install-hooks']; raise SystemExit(subprocess.call(command))"
	@$(PYTHON) scripts/setup_environment.py
	@echo "Setup concluído!"

verify:
	@echo "Validando ambiente..."
	@$(PYTHON) scripts/validate_env.py

# ---------------------------------------------------------------------------
# Qualidade
# ---------------------------------------------------------------------------
test:
	@echo "Executando testes..."
	uv run pytest

lint:
	@echo "Verificando código com ruff..."
	uv run ruff check .

format:
	@echo "Formatando código com ruff..."
	uv run ruff format .

# ---------------------------------------------------------------------------
# Avaliacao comparativa (MLflow)
# ---------------------------------------------------------------------------
baselines:
	@echo "Rodando pipeline de baselines (8 modelos) no MLflow..."
	uv run python -m techchallenge_fase2.pipelines.run_baselines

ease:
	@echo "Rodando pipeline dedicado do EASE^ no MLflow..."
	uv run python -m techchallenge_fase2.pipelines.run_ease

# ---------------------------------------------------------------------------
# Comparacao de modelos (min. 4 metricas: precision, recall, NDCG, MAP)
# ---------------------------------------------------------------------------
compare-models:
	@echo "Comparando modelos de recomendacao vs baselines..."
	uv run python -m techchallenge_fase2.pipelines.run_compare_models
	@echo "Comparacao concluida! Relatorio em reports/model_comparison_report.md"

# ---------------------------------------------------------------------------
# Model Registry (issue #16)
# ---------------------------------------------------------------------------
register:
	@echo "Registrando campeao no MLflow Model Registry (Staging)..."
	uv run python -m techchallenge_fase2.pipelines.register_model

promote:
	@echo "Validando Staging e promovendo para Production..."
	uv run python -m techchallenge_fase2.pipelines.promote_model

promote-dry-run:
	@echo "Validando Staging (dry-run, sem promover)..."
	uv run python -m techchallenge_fase2.pipelines.promote_model --dry-run

inference:
	@echo "Carregando modelo de Production para inferencia..."
	@read -p "user_id: " uid && \
		uv run python -m techchallenge_fase2.inference.load_model recommend --user-id $$uid

# ---------------------------------------------------------------------------
# Pipeline DVC
# ---------------------------------------------------------------------------
data:
	@echo "Baixando dataset RetailRocket para data/raw/ ..."
	uv run python scripts/download_dataset.py
	@echo "[OK] dataset disponível em data/raw/."

train:
	@echo "Reexecutando stage de treino do pipeline DVC (saída live)..."
	uv run dvc repro train -v

train-live:
	@echo "Rodando estagio de TREINO direto (sem DVC) com logs/tqdm live..."
	uv run python -m techchallenge_fase2.pipelines.training

pipeline:
	@echo "Reexecutando pipeline DVC completo (saída live)..."
	uv run dvc repro -v

pipeline-force:
	@echo "Reexecutando pipeline DVC completo (forcado, saída live)..."
	uv run dvc repro --force -v

pipeline-live:
	@echo "Rodando pipeline COMPLETO direto (sem DVC) com logs/tqdm live..."
	uv run python -m techchallenge_fase2.pipelines.preprocess
	uv run python -m techchallenge_fase2.pipelines.features
	uv run python -m techchallenge_fase2.pipelines.training
	uv run python -m techchallenge_fase2.pipelines.evaluation
	@echo "[OK] pipeline concluido. metricas em metrics/recommendation_metrics.json"

dvc-push:
	uv run dvc push

dvc-pull:
	uv run dvc pull

dvc-status:
	uv run dvc status

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
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

docker-train:
	@echo "Docker: Rodando pipeline completo no container CPU (perfil train)..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile train up --build train
	@$(PYTHON) -c "from pathlib import Path; values = dict(line.split('=', 1) for line in Path('.env').read_text().splitlines() if line.startswith('MLFLOW_PORT=')); port = values.get('MLFLOW_PORT', '5000').split('#', 1)[0].strip() or '5000'; print(f'[OK] Pipeline concluido. Veja runs em http://localhost:{port} e metricas em metrics/recommendation_metrics.json')"
