.PHONY: sync setup test lint verify data dvc-push dvc-pull dvc-status

sync:
	uv sync

# Setup inicial do ambiente (espelha tech-challenge-fase-1).
# Le DVC_ONEDRIVE_REMOTE_URL do .env e configura o remote OneDrive.
setup:
	@if [ ! -f .env ]; then \
		echo "Criando .env a partir de .env.example..."; \
		cp .env.example .env; \
	fi
	uv sync
	uv run pre-commit install
	@echo "Configurando DVC remote OneDrive (opcional - repo pode ser self-contained)..."
	@URL=$$(grep -E '^DVC_ONEDRIVE_REMOTE_URL=' .env | cut -d '=' -f2-); \
	if [ -n "$$URL" ]; then \
		uv run dvc remote remove onedrive_remote 2>/dev/null || true; \
		uv run dvc remote add -d onedrive_remote "$$URL"; \
		echo "[OK] DVC remote configurado para: $$URL"; \
	else \
		echo "[WARN] DVC_ONEDRIVE_REMOTE_URL vazio. Defina o caminho da pasta OneDrive no .env."; \
	fi
	@echo "Setup concluido!"

# Download do dataset RetailRocket via Kaggle (requer KAGGLE_USERNAME/KAGGLE_KEY no .env).
data:
	@echo "Baixando dataset RetailRocket para data/raw/ ..."
	uv run python scripts/download_dataset.py
	@echo "[OK] dataset disponivel em data/raw/."

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
	uv run pytest

lint:
	uv run ruff check .

verify:
	uv run python scripts/validate_env.py
