"""Download do dataset RetailRocket Ecommerce via API do Kaggle.

Baixa os arquivos events.csv, item_properties_part1.csv,
item_properties_part2.csv e category_tree.csv para data/raw/.
Credenciais lidas do .env (KAGGLE_USERNAME e KAGGLE_KEY).

Execucao:
    uv run python scripts/download_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

# Caminho raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Slug do dataset no Kaggle
KAGGLE_DATASET_SLUG = "retailrocket/ecommerce-dataset"

# Arquivos esperados apos o download
EXPECTED_FILES: tuple[str, ...] = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)


def load_kaggle_credentials() -> tuple[str, str]:
    """Carrega credenciais Kaggle do .env sem poluir os.environ.

    Returns:
        Tupla (username, key).

    Raises:
        SystemExit: Se as credenciais estiverem ausentes ou vazias.
    """
    env_path = PROJECT_ROOT / ".env"
    env_dict = dict(dotenv_values(env_path)) if env_path.exists() else {}
    username = (env_dict.get("KAGGLE_USERNAME") or "").strip()
    key = (env_dict.get("KAGGLE_KEY") or "").strip()

    if not username or not key:
        print("[ERROR] KAGGLE_USERNAME e KAGGLE_KEY ausentes no .env.")
        print("Crie um token em https://www.kaggle.com/settings -> API.")
        sys.exit(1)
    return username, key


def ensure_raw_dir() -> Path:
    """Garante que o diretorio data/raw/ exista e retorna seu caminho."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def dataset_already_downloaded(raw_dir: Path) -> bool:
    """Verifica se todos os arquivos esperados ja existem em data/raw/.

    Args:
        raw_dir: Caminho do diretorio data/raw/.

    Returns:
        True se todos os arquivos esperados existirem.
    """
    return all((raw_dir / name).is_file() for name in EXPECTED_FILES)


def download_dataset(raw_dir: Path) -> None:
    """Baixa e extrai o dataset RetailRocket via API do Kaggle.

    Args:
        raw_dir: Caminho do diretorio destino (data/raw/).

    Raises:
        SystemExit: Se o download falhar.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] pacote kaggle nao instalado. Rode: uv sync")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    print(f"Baixando {KAGGLE_DATASET_SLUG} para {raw_dir} ...")
    api.dataset_download_files(
        KAGGLE_DATASET_SLUG,
        path=str(raw_dir),
        unzip=True,
        quiet=False,
    )


def verify_download(raw_dir: Path) -> None:
    """Confirma que todos os arquivos esperados estao presentes.

    Args:
        raw_dir: Caminho do diretorio data/raw/.

    Raises:
        SystemExit: Se algum arquivo esperado estiver ausente.
    """
    missing = [n for n in EXPECTED_FILES if not (raw_dir / n).is_file()]
    if missing:
        joined = ", ".join(missing)
        print(f"[ERROR] arquivos ausentes apos download: {joined}")
        sys.exit(1)
    print("[OK] dataset baixado e verificado.")


def main() -> None:
    """Orquestra o download do dataset RetailRocket de forma idempotente."""
    username, key = load_kaggle_credentials()
    # A lib kaggle le as credenciais do ambiente.
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    raw_dir = ensure_raw_dir()
    if dataset_already_downloaded(raw_dir):
        print("[SKIP] dataset ja presente em data/raw/. Nada a fazer.")
        return

    download_dataset(raw_dir)
    verify_download(raw_dir)


if __name__ == "__main__":
    main()
