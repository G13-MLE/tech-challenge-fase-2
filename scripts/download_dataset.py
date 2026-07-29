"""Download do dataset RetailRocket Ecommerce via API do Kaggle.

Baixa os arquivos events.csv, item_properties_part1.csv,
item_properties_part2.csv e category_tree.csv para data/raw/.
Credenciais lidas do .env (KAGGLE_USERNAME e KAGGLE_KEY).

Execução:
    uv run python scripts/download_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import dotenv_values

# Caminho raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Slug do dataset no Kaggle
KAGGLE_DATASET_SLUG = "retailrocket/ecommerce-dataset"

# Arquivos esperados após o download
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
    """Garante que o diretório data/raw/ exista e retorna seu caminho."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def dataset_already_downloaded(raw_dir: Path) -> bool:
    """Verifica se todos os arquivos esperados já existem em data/raw/.

    Args:
        raw_dir: Caminho do diretório data/raw/.

    Returns:
        True se todos os arquivos esperados existirem.
    """
    return all((raw_dir / name).is_file() for name in EXPECTED_FILES)


def download_dataset(raw_dir: Path) -> None:
    """Baixa e extrai o dataset RetailRocket via API do Kaggle.

    Args:
        raw_dir: Caminho do diretório destino (data/raw/).

    Raises:
        SystemExit: Se o download falhar.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] pacote kaggle não instalado. Rode: uv sync")
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
    """Confirma que todos os arquivos esperados estão presentes.

    Args:
        raw_dir: Caminho do diretório data/raw/.

    Raises:
        SystemExit: Se algum arquivo esperado estiver ausente.
    """
    missing = [n for n in EXPECTED_FILES if not (raw_dir / n).is_file()]
    if missing:
        joined = ", ".join(missing)
        print(f"[ERROR] arquivos ausentes após download: {joined}")
        sys.exit(1)
    print("[OK] dataset baixado e verificado.")


@contextmanager
def kaggle_config_env(username: str, key: str) -> Iterator[None]:
    """Expoem credenciais Kaggle via kaggle.json temporario.

    Escreve ``kaggle.json`` em um diretório temporário com permissão 0600 e
    aponta ``KAGGLE_CONFIG_DIR`` para ele, mantendo as credenciais fora de
    ``os.environ`` (evitando vazamento para subprocessos e dumps de ambiente).
    Remove o arquivo ao sair do contexto.
    """
    with tempfile.TemporaryDirectory(prefix="kaggle-cfg-") as tmpdir:
        config_path = Path(tmpdir) / "kaggle.json"
        config_path.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
        config_path.chmod(0o600)
        previous_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        os.environ["KAGGLE_CONFIG_DIR"] = tmpdir
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        try:
            yield
        finally:
            if previous_config_dir is None:
                os.environ.pop("KAGGLE_CONFIG_DIR", None)
            else:
                os.environ["KAGGLE_CONFIG_DIR"] = previous_config_dir


def main() -> None:
    """Orquestra o download do dataset RetailRocket de forma idempotente."""
    username, key = load_kaggle_credentials()
    raw_dir = ensure_raw_dir()
    if dataset_already_downloaded(raw_dir):
        print("[SKIP] dataset já presente em data/raw/. Nada a fazer.")
        return

    with kaggle_config_env(username, key):
        download_dataset(raw_dir)
    verify_download(raw_dir)


if __name__ == "__main__":
    main()
