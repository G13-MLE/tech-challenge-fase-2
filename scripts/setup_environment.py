"""Setup local multiplataforma para o projeto."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

# Caminho raiz do projeto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_NAME = "onedrive_remote"


def ensure_env_file() -> Path:
    """Garante que o arquivo .env exista."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        return env_path

    example_path = PROJECT_ROOT / ".env.example"
    if not example_path.is_file():
        raise SystemExit("[ERROR] .env.example não encontrado.")

    print("Criando .env a partir de .env.example...")
    shutil.copyfile(example_path, env_path)
    return env_path


def load_remote_url(env_path: Path) -> str:
    """Carrega a URL do remote DVC a partir do .env."""
    env_dict = dict(dotenv_values(env_path))
    return (env_dict.get("DVC_ONEDRIVE_REMOTE_URL") or "").strip()


def run_dvc_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Executa um comando DVC usando o Python do ambiente atual."""
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def configure_dvc_remote(remote_url: str) -> None:
    """Configura o remote DVC local quando a URL estiver preenchida."""
    if not remote_url:
        print("[WARN] DVC_ONEDRIVE_REMOTE_URL vazio. Defina o caminho no .env.")
        return

    print("Configurando DVC remote OneDrive...")
    run_dvc_command(["remote", "remove", "--local", REMOTE_NAME])
    result = run_dvc_command(
        ["remote", "add", "-d", "--local", REMOTE_NAME, remote_url],
    )
    if result.returncode != 0:
        raise SystemExit((result.stderr or result.stdout).strip())

    print(f"[OK] DVC remote configurado para: {remote_url}")


def main() -> None:
    """Orquestra o setup local complementar ao uv e pre-commit."""
    env_path = ensure_env_file()
    configure_dvc_remote(load_remote_url(env_path))
    print("Setup concluído!")


if __name__ == "__main__":
    main()
