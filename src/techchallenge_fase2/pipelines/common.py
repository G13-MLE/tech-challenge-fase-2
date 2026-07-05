"""Funções utilitárias comuns para pipelines de ML."""

from __future__ import annotations

import os

import numpy as np


def load_dotenv_silent() -> None:
    """Carrega variáveis de ambiente do arquivo .env silenciosamente."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass


def set_global_seed(seed: int) -> None:
    """Define seed global para reprodutibilidade.

    Configura seed em NumPy para garantir resultados
    reproduzíveis em pipelines de ML.

    Args:
        seed: Valor da semente para geração de números aleatórios.
    """
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_experiment_name(
    cli_arg: str | None,
    env_var_name: str,
    default_name: str,
) -> str:
    """Obtém nome do experimento com prioridade: CLI > env var > default.

    Args:
        cli_arg: Argumento de linha de comando (maior prioridade).
        env_var_name: Nome da variável de ambiente.
        default_name: Valor padrão se nenhum outro for fornecido.

    Returns:
        Nome do experimento definido.
    """
    if cli_arg:
        return cli_arg

    env_value = os.getenv(env_var_name)
    if env_value:
        return env_value

    return default_name


def safe_get_dataset_version() -> str:
    """Obtém versão do dataset via DVC com fallback gracioso.

    Returns:
        String com versão do dataset ou "unknown".
    """
    try:
        dvc_dir = os.path.join("data", "raw")
        md5_file = os.path.join(dvc_dir, ".md5")
        if os.path.exists(md5_file):
            with open(md5_file) as f:
                return f.read().strip()[:8]

        dvc_files = [f for f in os.listdir(dvc_dir) if f.endswith(".dvc")]
        if dvc_files:
            return dvc_files[0].replace(".dvc", "")[:8]
    except (FileNotFoundError, ValueError, OSError):
        pass

    return "unknown"
