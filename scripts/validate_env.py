import sys
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings  # noqa: E402


def main() -> int:
    """Valida variaveis de ambiente e imprime resumo da configuracao.

    Returns:
        Codigo de saida (0 para sucesso, 1 para erro de validacao).
    """
    try:
        settings = get_settings()
    except ValidationError as error:
        print("[ERROR] Configuracao de ambiente invalida.")
        print(error)
        return 1

    print("[OK] Configuracao de ambiente valida.")
    print(f"ENV={settings.env.value}")
    print(f"RANDOM_SEED={settings.random_seed}")
    print(f"DATA_DIR={settings.data_dir}")
    print(f"MODELS_DIR={settings.models_dir}")
    print(f"CONFIGS_DIR={settings.configs_dir}")
    print(f"MLFLOW_TRACKING_URI={settings.mlflow_tracking_uri}")
    print(f"MLFLOW_EXPERIMENT_NAME={settings.mlflow_experiment_name}")
    print(f"DVC_REMOTE_URL={settings.dvc_remote_url or '<empty>'}")
    print(f"LEARNING_RATE={settings.learning_rate}")
    print(f"BATCH_SIZE={settings.batch_size}")
    print(f"EPOCHS={settings.epochs}")
    print(f"EMBEDDING_DIM={settings.embedding_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
