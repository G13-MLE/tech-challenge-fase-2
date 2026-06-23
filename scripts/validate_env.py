"""Validação de ambiente para o projeto tech-challenge-fase2.

Verifica se o ambiente está corretamente configurado para execução
do projeto, incluindo versão do Python, dependências, arquivo .env,
DVC, Docker e GPU. Imprime status OK/ERROR/AVISO para cada verificação
e retorna código de saída 0 se tudo OK, 1 se houver problemas.

Design pattern: Template Method + Registry via __init_subclass__.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

# Caminho raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Variáveis obrigatórias do .env (devem estar presentes e com valor)
REQUIRED_ENV_VARS: list[str] = [
    "ENV",
    "RANDOM_SEED",
    "DATA_DIR",
    "MODELS_DIR",
    "CONFIGS_DIR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "LEARNING_RATE",
    "BATCH_SIZE",
    "EPOCHS",
    "EMBEDDING_DIM",
]

# Variáveis opcionais do .env (devem estar presentes, valor pode ser vazio)
OPTIONAL_ENV_VARS: list[str] = [
    "DVC_REMOTE_URL",
]

# Mapeamento de nome de pacote para nome de importação
DEPENDENCY_MAP: dict[str, str] = {
    "torch": "torch",
    "scikit-learn": "sklearn",
    "mlflow": "mlflow",
    "dvc": "dvc",
}

# Versão mínima do Python
MIN_PYTHON_VERSION = (3, 13)


@dataclass(frozen=True)
class CheckResult:
    """Resultado de uma verificação de ambiente."""

    name: str
    passed: bool
    message: str
    required: bool


class EnvironmentCheck(ABC):
    """Classe base abstrata para verificações de ambiente.

    Implementa Template Method: ``run()`` orquestra o fluxo invariante
    (executar verificação e construir CheckResult), enquanto ``check()``
    define o passo variável implementado por cada subclasse.

    Registry: subclasses concretas se registram automaticamente via
    ``__init_subclass__``, eliminando listas hardcoded em main().
    """

    registry: list[type[EnvironmentCheck]] = []

    # Atributos de classe que subclasses devem definir
    name: str
    required: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Registra subclasses concretas automaticamente."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstract__", False):
            EnvironmentCheck.registry.append(cls)

    @abstractmethod
    def check(self) -> tuple[bool, str]:
        """Executa a verificação e retorna (passou, mensagem).

        Returns:
            Tupla (passou, mensagem_descritiva).
        """
        ...

    def run(self) -> CheckResult:
        """Template Method: orquestra verificação e construção do resultado."""
        passed, message = self.check()
        return CheckResult(
            name=self.name, passed=passed, message=message, required=self.required
        )

    @classmethod
    def discover_checks(cls) -> list[EnvironmentCheck]:
        """Retorna instâncias de todas as verificações registradas."""
        return [subcls() for subcls in cls.registry]


def format_status(passed: bool, required: bool) -> str:
    """Formata o status da verificação em português.

    Args:
        passed: Se a verificação passou.
        required: Se a verificação é obrigatória.

    Returns:
        String de status: "[OK]", "[ERROR]" ou "[AVISO]".
    """
    if passed:
        return "[OK]"
    return "[ERROR]" if required else "[AVISO]"


def check_dependency(pkg_name: str, import_name: str) -> tuple[bool, str]:
    """Verifica se uma dependência está instalada e retorna versão.

    Args:
        pkg_name: Nome do pacote (para exibição).
        import_name: Nome do módulo para importação.

    Returns:
        Tupla (instalado, mensagem).
    """
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "desconhecida")
        return True, f"{pkg_name} {version}"
    except ImportError:
        return False, f"{pkg_name} não encontrado"


def run_subprocess(command: list[str]) -> tuple[bool, str]:
    """Executa um comando subprocess de forma segura.

    Args:
        command: Lista de argumentos do comando.

    Returns:
        Tupla (sucesso, stdout_ou_mensagem_erro).
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, f"comando retornou código {result.returncode}"
    except FileNotFoundError:
        cmd = command[0]
        return False, f"comando {cmd} não encontrado"
    except subprocess.TimeoutExpired:
        cmd = command[0]
        return False, f"{cmd} demorou demais para responder"


class PythonVersionCheck(EnvironmentCheck):
    """Verifica se a versão do Python atende o requisito mínimo."""

    name = "Versão do Python"
    required = True

    def check(self) -> tuple[bool, str]:
        """Verifica versão do Python instalada."""
        current = sys.version_info[:2]
        passed = current >= MIN_PYTHON_VERSION
        version_str = f"{current[0]}.{current[1]}"
        required_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        if passed:
            return True, f"Python {version_str} (>= {required_str})"
        return False, f"Python {version_str} (requer >= {required_str})"


class DependencyCheck(EnvironmentCheck):
    """Verifica se as dependências obrigatórias estão instaladas."""

    name = "Dependências"
    required = True

    def check(self) -> tuple[bool, str]:
        """Verifica dependências instaladas e suas versões."""
        results = [check_dependency(pkg, imp) for pkg, imp in DEPENDENCY_MAP.items()]
        missing = [msg for ok, msg in results if not ok]
        if missing:
            return False, "Faltando: " + ", ".join(missing)
        found = [msg for ok, msg in results if ok]
        return True, ", ".join(found)


class EnvFileCheck(EnvironmentCheck):
    """Verifica se o arquivo .env está preenchido com variáveis obrigatórias."""

    name = "Arquivo .env"
    required = True

    def load_env_dict(self) -> dict[str, str | None]:
        """Carrega variáveis do .env sem modificar os.environ.

        Returns:
            Dicionário com chave=valor do .env, ou vazio se não encontrado.
        """
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return {}
        return dict(dotenv_values(env_path))

    def check_required_vars(self, env_dict: dict[str, str | None]) -> list[str]:
        """Retorna variáveis obrigatórias faltando ou com valor vazio.

        Args:
            env_dict: Dicionário de variáveis carregadas do .env.

        Returns:
            Lista de nomes de variáveis ausentes ou vazias.
        """
        return [
            var
            for var in REQUIRED_ENV_VARS
            if not (value := env_dict.get(var)) or value.strip() == ""
        ]

    def check_optional_vars(self, env_dict: dict[str, str | None]) -> list[str]:
        """Retorna variáveis opcionais ausentes do .env.

        Args:
            env_dict: Dicionário de variáveis carregadas do .env.

        Returns:
            Lista de nomes de variáveis opcionais ausentes.
        """
        return [var for var in OPTIONAL_ENV_VARS if var not in env_dict]

    def check(self) -> tuple[bool, str]:
        """Verifica arquivo .env com variáveis obrigatórias e opcionais."""
        env_dict = self.load_env_dict()
        if not env_dict and not (PROJECT_ROOT / ".env").exists():
            return False, "Arquivo .env não encontrado"

        missing_required = self.check_required_vars(env_dict)
        if missing_required:
            vars_str = ", ".join(missing_required)
            return False, f"Variáveis obrigatórias faltando ou vazias: {vars_str}"

        missing_optional = self.check_optional_vars(env_dict)
        if missing_optional:
            vars_str = ", ".join(missing_optional)
            return True, f"Variáveis opcionais ausentes: {vars_str}"

        msg = f"Variáveis OK ({len(REQUIRED_ENV_VARS)} obrigatórias"
        msg += f", {len(OPTIONAL_ENV_VARS)} opcionais)"
        return True, msg


class DvcConfiguredCheck(EnvironmentCheck):
    """Verifica se o DVC está instalado e o projeto está inicializado."""

    name = "DVC configurado"
    required = True

    def check_dvc_cli(self) -> tuple[bool, str]:
        """Verifica se o comando dvc está disponível via CLI.

        Returns:
            Tupla (disponível, versao_ou_erro).
        """
        ok, output = run_subprocess(["dvc", "version"])
        if not ok:
            return False, output
        # Extrair apenas a primeira linha com a versão
        version_line = output.split("\n")[0]
        return True, version_line

    def check_dvc_init(self) -> tuple[bool, str]:
        """Verifica se o projeto foi inicializado com dvc init.

        Returns:
            Tupla (inicializado, mensagem).
        """
        dvc_dir = PROJECT_ROOT / ".dvc"
        dvcignore = PROJECT_ROOT / ".dvcignore"

        if not dvc_dir.is_dir():
            return False, "diretório .dvc não encontrado"
        if not dvcignore.is_file():
            return False, "arquivo .dvcignore não encontrado"
        return True, "projeto inicializado com DVC"

    def check_dvc_remote(self) -> tuple[bool, str]:
        """Verifica se um remote DVC está configurado (se DVC_REMOTE_URL definido).

        Returns:
            Tupla (remote_ok, mensagem).
        """
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return True, "sem remote (self-contained)"

        env_dict = dict(dotenv_values(env_path))
        remote_url = env_dict.get("DVC_REMOTE_URL", "").strip()

        if not remote_url:
            return True, "sem remote (self-contained)"

        ok, output = run_subprocess(["dvc", "remote", "list"])
        if ok and output:
            return True, f"remote configurado: {remote_url}"
        return False, f"DVC_REMOTE_URL definido ({remote_url}) mas remote ausente"

    def check(self) -> tuple[bool, str]:
        """Verifica DVC instalado, inicializado e remote configurado."""
        cli_ok, cli_msg = self.check_dvc_cli()
        if not cli_ok:
            return False, cli_msg

        init_ok, init_msg = self.check_dvc_init()
        if not init_ok:
            return False, init_msg

        remote_ok, remote_msg = self.check_dvc_remote()
        if not remote_ok:
            return False, remote_msg

        return True, f"{cli_msg}; {init_msg}; {remote_msg}"


class DockerAvailableCheck(EnvironmentCheck):
    """Verifica se o Docker está disponível (verificação opcional)."""

    name = "Docker"
    required = False

    def check(self) -> tuple[bool, str]:
        """Verifica disponibilidade do Docker via CLI."""
        ok, output = run_subprocess(["docker", "--version"])
        if ok:
            return True, output
        return False, "não disponível (opcional)"


class GpuAvailableCheck(EnvironmentCheck):
    """Verifica se uma GPU está disponível via PyTorch (verificação opcional)."""

    name = "GPU"
    required = False

    def check(self) -> tuple[bool, str]:
        """Verifica disponibilidade de GPU (CUDA ou MPS)."""
        try:
            import torch

            available: list[str] = []
            if torch.cuda.is_available():
                available.append(f"CUDA ({torch.cuda.get_device_name(0)})")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append("MPS (Apple Silicon)")

            if available:
                return True, "Disponível: " + ", ".join(available)
            return False, "nenhuma GPU detectada (opcional)"
        except ImportError:
            return False, "torch não instalado (opcional)"


def main() -> None:
    """Executa todas as verificações de ambiente e imprime resultados.

    Descobre verificações automaticamente via Registry. Retorna código
    de saída 0 se todas as verificações obrigatórias passaram, ou 1 caso
    contrário.
    """
    checks = EnvironmentCheck.discover_checks()
    results = [check.run() for check in checks]

    # Imprimir cabeçalho
    print("=" * 60)
    print("Validação de Ambiente - tech-challenge-fase2")
    print("=" * 60)

    # Imprimir cada resultado
    for result in results:
        status = format_status(result.passed, result.required)
        print(f"  {status} {result.name}: {result.message}")

    # Imprimir resumo
    print("=" * 60)
    required_failed = [r for r in results if r.required and not r.passed]
    if required_failed:
        failed_names = ", ".join(r.name for r in required_failed)
        print(f"Falha em verificações obrigatórias: {failed_names}")
        print("Corrija os problemas acima e execute novamente.")
        sys.exit(1)
    else:
        print("Ambiente configurado corretamente!")
        sys.exit(0)


if __name__ == "__main__":
    main()
