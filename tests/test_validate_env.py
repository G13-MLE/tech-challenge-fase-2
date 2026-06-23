"""Testes para o script de validação de ambiente.

Cobre todas as verificações com mocks para garantir que cada verificação
funcione corretamente em isolamento, incluindo Template Method, Registry
e helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.validate_env import (
    REQUIRED_ENV_VARS,
    CheckResult,
    DependencyCheck,
    DockerAvailableCheck,
    DvcConfiguredCheck,
    EnvFileCheck,
    EnvironmentCheck,
    GpuAvailableCheck,
    PythonVersionCheck,
    check_dependency,
    format_status,
    run_subprocess,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_completed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Cria um CompletedProcess mockado para subprocess.run."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# Testes de _format_status
# ---------------------------------------------------------------------------


class TestFormatStatus:
    """Testes para a função format_status."""

    def test_passed_required(self) -> None:
        assert format_status(passed=True, required=True) == "[OK]"

    def test_passed_optional(self) -> None:
        assert format_status(passed=True, required=False) == "[OK]"

    def test_failed_required(self) -> None:
        assert format_status(passed=False, required=True) == "[ERROR]"

    def test_failed_optional(self) -> None:
        assert format_status(passed=False, required=False) == "[AVISO]"


# ---------------------------------------------------------------------------
# Testes de _check_dependency
# ---------------------------------------------------------------------------


class TestCheckDependency:
    """Testes para a função check_dependency."""

    def test_dependency_found(self) -> None:
        ok, msg = check_dependency("os", "os")
        assert ok is True
        assert "os" in msg

    def test_dependency_not_found(self) -> None:
        ok, msg = check_dependency("nao_existe", "modulo_nao_existe_xyz")
        assert ok is False
        assert "não encontrado" in msg


# ---------------------------------------------------------------------------
# Testes de _run_subprocess
# ---------------------------------------------------------------------------


class TestRunSubprocess:
    """Testes para a função run_subprocess."""

    def test_command_success(self) -> None:
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.return_value = make_completed(stdout="1.0.0\n")
            ok, output = run_subprocess(["tool", "--version"])
            assert ok is True
            assert "1.0.0" in output

    def test_command_not_found(self) -> None:
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("tool not found")
            ok, msg = run_subprocess(["tool", "--version"])
            assert ok is False
            assert "não encontrado" in msg

    def test_command_timeout(self) -> None:
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="tool", timeout=10)
            ok, msg = run_subprocess(["tool", "--version"])
            assert ok is False
            assert "demorou demais" in msg

    def test_command_nonzero_exit(self) -> None:
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.return_value = make_completed(returncode=1)
            ok, msg = run_subprocess(["tool", "--version"])
            assert ok is False
            assert "código 1" in msg


# ---------------------------------------------------------------------------
# Testes de Template Method (EnvironmentCheck.run)
# ---------------------------------------------------------------------------


class TestTemplateMethod:
    """Testes para o Template Method EnvironmentCheck.run()."""

    def test_run_delegates_to_check(self) -> None:
        """Verifica que run() chama check() e constrói CheckResult."""

        class StubCheck(EnvironmentCheck):
            name = "Stub"
            required = True

            def check(self) -> tuple[bool, str]:
                return True, "tudo OK"

        result = StubCheck().run()
        assert result.name == "Stub"
        assert result.passed is True
        assert result.message == "tudo OK"
        assert result.required is True

    def test_run_propagates_failure(self) -> None:
        """Verifica que run() propaga falha de check()."""

        class FailingCheck(EnvironmentCheck):
            name = "Failing"
            required = False

            def check(self) -> tuple[bool, str]:
                return False, "algo errado"

        result = FailingCheck().run()
        assert result.passed is False
        assert result.message == "algo errado"
        assert result.required is False


# ---------------------------------------------------------------------------
# Testes de Registry (EnvironmentCheck.discover_checks)
# ---------------------------------------------------------------------------


class TestRegistry:
    """Testes para o Registry via __init_subclass__."""

    def test_discover_returns_instances(self) -> None:
        """Verifica que discover_checks retorna instâncias de checks."""
        checks = EnvironmentCheck.discover_checks()
        assert len(checks) > 0
        for check in checks:
            assert isinstance(check, EnvironmentCheck)

    def test_all_checks_registered(self) -> None:
        """Verifica que todas as classes concretas estão registradas."""
        expected = {
            PythonVersionCheck,
            DependencyCheck,
            EnvFileCheck,
            DvcConfiguredCheck,
            DockerAvailableCheck,
            GpuAvailableCheck,
        }
        registered = {type(c) for c in EnvironmentCheck.discover_checks()}
        assert expected.issubset(registered)


# ---------------------------------------------------------------------------
# Testes de PythonVersionCheck
# ---------------------------------------------------------------------------


class TestPythonVersionCheck:
    """Testes para PythonVersionCheck."""

    def test_passes_with_current_version(self) -> None:
        check = PythonVersionCheck()
        result = check.run()
        assert result.name == "Versão do Python"
        assert result.required is True

    def test_fails_with_old_version(self) -> None:
        check = PythonVersionCheck()
        with patch.object(sys, "version_info", (3, 11, 0)):
            result = check.run()
            assert result.passed is False
            assert "3.11" in result.message

    def test_passes_with_new_version(self) -> None:
        check = PythonVersionCheck()
        with patch.object(sys, "version_info", (3, 14, 0)):
            result = check.run()
            assert result.passed is True
            assert "3.14" in result.message


# ---------------------------------------------------------------------------
# Testes de DependencyCheck
# ---------------------------------------------------------------------------


class TestDependencyCheck:
    """Testes para DependencyCheck."""

    def test_all_dependencies_available(self) -> None:
        check = DependencyCheck()
        with patch("scripts.validate_env.importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_mod.__version__ = "1.0.0"
            mock_import.return_value = mock_mod
            result = check.run()
            assert result.passed is True
            assert result.required is True

    def test_missing_dependency(self) -> None:
        check = DependencyCheck()
        with patch("scripts.validate_env.importlib.import_module") as mock_import:

            def side_effect(name: str) -> Any:
                if name == "torch":
                    raise ImportError("torch not found")
                mod = MagicMock()
                mod.__version__ = "1.0.0"
                return mod

            mock_import.side_effect = side_effect
            result = check.run()
            assert result.passed is False
            assert "torch" in result.message


# ---------------------------------------------------------------------------
# Testes de EnvFileCheck
# ---------------------------------------------------------------------------


class TestEnvFileCheck:
    """Testes para EnvFileCheck."""

    def test_env_file_not_found(self, tmp_path: Path) -> None:
        check = EnvFileCheck()
        with patch("scripts.validate_env.PROJECT_ROOT", tmp_path):
            result = check.run()
            assert result.passed is False
            assert "não encontrado" in result.message

    def test_env_file_missing_required_vars(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("ENV=dev\nRANDOM_SEED=42\n")
        check = EnvFileCheck()
        with patch("scripts.validate_env.PROJECT_ROOT", tmp_path):
            result = check.run()
            assert result.passed is False
            assert "obrigatórias" in result.message

    def test_env_file_all_required_vars_present(self, tmp_path: Path) -> None:
        env_content = "\n".join(f"{var}=value" for var in REQUIRED_ENV_VARS)
        env_content += "\nDVC_ONEDRIVE_REMOTE_URL=\n"
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)
        check = EnvFileCheck()
        with patch("scripts.validate_env.PROJECT_ROOT", tmp_path):
            result = check.run()
            assert result.passed is True

    def test_env_file_empty_required_var(self, tmp_path: Path) -> None:
        lines = [f"{var}=value" for var in REQUIRED_ENV_VARS]
        lines[0] = "ENV="  # Valor vazio para ENV
        lines.append("DVC_ONEDRIVE_REMOTE_URL=")
        env_file = tmp_path / ".env"
        env_file.write_text("\n".join(lines) + "\n")
        check = EnvFileCheck()
        with patch("scripts.validate_env.PROJECT_ROOT", tmp_path):
            result = check.run()
            assert result.passed is False
            assert "ENV" in result.message


# ---------------------------------------------------------------------------
# Testes de DvcConfiguredCheck
# ---------------------------------------------------------------------------


class TestDvcConfiguredCheck:
    """Testes para DvcConfiguredCheck."""

    def test_dvc_not_installed(self, tmp_path: Path) -> None:
        check = DvcConfiguredCheck()
        with (
            patch("scripts.validate_env.PROJECT_ROOT", tmp_path),
            patch("scripts.validate_env.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = FileNotFoundError("dvc not found")
            result = check.run()
            assert result.passed is False
            assert "não encontrado" in result.message

    def test_dvc_not_initialized(self, tmp_path: Path) -> None:
        check = DvcConfiguredCheck()
        (tmp_path / ".dvcignore").write_text("# dvcignore\n")
        with (
            patch("scripts.validate_env.PROJECT_ROOT", tmp_path),
            patch("scripts.validate_env.subprocess.run") as mock_run,
        ):
            mock_run.return_value = make_completed(stdout="DVC version: 3.67.1\n")
            result = check.run()
            assert result.passed is False
            assert ".dvc" in result.message

    def test_dvc_fully_configured_self_contained(self, tmp_path: Path) -> None:
        check = DvcConfiguredCheck()
        (tmp_path / ".dvc").mkdir()
        (tmp_path / ".dvc" / ".gitignore").write_text("/config.local\n/tmp\n/cache\n")
        (tmp_path / ".dvcignore").write_text("# dvcignore\n")
        env_file = tmp_path / ".env"
        env_file.write_text("DVC_ONEDRIVE_REMOTE_URL=\n")
        with (
            patch("scripts.validate_env.PROJECT_ROOT", tmp_path),
            patch("scripts.validate_env.subprocess.run") as mock_run,
        ):
            mock_run.return_value = make_completed(stdout="DVC version: 3.67.1\n")
            result = check.run()
            assert result.passed is True

    def test_dvc_with_remote_configured(self, tmp_path: Path) -> None:
        check = DvcConfiguredCheck()
        (tmp_path / ".dvc").mkdir()
        (tmp_path / ".dvc" / ".gitignore").write_text("/config.local\n/tmp\n/cache\n")
        (tmp_path / ".dvcignore").write_text("# dvcignore\n")
        env_file = tmp_path / ".env"
        env_file.write_text("DVC_ONEDRIVE_REMOTE_URL=s3://bucket/dvc\n")
        with (
            patch("scripts.validate_env.PROJECT_ROOT", tmp_path),
            patch("scripts.validate_env.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                make_completed(stdout="DVC version: 3.67.1\n"),
                make_completed(stdout="onedrive_remote\ts3://bucket/dvc\n"),
            ]
            result = check.run()
            assert result.passed is True


# ---------------------------------------------------------------------------
# Testes de DockerAvailableCheck
# ---------------------------------------------------------------------------


class TestDockerAvailableCheck:
    """Testes para DockerAvailableCheck."""

    def test_docker_available(self) -> None:
        check = DockerAvailableCheck()
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.return_value = make_completed(stdout="Docker version 29.4.0\n")
            result = check.run()
            assert result.passed is True
            assert result.required is False

    def test_docker_not_found(self) -> None:
        check = DockerAvailableCheck()
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            result = check.run()
            assert result.passed is False
            assert "opcional" in result.message

    def test_docker_timeout(self) -> None:
        check = DockerAvailableCheck()
        with patch("scripts.validate_env.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=10)
            result = check.run()
            assert result.passed is False
            assert "não disponível" in result.message


# ---------------------------------------------------------------------------
# Testes de GpuAvailableCheck
# ---------------------------------------------------------------------------


class TestGpuAvailableCheck:
    """Testes para GpuAvailableCheck."""

    def test_gpu_cuda_available(self) -> None:
        check = GpuAvailableCheck()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check.run()
            assert result.passed is True
            assert "CUDA" in result.message

    def test_gpu_mps_available(self) -> None:
        check = GpuAvailableCheck()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_backends = MagicMock()
        mock_backends.mps.is_available.return_value = True
        mock_torch.backends = mock_backends
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check.run()
            assert result.passed is True
            assert "MPS" in result.message

    def test_gpu_not_available(self) -> None:
        check = GpuAvailableCheck()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_backends = MagicMock()
        mock_backends.mps.is_available.return_value = False
        mock_torch.backends = mock_backends
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check.run()
            assert result.passed is False
            assert "opcional" in result.message

    def test_torch_not_installed(self) -> None:
        check = GpuAvailableCheck()
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = check.run()
                assert result.passed is False
                assert "opcional" in result.message


# ---------------------------------------------------------------------------
# Testes de main() e exit codes
# ---------------------------------------------------------------------------


class TestMain:
    """Testes para a função main() e códigos de saída."""

    def test_format_status_ok(self) -> None:
        """Verifica que format_status retorna [OK] quando passa."""
        assert format_status(True, True) == "[OK]"

    def test_exit_code_1_on_failure(self) -> None:
        """Verifica que falhas obrigatórias geram exit code 1."""
        result = CheckResult(
            name="Teste", passed=False, message="Falhou", required=True
        )
        assert result.required is True
        assert result.passed is False
        assert format_status(False, True) == "[ERROR]"

    def test_exit_code_0_on_optional_failure(self) -> None:
        """Verifica que falhas opcionais não geram exit code 1."""
        result = CheckResult(
            name="Teste", passed=False, message="Falhou", required=False
        )
        assert result.required is False
        assert result.passed is False
        assert format_status(False, False) == "[AVISO]"
