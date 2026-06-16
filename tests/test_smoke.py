import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_main_prints_expected_message() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "Hello from fase2!"


def test_docker_files_exist() -> None:
    expected_paths = [
        "docker/Dockerfile",
        "docker/Dockerfile.mlflow",
        "docker/docker-compose.yml",
        ".dockerignore",
    ]

    for path in expected_paths:
        assert (PROJECT_ROOT / path).is_file()


def test_dockerfile_has_cpu_and_gpu_targets() -> None:
    content = (PROJECT_ROOT / "docker/Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim AS builder-cpu" in content
    assert "FROM python:3.13-slim AS builder-gpu" in content
    assert "FROM runtime-base AS cpu" in content
    assert "FROM runtime-base AS gpu" in content
    assert "torch==2.12.0+cpu" in content


def test_compose_defines_mlflow_stack() -> None:
    content = (PROJECT_ROOT / "docker/docker-compose.yml").read_text(
        encoding="utf-8",
    )

    for service in ["postgres:", "minio:", "mlflow-server:", "minio-setup:"]:
        assert service in content

    assert "service_completed_successfully" in content
