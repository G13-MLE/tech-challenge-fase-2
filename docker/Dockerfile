# =============================================================================
# Dockerfile multi-stage: builder + runtime
# Targets: cpu (default), gpu
#
# Build:
#   make docker-build              # CPU (default, menor)
#   make docker-build-gpu          # GPU (com CUDA)
#
# Run:
#   docker run --rm techchallenge-fase2
#   docker run --rm --gpus all techchallenge-fase2:gpu
# =============================================================================

# -----------------------------------------------------------------------------
# Estágio builder-cpu: instala dependências com torch CPU-only
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder-cpu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_HTTP_TIMEOUT=120

# Copiar metadados de dependências primeiro (cache de camada Docker)
COPY pyproject.toml uv.lock ./

# Criar ambiente virtual com pip para instalação resiliente em redes lentas
RUN uv venv --seed /app/.venv

# Instalar dependências de produção travadas e torch CPU-only pelo índice dedicado.
# O lock padrão contém pacotes CUDA para Linux, então eles são filtrados no CPU.
RUN uv export --frozen --no-dev --format requirements-txt --no-hashes \
        | grep -Ev '^(torch|nvidia-|cuda-|triton)' \
        > /tmp/requirements-cpu.txt && \
    /app/.venv/bin/python -m pip install \
        --retries 10 \
        --timeout 120 \
        -r /tmp/requirements-cpu.txt && \
    /app/.venv/bin/python -m pip install \
        --retries 10 \
        --timeout 120 \
        "torch==2.12.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
        --no-deps && \
    rm /tmp/requirements-cpu.txt

# Limpar caches e arquivos desnecessários do .venv
RUN find /app/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name '*.pyc' -delete 2>/dev/null; \
    find /app/.venv -name '*.pyo' -delete 2>/dev/null; \
    true

# -----------------------------------------------------------------------------
# Estágio builder-gpu: instala dependências com torch CUDA
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder-gpu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_HTTP_TIMEOUT=120

# Copiar metadados de dependências primeiro (cache de camada Docker)
COPY pyproject.toml uv.lock ./

# Instalar dependências de produção com torch padrão (inclui CUDA)
# --frozen: respeita o lock file, não atualiza
# --no-dev: exclui pytest, ruff, pre-commit
# --no-install-project: não instala o projeto como pacote editável
RUN uv sync --frozen --no-dev --no-install-project

# Limpar caches e arquivos desnecessários do .venv
RUN find /app/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name '*.pyc' -delete 2>/dev/null; \
    find /app/.venv -name '*.pyo' -delete 2>/dev/null; \
    true

# -----------------------------------------------------------------------------
# Estágio runtime-base: configurações comuns entre cpu e gpu
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime-base

# Criar usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin appuser

WORKDIR /home/appuser/app

# Configurar ambiente
ENV PATH="/home/appuser/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# Estágio gpu: imagem final para GPU (deve vir ANTES de cpu para não ser default)
# -----------------------------------------------------------------------------
FROM runtime-base AS gpu

# Copiar ambiente virtual do builder-gpu
COPY --from=builder-gpu /app/.venv /home/appuser/.venv

# Copiar código da aplicação
COPY main.py ./
COPY src/ ./src/
COPY configs/ ./configs/

# Ajustar permissões
RUN chown -R appuser:appuser /home/appuser

USER appuser

CMD ["python", "main.py"]

# -----------------------------------------------------------------------------
# Estágio cpu: imagem final para CPU (default - último estágio)
# -----------------------------------------------------------------------------
FROM runtime-base AS cpu

# Copiar ambiente virtual do builder-cpu
COPY --from=builder-cpu /app/.venv /home/appuser/.venv

# Copiar código da aplicação
COPY main.py ./
COPY src/ ./src/
COPY configs/ ./configs/

# Ajustar permissões
RUN chown -R appuser:appuser /home/appuser

USER appuser

CMD ["python", "main.py"]
