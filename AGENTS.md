# AGENTS.md — Tech Challenge Fase 02

## Objetivo
Sistema de recomendação de produtos com rede neural PyTorch, pipeline reprodutível (DVC + MLflow) e containerização Docker.

## Estrutura do Projeto
```
src/        — Código-fonte modular (módulos curtos, SOLID)
tests/      — Testes automatizados
data/       — Datasets versionados pelo DVC
models/     — Artefatos de modelo
configs/    — Configurações externalizadas
scripts/    — Scripts utilitários (ex: validate_env.py)
docker/     — Docker Images e Docker Composes
dvc.yaml    — Pipeline reprodutível
```

## Clean Code
- Funções ≤ 20 linhas; nomes descritivos; type hints em todas as funções públicas.
- Docstrings no estilo Google.
- Aplicar ≥ 1 design pattern: Factory (modelos), Strategy (preprocessors) ou Template Method.
- Commits semânticos; `.gitignore`, `.dockerignore`, `.env.example` configurados.

## Dependências & Ambiente
- `pyproject.toml` com uv; dependências prod e dev separadas.
- Lock file (`uv.lock`) commitado; instalação limpa validada (`uv sync`).
- Configurações via `.env` + Pydantic Settings; seeds fixados.

## Linting & Qualidade
- `ruff` sem erros; pre-commit hooks ativos.
