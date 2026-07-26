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

## Fluxo do Model Registry (issue #16)
O campeao de `make compare-models` (maior `harmonic_mean_at_10`) e
registrado no MLflow Model Registry seguindo Staging -> Production:
- `make register` - descobre o campeao, empacota como pyfunc e registra em Staging.
- `make promote-dry-run` - avalia Staging vs `models/model_comparison.csv` sem promover.
- `make promote` - valida Staging (tolerancia `MLFLOW_REGISTRY_STAGING_TOLERANCE`)
  e promove para Production, arquivando versoes anteriores.
- `make inference` - carrega de Production via CLI para recomendar por `user_id`.

Codigo: `src/techchallenge_fase2/inference/` (pyfunc wrapper + CLI de
inferencia) e `src/techchallenge_fase2/pipelines/register_model.py`,
`promote_model.py`.

## Filosofia: Simplicidade em Primeiro Lugar
- **Prefira soluções simples** - evite complexidade excessiva
- **Questione cada adição** - isso realmente precisa ser adicionado?
- **Função acima da forma** - código funcional é melhor que arquitetura perfeita
- **Menos é mais** - menos linhas, menos arquivos, menos dependências
- **Prefira imutabilidade** - evite utilização desnecessária de estados em classes
- **Código em inglês, comentários em português** - utilize sempre esse estilo, códigos em inglês e comentários e documentações em português

## Clean Code
- Funções ≤ 20 linhas; nomes descritivos; type hints em todas as funções públicas.
- Docstrings no estilo Google.
- Aplicar ≥ 1 design pattern: Factory (modelos), Strategy (preprocessors) ou Template Method.
- Commits semânticos; `.gitignore`, `.dockerignore`, `.env.example` configurados.

## Dependências & Ambiente
- `pyproject.toml` com uv; dependências prod e dev separadas.
- Lock file (`uv.lock`) commitado; instalação limpa validada (`uv sync`).
- Configurações via `.env` + Pydantic Settings; seeds fixados.
- Use `python-dotenv` para carregar variáveis de ambiente
- Use arquivo `.env` para configuração local (copie de `.env.example`)
- Nunca faça commit nos arquivos `.env`
- Docker Compose usa arquivo `.env` automaticamente
- **NÃO crie documentação não solicitada** - não crie arquivos .md, CHANGELOGs ou documentos a menos que seja explicitamente solicitado.
- **NÃO guarde código obsoleto** - se o código não for usado, remova-o; não o mantenha "para uso futuro" ou "compatibilidade com versões anteriores".
- **NÃO utilize emojis em nenhum arquivo** - isso inclui código-fonte, documentação, comentários, mensagens de commit e scripts de shell.

## Linting & Qualidade
- `ruff` sem erros; pre-commit hooks ativos.

## Fluxo de Trabalho Git
- **NUNCA crie commits ou faça `git push` sozinho.** Sempre aguarde o usuário pedir ou deixe para que o usuário faça o commit e o push.
- Hooks do pre-commit rodam automaticamente no commit
- CI corrige automaticamente PRs com pre-commit
- Use mensagens de commit convencionais

## Importante
- Sempre execute `make test` e `make lint` após qualquer atualização de código.
- `make pipeline-live` e `make train-live` executam os estágios via Python
  direto (sem DVC) e exibem `tqdm` + logs em tempo real; prefira-os para
  iterar rápido. `make pipeline` / `make train` usam DVC (`dvc repro -v`).
- Para treinar no Docker reaproveitando a stack MLflow: `make docker-train`
  (perfil `train` no `docker/docker-compose.yml`).
- `make inference` carrega de `models:/TechChallengeFase2Recommender/Production`.
- O Model Card humano-legível está em `docs/MODEL_CARD.md`; o JSON logado
  em cada run do MLflow é gerado por `src/techchallenge_fase2/training/model_card.py`.
- Sempre que utilizado o português utilize linguagem técnica e português correto, com acentuação e sem erros.
