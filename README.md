# Tech Challenge FIAP - Fase 2

Sistema de recomendação de produtos para e-commerce baseado no comportamento de
navegação dos usuários, com rede neural **PyTorch (NCF)**, **Scikit-Learn**
baselines, pipeline reprodutível com **DVC** e experimentos rastreados com
**MLflow** + **Model Registry**. Tudo containerizado em **Docker**.

> Projeto do grupo **G13-MLE** para o Tech Challenge da Fase 02 (PÓS TECH FIAP).
> Dataset: **RetailRocket E-Commerce** (2.756.101 interações eventos).

---

## Sumario

- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Entregáveis por etapa](#entregáveis-por-etapa)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Setup rápido](#setup-rápido)
- [Como executar o pipeline](#como-executar-o-pipeline)
- [Avaliação comparativa e Model Registry](#avaliação-comparativa-e-model-registry)
- [Inferência](#inferência)
- [Docker](#docker)
- [Testes e lint](#testes-e-lint)
- [Documentação complementar](#documentação-complementar)
- [Dataset](#dataset)
- [Creditos](#creditos)

## Visão geral

| Item            | Valor                                                               |
|-----------------|---------------------------------------------------------------------|
| Problema        | Recomendação top-K de produtos para usuários de e-commerce          |
| Dataset         | [RetailRocket E-Commerce](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| Modelo campeão  | Neural Collaborative Filtering (GMF + MLP) em PyTorch               |
| Baselines       | Popularity, RecentItems, Random, ItemKNN, Logistic Regression, EASE^|
| Métricas        | Precision, Recall, NDCG, MAP, HitRate @10 + `harmonic_mean_at_10`  |
| Orquestração    | DVC (preprocess → feature_eng → train → evaluate)                    |
| Tracking        | MLflow (params / métricas / artefatos / Model Card JSON)             |
| Registry        | MLflow Model Registry (Staging → Production)                       |
| Container       | Docker multi-stage (CPU/GPU) + compose (Postgres + MinIO + MLflow)  |
| Build           | uv + `pyproject.toml` com deps prod/dev separadas                   |
| Lint            | ruff + pre-commit                                                   |
| Testes          | pytest conforme `--cov` (268+ testes)                              |

## Arquitetura

```
                    +----------------------+
 events.csv         |  DVC pipeline        |        +----------------+
 (RetailRocket) --> | preprocess ->        | -----> | MLflow tracking|
                    | feature_eng ->       |        | (params, metrics,|
                    | train -> evaluate    |        |  artifacts,      |
                    +----------------------+        |  model_card.json)|
                           |                        +----------------+
                           v                                |
                  models/torch_embedding_                    |
                  recommender.pt                              |
                           |                                v
                           v                        +------------------+
                  metrics/recommendation_          | MLflow Model     |
                  metrics.json                     | Registry          |
                                                   | Staging ->        |
                                                   | Production       |
                                                   +------------------+
                                                           | v
                                                           v
                                                  inference CLI (`load_model`)
                                                  recommends via models:/.../Production
```

Design patterns aplicados:
- **Factory** — `src/techchallenge_fase2/models/factory.py` instancia modelos por nome.
- **Strategy** — preprocessadores intercambiaveis via `params.features.event_weights`.
- **Template Method** — `Trainer._run_loop` orquestra treino com early stopping + checkpoints.

## Entregáveis por etapa

| Etapa | Requisito do edital                                                | Status        | Localização                                                |
|-------|--------------------------------------------------------------------|---------------|-----------------------------------------------------------|
| 1     | Estrutura `src/ tests/ data/ models/ configs/`                      | ✔ Concluído   | raiz do repo                                               |
| 1     | Naming, SOLID, type hints, docstrings Google                        | ✔ Concluído   | `src/techchallenge_fase2/`                                 |
| 1     | Design pattern (Factory / Strategy / Template Method)              | ✔ Concluído   | `models/factory.py`, `pipelines/preprocess.py`, `training/trainer.py` |
| 1     | ruff + pre-commit sem erros                                          | ✔ Concluído   | `pyproject.toml`, `.pre-commit-config.yaml`              |
| 2     | `pyproject.toml` (uv) com deps prod/dev separadas e lock commitado  | ✔ Concluído   | `pyproject.toml`, `uv.lock`                                |
| 2     | `.env` + Pydantic Settings                                          | ✔ Concluído   | `configs/settings.py`, `.env.example`                     |
| 2     | `scripts/validate_env.py`                                            | ✔ Concluído   | `scripts/validate_env.py`                                  |
| 2     | Instalação limpa validada                                            | ✔ Concluído   | `make sync && make verify`                                |
| 3     | Dockerfile multi-stage (builder + runtime)                          | ✔ Concluído   | `docker/Dockerfile` (targets cpu/gpu)                     |
| 3     | `docker-compose.yml` (MLflow + Postgres + MinIO + serviço treino)  | ✔ Concluído   | `docker/docker-compose.yml`                               |
| 3     | DVC init, dataset versionado (remote OneDrive sincronizado)           | ✔ Validado    | `data/raw.dvc`, `.dvc/config.local`, `scripts/setup_environment.py` |
| 3     | Pipeline DVC ≥ 3 stages (preprocess/feature_eng/train/evaluate)    | ✔ Reprodutível | `dvc.yaml` (4 stages), `dvc.lock` atualizado c/ dataset real; `dvc pull` testado em clone limpo |
| 3     | MLflow tracking (params/métricas/artefatos/Model Card)              | ✔ Concluído   | `training/mlflow_tracking.py`                              |
| 4     | MLP/NCF em PyTorch + early stopping                                 | ✔ Treinado     | `models/ncf.py`, `training/trainer.py`, `training/early_stopping.py` |
| 4     | Comparação com baselines scikit-learn usando ≥ 4 métricas          | ⚠ Pendente     | `pipelines/run_baselines.py`, `run_compare_models.py`; NCF avaliado (5 métricas @10), baselines + `make compare-models` pendentes |
| 4     | Model Registry Staging → Production                                 | ⚠ Pendente     | `pipelines/register_model.py`, `promote_model.py`, `inference/load_model.py`; depende de `make compare-models` declarar o campeao |
| 4     | Model Card                                                          | ✔ Concluído   | `docs/MODEL_CARD.md` (gerado tambem como JSON no MLflow)  |
| 4     | README completo                                                     | ✔ Concluído   | este arquivo                                              |
| 4     | Vídeo STAR de 5 minutos                                             | ⚠ Em progresso | link externo: TBD                                          |
| Bônus | Deploy em nuvem via Docker (URL pública + /health)                  | ⚠ Opcional    | issue #20 aberta                                          |

## Estrutura do projeto

```
.
├── src/techchallenge_fase2/
│   ├── models/           # NCF, EASE^, baselines, factory
│   ├── pipelines/        # preprocess, features, training, evaluation, run_*
│   ├── training/         # trainer, early_stopping, checkpoint, mlflow_tracking, model_card
│   ├── inference/        # load_model (CLI), mlflow_wrapper (pyfunc)
│   └── data/             # InteractionData dataclass
├── tests/                # 268+ testes pytest alinhados com src/
├── docker/               # Dockerfile (cpu/gpu), Dockerfile.mlflow, docker-compose.yml
├── configs/              # settings.py (Pydantic Settings)
├── data/{raw,processed,features}/  # versionados via DVC
├── models/               # artefatos .pt, checkpoints/ (DVC)
├── metrics/              # recommendation_metrics.json (DVC metrics)
├── reports/              # model_comparison_report.md (gerado por `make compare-models`)
├── notebooks/            # 4 notebooks de EDA/baseline
├── docs/                 # tech_challenge_fase02.md, MODEL_CARD.md, ML_CANVAS.pdf
├── scripts/              # download_dataset.py, setup_environment.py, validate_env.py
├── params.yaml           # hiperparametros do pipeline (lidos pelo DVC)
├── dvc.yaml              # pipeline de 4 stages
├── Makefile              # atalhos para setup/lint/test/pipeline/registry
└── pyproject.toml        # uv + deps prod/dev
```

## Pré-requisitos

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (gerenciador de dependências)
- [DVC](https://dvc.org/) (instalado via `uv sync` como dep de prod)
- [Docker](https://www.docker.com/) + Docker Compose (para o stack MLflow)
- Conta Kaggle + token (para `make data`)

## Setup rápido

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/G13-MLE/tech-challenge-fase-2.git
cd tech-challenge-fase-2

# 2. Copiar variaveis de ambiente e preencher credenciais
cp .env.example .env
#   Edite .env e preencha:
#     - KAGGLE_USERNAME, KAGGLE_KEY          (para baixar o dataset)
#     - DVC_ONEDRIVE_REMOTE_URL              (obrigatorio para dvc push/pull;
#                                            caminho local sincronizado pelo
#                                            cliente OneDrive, ex. Linux:
#                                            /home/<user>/OneDrive/techchallenge-fase2/files/
#                                            macOS:
#                                            /Users/<user>/OneDrive - FIAP/techchallenge-fase2/files/
#                                            Windows:
#                                            C:/Users/<user>/OneDrive/techchallenge-fase2/files/)
#     - MLFLOW_TRACKING_URI                  (default http://localhost:5000 ou file:./mlruns)

# 3. Sincronizar dependencias e configurar pre-commit + DVC remote
make setup                       # uv sync + pre-commit install + setup_environment.py
make verify                      # valida Python, deps, .env, DVC e Docker

# 4. Baixar o dataset RetailRocket via Kaggle
make data
```

## Como executar o pipeline

Existem dois caminhos concomitantes. O **A** usa DVC (rastreabilidade), o **B**
invoca os modulos diretamente (maximo progresso visual via tqdm).

### Caminho A — pipeline reprodutivel via DVC

```bash
# Inicia o stack MLflow (Postgres + MinIO + mlflow-server) em background
make mlflow-up

# Executa os 4 stages com saída ao vivo (-v):
#   preprocess   -> data/processed/interactions.parquet
#   feature_eng  -> data/features/{train,validation,test}.parquet + mappings.json + dataset_stats.json
#   train        -> models/torch_embedding_recommender.pt + models/checkpoints/ + MLflow run
#   evaluate     -> metrics/recommendation_metrics.json (DVC metrics) + MLflow run
make pipeline

# Para refazer tudo do zero (ignora cache DVC):
make pipeline-force

# Persistir artefatos no remote DVC (OneDrive):
make dvc-push
```

### Caminho B — pipeline direto (mais rápido para iteração visualize)

Reaproveita os parquets ja gerados se existirem; pula o DVC e exibe tqdm/logs em
tempo real sem buffer do DVC.

```bash
# Tudo de uma vez:
make pipeline-live

# Ou apenas o treino (avaliacao depois, opcional):
make train-live
```

Você verá:
```
2026-07-26 17:04:44 - __main__ - INFO - [STAGE 3/4] TRAIN - treino do NCF (PyTorch) com MLflow tracking
2026-07-26 17:04:44 - __main__ - INFO -   catalogo: usuarios=1407580 itens=235061 embedding_dim=64
2026-07-26 17:04:44 - __main__ - INFO -   treino: 1929270 interacoes | validacao: 413415 interacoes
Epochs:   0%|          | 0/20 [00:00<?<?, ?ep/s]                  # tqdm global
  ep  1/20:   1%|          | 38/3000 [00:01<-01:00, 37.5batch/s]  # tqdm por epoca
  ep  1/20 val: 50%|#####     | 870/1740 [00:00<00:01, 775.4batch/s]
2026-07-26 17:04:48 - techchallenge_fase2.training.trainer - INFO -   [BEST] ep 1/20  loss=0.6234  val_auc=0.7112
```

### Hiperparametros

Todos os hiperparametros ficam em [`params.yaml`](params.yaml) e são lidos pelo
DVC. Os principais:

| Bloco        | Parametro        | Default | Descrição                                                    |
|--------------|------------------|---------|-------------------------------------------------------------|
| `preprocess` | `sample_size`    | 0       | 0 = dataset completo. >0 = amostragem deterministica        |
| `preprocess` | `random_seed`    | 42      | Seed reprodutivel                                           |
| `features`   | `train_ratio`    | 0.70    | % cronologica para treino                                   |
| `features`   | `validation_ratio` | 0.15  | % cronologica para validacao                                |
| `training`   | `epochs`         | 20      | Maximo de épocas (early stopping com `patience=5`)           |
| `training`   | `batch_size`     | 1024    | Tamanho do mini-batch                                       |
| `training`   | `embedding_dim`  | 64      | Dimensão dos embeddings GMF/MLP                             |
| `training`   | `learning_rate` | 0.001   | LR do Adam                                                  |
| `training`   | `negative_samples`| 1      | Negativos por usuário (rejeicao amostral O(1))             |
| `evaluation` | `top_k`          | 10      | K para métricas Top-K                                       |
| `evaluation` | `max_users`      | 1000    | Amostra de usuários avaliados (warm-start)                  |

## Avaliação comparativa e Model Registry

O fluxo completo Staging → Production segue três etapas:
`compare-models` → `register` → `promote`.

```bash
# 1. Avaliar NCF vs EASE^ vs baselines scikit-learn com >=4 metricas + harmonic@10
make compare-models
#   - Roda modelos (==3 runs no MLflow) sobre validation/teste
#   - Gera models/model_comparison.csv e reports/model_comparison_report.md (markdown)
#   - O campeao e o de maior harmonic_mean_at_10

# 2. Empacotar o campeao como pyfunc e registar no Model Registry em Staging
make register

# 3. Validar Staging vs baseline atual (com tolerancia configuravel)
make promote-dry-run          # sem alterar Production
#   - Verifica se Staging esta dentro de MLFLOW_REGISTRY_STAGING_TOLERANCE (default 0.05)

# 4. Promover para Production (arquiva versões anteriores)
make promote
```

Tolerâncias e nomes são configurados no `.env`:

- `MLFLOW_MODEL_NAME` — nome do modelo no Registry (default `TechChallengeFase2Recommender`).
- `MLFLOW_REGISTRY_STAGING_TOLERANCE` — desvio maximo aceitavel vs `model_comparison.csv`.

## Inferência

A inferência carrega a versão `Production` do Model Registry (sem path local):

```bash
# Interativo (prompta por user_id):
make inference

# Programático:
uv run python -m techchallenge_fase2.inference.load_model recommend --user-id 123
uv run python -m techchallenge_fase2.inference.load_model list-versions
```

## Docker

### Imagem da aplicacao (multi-stage, CPU/GPU)

```bash
make docker-build            # target cpu (default)
make docker-build-gpu        # target gpu (CUDA)
```

### Stack MLflow (Postgres + MinIO + mlflow-server)

```bash
make mlflow-up               # sobe mlflow-server em http://localhost:5000
make mlflow-down             # para os containers
```

Saúde do MLflow: <http://localhost:5000/health>.

## Testes e lint

```bash
make test                    # uv run pytest
make lint                    # uv run ruff check .
make format                  # uv run ruff format .
```

## Documentação complementar

- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — Model Card (framework Mitchell et al. 2019).
- [`docs/ML_CANVAS.pdf`](docs/ML_CANVAS.pdf) — ML Canvas do projeto.
- [`docs/tech_challenge_fase02.md`](docs/tech_challenge_fase02.md) — enunciado oficial do desafio.
- `reports/model_comparison_report.md` — relatório comparativo gerado por `make compare-models`.

## Dataset

**RetailRocket E-Commerce** ([Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)):

| Arquivo                       | Linhas / Tamanho    | Conteudo                              |
|-------------------------------|---------------------|---------------------------------------|
| `events.csv`                  | 2.756.101 linhas (~94 MB) | `timestamp, visitorid, event, itemid`  |
| `item_properties_part1.csv`   | 4.552.608 (~484 MB) | `itemid, property, value, timestamp`  |
| `item_properties_part2.csv`   | 3.849.749 (~408 MB) | `itemid, property, value, timestamp`  |
| `category_tree.csv`           | 332 (~14 KB)        | `categoryid, parentid`                |

Eventos: `view` (96.7%), `addtocart` (2.5%), `transaction` (0.8%).
Janela temporal: ~2 meses (jun/jul 2015).

Os CSVs são versionados via DVC (`.dvc` em `data/raw.dvc`) e NÃO commitados no
Git (ver `.gitignore`). Use `make data` para baixar, e `make dvc-push` /
`make dvc-pull` para sincronizar com o remote OneDrive compartilhado.

## Creditos

**G13-MLE** — Grupo 13 (PÓS TECH FIAP) - Tech Challenge Fase 02:

- Eduardo Nunes Pereira
- Fernando (autor local deste repositorio)
- Ygor Martinelli

## Licença

Uso acadêmico restrito aos participantes do Tech Challenge FIAP.