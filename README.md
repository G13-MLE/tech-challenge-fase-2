# Tech Challenge FIAP - Fase 2

Sistema de recomendacao de produtos para e-commerce, com foco em PyTorch,
DVC, MLflow, Docker e boas praticas de engenharia.

## Setup

```powershell
$env:UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
$env:PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
uv --cache-dir .uv-cache sync
git config core.hooksPath .githooks
```

Os caches do projeto devem ficar no proprio workspace em `D:\POS\tech-challenge-fase-2`:

- `.uv-cache/`: cache de pacotes do uv.
- `.uv-python/`: instalacoes Python gerenciadas pelo uv.
- `.pre-commit-cache/`: repositorios/cache dos hooks de pre-commit.

O hook versionado em `.githooks/pre-commit` exporta essas variaveis antes de
rodar o `pre-commit`, evitando cache no perfil do usuario em `C:\`.

Configuracao opcional via `.env`:

```powershell
RETAILROCKET_DATA_DIR=D:/Dataset/archive
EDA_OUTPUT_DIR=reports/eda
REQUIRED_INTERACTIONS=10000
```

Comandos principais:

```powershell
make run-eda
make test
make lint
```

## EDA RetailRocket

A primeira entrega analisa o dataset RetailRocket Ecommerce sem versionar os
CSVs brutos no Git.

Dataset oficial:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Arquivos esperados:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

Execute a EDA:

```powershell
uv --cache-dir .uv-cache run python scripts/run_retailrocket_eda.py --dataset-dir "D:/Dataset/archive"
```

Artefatos gerados:

- `reports/eda/retailrocket_eda.md`
- `reports/eda/retailrocket_metrics.json`

Notebook de visualizacao:

- `notebooks/01_retailrocket_eda.ipynb`

O notebook consome `reports/eda/retailrocket_metrics.json`. Ele serve para
exploracao visual; a geracao dos artefatos finais deve continuar pelo script.

Abra pelo VS Code/Jupyter ou rode:

```powershell
uv --cache-dir .uv-cache run jupyter notebook notebooks/01_retailrocket_eda.ipynb
```

Os dados brutos devem ser rastreados futuramente pelo DVC, nao commitados
diretamente no Git.
