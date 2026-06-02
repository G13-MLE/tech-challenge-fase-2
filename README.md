# Tech Challenge FIAP - Fase 2

Sistema de recomendacao de produtos para e-commerce, com foco em PyTorch,
DVC, MLflow, Docker e boas praticas de engenharia.

## Setup

```powershell
uv --cache-dir .uv-cache sync
uv --cache-dir .uv-cache run pre-commit install
```

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
