# Tech Challenge FIAP - Fase 2

Sistema de recomendacao de produtos para e-commerce.

## EDA RetailRocket

A primeira entrega explora manualmente o dataset RetailRocket Ecommerce para
entender volume, qualidade, comportamento dos usuarios e caminhos provaveis para
o sistema de recomendacao.

Dataset oficial:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Arquivos esperados:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

Por padrao, o notebook procura os CSVs em `data/raw/retailrocket/`. Se os dados
estiverem em outro lugar, informe o caminho pela variavel `RETAILROCKET_DATA_DIR`.

PowerShell:

```powershell
$env:RETAILROCKET_DATA_DIR="C:\caminho\para\retailrocket"
uv sync
uv run jupyter notebook notebooks/01_retailrocket_eda.ipynb
```

Bash:

```bash
export RETAILROCKET_DATA_DIR="/caminho/para/retailrocket"
uv sync
uv run jupyter notebook notebooks/01_retailrocket_eda.ipynb
```

Os CSVs brutos nao devem ser commitados no Git. Eles serao versionados em etapa
posterior com DVC.
