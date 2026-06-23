# Tech Challenge FIAP - Fase 2

Sistema de recomendação de produtos para e-commerce.

## EDA RetailRocket

A primeira entrega explora manualmente o dataset RetailRocket Ecommerce para
entender volume, qualidade, comportamento dos usuários e caminhos prováveis para
o sistema de recomendação.

Dataset oficial:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Arquivos esperados:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

Por padrão, o notebook procura os CSVs em `data/raw/retailrocket/`. Se os dados
estiverem em outro lugar, informe o caminho pela variável `RETAILROCKET_DATA_DIR`.

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

Notebooks da análise:

- `notebooks/01_retailrocket_eda.ipynb`: EDA principal, sinais de eventos,
  perfis comportamentais e cenários de recomendação.
- `notebooks/02_retailrocket_field_value_analysis.ipynb`: leitura da coluna
  `value` em `item_properties` e classificação dos tipos de propriedades.
- `notebooks/03_retailrocket_feature_flow_analysis.ipynb`: fluxo entre tabelas,
  plano de join temporal e schema candidato para a tabela de treino.

Os CSVs brutos não devem ser commitados no Git. Eles serão versionados em etapa
posterior com DVC. Em Docker, esse mesmo diretório deve ser disponibilizado ao
container por volume, `dvc pull` ou etapa do pipeline.
