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

## Versionamento de Dados (DVC + OneDrive)

O dataset RetailRocket (~1.4 GB) é versionado com DVC e armazenado em uma
pasta compartilhada do OneDrive (remote "local"). Os arquivos `.dvc` são
commitados no Git como metadado de versão; o conteúdo binário fica no cache
do DVC sincronizado pelo cliente OneDrive.

### Setup

1. Criar (uma vez, pela equipe) uma pasta compartilhada no OneDrive, ex.:
   `techchallenge-fase2/dvcstore`. Todos os membros devem sincronizá-la
   localmente via cliente OneDrive.

2. Copiar `.env.example` para `.env` e preencher:
   - `DVC_ONEDRIVE_REMOTE_URL`: caminho local da pasta sincronizada
     (ex.: `/Users/<usuario>/OneDrive - FIAP/techchallenge-fase2/dvcstore`).
   - `KAGGLE_USERNAME` e `KAGGLE_KEY`: token criado em
     https://www.kaggle.com/settings -> API -> Create New Token.

3. Executar o setup (instala deps, pre-commit e configura o remote DVC):
   ```bash
   make setup
   ```

### Obter o dataset

```bash
make data        # baixa do Kaggle para data/raw/ via API
dvc add data/raw # versiona com DVC (gera data/raw.dvc + data/raw/.gitignore)
git add data/raw.dvc data/raw/.gitignore
```

### Compartilhar / restaurar via OneDrive

```bash
make dvc-push    # envia o cache do DVC para a pasta do OneDrive
make dvc-pull    # baixa o cache do OneDrive e restaura data/raw/
make dvc-status  # verifica o estado do versionamento
```
