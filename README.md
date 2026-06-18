# Tech Challenge Fiap - Fase 2

WIP

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
