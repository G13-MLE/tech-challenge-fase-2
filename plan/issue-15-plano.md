# Plano - Issue #15: Comparar com baselines Scikit-Learn usando no minimo 4 metricas

## Objetivo

Avaliar EASE^ (Embarrassingly Shallow Autoencoder) implementado em PyTorch como candidato a modelo principal (campeao) do projeto, comparando-o com baselines Scikit-Learn e modelos neurais existentes (NCF/MLP) usando no minimo 4 metricas de recomendacao. EASE^ sera declarado campeao se superar empiricamente todos os outros modelos; caso contrario, o vencedor do benchmark sera o principal.

## Referencias

- Paper original: Steck, 2019 - "Embarrassingly Shallow Autoencoders for Sparse Data" (arxiv.org/abs/1905.03375)
- Implementacao PyTorch de referencia: github.com/franckjay/TorchEASE
- Tutorial: stepbystepdatascience.com/ease
- Implementacao exploratoria atual: scripts/eval_ease_only.py (branch eda-test, PR #38)

## Decisoes Tomadas

1. **EASE^ como campeao condicional**: entra como candidato, nao como baseline. Vence se superar todos os outros modelos nas metricas; caso contrario, o modelo vencedor sera o principal.
2. **Manter MLP/NCF existente** (PR #37) como baseline neural forte na comparacao.
3. **CPU para inversao de matriz** (nao MPS). Micro-benchmark em macOS/Apple Silicon mostrou que MPS e 14x mais lento que CPU para 2000x2000 e equivalente para 5000x5000. Backend MPS nao esta otimizado para `torch.linalg.inv`. Usar CPU (LAPACK) para a inversao. MPS pode ser usado para matmul de predicoes se beneficio futuro for validado.
4. **Implementacao do zero** em src/ seguindo as regras de AGENTS.md (SOLID, funcoes <=20 linhas, type hints, docstrings Google, Factory pattern, codigo em ingles, comentarios em portugues, sem emojis).
5. **MAP@K** adicionada ao pipeline (ja existe em training/metrics.py, basta usar).

## Criterio de Vitoria

EASE^ sera declarado campeao se, em K=10, a **media harmonica** de Precision@10, Recall@10, NDCG@10 e MAP@10 for a mais alta entre todos os modelos, com vantagem relativa >=1% sobre o segundo colocado. Em caso de empate tecnico (<1%), EASE^ vence por simplicidade e velocidade de treino (principio Occam).

Metricas canonicas: Precision@K, Recall@K, NDCG@K, MAP@K (4 obrigarias). HitRate@K como bonus. K de referencia: 10. Ks complementares: 5, 20.

## Divisao de Dados

Migrar de `temporal_holdout_split` (shuffle por usuario) para **split cronologico global 70/15/15** (treino/validacao/teste), conforme ja implementado em `scripts/eval_ease_only.py`. Justificativa: recomendaçao e intrinsecamente temporal; split cronologico evita leakage e reproduz o cenario real de previsao do futuro. Todos os modelos usarao o mesmo split para comparacao justa.

Limite de itens por popularidade: `max_items=20000` (mantem compatibilidade com a prova de conceito do `eval_ease_only.py` e viabiliza inversao de matriz em memoria).

## Arquitetura

### Etapa A - EASE^ PyTorch

**Arquivo**: `src/techchallenge_fase2/models/ease_torch.py`

Conteudo:
- `EASEConfig` (dataclass frozen): `lambda_reg: float = 250.0`, `max_items: int = 20000`, `batch_size: int = 1000`, `popularity_blending: float = 0.0`, `device: str = "auto"`.
- `EASETorchRecommender(RecommenderModel)`: implementacao do contrato ABC existente.
- Auto-device: detecta MPS, fallback CPU. Para matrizes G > 20k itens, valida empiricamente se MPS traz ganho; senao, força CPU.
- Mapeamentos str↔int internos (compatibiliza contrato `tuple[str, str]` com matrizes numericas).
- `fit`: matriz esparsa scipy → G = X^T·X (CPU) → move G para device → `torch.linalg.inv` → `B = I - P / diag(P)` → `fill_diagonal(B, 0)` → retorna B para CPU.
- `recommend`: scoring em batches (`X_eval @ B`), exclui itens ja vistos, cold-start fallback para MostPopular, popularity blending opcional.
- Funcoes <=20 linhas; quebrar `fit` em `_build_gram`, `_invert`, `_build_b`.

### Etapa B - Baselines Scikit-Learn

**Arquivo**: `src/techchallenge_fase2/models/sklearn_baselines.py`

Conteudo:
- `ItemKNNRecommender(RecommenderModel)`: `sklearn.neighbors.NearestNeighbors` sobre vetor de itens (similaridade cosseno). Gera ranking por agregacao de vizinhos.
- `LogisticRegressionRecommender(RecommenderModel)`: formulacao binaria user×item com sampling de negativos. Para cada user, monta exemplos (positivos + negativos amostrados) e treina `LogisticRegression` para preder score de cada item candidato.

Nao implementar RandomForest nesta etapa (custo computacional alto, ganho marginal esperado). ItemKNN e LogisticRegression cobrem os dois paradigmas sklearn relevantes (vizinhanca e linear supervisionado).

### Etapa C - Integracao no Factory

**Arquivos**: `src/techchallenge_fase2/models/config.py`, `src/techchallenge_fase2/models/factory.py`

- Adicionar ao `ModelType` enum: `EASE_TORCH = "ease_torch"`, `ITEM_KNN = "item_knn"`, `LOGISTIC_REGRESSION = "logistic_regression"`.
- Estender `ModelConfig` com campos: `lambda_reg: float = 250.0`, `max_items: int = 20000`, `batch_size: int = 1000`, `popularity_blending: float = 0.0`.
- Registrar creators em `RecommenderModelFactory.default()`: `create_ease_torch_model`, `create_item_knn_model`, `create_logistic_regression_model`.

### Etapa D - Pipeline Unificado

**Arquivo**: `src/techchallenge_fase2/pipelines/run_baselines.py` (refatorar existente)

Mudancas:
- Substituir `temporal_holdout_split` por `chronological_split` 70/15/15 (extrair de `scripts/eval_ease_only.py` para uma funcao reutilizavel em `pipelines/common.py` ou novo `pipelines/splits.py`).
- Adicionar EASE^, ItemKNN, LogisticRegression ao loop de avaliacao.
- Tags MLflow: `model_role` ∈ {`baseline`, `baseline_neural`, `champion_candidate`}.
- CSV comparativo: adicionar coluna `model_role` e ordenar por media harmonica das 4 metricas @K=10.
- Manter MLflow tracking, plots, model card ja existentes.
- Logar tempo de treino e de inferencia por modelo (importante para desempate por simplicidade).

### Etapa E - Testes

**Arquivos**:
- `tests/test_ease_torch.py`: sanity check - matriz pequena, B tem diagonal zero, predicoes nao vazias, exclude_seen_items funciona, cold-start fallback.
- `tests/test_sklearn_baselines.py`: sanity check ItemKNN e LogisticRegression.
- Atualizar `tests/test_model_factory.py`: cobrir novos tipos registrados.
- Atualizar `tests/test_recommendation_metrics.py`: validar MAP@K no pipeline integrado.

## Checklist de Entrega (Expectativas da Issue)

| Expectativa | Artefato |
|---|---|
| Implementar baselines (popularidade, regressao logistica, random forest) | ItemKNN + LogisticRegression em sklearn_baselines.py (RF omitido por custo/beneficio) |
| Avaliar todos os modelos com as mesmas metricas (>= 4) | Precision@K, Recall@K, NDCG@K, MAP@K em run_baselines.py |
| Salvar resultados comparativos | CSV em models/baseline_comparison.csv + MLflow |
| Script src/pipelines/run_baselines.py funcional | Refatoracao do existente |
| Resultados salvos em CSV e/ou no MLflow | Ja coberto pela integracao existente |
| Comparacao justa: mesmos dados de treino/teste | Split cronologico unico para todos os modelos |
| Analise comparativa documentada | Model card por modelo + grafico comparativo |

## Ordem de Execucao

1. Validar empiricamente ganho do MPS para `torch.inverse()` com ~20k itens (micro-benchmark, 15 min). **CONCLUIDO**: MPS 14x mais lento que CPU para 2000x2000; CPU sera usada para inversao.
2. Etapa A: EASE^ PyTorch + testes unitarios.
3. Etapa C: Integracao no Factory.
4. Etapa B: Baselines sklearn + testes unitarios.
5. Etapa D: Refatorar run_baselines.py com split cronologico + todos os modelos.
6. Etapa E: Testes integrados.
7. `make lint && make test`.
8. Executar pipeline completa e coletar resultados.
9. Declarar campeao com base no criterio de vitoria.

## Riscos e Mitigacoes

- **MPS lento para `torch.inverse`**: mitigar com micro-benchmark previo e fallback CPU automatico.
- **Estouro de memoria** (matriz G 20k×20k = 3.2GB float64): manter max_items=20000 e usar float32 quando seguro.
- **LogisticRegression com catalogo grande**: sampling de negativos por usuario (ex: 10 negativos por positivo) para viabilizar treino.
- **Comparacao injusta por split divergente**: usar funcao unica de split para todos os modelos.
- **EASE^ nao bater NCF**: plano contempla essa contingencia - o vencedor do benchmark sera o principal, sem retrabalho de arquitetura.

## Fora de Escopo (nesta issue)

- Tuning fino de hiperparametros via Optuna (ja explorado em eval_ease_only.py; sera ativado se necessario para fazer EASE^ vencer).
- RandomForest (omitido por custo/beneficio; pode ser adicionado em follow-up).
- Visualizacao UMAP de itens (mencionado no tutorial, fora do escopo da issue).
- Model Registry / Staging (issue #16 separada).
