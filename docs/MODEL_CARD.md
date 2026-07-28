# Model Card - Tech Challenge Fase 2

**Framework:** *Model Cards for Model Reporting* (Mitchell et al., ACM FAccT 2019)
**Modelo campeao:** EASE^ (Embarrassingly Shallow Auto-Encoders) - campeao do `make compare-models` (H-Mean@10=0.0325)
**Modelo neural:** Neural Collaborative Filtering (NCF) - GMF + MLP em PyTorch (treinado via DVC, H-Mean@10=0.0087)
**Status:** Catalogo filtrado (10k itens, usuarios>=5 interacoes); 6 modelos avaliados via `make compare-models`; EASE^ campeao; `make register && make promote` pendentes (ver Secao 4.1)
**Ultima atualizacao:** 27/07/2026 (apos `make pipeline-live` + `make compare-models` no dataset RetailRocket filtrado)

---

## 1. Model Details

| Campo              | Valor                                                              |
|--------------------|--------------------------------------------------------------------|
| Nome               | TechChallengeFase2Recommender                                      |
| Tipo               | Neural Collaborative Filtering (`neural_ncf`)                      |
| Framework          | PyTorch (CPU/GPU)                                                  |
| Arquitetura        | GMF + MLP fundidos (He et al., 2017), losses via BCEWithLogitsLoss |
| Otimizador         | Adam (lr=0.001)                                                    |
| Embedding dim      | 64                                                                 |
| Batch size         | 1024                                                               |
| Epocas             | 20 max (early stopping em ep 8; best_val_auc=0.7250 em ep 3)        |
| Negative samples   | 4 por usuario (implicit feedback via rejeicao amostral O(1))       |
| Seed               | 42                                                                 |
| Versionamento      | DVC (dataset/params) + MLflow Model Registry (Staging -> Production)|
| Quem treinou       | G13-MLE Grupo 13                                                   |

A arquitetura combina duas torres de embeddings:

1. **GMF (Generalized Matrix Factorization):** produto Hadamard de embeddings lineares de usuario e item.
2. **MLP (Multi-Layer Perceptron):** concatenacao de embeddings seguida de camadas densas.
3. **Fusao:** saidas do GMF e do MLP sao concatenadas e passadas por uma camada linear final.

O modelo final e serializado como `models/torch_embedding_recommender.pt` (weights-only checkpoint) e empacotado como `pyfunc` do MLflow para inferencia em producao via `models:/TechChallengeFase2Recommender/Production`.

## 2. Intended Use

**Uso primario:** Recomendar produtos para usuarios de e-commerce com base em
padroes de interacao historicos (view, addtocart, transaction).

**Usuarios do modelo:**

- Equipes de produto e data science responsaveis por personalizacao.
- Sistemas automatizados de recomendacao online (ranking top-K).

**Fora de escopo (out of scope):**

- Recomendacao personalizada em cold-start puro sem nenhuma interacao historica
  (o modelo nao consegue personalizar para usuarios/itens nao vistos no treino).
- Aplicacao direta em outros dominios (ex.: streaming, noticias) sem revalidacao.
- Unica base para decisoes criticas de negocio (precisa ser avaliado em A/B test).

## 3. Factors (Fatores de Avaliacao)

| Fator                  | Descricao                                                                                      | Impacto                                                              |
|------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `popularity_bias`      | Modelos tendem a recomendar itens populares, perpetuando o viés e limitando descoberta de cauda longa. | Itens populares dominam; itens nicho raramente aparecem mesmo quando relevantes. |
| `cold_start`           | Usuarios e itens novos sem interacoes historicas nao recebem/geram recomendacoes personalizadas. | Baselines tratam todos os usuarios de forma idêntica; o NCF tambem nao personaliza fora do catalogo treino. |
| `data_sparsity`        | A maioria dos usuarios interage com poucos itens; matriz de interacoes muito esparsa.         | Recall tende a ser baixo quando o conjunto de itens relevantes e grande. |
| `temporal_dynamics`    | Preferencias e popularidade variam no tempo.                                                   | Modelos treinados em dados historicos podem nao capturar tendencias recentes ( RetailRocket cobre 2 meses). |

## 4. Metrics

Métricas top-K avaliadas em `data/features/test.parquet` (15% final, cronologico):

| Metrica        | Formula / Definicao                                                | Justificativa                                                     |
|----------------|--------------------------------------------------------------------|-------------------------------------------------------------------|
| Precision@10   | `|relevantes ∩ recomendados| / |recomendados|`                    | Acerto na lista curta.                                            |
| Recall@10      | `|relevantes ∩ recomendados| / |relevantes|`                      | Cobertura dos itens realmente relevantes.                         |
| NDCG@10        | DCG normalizado com ganho posicional.                              | Premia ranking correto (relevantes no topo).                      |
| MAP@10         | Mean Average Precision.                                            | Media sobre posicoes precisas.                                     |
| Hit Rate@10    | `1 se |relevantes ∩ recomendados| > 0 else 0`.                     | Indicador binario de acerto (engajamento minimo).                 |
| `harmonic_mean_at_10` | Media harmonica das cinco metricas acima.                    | Criterio unico de campeao (balanceia as dimensoes).               |

Estas sao as cinco metricas canonicas de sistemas de recomendação top-K, mais a
media harmonica usada como criterio de desempate no `make compare-models`. Os
baselines scikit-learn (Popularity, RecentItems, Random, ItemKNN, Logistic
Regression) sao avaliados com o mesmo protocolo para comparacao justa.

### 4.1 Resultados quantitativos

Os numeros do **NCF** abaixo sao preenchidos apos `make pipeline` (DVC, dataset
RetailRocket completo). As linhas de **baselines** (Popularity, RecentItems,
Random, ItemKNN, LogisticRegression, EASE^) serao preenchidas apos
`make compare-models`. Veja `models/model_comparison.csv` e
`reports/model_comparison_report.md` para a tabela oficial atualizada a cada
promocao a Production.

| Modelo              | Precision@10 | Recall@10  | NDCG@10    | MAP@10   | HitRate@10 | harmonic@10 |
|---------------------|--------------|------------|------------|----------|------------|-------------|
| Popularity          | 0.0016       | 0.0064     | 0.0030     | 0.0012  | 0.0140     | 0.0021      |
| RecentItems         | 0.0006       | 0.0036     | 0.0017     | 0.0009  | 0.0060     | 0.0011      |
| Random              | 0.0001       | 0.0005     | 0.0002     | 0.0001  | 0.0010     | 0.0001      |
| ItemKNN             | 0.0097       | 0.0479     | 0.0297     | 0.0193  | 0.0780     | 0.0191      |
| LogisticRegression  | 0.0022       | 0.0088     | 0.0042     | 0.0017  | 0.0210     | 0.0029      |
| **EASE^ (campeao)** | **0.0166**   | **0.0737** | **0.0501** | **0.0341** | **0.1310** | **0.0325**   |
| NCF (DVC pipeline)  | 0.0027       | 0.0107     | 0.0069     | 0.0037  | 0.0250     | 0.0087      |

Catalogo filtrado a 10k itens mais populares + usuarios com >=5 interacoes
(decisao de design documentada: ver Secao 5). Avaliacao em 1000 usuarios
warm-start do conjunto teste cronologico, top_k=10.

NCF executado via `dvc repro -v` no catalogo completo:
- Treino: 7 epocas rodadas (early stopping acionado em patience=5 apos best em ep 2)
- `best_val_auc=0.7250` | `final_train_loss=0.0142`
- Avaliacao Top-K=10 em 1000 usuarios warm-start do conjunto de teste cronologico

**Campeao:** `ease_torch` (H-Mean@10=0.0325), com vantagem de +0.0134 sobre
o `item_knn` (H-Mean@10=0.0191). Os numeros acima foram coletados pelo
`make compare-models` no catalogo filtrado (10k itens, usuarios>=5
interacoes), 3 runs no experimento `tech-challenge-comparison` do MLflow.
Veja `models/model_comparison.csv` e `reports/model_comparison_report.md`.

**Proximos passos:** apos o merge deste PR, rodar `make register &&
make promote-dry-run && make promote` para declarar o `ease_torch` campeao
no MLflow Model Registry em Production. O NCF treinado pelo DVC pipeline ja
tem checkpoint em `models/torch_embedding_recommender.pt` e pode ser
registrado separadamente se desejado.

### 4.2 Usuarios avaliados

| Campo                | Valor                          |
|----------------------|--------------------------------|
| `max_users` (params) | 1000                           |
| Selecao              | Primeiros 1000 usuarios warm-start em `test.parquet` |
| Catalogo de itens    | Itens vistos no treino (~235.061) |

## 5. Evaluation Data

| Campo             | Valor                                                              |
|-------------------|--------------------------------------------------------------------|
| Dataset           | RetailRocket E-Commerce (`retailrocket/ecommerce-dataset` no Kaggle)|
| Schema            | `timestamp, visitorid, event, itemid` (events.csv)                 |
| Eventos           | `view` (1.0), `addtocart` (3.0), `transaction` (5.0) - event weights|
| Linhas brutas     | 2.756.101 interacoes                                               |
| Janela temporal   | ~2 meses (jun-jul/2015)                                            |
| Divisao           | Treino 70% / Validacao 15% / Teste 15% (split cronologico)         |
| Estrategia split  | `chronological_3way` por timestamp (sem vazamento temporal)        |
| Sessoes           | `session_gap_minutes=30` para identificacao de sessoes             |
| Encoding          | `visitorid` e `itemid` mapeados para inteiros estaveis            |

## 6. Training Data

| Campo                       | Valor                                       |
|-----------------------------|---------------------------------------------|
| Treino                      | 1.929.270 interacoes (70%) ~1.4M usuarios   |
| Validacao                   | 413.415 interacoes (15%)                    |
| Teste (holdout cronologico) | 413.416 interacoes (15%)                     |
| Usuarios unicos             | 1.407.580                                   |
| Itens unicos                | 235.061                                     |
| Esparsidade                 | ~1 - (2.756.101 / (1.4M * 235k)) ≈ 0.999992 |
| Negative sampling           | 1 negativo/usuario (rejeicao amostral O(1)) |
| Ambiente de treino          | CPU single-thread (configuravel para CUDA)  |
| Tracking                    | MLflow `tech-challenge-fase2-ncf`           |

## 7. Ethical Considerations

### 7.1 Vieses identificados

| Vies              | Descricao                                                                   | Mitigacao                                                                              |
|-------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Popularidade      | Recomendar itens populares amplifica sua visibilidade, criando feedback que marginaliza itens menos populares. | Combinar NCF com modelos baseados em conteudo; diversificar com MMR. |
| Filter bubble     | Recomendar so com base no passado limita exposicao a novidades.            | Exploracao epsilon-greedy; diversificacao forçada; incluir surface cold-start.          |
| Exclusao por cold-start | Usuarios novos nao recebem recomendacao personalizada.                | Backfill com Popularidade/RecentItems como fallback; nao bloquear onboarding.          |
| Confiança em heuristicas de implicit feedback | `view=1, addtocart=3, transaction=5` são pesos arbitrarios; podem sobrerrepresentar compra. | Sensitivity analysis dos pesos; considerar tempo de permanencia e sequencia. |

### 7.2 Mitigacoes gerais

- Monitorar diversidade (cobertura de catalogo) e nao so acuracia.
- Avaliar metricas por segmento (usuario novo vs. recorrente, popular vs. nicho).
- Combinar multiplas estrategias de recomendacao.
- Auditar periodicamente vieses nos resultados.
- Documentar e versionar experimentos (MLflow + DVC).

## 8. Caveats and Recommendations

### 8.1 Limitacoes

- O modelo nao personaliza para usuarios ou itens fora do catalogo de treino (cold-start).
- A qualidade depende da completude das interacoes; itens sem eventos nao sao recomendaveis.
- Metricas offline (precision/recall/NDCG/MAP/hit_rate) podem nao refletir satisfacao real do usuario.
- O RetailRocket cobre apenas ~2 meses; tendencias sazonais nao sao capturadas.
- O batch de avaliacao faz forward do catalogo inteiro por usuario (enviesado para usuarios de catalogo menor em producao).

### 8.2 Recomendacoes

- Tratar o NCF aqui como baseline neural; considerar comparacao com EASE^ (ja no pipeline) e modelos baseados em conteudo.
- Implementar A/B testing antes de deploy em producao (engajamento, CTR, conversao).
- Re-treinar periodicamente (window deslizante) para capturar dinamica temporal.
- Combinar com informações contextuais (horario, dispositivo) quando disponivel.
- Monitorar drift de distribuicao em producao (ex.: KS testes nos embeddings).

## 9. Cenarios de Falha

| Categoria | Cenario                                              | Sinais                                          | Acao                                                              |
|-----------|------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------|
| Dados     | Queda na coleta de eventos; alta esparsidade         | Recall@10 cai; cobertura do catalogo caiu > 30% | Investigar pipeline de ingestao; reavaliar treino com novas split. |
| Modelo    | Drift de preferencias sem retreino                   | harmonic_mean@10 caiu > tolerancia no `promote-dry-run` | Retreinar com janela mais recente; arquivar versao Production.   |
| Negocio   | Usuario ou item novo (cold-start) recebe top popular  | Reclamacoes de personalizacao; metricas de negocio distorcem | Ativar fallback Popularidade/RecentItems ate NCF aprender.       |
| Infra     | MLflow server fora do ar                             | `promote`/`inference` falha                     | Subir `make mlflow-up`;_health redisponivel em `http://localhost:5000`. |
| Infra     | Catalogo cresce além do embedding size treinado       | Treino com num_items diferente do checkpoint    | Recriar mapeamento em `data/features/mappings.json` e retreinar.  |

## 10. Como usar o modelo

```bash
# inference interativa via CLI:
make inference                # prompta por user_id e retorna top-K

# inference programatica via pyfunc do MLflow:
uv run python -m techchallenge_fase2.inference.load_model recommend --user-id 123
```

## 11. Como NAO usar o modelo

- **Nao use** diretamente para usuarios sem interacoes no periodo de treino (cold-start puro).
- **Nao use** como unica fonte de decisao em promocoes de marketing de alto valor sem A/B test.
- **Nao use** em outro dominio sem retreino; o embedding e o mapeamento de IDs
  sao exclusivos do catalogo do RetailRocket.
- **Nao use** a versao Staging em producao; use sempre `models:/.../Production`.

## 12. Referencias

- He, Liao, Zhang, Nie, Hu, Chua. *Neural Collaborative Filtering* (WWW 2017).
- Mitchell et al. *Model Cards for Model Reporting* (ACM FAccT 2019).
- Steck et al. *EASE^: Embarrassingly Shallow Auto-Encoders for Sparse Data* (ICDM WSDM 2019).

## Apêndice A - Artefatos relacionados

| Artefato                              | Local                                          |
|---------------------------------------|------------------------------------------------|
| Checkpoint PyTorch                    | `models/torch_embedding_recommender.pt`        |
| Comparativo oficial                   | `models/model_comparison.csv`                  |
| Relatorio markdown de comparacao      | `reports/model_comparison_report.md`           |
| Historico de treino                   | `models/training_history.json`                 |
| Metricas Top-K (DVC metrics)          | `metrics/recommendation_metrics.json`          |
| Model Card (JSON, logado no MLflow)   | `model_card.json` no artifact da run            |
| MLflow Registry Production            | `models:/TechChallengeFase2Recommender/Production`|
| ML Canvas                             | `docs/ML_CANVAS.pdf`                           |