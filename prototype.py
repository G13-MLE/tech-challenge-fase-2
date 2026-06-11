#!/usr/bin/env python3
"""
Protótipo de sistema de recomendação para o dataset Retailrocket.

Estratégias implementadas:
1. Popularidade (Best Sellers) - itens mais comprados com decaimento temporal
2. Co-ocorrência (Viewed-then-Bought) - itens comprados após visualização
3. Filtragem Colaborativa Item-Item - similaridade entre itens via co-interação
4. Baseado em Categoria - itens populares da mesma categoria

Modelo neural:
- Embeddings de usuário e item + MLP para filtragem colaborativa
- Feedback implícito: view=1, addtocart=2, transaction=3

Métricas de avaliação: Hit Rate@K, NDCG@K, Precision@K, Coverage

Para obter os dados:
1. Instale o kaggle CLI: pip install kaggle
2. Configure suas credenciais em ~/.kaggle/kaggle.json
3. Execute: kaggle datasets download -d retailrocket/ecommerce-dataset -p tmp/ --unzip
   Ou baixe manualmente de: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
   e extraia os CSVs no diretório tmp/ do projeto
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# --- Reprodutibilidade ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).parent / "tmp"
K_VALUES = [5, 10, 20]
EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}
SAMPLE_SIZE = 100_000  # Amostra para o modelo neural


# =============================================================================
# Carregamento e preparação dos dados
# =============================================================================

def load_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Carrega os arquivos CSV do dataset Retailrocket."""
    events = pd.read_csv(data_dir / "events.csv")
    # Colunas do dataset: timestamp, visitorid, event, itemid, transactionid

    # Carrega propriedades dos itens (2 partes)
    parts = []
    for f in sorted(data_dir.glob("item_properties_part*.csv")):
        parts.append(pd.read_csv(f))
    items = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Árvore de categorias
    cat_tree = pd.read_csv(data_dir / "category_tree.csv")

    return {"events": events, "items": items, "categories": cat_tree}


def prepare_data(events: pd.DataFrame) -> pd.DataFrame:
    """Prepara o dataframe de eventos com pesos de interação."""
    df = events.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.dropna(subset=["itemid"])  # Remove interações sem item
    df["itemid"] = df["itemid"].astype(int)
    df["visitorid"] = df["visitorid"].astype(int)
    return df


def train_test_split_by_time(
    df: pd.DataFrame, min_interactions: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide treino/teste por tempo: última interação de cada usuário é teste."""
    user_counts = df.groupby("visitorid").size()
    eligible = user_counts[user_counts >= min_interactions].index
    df_eligible = df[df["visitorid"].isin(eligible)]

    # Última interação de cada usuário vai para teste
    idx_last = df_eligible.groupby("visitorid")["timestamp"].idxmax()
    test = df_eligible.loc[idx_last]
    train = df_eligible.drop(idx_last)
    return train, test


# =============================================================================
# Estratégia 1: Popularidade (Best Sellers) com decaimento temporal
# =============================================================================

def recommend_popularity(
    train: pd.DataFrame, k: int = 10, decay_days: float = 30.0
) -> list[int]:
    """Recomenda os itens mais comprados com decaimento temporal.

    Score = contagem * exp(-lambda * dias_desde_última_compra)
    """
    transactions = train[train["event"] == "transaction"]
    if transactions.empty:
        return []

    max_time = transactions["timestamp"].max()
    item_stats = transactions.groupby("itemid").agg(
        count=("itemid", "size"),
        last_time=("timestamp", "max"),
    )
    days_since = (max_time - item_stats["last_time"]).dt.total_seconds() / 86400
    decay = np.exp(-days_since / decay_days)
    item_stats["score"] = item_stats["count"] * decay

    return item_stats.nlargest(k, "score").index.tolist()


# =============================================================================
# Estratégia 2: Co-ocorrência (Viewed-then-Bought)
# =============================================================================

def build_viewed_bought_matrix(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    """Constrói matriz de co-ocorrência: itens visualizados antes de uma compra."""
    # Para cada transação, encontra itens visualizados pelo mesmo usuário antes
    co_occurrence: dict[int, dict[int, float]] = {}

    transactions = train[train["event"] == "transaction"]
    views = train[train["event"] == "view"]

    for _, txn in transactions.iterrows():
        user_views = views[
            (views["visitorid"] == txn["visitorid"])
            & (views["timestamp"] < txn["timestamp"])
        ]
        for _, vw in user_views.iterrows():
            viewed_item = int(vw["itemid"])
            bought_item = int(txn["itemid"])
            if viewed_item not in co_occurrence:
                co_occurrence[viewed_item] = {}
            co_occurrence[viewed_item][bought_item] = (
                co_occurrence[viewed_item].get(bought_item, 0) + 1
            )

    return co_occurrence


def recommend_co_occurrence(
    item_id: int, co_matrix: dict[int, dict[int, float]], k: int = 10
) -> list[int]:
    """Para um item visualizado, recomenda itens comprados junto."""
    if item_id not in co_matrix:
        return []
    sorted_items = sorted(
        co_matrix[item_id].items(), key=lambda x: x[1], reverse=True
    )
    return [item for item, _ in sorted_items[:k]]


# =============================================================================
# Estratégia 3: Filtragem Colaborativa Item-Item
# =============================================================================

def build_item_similarity_matrix(
    train: pd.DataFrame, top_k: int = 100, max_items: int = 5000
) -> dict[int, list[tuple[int, float]]]:
    """Constrói similaridade item-item via cosseno na matriz usuário-item."""
    # Amostra itens se necessário para caber na memória
    item_counts = train.groupby("itemid")["weight"].sum()
    top_items = item_counts.nlargest(min(max_items, len(item_counts))).index

    filtered = train[train["itemid"].isin(top_items)]
    user_item = filtered.pivot_table(
        index="visitorid", columns="itemid", values="weight", fill_value=0
    )

    if user_item.empty:
        return {}

    # Similaridade cosseno entre itens
    sim_matrix = cosine_similarity(user_item.T.values)
    item_ids = user_item.columns.tolist()

    # Para cada item, guarda os top_k mais similares
    item_neighbors: dict[int, list[tuple[int, float]]] = {}
    for i, item in enumerate(item_ids):
        sims = sim_matrix[i]
        top_indices = np.argsort(sims)[::-1][1 : top_k + 1]  # Exclui o próprio item
        item_neighbors[item] = [
            (item_ids[j], sims[j]) for j in top_indices if sims[j] > 0
        ]

    return item_neighbors


def recommend_item_cf(
    item_id: int, neighbors: dict[int, list[tuple[int, float]]], k: int = 10
) -> list[int]:
    """Recomenda itens similares baseado em filtragem colaborativa."""
    if item_id not in neighbors:
        return []
    return [item for item, _ in neighbors[item_id][:k]]


# =============================================================================
# Estratégia 4: Baseado em Categoria
# =============================================================================

def build_category_popularity(
    train: pd.DataFrame, items: pd.DataFrame, cat_tree: pd.DataFrame
) -> dict[int, list[int]]:
    """Constrói mapa de categoria -> itens mais populares."""
    # Extrai categoria de cada item (propriedade "categoryid")
    cat_props = items[items["property"] == "categoryid"][["itemid", "value"]].copy()
    cat_props["categoryid"] = cat_props["value"].astype(int)
    cat_props = cat_props.drop_duplicates(subset=["itemid"], keep="last")

    # Popularidade por item (soma de pesos)
    item_pop = train.groupby("itemid")["weight"].sum().reset_index()
    item_pop.columns = ["itemid", "popularity"]

    # Junta com categorias
    merged = cat_props.merge(item_pop, on="itemid", how="left")
    merged = merged[merged["popularity"].notna()]

    # Constrói mapa categoria -> itens ordenados por popularidade
    cat_map: dict[int, list[int]] = {}
    for cat_id, group in merged.groupby("categoryid"):
        sorted_items = group.nlargest(100, "popularity")["itemid"].tolist()
        cat_map[cat_id] = sorted_items

    return cat_map


def recommend_category(
    item_id: int,
    cat_pop: dict[int, list[int]],
    item_to_cat: dict[int, int],
    k: int = 10,
) -> list[int]:
    """Recomenda itens populares da mesma categoria do item dado."""
    cat_id = item_to_cat.get(item_id)
    if cat_id is None or cat_id not in cat_pop:
        return []
    # Exclui o próprio item
    return [i for i in cat_pop[cat_id] if i != item_id][:k]


# =============================================================================
# Modelo Neural: Co-ocorrência (Viewed → Bought)
# =============================================================================

class NeuralCoOccurrence(nn.Module):
    """Modelo neural item-to-item baseado em co-ocorrência.

    Dado um item visualizado, prevê qual item será comprado.
    Dois embeddings separados: um para o item de origem (viewed),
    outro para o item alvo (bought), concatenados num MLP.
    """

    def __init__(self, n_items: int, dim: int = 32):
        super().__init__()
        # Embedding do item visualizado (origem)
        self.source_emb = nn.Embedding(n_items, dim)
        # Embedding do item comprado (alvo)
        self.target_emb = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # Inicialização Xavier
        nn.init.xavier_uniform_(self.source_emb.weight)
        nn.init.xavier_uniform_(self.target_emb.weight)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        s = self.source_emb(source_ids)
        t = self.target_emb(target_ids)
        x = torch.cat([s, t], dim=-1)
        return self.mlp(x).squeeze(-1)


def extract_cooccurrence_pairs(train: pd.DataFrame) -> pd.DataFrame:
    """Extrai pares (item_visualizado, item_comprado) a partir dos eventos.

    Para cada transação, encontra todos os itens que o mesmo usuário
    visualizou antes da compra. Cada par (viewed, bought) é um exemplo positivo.
    """
    transactions = train[train["event"] == "transaction"][["visitorid", "itemid", "timestamp"]]
    views = train[train["event"] == "view"][["visitorid", "itemid", "timestamp"]]

    pairs = []
    for _, txn in transactions.iterrows():
        user_views = views[
            (views["visitorid"] == txn["visitorid"])
            & (views["timestamp"] < txn["timestamp"])
        ]
        for viewed_item in user_views["itemid"].values:
            pairs.append({"source": int(viewed_item), "target": int(txn["itemid"])})

    return pd.DataFrame(pairs)


def prepare_cooccurrence_data(
    pairs: pd.DataFrame, item_map: dict[int, int], n_neg: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepara tensores de treino com amostragem negativa.

    Para cada par positivo (viewed, bought), gera n_neg pares negativos
    trocando o item alvo por um item aleatório que não foi comprado.
    """
    sources, targets, labels = [], [], []
    n_items = len(item_map)

    for _, row in pairs.iterrows():
        src_idx = item_map.get(row["source"])
        tgt_idx = item_map.get(row["target"])
        if src_idx is None or tgt_idx is None:
            continue

        # Positivo
        sources.append(src_idx)
        targets.append(tgt_idx)
        labels.append(1.0)

        # Negativos: itens aleatórios como alvo (não comprados após a visualização)
        neg_indices = np.random.choice(n_items, size=n_neg, replace=False)
        for ni in neg_indices:
            if ni != tgt_idx:
                sources.append(src_idx)
                targets.append(ni)
                labels.append(0.0)

    return (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_cooccurrence_model(
    train: pd.DataFrame, epochs: int = 5, lr: float = 0.001, batch_size: int = 1024
) -> tuple[NeuralCoOccurrence | None, dict[int, int]]:
    """Treina o modelo neural de co-ocorrência.

    Constrói os pares viewed→bought, mapeia itens para índices,
    e treina o modelo com BCE e amostragem negativa.
    """
    print("  Extraindo pares viewed→bought...")
    pairs = extract_cooccurrence_pairs(train)
    print(f"  Pares positivos: {len(pairs):,}")

    if pairs.empty:
        print("  AVISO: nenhum par encontrado. Verifique os dados.")
        return None, {}

    # Mapeia todos os itens que aparecem nos pares
    all_items_set = set(pairs["source"].unique()) | set(pairs["target"].unique())
    item_map = {item: idx for idx, item in enumerate(sorted(all_items_set))}
    print(f"  Itens únicos nos pares: {len(item_map):,}")

    # Amostra se houver muitos pares (para velocidade)
    if len(pairs) > SAMPLE_SIZE:
        pairs = pairs.sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"  Amostra de pares: {len(pairs):,}")

    print("  Preparando dados com amostragem negativa...")
    data = prepare_cooccurrence_data(pairs, item_map, n_neg=4)

    model = NeuralCoOccurrence(n_items=len(item_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(data[0], data[1], data[2])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_src, batch_tgt, batch_labels in loader:
            optimizer.zero_grad()
            preds = model(batch_src, batch_tgt)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"  Época {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, item_map


def recommend_coocc_neural(
    source_item: int,
    model: NeuralCoOccurrence,
    item_map: dict[int, int],
    rev_item_map: dict[int, int],
    k: int = 10,
) -> list[int]:
    """Dado um item visualizado, recomenda itens com maior probabilidade de compra."""
    model.eval()
    if source_item not in item_map:
        return []

    src_idx = torch.tensor([item_map[source_item]], dtype=torch.long)
    # Pontua todos os itens candidatos como alvo
    candidate_indices = torch.arange(len(rev_item_map), dtype=torch.long)
    # Exclui o próprio item de origem
    mask = candidate_indices != item_map[source_item]
    candidate_indices = candidate_indices[mask]

    src_expanded = src_idx.expand(len(candidate_indices))

    with torch.no_grad():
        scores = model(src_expanded, candidate_indices)

    top_k_indices = torch.topk(scores, min(k, len(candidate_indices))).indices
    return [rev_item_map[idx.item()] for idx in top_k_indices]


# =============================================================================
# Métricas de avaliação
# =============================================================================

def hit_rate_at_k(recommendations: list[int], relevant: set[int], k: int) -> float:
    """Fração de usuários cujo item relevante está no top-K."""
    return 1.0 if relevant & set(recommendations[:k]) else 0.0


def ndcg_at_k(recommendations: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K: ganho descontado normalizado."""
    dcg = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 porque começa em 0
    # Ideal: item relevante na primeira posição
    idcg = 1.0  # Melhor caso: 1 item relevante no topo
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommendations: list[int], relevant: set[int], k: int) -> float:
    """Precisão@K: fração dos K recomendados que são relevantes."""
    if k == 0:
        return 0.0
    return len(relevant & set(recommendations[:k])) / k


def coverage(all_recs: list[list[int]], n_total_items: int) -> float:
    """Fração dos itens totais que foram recomendados ao menos uma vez."""
    recommended = set()
    for recs in all_recs:
        recommended.update(recs)
    return len(recommended) / n_total_items if n_total_items > 0 else 0.0


# =============================================================================
# Avaliação completa
# =============================================================================

def evaluate_strategy(
    strategy_name: str,
    recommend_fn,
    test_users: pd.DataFrame,
    train: pd.DataFrame,
    all_items: np.ndarray,
    **kwargs,
) -> dict[str, dict[int, float]]:
    """Avalia uma estratégia de recomendação com todas as métricas."""
    results: dict[str, dict[int, list[float]]] = {
        "hit_rate": {k: [] for k in K_VALUES},
        "ndcg": {k: [] for k in K_VALUES},
        "precision": {k: [] for k in K_VALUES},
    }
    all_recs: dict[int, list[list[int]]] = {k: [] for k in K_VALUES}

    # Amostra de usuários para avaliação (limita para velocidade)
    sample_users = test_users["visitorid"].unique()[:500]

    for user_id in sample_users:
        # Item relevante: último item com o qual o usuário interagiu (do teste)
        user_test = test_users[test_users["visitorid"] == user_id]
        relevant_items = set(user_test["itemid"].values)
        if not relevant_items:
            continue

        # Item de referência para estratégias baseadas em item
        ref_item = user_test["itemid"].iloc[0]

        # Gera recomendações
        recs = recommend_fn(user_id=user_id, item_id=ref_item, train=train, **kwargs)

        for k in K_VALUES:
            top_k = recs[:k]
            results["hit_rate"][k].append(hit_rate_at_k(top_k, relevant_items, k))
            results["ndcg"][k].append(ndcg_at_k(top_k, relevant_items, k))
            results["precision"][k].append(precision_at_k(top_k, relevant_items, k))
            all_recs[k].append(top_k)

    # Agrega métricas
    final: dict[str, dict[int, float]] = {}
    for metric in ["hit_rate", "ndcg", "precision"]:
        final[metric] = {k: np.mean(results[metric][k]) for k in K_VALUES}
    final["coverage"] = {
        k: coverage(all_recs[k], len(all_items)) for k in K_VALUES
    }

    print(f"\n--- {strategy_name} ---")
    for metric, values in final.items():
        for k, v in values.items():
            print(f"  {metric}@{k}: {v:.4f}")

    return final


# =============================================================================
# Visualização dos resultados
# =============================================================================

def plot_results(all_results: dict[str, dict[str, dict[int, float]]]):
    """Gera gráficos comparando as estratégias de recomendação."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["hit_rate", "ndcg", "precision", "coverage"]
    colors = sns.color_palette("husl", len(all_results))

    for ax, metric in zip(axes.flat, metrics):
        strategies = list(all_results.keys())
        x = np.arange(len(K_VALUES))
        width = 0.15
        for i, strategy in enumerate(strategies):
            values = [all_results[strategy][metric][k] for k in K_VALUES]
            ax.bar(x + i * width, values, width, label=strategy, color=colors[i])
        ax.set_xlabel("K")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}@K")
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.legend(fontsize=8)

    plt.suptitle("Comparação de Estratégias de Recomendação - Retailrocket", fontsize=14)
    plt.tight_layout()
    plt.savefig("recommendation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGráfico salvo em: recommendation_results.png")


# =============================================================================
# Função principal
# =============================================================================

def main():
    """Executa o pipeline completo de recomendação."""
    print("=" * 60)
    print("Sistema de Recomendação - Retailrocket Dataset")
    print("=" * 60)

    # --- Carregamento dos dados ---
    print("\n[1/6] Carregando dados...")
    if not DATA_DIR.exists():
        print(
            f"Diretório '{DATA_DIR}' não encontrado. "
            "Baixe o dataset do Kaggle conforme instruções no cabeçalho."
        )
        return

    data = load_data(DATA_DIR)
    events = prepare_data(data["events"])
    print(f"  Eventos: {len(events):,} registros")
    print(f"  Usuários: {events['visitorid'].nunique():,}")
    print(f"  Itens: {events['itemid'].nunique():,}")

    # --- Divisão treino/teste ---
    print("\n[2/6] Dividindo treino/teste...")
    train, test = train_test_split_by_time(events, min_interactions=3)
    print(f"  Treino: {len(train):,} | Teste: {len(test):,}")

    all_items = events["itemid"].unique()

    # --- Pré-computação de estruturas ---
    print("\n[3/6] Construindo estruturas auxiliares...")

    # Co-ocorrência (usa amostra para velocidade)
    train_sample = train.sample(n=min(50_000, len(train)), random_state=SEED)
    co_matrix = build_viewed_bought_matrix(train_sample)
    print(f"  Co-ocorrência: {len(co_matrix):,} itens com dados")

    # Similaridade item-item
    item_neighbors = build_item_similarity_matrix(train, max_items=3000)
    print(f"  Similaridade item-item: {len(item_neighbors):,} itens com vizinhos")

    # Popularidade por categoria
    cat_pop = build_category_popularity(train, data["items"], data["categories"])
    # Mapa item -> categoria
    cat_props = data["items"][data["items"]["property"] == "categoryid"][
        ["itemid", "value"]
    ].copy()
    cat_props["categoryid"] = pd.to_numeric(cat_props["value"], errors="coerce")
    cat_props = cat_props.drop_duplicates(subset=["itemid"], keep="last")
    item_to_cat = dict(zip(cat_props["itemid"].astype(int), cat_props["categoryid"]))
    print(f"  Categorias: {len(cat_pop):,} categorias com itens")

    # --- Avaliação das estratégias ---
    print("\n[4/6] Avaliando estratégias clássicas...")

    # Estratégia 1: Popularidade
    popular_items = recommend_popularity(train, k=max(K_VALUES))

    def pop_recommend(user_id, item_id, train, **kw):
        return popular_items

    results_popularity = evaluate_strategy(
        "Popularidade", pop_recommend, test, train, all_items
    )

    # Estratégia 2: Co-ocorrência
    def coocc_recommend(user_id, item_id, train, **kw):
        recs = recommend_co_occurrence(item_id, co_matrix, k=max(K_VALUES))
        # Se não houver co-ocorrência, retorna populares
        return recs if recs else popular_items

    results_coocc = evaluate_strategy(
        "Co-ocorrência", coocc_recommend, test, train, all_items
    )

    # Estratégia 3: Item-Item CF
    def cf_recommend(user_id, item_id, train, **kw):
        recs = recommend_item_cf(item_id, item_neighbors, k=max(K_VALUES))
        return recs if recs else popular_items

    results_cf = evaluate_strategy(
        "Item-Item CF", cf_recommend, test, train, all_items
    )

    # Estratégia 4: Categoria
    def cat_recommend(user_id, item_id, train, **kw):
        recs = recommend_category(item_id, cat_pop, item_to_cat, k=max(K_VALUES))
        return recs if recs else popular_items

    results_category = evaluate_strategy(
        "Categoria", cat_recommend, test, train, all_items
    )

    # --- Modelo Neural de Co-ocorrência ---
    print("\n[5/6] Treinando modelo Neural Co-ocorrência...")
    coocc_model, coocc_item_map = train_cooccurrence_model(
        train, epochs=5, lr=0.001
    )

    # Mapa reverso: índice -> item_id
    coocc_rev_map = {v: k for k, v in coocc_item_map.items()}

    def neural_coocc_recommend(user_id, item_id, train, **kw):
        recs = recommend_coocc_neural(
            item_id, coocc_model, coocc_item_map, coocc_rev_map, k=max(K_VALUES)
        )
        return recs if recs else popular_items

    results_neural_coocc = evaluate_strategy(
        "Neural Co-ocorrência", neural_coocc_recommend, test, train, all_items
    )

    # --- Visualização ---
    print("\n[6/6] Gerando visualização...")
    all_results = {
        "Popularidade": results_popularity,
        "Co-ocorrência": results_coocc,
        "Item-Item CF": results_cf,
        "Categoria": results_category,
        "Neural Co-ocorrência": results_neural_coocc,
    }
    plot_results(all_results)

    # --- Resumo: melhor estratégia por métrica ---
    print("\n" + "=" * 60)
    print("RESUMO - MELHORES ESTRATÉGIAS POR MÉTRICA")
    print("=" * 60)
    metrics = ["hit_rate", "ndcg", "precision", "coverage"]
    for metric in metrics:
        # Usa K=10 como referência para ranking
        best_strategy = max(
            all_results, key=lambda s: all_results[s][metric][10]
        )
        best_value = all_results[best_strategy][metric][10]
        print(f"  {metric}@10: {best_strategy} ({best_value:.4f})")

    # Estratégia geral: mais vitórias entre todas as métricas e Ks
    win_counts: dict[str, int] = {s: 0 for s in all_results}
    for metric in metrics:
        for k in K_VALUES:
            winner = max(all_results, key=lambda s: all_results[s][metric][k])
            win_counts[winner] += 1
    overall_best = max(win_counts, key=lambda s: win_counts[s])
    print(f"\n  MELHOR ESTRATÉGIA GERAL: {overall_best}")
    print(f"  Vitórias: {win_counts[overall_best]}/{len(metrics) * len(K_VALUES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()