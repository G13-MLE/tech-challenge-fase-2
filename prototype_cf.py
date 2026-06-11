#!/usr/bin/env python3
"""
Protótipo de sistema de recomendação Neural CF (MLP item-to-item).

Usa matrizes esparsas (scipy.sparse) para processar TODOS os itens do catálogo
ao invés de limitar aos mais populares. Memória: ~90 MB ao invés de ~4.4 TB.

Estratégia: Neural CF - embeddings + MLP item-to-item
Feedback implícito: view=1, addtocart=5, transaction=10
Métricas: Hit Rate@K, NDCG@K, Precision@K, Coverage

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
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

# --- Reprodutibilidade ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).parent / "tmp"
K_VALUES = [5, 10, 20]
EVENT_WEIGHTS = {"view": 1, "addtocart": 5, "transaction": 10}
SAMPLE_SIZE = 200_000  # Amostra para modelos neurais


def load_and_prepare(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Carrega e prepara o dataframe de eventos com pesos de interação."""
    df = pd.read_csv(data_dir / "events.csv")
    df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.dropna(subset=["itemid"])
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
    idx_last = df_eligible.groupby("visitorid")["timestamp"].idxmax()
    test = df_eligible.loc[idx_last]
    train = df_eligible.drop(idx_last)
    return train, test


def build_sparse_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int], dict[int, int]]:
    """Constrói matriz esparsa CSR usuário-item com TODOS os itens.

    Retorna:
        - Matriz CSR esparsa (n_users x n_items)
        - user_id -> índice da linha
        - item_id -> índice da coluna
        - índice -> item_id (reverso)

    Memória: ~90 MB para o dataset completo (ao invés de 4.4 TB na densa).
    """
    agg = train.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()

    unique_users = sorted(agg["visitorid"].unique())
    unique_items = sorted(agg["itemid"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    item_map = {iid: i for i, iid in enumerate(unique_items)}
    rev_item_map = {i: iid for iid, i in item_map.items()}

    row_indices = agg["visitorid"].map(user_map).values
    col_indices = agg["itemid"].map(item_map).values
    data = agg["weight"].values.astype(np.float32)

    matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_map), len(item_map)),
    )
    return matrix, user_map, item_map, rev_item_map


# --- Neural CF - MLP item-to-item ---


class NeuralCF(nn.Module):
    """Modelo neural item-to-item para filtragem colaborativa.

    Dado um item de origem, prevê quais itens são relevantes.
    Dois embeddings: source (origem) e target (alvo), concatenados num MLP.
    """

    def __init__(self, n_items: int, dim: int = 32) -> None:
        super().__init__()
        self.source_emb = nn.Embedding(n_items, dim)
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
        nn.init.xavier_uniform_(self.source_emb.weight)
        nn.init.xavier_uniform_(self.target_emb.weight)

    def forward(
        self, source_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        s = self.source_emb(source_ids)
        t = self.target_emb(target_ids)
        x = torch.cat([s, t], dim=-1)
        return self.mlp(x).squeeze(-1)


def extract_cf_pairs(sparse_matrix: csr_matrix) -> list[tuple[int, int]]:
    """Extrai pares (item_origem, item_alvo) a partir da matriz esparsa.

    Para cada usuário, todos os itens com os quais interagiu formam
    pares entre si: se acessou A e B, (A, B) e (B, A) são pares positivos.
    """
    pairs = []
    for u in range(sparse_matrix.shape[0]):
        row = sparse_matrix.getrow(u)
        items = row.indices.tolist()
        if len(items) < 2:
            continue
        # Limita a 10 itens por usuário para não explodir pares
        items = items[:10]
        for i in items:
            for j in items:
                if i != j:
                    pairs.append((i, j))
        if len(pairs) > SAMPLE_SIZE:
            np.random.shuffle(pairs)
            pairs = pairs[:SAMPLE_SIZE]
            break
    return pairs


def train_neuralcf(
    sparse_matrix: csr_matrix,
    item_map: dict[int, int],
    n_neg: int = 4,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 1024,
) -> NeuralCF | None:
    """Treina o modelo Neural CF item-to-item com pares da matriz esparsa."""
    n_items = len(item_map)

    print("  Extraindo pares de itens co-interagidos...")
    pair_indices = extract_cf_pairs(sparse_matrix)
    print(f"  Pares positivos: {len(pair_indices):,}")

    if not pair_indices:
        print("  AVISO: nenhum par encontrado.")
        return None

    # Prepara tensores com amostragem negativa
    sources, targets, labels = [], [], []
    for src_idx, tgt_idx in pair_indices:
        sources.append(src_idx)
        targets.append(tgt_idx)
        labels.append(1.0)
        neg_indices = np.random.choice(n_items, size=n_neg, replace=False)
        for ni in neg_indices:
            if ni != tgt_idx:
                sources.append(src_idx)
                targets.append(ni)
                labels.append(0.0)

    src_t = torch.tensor(sources, dtype=torch.long)
    tgt_t = torch.tensor(targets, dtype=torch.long)
    lab_t = torch.tensor(labels, dtype=torch.float32)

    model = NeuralCF(n_items=n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(src_t, tgt_t, lab_t)
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
        print(f"  Época {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

    return model


def recommend_neuralcf(
    source_item: int,
    model: NeuralCF,
    item_map: dict[int, int],
    rev_item_map: dict[int, int],
    k: int = 10,
) -> list[int]:
    """Dado um item, recomenda itens com maior probabilidade via MLP."""
    model.eval()
    if source_item not in item_map:
        return []

    src_idx = torch.tensor([item_map[source_item]], dtype=torch.long)
    n_items = len(rev_item_map)
    candidate_indices = torch.arange(n_items, dtype=torch.long)
    mask = candidate_indices != item_map[source_item]
    candidate_indices = candidate_indices[mask]

    # Pontua em mini-batches para não estourar memória da GPU
    infer_batch = 5000
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(candidate_indices), infer_batch):
            end = min(start + infer_batch, len(candidate_indices))
            c_batch = candidate_indices[start:end]
            s_batch = src_idx.expand(len(c_batch))
            all_scores.append(model(s_batch, c_batch))

    all_scores = torch.cat(all_scores)
    top_k_indices = torch.topk(all_scores, min(k, len(all_scores))).indices
    return [rev_item_map[candidate_indices[i].item()] for i in top_k_indices.tolist()]


# --- Métricas de avaliação ---


def hit_rate_at_k(recommendations: list[int], relevant: set[int], k: int) -> float:
    """Fração de usuários cujo item relevante está no top-K."""
    return 1.0 if relevant & set(recommendations[:k]) else 0.0


def ndcg_at_k(recommendations: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K: ganho descontado normalizado."""
    dcg = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg  # idcg = 1.0 para um único item relevante


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


def evaluate(
    recommend_fn,
    test_users: pd.DataFrame,
    n_items: int,
    **kwargs,
) -> dict[str, dict[int, float]]:
    """Avalia a estratégia de recomendação com todas as métricas."""
    results: dict[str, dict[int, list[float]]] = {
        "hit_rate": {k: [] for k in K_VALUES},
        "ndcg": {k: [] for k in K_VALUES},
        "precision": {k: [] for k in K_VALUES},
    }
    all_recs: dict[int, list[list[int]]] = {k: [] for k in K_VALUES}

    for user_id in test_users["visitorid"].unique()[:500]:
        user_test = test_users[test_users["visitorid"] == user_id]
        relevant_items = set(user_test["itemid"].values)
        if not relevant_items:
            continue

        recs = recommend_fn(user_id=int(user_id), **kwargs)

        for k in K_VALUES:
            top_k = recs[:k]
            results["hit_rate"][k].append(hit_rate_at_k(top_k, relevant_items, k))
            results["ndcg"][k].append(ndcg_at_k(top_k, relevant_items, k))
            results["precision"][k].append(precision_at_k(top_k, relevant_items, k))
            all_recs[k].append(top_k)

    final: dict[str, dict[int, float]] = {}
    for metric in ["hit_rate", "ndcg", "precision"]:
        final[metric] = {k: np.mean(results[metric][k]) for k in K_VALUES}
    final["coverage"] = {k: coverage(all_recs[k], n_items) for k in K_VALUES}

    print("\n--- Neural CF ---")
    for metric, values in final.items():
        for k, v in values.items():
            print(f"  {metric}@{k}: {v:.4f}")

    return final


def plot_results(results: dict[str, dict[int, float]]) -> None:
    """Gera gráfico com as métricas do modelo Neural CF."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["hit_rate", "ndcg", "precision", "coverage"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    for ax, metric in zip(axes, metrics):
        values = [results[metric][k] for k in K_VALUES]
        ax.bar([str(k) for k in K_VALUES], values, color=colors)
        ax.set_xlabel("K")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}@K")

    plt.suptitle("Neural CF (MLP item-to-item) - Retailrocket", fontsize=14)
    plt.tight_layout()
    plt.savefig("cf_sparse_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGráfico salvo em: cf_sparse_results.png")


# --- Pipeline principal ---


def main() -> None:
    """Executa o pipeline Neural CF (MLP item-to-item) com matrizes esparsas."""
    print("=" * 60)
    print("Neural CF (MLP item-to-item) - Retailrocket Dataset")
    print("=" * 60)

    # Carregamento dos dados
    print("\n[1/3] Carregando dados...")
    if not DATA_DIR.exists():
        print(
            f"Diretório '{DATA_DIR}' não encontrado. "
            "Baixe o dataset do Kaggle conforme instruções no cabeçalho."
        )
        return

    events = load_and_prepare(DATA_DIR)
    print(f"  Eventos: {len(events):,} registros")
    print(f"  Usuários: {events['visitorid'].nunique():,}")
    print(f"  Itens: {events['itemid'].nunique():,}")

    # Divisão treino/teste
    print("\n[2/3] Dividindo treino/teste...")
    train, test = train_test_split_by_time(events, min_interactions=3)
    print(f"  Treino: {len(train):,} | Teste: {len(test):,}")

    # Matriz esparsa
    print("\n  Construindo matriz esparsa...")
    sparse_mat, user_map, item_map, rev_item_map = build_sparse_matrix(train)
    print(f"  Matriz CSR: {sparse_mat.shape[0]:,} x {sparse_mat.shape[1]:,}")
    print(f"  Não-zero: {sparse_mat.nnz:,} | Memória: {sparse_mat.data.nbytes / 1024 / 1024:.1f} MB")

    # Popularidade (fallback)
    popular_items = (
        train[train["event"] == "transaction"]
        .groupby("itemid")
        .size()
        .nlargest(max(K_VALUES))
        .index.tolist()
    )

    # Treinamento
    print("\n[3/3] Treinando Neural CF (MLP item-to-item)...")
    model = train_neuralcf(sparse_mat, item_map, epochs=10, lr=0.001)
    if model is None:
        return

    def recommend(user_id: int, **kw) -> list[int]:
        row = sparse_mat.getrow(user_map.get(user_id, -1))
        if row.nnz == 0:
            return popular_items
        source_item = rev_item_map[int(row.indices[-1])]
        recs = recommend_neuralcf(source_item, model, item_map, rev_item_map, k=max(K_VALUES))
        return recs if recs else popular_items

    results = evaluate(recommend, test, len(item_map))

    # Visualização e resumo
    plot_results(results)

    print("\n" + "=" * 60)
    print("RESUMO - NEURAL CF (MLP ITEM-TO-ITEM)")
    print("=" * 60)
    for metric in ["hit_rate", "ndcg", "precision", "coverage"]:
        for k in K_VALUES:
            print(f"  {metric}@{k}: {results[metric][k]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()