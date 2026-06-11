#!/usr/bin/env python3
"""
Protótipo de sistema de recomendação baseado em Filtragem Colaborativa (CF).

Usa matrizes esparsas (scipy.sparse) para processar TODOS os itens do catálogo
ao invés de limitar aos mais populares. Memória: ~90 MB ao invés de ~4.4 TB.

Estratégias CF implementadas:
1. Item-based CF - itens similares via cosseno em batches
2. Matrix Factorization (ALS) - fatoração com iteração esparsa
3. Neural CF - embeddings + MLP item-to-item
4. Híbrida - combina Item-based + MF

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
import seaborn as sns
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --- Reprodutibilidade ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).parent / "tmp"
K_VALUES = [5, 10, 20]
EVENT_WEIGHTS = {"view": 1, "addtocart": 5, "transaction": 10}
SAMPLE_SIZE = 200_000  # Amostra para modelos neurais
SIM_BATCH_SIZE = 2000  # Itens por batch na similaridade


# =============================================================================
# Carregamento e preparação dos dados
# =============================================================================

def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Carrega o arquivo de eventos do dataset Retailrocket."""
    return pd.read_csv(data_dir / "events.csv")


def prepare_data(events: pd.DataFrame) -> pd.DataFrame:
    """Prepara o dataframe de eventos com pesos de interação."""
    df = events.copy()
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


# =============================================================================
# Matriz esparsa usuário-item (TODOS os itens, sem limite)
# =============================================================================

def build_sparse_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Constrói matriz esparsa CSR usuário-item com TODOS os itens.

    Retorna:
        - Matriz CSR esparsa (n_users x n_items)
        - user_id -> índice da linha
        - item_id -> índice da coluna
        - índice -> user_id (reverso)
        - índice -> item_id (reverso)

    Memória: ~90 MB para o dataset completo (ao invés de 4.4 TB na densa).
    """
    # Agrega pesos por (usuário, item) — soma interações duplicadas
    agg = train.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()

    # Mapeia ids para índices consecutivos
    unique_users = sorted(agg["visitorid"].unique())
    unique_items = sorted(agg["itemid"].unique())
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    item_map = {iid: i for i, iid in enumerate(unique_items)}
    rev_user_map = {i: uid for uid, i in user_map.items()}
    rev_item_map = {i: iid for iid, i in item_map.items()}

    # Constrói CSR: só armazena valores não-zero
    row_indices = agg["visitorid"].map(user_map).values
    col_indices = agg["itemid"].map(item_map).values
    data = agg["weight"].values.astype(np.float32)

    matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_map), len(item_map)),
    )
    return matrix, user_map, item_map, rev_user_map, rev_item_map


# =============================================================================
# CF 1: Item-based (batch) - "Itens parecidos com os que você acessou"
# =============================================================================

def build_item_similarity_batched(
    sparse_matrix: csr_matrix, item_map: dict[int, int],
    rev_item_map: dict[int, int], top_k: int = 100,
) -> dict[int, list[tuple[int, float]]]:
    """Constrói similaridade item-item em batches para não estourar memória.

    Para cada batch de itens, computa similaridade cosseno com todos os outros,
    guarda apenas os top_k vizinhos. Memória por batch: ~batch_size * n_items * 4 bytes.
    """
    n_items = sparse_matrix.shape[1]
    item_neighbors: dict[int, list[tuple[int, float]]] = {}

    # Transpõe uma vez: itens x usuários
    item_user = sparse_matrix.T.tocsr()

    for start in range(0, n_items, SIM_BATCH_SIZE):
        end = min(start + SIM_BATCH_SIZE, n_items)
        # Similaridade do batch com todos os itens
        batch_sim = cosine_similarity(item_user[start:end], item_user)

        for i in range(batch_sim.shape[0]):
            item_idx = start + i
            item_id = rev_item_map[item_idx]
            sims = batch_sim[i]
            # Exclui o próprio item e guarda top_k
            sims[item_idx] = 0
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
            neighbors = [
                (rev_item_map[int(j)], float(sims[j]))
                for j in top_indices if sims[j] > 0
            ]
            item_neighbors[item_id] = neighbors

        if (start // SIM_BATCH_SIZE + 1) % 10 == 0:
            print(f"    Batch {start}-{end} de {n_items} itens")

    return item_neighbors


def recommend_item_based(
    user_id: int,
    sparse_matrix: csr_matrix,
    user_map: dict[int, int],
    rev_item_map: dict[int, int],
    item_neighbors: dict[int, list[tuple[int, float]]],
    k: int = 10,
) -> list[int]:
    """Recomenda itens similares aos que o usuário já acessou (via esparsa)."""
    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    # Extrai itens acessados diretamente da linha esparsa
    row = sparse_matrix.getrow(user_idx)
    interacted_indices = set(row.indices)
    interacted_items = {rev_item_map[i] for i in interacted_indices}

    # Pontua candidatos via similaridade dos itens já acessados
    item_scores: dict[int, float] = {}
    for idx in interacted_indices:
        src_item = rev_item_map[idx]
        for neighbor_item, sim in item_neighbors.get(src_item, []):
            if neighbor_item not in interacted_items:
                item_scores[neighbor_item] = item_scores.get(neighbor_item, 0) + sim

    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items[:k]]


# =============================================================================
# CF 2: Matrix Factorization (ALS) esparsa
# =============================================================================

def train_mf_als_sparse(
    sparse_matrix: csr_matrix,
    n_factors: int = 64, n_iterations: int = 20, reg: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Treina Matrix Factorization via ALS usando iteração esparsa.

    Só processa entradas não-zero, sem alocar a matriz densa.
    R ≈ U * V^T, onde U e V são aprendidos via mínimos quadrados alternados.
    """
    n_users, n_items = sparse_matrix.shape

    U = np.random.randn(n_users, n_factors) * 0.01
    V = np.random.randn(n_items, n_factors) * 0.01

    # Converte para CSC para acesso eficiente por coluna
    matrix_csc = sparse_matrix.tocsc()
    I_eye = reg * np.eye(n_factors)

    for iteration in range(n_iterations):
        # Fixa V, otimiza U — itera sobre linhas (usuários) não-vazias
        for u in range(n_users):
            row_start = sparse_matrix.indptr[u]
            row_end = sparse_matrix.indptr[u + 1]
            if row_start == row_end:
                continue
            indices = sparse_matrix.indices[row_start:row_end]
            values = sparse_matrix.data[row_start:row_end]
            V_sub = V[indices]
            A = V_sub.T @ V_sub + I_eye
            b = V_sub.T @ values
            U[u] = np.linalg.solve(A, b)

        # Fixa U, otimiza V — itera sobre colunas (itens) não-vazias
        for i in range(n_items):
            col_start = matrix_csc.indptr[i]
            col_end = matrix_csc.indptr[i + 1]
            if col_start == col_end:
                continue
            indices = matrix_csc.indices[col_start:col_end]
            values = matrix_csc.data[col_start:col_end]
            U_sub = U[indices]
            A = U_sub.T @ U_sub + I_eye
            b = U_sub.T @ values
            V[i] = np.linalg.solve(A, b)

        # Erro de reconstrução em amostra (não percorre toda a matriz)
        sample_users = np.random.choice(n_users, size=min(1000, n_users), replace=False)
        sample_rows = sparse_matrix[sample_users]
        pred = U[sample_users] @ V.T
        mask = sample_rows.toarray() > 0
        if mask.any():
            error = np.sum((sample_rows.toarray()[mask] - pred[mask]) ** 2)
        else:
            error = 0.0
        print(f"  Iteração {iteration + 1}/{n_iterations} - Erro (amostra): {error:.2f}")

    return U, V


def recommend_mf(
    user_id: int,
    U: np.ndarray, V: np.ndarray,
    user_map: dict[int, int],
    rev_item_map: dict[int, int],
    sparse_matrix: csr_matrix,
    k: int = 10,
) -> list[int]:
    """Recomenda itens com maior score predito pela fatoração de matriz."""
    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    scores = U[user_idx] @ V.T

    # Exclui itens já acessados (direto da esparsa)
    row = sparse_matrix.getrow(user_idx)
    interacted = set(row.indices)

    # Ordena por score, excluindo interagidos
    item_indices = np.argsort(scores)[::-1]
    result = []
    for idx in item_indices:
        if idx not in interacted:
            result.append(rev_item_map[int(idx)])
            if len(result) >= k:
                break
    return result


# =============================================================================
# CF 3: Neural CF - MLP item-to-item
# =============================================================================

class NeuralCF(nn.Module):
    """Modelo neural item-to-item para filtragem colaborativa.

    Dado um item de origem, prevê quais itens são relevantes.
    Dois embeddings: source (origem) e target (alvo), concatenados num MLP.
    """

    def __init__(self, n_items: int, dim: int = 32):
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

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        s = self.source_emb(source_ids)
        t = self.target_emb(target_ids)
        x = torch.cat([s, t], dim=-1)
        return self.mlp(x).squeeze(-1)


def extract_cf_pairs_sparse(
    sparse_matrix: csr_matrix,
    user_map: dict[int, int],
    rev_item_map: dict[int, int],
) -> list[tuple[int, int]]:
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
        # Amostra se muitos pares
        if len(pairs) > SAMPLE_SIZE:
            np.random.shuffle(pairs)
            pairs = pairs[:SAMPLE_SIZE]
            break
    return pairs


def train_neuralcf(
    sparse_matrix: csr_matrix,
    item_map: dict[int, int],
    n_neg: int = 4,
    epochs: int = 10, lr: float = 0.001, batch_size: int = 1024,
) -> tuple[NeuralCF | None, dict[int, int]]:
    """Treina o modelo Neural CF item-to-item com pares da matriz esparsa."""
    n_items = len(item_map)

    print("  Extraindo pares de itens co-interagidos...")
    pair_indices = extract_cf_pairs_sparse(sparse_matrix, {}, {})
    print(f"  Pares positivos: {len(pair_indices):,}")

    if not pair_indices:
        print("  AVISO: nenhum par encontrado.")
        return None, item_map

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

    data = (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )

    model = NeuralCF(n_items=n_items)
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
    batch = 5000
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(candidate_indices), batch):
            end = min(start + batch, len(candidate_indices))
            c_batch = candidate_indices[start:end]
            s_batch = src_idx.expand(len(c_batch))
            scores = model(s_batch, c_batch)
            all_scores.append(scores)

    all_scores = torch.cat(all_scores)
    top_k_indices = torch.topk(all_scores, min(k, len(all_scores))).indices
    # Mapeia de volta: posição no candidato -> índice do item -> item_id
    return [rev_item_map[candidate_indices[i].item()] for i in top_k_indices.tolist()]


# =============================================================================
# CF 4: Híbrida - Combina Item-based + MF
# =============================================================================

def recommend_hybrid(
    user_id: int,
    sparse_matrix: csr_matrix,
    user_map: dict[int, int],
    rev_item_map: dict[int, int],
    item_neighbors: dict[int, list[tuple[int, float]]],
    U: np.ndarray, V: np.ndarray,
    weights: tuple[float, float] = (0.4, 0.6),
    k: int = 10,
) -> list[int]:
    """Combina scores de Item-based CF e Matrix Factorization.

    score = w1 * score_item + w2 * score_mf (normalizados para [0,1])
    """
    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    row = sparse_matrix.getrow(user_idx)
    interacted_indices = set(row.indices)
    interacted_items = {rev_item_map[i] for i in interacted_indices}

    # --- Score Item-based ---
    item_scores: dict[int, float] = {}
    for idx in interacted_indices:
        src_item = rev_item_map[idx]
        for neighbor_item, sim in item_neighbors.get(src_item, []):
            if neighbor_item not in interacted_items:
                item_scores[neighbor_item] = item_scores.get(neighbor_item, 0) + sim
    if item_scores:
        max_is = max(item_scores.values())
        if max_is > 0:
            item_scores = {k: v / max_is for k, v in item_scores.items()}

    # --- Score MF ---
    scores_arr = U[user_idx] @ V.T
    mf_scores: dict[int, float] = {}
    top_indices = np.argsort(scores_arr)[::-1][:k * 10]
    for idx in top_indices:
        item_id = rev_item_map.get(int(idx))
        if item_id is not None and item_id not in interacted_items:
            mf_scores[item_id] = scores_arr[idx]
    if mf_scores:
        max_mf = max(mf_scores.values())
        if max_mf > 0:
            mf_scores = {k: v / max_mf for k, v in mf_scores.items()}

    # --- Combina ---
    w1, w2 = weights
    all_candidates = set(item_scores) | set(mf_scores)
    combined: dict[int, float] = {}
    for item_id in all_candidates:
        combined[item_id] = w1 * item_scores.get(item_id, 0) + w2 * mf_scores.get(item_id, 0)

    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items[:k]]


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
            dcg += 1.0 / np.log2(i + 2)
    idcg = 1.0
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

    sample_users = test_users["visitorid"].unique()[:500]

    for user_id in sample_users:
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
        width = 0.18
        for i, strategy in enumerate(strategies):
            values = [all_results[strategy][metric][k] for k in K_VALUES]
            ax.bar(x + i * width, values, width, label=strategy, color=colors[i])
        ax.set_xlabel("K")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}@K")
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.legend(fontsize=7)

    plt.suptitle("CF Esparsa - Comparação de Estratégias - Retailrocket", fontsize=14)
    plt.tight_layout()
    plt.savefig("cf_sparse_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGráfico salvo em: cf_sparse_results.png")


# =============================================================================
# Função principal
# =============================================================================

def main():
    """Executa o pipeline de filtragem colaborativa com matrizes esparsas."""
    print("=" * 60)
    print("CF Esparsa (scipy.sparse) - Retailrocket Dataset")
    print("=" * 60)

    # --- Carregamento dos dados ---
    print("\n[1/4] Carregando dados...")
    if not DATA_DIR.exists():
        print(
            f"Diretório '{DATA_DIR}' não encontrado. "
            "Baixe o dataset do Kaggle conforme instruções no cabeçalho."
        )
        return

    events = prepare_data(load_data(DATA_DIR))
    print(f"  Eventos: {len(events):,} registros")
    print(f"  Usuários: {events['visitorid'].nunique():,}")
    print(f"  Itens: {events['itemid'].nunique():,}")

    # --- Divisão treino/teste ---
    print("\n[2/4] Dividindo treino/teste...")
    train, test = train_test_split_by_time(events, min_interactions=3)
    print(f"  Treino: {len(train):,} | Teste: {len(test):,}")

    all_items = events["itemid"].unique()

    # --- Matriz esparsa (TODOS os itens) ---
    print("\n  Construindo matriz esparsa...")
    sparse_mat, user_map, item_map, rev_user_map, rev_item_map = build_sparse_matrix(train)
    print(f"  Matriz CSR: {sparse_mat.shape[0]:,} usuários x {sparse_mat.shape[1]:,} itens")
    print(f"  Não-zero: {sparse_mat.nnz:,} entradas")
    print(f"  Memória: {sparse_mat.data.nbytes / 1024 / 1024:.1f} MB")

    # Popularidade (fallback)
    popular_items = (
        train[train["event"] == "transaction"]
        .groupby("itemid").size()
        .nlargest(max(K_VALUES))
        .index.tolist()
    )

    # --- CF 1: Item-based (batch) ---
    print("\n[3/4] Construindo similaridade item-item em batches...")
    item_neighbors = build_item_similarity_batched(
        sparse_mat, item_map, rev_item_map, top_k=100
    )
    print(f"  Vizinhos construídos para {len(item_neighbors):,} itens")

    results_item_cf = evaluate_strategy(
        "Item-based CF",
        lambda user_id: recommend_item_based(
            user_id, sparse_mat, user_map, rev_item_map, item_neighbors, k=max(K_VALUES)
        ) or popular_items,
        test, train, all_items,
    )

    # --- CF 2: Matrix Factorization (ALS esparsa) ---
    print("\n  Treinando Matrix Factorization (ALS esparsa)...")
    U, V = train_mf_als_sparse(sparse_mat, n_factors=64, n_iterations=20, reg=0.05)

    results_mf = evaluate_strategy(
        "Matrix Factorization",
        lambda user_id: recommend_mf(
            user_id, U, V, user_map, rev_item_map, sparse_mat, k=max(K_VALUES)
        ) or popular_items,
        test, train, all_items,
    )

    # --- CF 3: Neural CF (MLP item-to-item) ---
    print("\n[4/4] Treinando Neural CF (MLP item-to-item)...")
    neuralcf_model, neuralcf_item_map = train_neuralcf(
        sparse_mat, item_map, epochs=10, lr=0.001
    )
    neuralcf_rev_map = {v: k for k, v in neuralcf_item_map.items()}

    def neural_cf_recommend(user_id, **kw):
        row = sparse_mat.getrow(user_map.get(user_id, -1))
        if row.nnz == 0:
            return popular_items
        # Último item acessado como origem
        source_item = rev_item_map[int(row.indices[-1])]
        recs = recommend_neuralcf(
            source_item, neuralcf_model, neuralcf_item_map, neuralcf_rev_map, k=max(K_VALUES)
        )
        return recs if recs else popular_items

    results_neural = evaluate_strategy(
        "Neural CF",
        neural_cf_recommend,
        test, train, all_items,
    )

    # --- CF 4: Híbrida ---
    print("\n  Avaliando estratégia Híbrida...")
    results_hybrid = evaluate_strategy(
        "Híbrida (IB+MF)",
        lambda user_id: recommend_hybrid(
            user_id, sparse_mat, user_map, rev_item_map, item_neighbors,
            U, V, weights=(0.4, 0.6), k=max(K_VALUES),
        ) or popular_items,
        test, train, all_items,
    )

    # --- Visualização ---
    all_results = {
        "Item-based CF": results_item_cf,
        "Matrix Factorization": results_mf,
        "Neural CF": results_neural,
        "Híbrida": results_hybrid,
    }
    plot_results(all_results)

    # --- Resumo ---
    print("\n" + "=" * 60)
    print("RESUMO - MELHORES ESTRATÉGIAS CF POR MÉTRICA")
    print("=" * 60)
    metrics = ["hit_rate", "ndcg", "precision", "coverage"]
    for metric in metrics:
        best_strategy = max(
            all_results, key=lambda s: all_results[s][metric][10]
        )
        best_value = all_results[best_strategy][metric][10]
        print(f"  {metric}@10: {best_strategy} ({best_value:.4f})")

    win_counts: dict[str, int] = {s: 0 for s in all_results}
    for metric in metrics:
        for k in K_VALUES:
            winner = max(all_results, key=lambda s: all_results[s][metric][k])
            win_counts[winner] += 1
    overall_best = max(win_counts, key=lambda s: win_counts[s])
    print(f"\n  MELHOR ESTRATÉGIA CF GERAL: {overall_best}")
    print(f"  Vitórias: {win_counts[overall_best]}/{len(metrics) * len(K_VALUES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()