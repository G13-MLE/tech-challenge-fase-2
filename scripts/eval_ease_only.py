"""
Avaliacao otimizada de EASE^ com scoring em batch.

Abordagem vetorizada para evitar o loop por usuario que causava timeout.
Objetivo: superar Item-KNN baseline Recall@10 = 0.01060 na validacao.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# --- Parametros ---
RANDOM_SEED = 42
TOP_K = [5, 10, 20]
IMPLICIT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MAX_EVAL_USERS = 5000

# EASE^ hiperparametros - lambda mais promissores primeiro
EASE_LAMBDAS = [250, 500, 100, 1000, 2500]
EASE_MAX_ITEMS = 20_000

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RETAILROCKET_DATA_DIR", str(PROJECT_ROOT / "data")))
events_path = DATA_DIR / "events.csv"

np.random.seed(RANDOM_SEED)


# ============================================================================
# FUNCOES DE METRICAS (vetorizadas)
# ============================================================================


def compute_metrics_batch(
    recs: np.ndarray,
    relevant: dict[int, set[int]],
    user_list: list[int],
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Calcula metricas em batch para todos os usuarios.

    Args:
        recs: Array (n_users, max_k) com itens recomendados por usuario.
        relevant: Dict user_idx -> set de itens relevantes.
        user_list: Lista de user_idx na mesma ordem de recs.
        k_values: Lista de valores de K.

    Returns:
        Dict mapeando K -> {metrica: valor_medio}.
    """
    _ = max(k_values)  # usado indiretamente via recs
    results = {
        k: {"precision": [], "recall": [], "ndcg": [], "hit_rate": []} for k in k_values
    }

    for i, user_idx in enumerate(user_list):
        rel = relevant.get(user_idx, set())
        if not rel:
            continue

        user_recs = recs[i]

        for k in k_values:
            rec_k = set(user_recs[:k])
            hits = rec_k & rel
            n_hits = len(hits)
            n_rel = len(rel)

            results[k]["precision"].append(n_hits / k if k > 0 else 0.0)
            results[k]["recall"].append(n_hits / n_rel if n_rel > 0 else 0.0)
            results[k]["hit_rate"].append(1 if n_hits > 0 else 0)

            # NDCG
            dcg = 0.0
            for j, item in enumerate(user_recs[:k]):
                if item in rel:
                    dcg += 1.0 / np.log2(j + 2)
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(n_rel, k)))
            results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

    # Calcular medias
    final = {}
    for k in k_values:
        final[k] = {}
        for metric in ["precision", "recall", "ndcg", "hit_rate"]:
            vals = results[k][metric]
            final[k][metric] = np.mean(vals) if vals else 0.0
        final[k]["n_users"] = len(results[k]["precision"])

    return final


# ============================================================================
# EASE^ TREINAMENTO E AVALIACAO
# ============================================================================


def train_ease(
    x_train: sp.csr_matrix,
    lam: float = 250.0,
    max_items: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Treina EASE^ (forma fechada).

    Args:
        x_train: Matriz esparsa usuario-item.
        lam: Parametro de regularizacao lambda.
        max_items: Numero maximo de itens (top-N por popularidade).

    Returns:
        Tupla (b_matrix, top_item_indices).
    """
    item_pop = np.asarray(x_train.sum(axis=0)).ravel()
    if max_items < x_train.shape[1]:
        top_item_indices = np.argsort(-item_pop)[:max_items]
    else:
        top_item_indices = np.arange(x_train.shape[1])

    x_sub = x_train[:, top_item_indices].astype(np.float64).tocsr()
    n_top = len(top_item_indices)

    print(f"  EASE^: Calculando para {n_top:,} itens (lambda={lam})...")

    # G = X^T X + lambda * I
    t0 = time.time()
    g_matrix = x_sub.T.dot(x_sub).toarray()
    g_matrix += lam * np.eye(n_top)
    t1 = time.time()
    print(f"  G calculado em {t1 - t0:.1f}s")

    # P = G^{-1}
    p_inv = np.linalg.inv(g_matrix)
    t2 = time.time()
    print(f"  P = inv(G) calculado em {t2 - t1:.1f}s")

    # B = I - P / diag(P) (zera diagonal)
    b_matrix = np.eye(n_top) - p_inv / np.diag(p_inv)
    np.fill_diagonal(b_matrix, 0)

    print(f"  B: norm={np.linalg.norm(b_matrix):.2f}, max={np.abs(b_matrix).max():.6f}")
    print(f"  Treino total: {time.time() - t0:.1f}s")

    return b_matrix, top_item_indices


def evaluate_ease_batch(
    x_train: sp.csr_matrix,
    b_matrix: np.ndarray,
    top_item_indices: np.ndarray,
    ground_truth: dict[int, set[int]],
    item_popularity: np.ndarray,
    item_log_popularity: np.ndarray,
    most_popular_items: np.ndarray,
    max_eval_users: int = 5000,
    pop_weights: list[float] | None = None,
    k_values: list[int] | None = None,
) -> dict[float, dict[int, dict[str, float]]]:
    """Avalia EASE^ com scoring em batch para multiplos popularity weights.

    Args:
        x_train: Matriz esparsa completa.
        b_matrix: Matriz de similaridade EASE^ (n_top x n_top).
        top_item_indices: Indices dos itens selecionados.
        ground_truth: Dict user_idx -> itens relevantes.
        item_popularity: Popularidade de cada item.
        item_log_popularity: Log-popularidade de cada item.
        most_popular_items: Itens ordenados por popularidade.
        max_eval_users: Maximo de usuarios para avaliar.
        pop_weights: Lista de popularity blending weights.
        k_values: Valores de K para metricas.

    Returns:
        Dict mapeando pw -> {K -> {metrica: valor}}.
    """
    if pop_weights is None:
        pop_weights = [0.0, 0.3, 0.5, 1.0]
    if k_values is None:
        k_values = [5, 10, 20]

    n_top = len(top_item_indices)
    max_k = max(k_values)

    # Amostrar usuarios
    user_list = list(ground_truth.keys())
    if len(user_list) > max_eval_users:
        rng = np.random.default_rng(RANDOM_SEED)
        user_list = sorted(
            rng.choice(user_list, size=max_eval_users, replace=False).tolist()
        )

    # Extrair sub-matriz para usuarios avaliados
    user_indices = np.array(user_list)
    x_eval = x_train[user_indices]
    x_eval_sub = x_eval[:, top_item_indices].astype(np.float64).tocsr()

    print(f"  Avaliando {len(user_list):,} usuarios...")

    # Pre-computar popularidade dos itens top
    pop_scores = item_log_popularity[top_item_indices]

    # Scoring em batch: (n_users, n_top) = X_eval_sub @ B
    # Isso e o mais caro mas muito mais rapido que loop por usuario
    print(f"  Calculando scores em batch ({len(user_list):,} x {n_top:,})...")
    t0 = time.time()

    # Processar em blocos para economizar memoria
    batch_size = 1000
    n_batches = (len(user_list) + batch_size - 1) // batch_size
    all_scores = np.zeros((len(user_list), n_top), dtype=np.float64)

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, len(user_list))
        block = x_eval_sub[start:end].toarray()
        all_scores[start:end] = block @ b_matrix

    print(f"  Scoring em batch: {time.time() - t0:.1f}s")

    # Marcar itens ja vistos (excluir)
    print("  Marcando itens ja vistos...")
    t1 = time.time()
    for i in range(len(user_list)):
        row = x_eval_sub[i]
        if row.nnz > 0:
            for j in range(row.nnz):
                all_scores[i, row.indices[j]] = -np.inf
    print(f"  Marcacao: {time.time() - t1:.1f}s")

    # Pre-computar recomendacoes por popularity weight
    results_all = {}

    for pw in pop_weights:
        print(f"  pw={pw:.1f}: calculando recomendacoes...")
        t2 = time.time()

        # Score final = EASE^ score + pw * popularity
        if pw > 0:
            scores = all_scores + pw * pop_scores[np.newaxis, :]
        else:
            scores = all_scores

        # Top-K recomendacoes
        # argpartition e mais rapido que argsort para top-K
        recs = np.zeros((len(user_list), max_k), dtype=np.int64)
        fallback_count = 0

        for i in range(len(user_list)):
            # Verificar se usuario tem interacoes (warm-start)
            row_orig = x_eval[i]
            if row_orig.nnz == 0:
                # Cold-start: MostPopular fallback
                rec_idx = 0
                for pop_item in most_popular_items:
                    if rec_idx >= max_k:
                        break
                    recs[i, rec_idx] = int(pop_item)
                    rec_idx += 1
                fallback_count += 1
                continue

            # Warm-start: ordenar scores
            user_scores = scores[i]
            top_k_pos = np.argpartition(-user_scores, max_k)[:max_k]
            # Ordenar os top-K por score
            top_k_pos = top_k_pos[np.argsort(-user_scores[top_k_pos])]

            rec_idx = 0
            for pos in top_k_pos:
                if user_scores[pos] > -np.inf:
                    recs[i, rec_idx] = int(top_item_indices[pos])
                    rec_idx += 1
                if rec_idx >= max_k:
                    break

            # Completar com populares se necessario
            if rec_idx < max_k:
                seen = set(int(x) for x in row_orig.indices)
                rec_set = set(recs[i, :rec_idx])
                for pop_item in most_popular_items:
                    if rec_idx >= max_k:
                        break
                    if int(pop_item) not in seen and int(pop_item) not in rec_set:
                        recs[i, rec_idx] = int(pop_item)
                        rec_idx += 1

        elapsed = time.time() - t2
        print(f"  pw={pw:.1f}: {elapsed:.1f}s, cold-start={fallback_count}")

        # Calcular metricas
        metrics = compute_metrics_batch(recs, ground_truth, user_list, k_values)
        results_all[pw] = metrics

    return results_all


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("Avaliacao EASE^ - scoring em batch (otimizado)")
    print("=" * 70)

    # --- 1. Carregar dados ---
    print("\n[1/4] Carregando dados...")
    t_start = time.time()
    events = pd.read_csv(events_path)
    events["event_time"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    events = events.sort_values(["event_time", "visitorid", "itemid"]).reset_index(
        drop=True
    )
    events["implicit_weight"] = (
        events["event"].map(IMPLICIT_WEIGHTS).fillna(0).astype(int)
    )
    print(f"  Total eventos: {len(events):,}")

    # --- 2. Split e matriz ---
    print("\n[2/4] Construindo split e matriz...")
    time_min = events["event_time"].min()
    time_max = events["event_time"].max()
    time_range = time_max - time_min
    train_end = time_min + time_range * SPLIT_RATIOS[0]
    val_end = train_end + time_range * SPLIT_RATIOS[1]

    train_df = events[events["event_time"] < train_end].copy()
    val_df = events[
        (events["event_time"] >= train_end) & (events["event_time"] < val_end)
    ].copy()
    test_df = events[events["event_time"] >= val_end].copy()

    visitor_ids = train_df["visitorid"].unique()
    item_ids = train_df["itemid"].unique()
    visitor_to_idx = {v: i for i, v in enumerate(visitor_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    n_users = len(visitor_ids)
    n_items = len(item_ids)

    train_df_local = train_df.assign(
        visitor_idx=train_df["visitorid"].map(visitor_to_idx),
        item_idx=train_df["itemid"].map(item_to_idx),
    )

    interaction_weights = (
        train_df_local.groupby(["visitor_idx", "item_idx"], observed=True)[
            "implicit_weight"
        ]
        .sum()
        .reset_index()
    )
    interaction_weights["implicit_weight"] = interaction_weights[
        "implicit_weight"
    ].clip(upper=50)

    sparse_matrix = sp.csr_matrix(
        (
            interaction_weights["implicit_weight"].values,
            (
                interaction_weights["visitor_idx"].values,
                interaction_weights["item_idx"].values,
            ),
        ),
        shape=(n_users, n_items),
    )
    print(f"  Matriz: {n_users:,} x {n_items:,}, NNZ: {sparse_matrix.nnz:,}")

    item_popularity = np.asarray(sparse_matrix.sum(axis=0)).ravel()
    item_log_popularity = np.log1p(item_popularity)
    most_popular_items = np.argsort(-item_popularity)

    # --- 3. Ground truth ---
    print("\n[3/4] Construindo ground truth...")

    def build_ground_truth(
        df: pd.DataFrame,
        v2i: dict,
        i2i: dict,
        min_weight: int = 1,
    ) -> dict[int, set[int]]:
        df_known = df[df["visitorid"].isin(v2i) & df["itemid"].isin(i2i)].copy()
        df_known["visitor_idx"] = df_known["visitorid"].map(v2i)
        df_known["item_idx"] = df_known["itemid"].map(i2i)
        ui = (
            df_known.groupby(["visitor_idx", "item_idx"], observed=True)[
                "implicit_weight"
            ]
            .sum()
            .reset_index()
        )
        ui = ui[ui["implicit_weight"] >= min_weight]
        gt: dict[int, set[int]] = defaultdict(set)
        for _, row in ui.iterrows():
            gt[int(row["visitor_idx"])].add(int(row["item_idx"]))
        return dict(gt)

    val_ground_truth = build_ground_truth(val_df, visitor_to_idx, item_to_idx)
    test_ground_truth = build_ground_truth(test_df, visitor_to_idx, item_to_idx)
    print(
        f"  Val: {len(val_ground_truth):,} usuarios, "
        f"Test: {len(test_ground_truth):,} usuarios"
    )

    # --- 4. EASE^ grid search ---
    print("\n[4/4] Avaliando EASE^...")
    all_results = {}

    for lam in EASE_LAMBDAS:
        print(f"\n{'=' * 60}")
        print(f"  EASE^ lambda={lam}")
        print(f"{'=' * 60}")

        t0 = time.time()
        b_mat, top_idx = train_ease(sparse_matrix, lam=lam, max_items=EASE_MAX_ITEMS)
        train_time = time.time() - t0
        print(f"  Treino total: {train_time:.1f}s")

        # Avaliar com multiplos popularity weights
        pw_list = [0.0, 0.3, 0.5, 1.0]
        results = evaluate_ease_batch(
            sparse_matrix,
            b_mat,
            top_idx,
            val_ground_truth,
            item_popularity,
            item_log_popularity,
            most_popular_items,
            max_eval_users=MAX_EVAL_USERS,
            pop_weights=pw_list,
            k_values=TOP_K,
        )

        for pw, metrics in results.items():
            name = f"EASE^(lam={lam},pw={pw})"
            recall_10 = metrics[10]["recall"]
            ndcg_10 = metrics[10]["ndcg"]
            all_results[name] = {"Recall@10": recall_10, "NDCG@10": ndcg_10}
            marker = " ***" if recall_10 > 0.01060 else ""
            print(f"  {name}: Recall@10={recall_10:.5f}, NDCG@10={ndcg_10:.5f}{marker}")

        eval_time = time.time() - t0
        print(f"  Lambda {lam} completo em {eval_time:.1f}s")

    # --- Resumo ---
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print("\nBaseline Item-KNN: Recall@10 = 0.01060")
    print("Baseline MostPopular: Recall@10 = 0.00292\n")

    for name, metrics in sorted(
        all_results.items(),
        key=lambda x: x[1]["Recall@10"],
        reverse=True,
    ):
        marker = " ***" if metrics["Recall@10"] > 0.01060 else ""
        print(
            f"  {name}: Recall@10={metrics['Recall@10']:.5f}, "
            f"NDCG@10={metrics['NDCG@10']:.5f}{marker}"
        )

    best_name = max(all_results, key=lambda k: all_results[k]["Recall@10"])
    best_recall = all_results[best_name]["Recall@10"]
    print(f"\nMelhor: {best_name} com Recall@10={best_recall:.5f}")

    if best_recall > 0.01060:
        print("\n*** SUCESSO: Superou Item-KNN baseline! ***")
    else:
        gap = (0.01060 - best_recall) / 0.01060 * 100
        print(f"\n*** Ainda nao superou Item-KNN. Gap: {gap:.1f}% ***")

    print(f"\nTempo total: {time.time() - t_start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
